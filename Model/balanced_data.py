#!/usr/bin/env python3
"""
balance_data.py - turn the raw, lopsided dataset into a balanced, leak-free
train / validation / test set.

Reads:   data/processed/dataset.csv          (from collect_data.py)
         columns: url, label, date_collected, source     label: 1 = phishing, 0 = legit
Writes:  data/processed/train.csv
         data/processed/val.csv
         data/processed/test.csv
         data/processed/dataset_balanced.csv  (all three parts together)

Steps
  1. Remove duplicate URLs and rows without a usable host name.
  2. Cap the number of URLs per host (phishing lists contain thousands of URLs on
     the same few hosts; keeping them all just teaches the model those hosts).
  3. Downsample so the classes match the requested ratio (default 50/50).
  4. Split into train / val / test (default 70 / 15 / 15):
       - phishing: by DATE (oldest -> train, newest -> test) when the data spans
         at least 3 different collection dates, otherwise by host.
       - legitimate: by host (they have no natural date).
     Splitting by host means the same website never appears on both sides, which
     would make the test score look better than it really is.

Usage:
    python balance_data.py
    python balance_data.py --max-phish 100000 --max-per-host 5
    python balance_data.py --ratio 0.5          # 1 phishing for every 2 legit
    python balance_data.py --split random      # force host-based split (no dates)

Install:
    pip install pandas numpy
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

COLUMNS = ["url", "label", "date_collected", "source"]


# --------------------------------------------------------------------------- helpers
def log(msg):
    print(msg, flush=True)


def get_host(url):
    """Return the lower-case host name of a URL ('' if it cannot be parsed)."""
    try:
        return urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    except Exception:
        return ""


def cap_per_host(df, max_per_host, seed):
    """Keep at most `max_per_host` randomly chosen rows for every host."""
    df = df.sample(frac=1, random_state=seed)
    return df[df.groupby("host").cumcount() < max_per_host]


def split_by_host(df, val_frac, test_frac, seed):
    """Random split where every host lands entirely in one part."""
    rng = np.random.default_rng(seed)
    counts = df["host"].value_counts()
    hosts = counts.index.to_numpy().copy()
    rng.shuffle(hosts)
    cum = np.cumsum(counts.loc[hosts].to_numpy()) / len(df)
    part = np.where(cum <= 1 - val_frac - test_frac, 0,
                    np.where(cum <= 1 - test_frac, 1, 2))
    lookup = dict(zip(hosts, part))
    which = df["host"].map(lookup)
    return df[which == 0], df[which == 1], df[which == 2]


def split_by_time(df, val_frac, test_frac, seed):
    """Oldest rows -> train, next -> val, newest -> test."""
    df = df.sample(frac=1, random_state=seed).sort_values("date_collected", kind="stable")
    n = len(df)
    a = int(n * (1 - val_frac - test_frac))
    b = int(n * (1 - test_frac))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def date_span(df):
    d = df["date_collected"].dropna().astype(str)
    d = d[d != ""]
    return f"{d.min()} .. {d.max()}" if len(d) else "-"


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Balance and split the phishing dataset.")
    ap.add_argument("--input", default="data/processed/dataset.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--max-per-host", type=int, default=5,
                    help="max phishing URLs kept per host (default 5)")
    ap.add_argument("--max-per-host-legit", type=int, default=20,
                    help="max legitimate URLs kept per host (default 20)")
    ap.add_argument("--max-phish", type=int, default=100_000,
                    help="upper limit on phishing rows kept (default 100000)")
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="phishing rows per legitimate row (default 1.0 = 50/50)")
    ap.add_argument("--val", type=float, default=0.15, help="validation share (default 0.15)")
    ap.add_argument("--test", type=float, default=0.15, help="test share (default 0.15)")
    ap.add_argument("--split", choices=["auto", "time", "random"], default="auto",
                    help="how to split phishing rows (default auto)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not Path(args.input).exists():
        sys.exit(f"Input file not found: {args.input}  (run collect_data.py first)")
    if args.val + args.test >= 0.9 or args.ratio <= 0:
        sys.exit("Invalid --val/--test/--ratio values.")

    # ---- load and clean
    df = pd.read_csv(args.input)
    for col in ("url", "label"):
        if col not in df.columns:
            sys.exit(f"Input file has no '{col}' column.")
    if "date_collected" not in df.columns:
        df["date_collected"] = ""
    if "source" not in df.columns:
        df["source"] = ""
    df = df.dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates("url")
    df["host"] = df["url"].map(get_host)
    df = df[df["host"] != ""]

    phish = df[df.label == 1]
    legit = df[df.label == 0]
    log(f"Loaded: {len(df):,} unique URLs  (phishing {len(phish):,}, legitimate {len(legit):,})")
    if len(phish) == 0 or len(legit) == 0:
        sys.exit("Need both phishing and legitimate rows. Run collect_data.py again.")

    # ---- cap per host
    phish = cap_per_host(phish, args.max_per_host, args.seed)
    legit = cap_per_host(legit, args.max_per_host_legit, args.seed)
    log(f"After per-host cap: phishing {len(phish):,} ({phish.host.nunique():,} hosts), "
        f"legitimate {len(legit):,} ({legit.host.nunique():,} hosts)")

    # ---- balance to the requested ratio
    n_phish = min(len(phish), args.max_phish, int(len(legit) * args.ratio))
    n_legit = min(len(legit), int(round(n_phish / args.ratio)))
    if n_phish == 0 or n_legit == 0:
        sys.exit("Not enough data left after capping. Collect more legitimate URLs.")
    phish = phish.sample(n_phish, random_state=args.seed)
    legit = legit.sample(n_legit, random_state=args.seed)
    log(f"Balanced: phishing {n_phish:,}, legitimate {n_legit:,}")
    if n_legit < 50_000:
        log("  NOTE: fewer than 50,000 legitimate rows. Run collect_data.py with a larger "
            "--legit-domains for a stronger model.")

    # ---- split
    n_dates = phish["date_collected"].astype(str).replace("", np.nan).dropna().nunique()
    use_time = args.split == "time" or (args.split == "auto" and n_dates >= 3)
    if args.split == "time" and n_dates < 2:
        sys.exit("--split time needs phishing rows from at least 2 different dates.")
    if use_time:
        log(f"Splitting phishing by date ({n_dates} distinct dates); legitimate by host.")
        p_parts = split_by_time(phish, args.val, args.test, args.seed)
    else:
        log(f"Splitting phishing by host (only {n_dates} distinct collection date(s)). "
            "Re-run collect_data.py over a few weeks to enable the time-based split.")
        p_parts = split_by_host(phish, args.val, args.test, args.seed)
    l_parts = split_by_host(legit, args.val, args.test, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["train", "val", "test"]
    parts = {}
    for name, p, l in zip(names, p_parts, l_parts):
        part = pd.concat([p, l]).sample(frac=1, random_state=args.seed)
        parts[name] = part
        part[COLUMNS].to_csv(out_dir / f"{name}.csv", index=False)
    full = pd.concat(parts.values()).sample(frac=1, random_state=args.seed)
    full[COLUMNS].to_csv(out_dir / "dataset_balanced.csv", index=False)

    # ---- summary
    log("\nSplit summary")
    log(f"{'part':<6}{'rows':>10}{'phishing':>10}{'legit':>10}{'phish %':>9}   phishing dates")
    for name in names:
        part = parts[name]
        ph = int((part.label == 1).sum())
        lg = int((part.label == 0).sum())
        share = 100 * ph / max(len(part), 1)
        log(f"{name:<6}{len(part):>10,}{ph:>10,}{lg:>10,}{share:>8.1f}%   "
            f"{date_span(part[part.label == 1])}")

    leak = len(set(parts["train"].host) & set(parts["test"].host))
    log(f"\nHosts shared between train and test: {leak:,}"
        + ("  (expected to be small; phishing kits reuse some hosts over time)" if use_time else ""))
    log(f"Files written to {out_dir}/: train.csv, val.csv, test.csv, dataset_balanced.csv")


if __name__ == "__main__":
    main()