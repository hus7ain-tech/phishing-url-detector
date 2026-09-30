#!/usr/bin/env python3
"""
collect_data.py - download and label data for a phishing-URL detector.

What it does
  1. Downloads phishing URL feeds (OpenPhish, Phishing.Database, optionally PhishTank).
     It only downloads the TEXT of the addresses. It never visits them.
  2. Downloads the Tranco top-sites list and builds legitimate URLs by reading the
     public homepages of a random sample of those sites and collecting their links.
  3. Saves raw downloads (dated) to data/raw/ and merges everything into ONE file:
        data/processed/dataset.csv   columns: url, label, date_collected, source
        label: 1 = phishing, 0 = legitimate

Run it again every few days: new phishing URLs are appended and old ones keep their
original collection date (needed for a time-based train/test split later).

Install:
    pip install requests pandas beautifulsoup4

Usage:
    python collect_data.py
    python collect_data.py --legit-domains 8000 --links-per-site 10 --balance
    python collect_data.py --skip-legit-crawl          # phishing feeds + homepages only
    PHISHTANK_KEY=yourkey python collect_data.py       # also use PhishTank
"""

import argparse
import io
import os
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Put a real contact address here. It is polite and some sites require it.
USER_AGENT = "phishing-research-bot/1.0 (contact: you@example.com)"
HEADERS = {"User-Agent": USER_AGENT}
TODAY = date.today().isoformat()

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

OPENPHISH_URL = "https://openphish.com/feed.txt"
PHISHING_DB_URL = (
    "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/"
    "master/phishing-links-ACTIVE.txt"
)
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


# --------------------------------------------------------------------------- helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_urls(urls):
    """Keep only sane http(s) URLs, strip whitespace, remove duplicates."""
    seen, out = set(), []
    for u in urls:
        u = u.strip()
        if not u.lower().startswith(("http://", "https://")):
            continue
        if len(u) > 2000 or " " in u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download_text(url, raw_name, timeout=120, headers=None):
    """Download a text file, save a dated copy in data/raw/, return the text."""
    r = requests.get(url, headers=headers or HEADERS, timeout=timeout)
    r.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{raw_name}_{TODAY}.txt").write_text(r.text, encoding="utf-8")
    return r.text


# ----------------------------------------------------------------- phishing sources
def get_openphish():
    text = download_text(OPENPHISH_URL, "openphish")
    return clean_urls(text.splitlines())


def get_phishing_database():
    text = download_text(PHISHING_DB_URL, "phishing_database", timeout=300)
    return clean_urls(text.splitlines())


def get_phishtank():
    key = os.environ.get("PHISHTANK_KEY")
    if not key:
        log("PhishTank skipped (set PHISHTANK_KEY to enable it).")
        return []
    url = f"http://data.phishtank.com/data/{key}/online-valid.csv"
    text = download_text(url, "phishtank", timeout=300,
                         headers={"User-Agent": "phishtank/research-project"})
    df = pd.read_csv(io.StringIO(text))
    return clean_urls(df["url"].astype(str).tolist())


def collect_phishing():
    sources = {
        "openphish": get_openphish,
        "phishing_database": get_phishing_database,
        "phishtank": get_phishtank,
    }
    frames = []
    for name, fn in sources.items():
        try:
            urls = fn()
            log(f"{name}: {len(urls):,} phishing URLs")
            if urls:
                frames.append(pd.DataFrame({"url": urls, "source": name}))
        except Exception as e:  # one broken source must not stop the run
            log(f"{name}: FAILED ({e})")
    if not frames:
        return pd.DataFrame(columns=["url", "source"])
    df = pd.concat(frames).drop_duplicates("url")
    df["label"] = 1
    return df


# --------------------------------------------------------------- legitimate sources
def get_tranco_domains(top_n=100_000):
    log("Downloading Tranco list...")
    r = requests.get(TRANCO_URL, headers=HEADERS, timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = z.namelist()[0]
        text = z.read(name).decode("utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"tranco_{TODAY}.csv").write_text(text, encoding="utf-8")
    df = pd.read_csv(io.StringIO(text), header=None, names=["rank", "domain"], nrows=top_n)
    return df["domain"].astype(str).tolist()


def crawl_homepage(domain, links_per_site, delay):
    """Fetch a site's public homepage and return the homepage URL plus a few internal links."""
    home = f"https://{domain}/"
    found = [home]
    try:
        r = requests.get(home, headers=HEADERS, timeout=6, allow_redirects=True)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code == 200 and "html" in ctype:
            soup = BeautifulSoup(r.text[:500_000], "html.parser")
            base_host = urlparse(r.url).netloc.lower()
            links = set()
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("mailto:", "javascript:", "tel:", "#")):
                    continue
                full = urljoin(r.url, href).split("#")[0]
                p = urlparse(full)
                if p.scheme in ("http", "https") and p.netloc.lower() == base_host and p.path not in ("", "/"):
                    links.add(full)
            found += random.sample(sorted(links), min(links_per_site, len(links)))
    except Exception:
        pass  # dead site, timeout, TLS problem: keep just the homepage
    time.sleep(delay)
    return found


def collect_legit(n_domains, links_per_site, workers, skip_crawl, seed=42):
    domains = get_tranco_domains()
    random.seed(seed)
    sample = random.sample(domains, min(n_domains, len(domains)))
    log(f"Legitimate: using {len(sample):,} domains sampled from the Tranco top 100k")

    urls = []
    if skip_crawl:
        urls = [f"https://{d}/" for d in sample]
    else:
        log("Crawling homepages for internal links (this is the slow part)...")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(crawl_homepage, d, links_per_site, 0.5) for d in sample]
            for f in as_completed(futures):
                urls += f.result()
                done += 1
                if done % 250 == 0:
                    log(f"  crawled {done:,}/{len(sample):,} sites, {len(urls):,} URLs so far")

    urls = clean_urls(urls)
    log(f"tranco: {len(urls):,} legitimate URLs")
    df = pd.DataFrame({"url": urls})
    df["source"] = "tranco"
    df["label"] = 0
    return df


# ------------------------------------------------------------------------- merging
def merge_and_save(new_df, balance):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PROC_DIR / "dataset.csv"

    new_df = new_df.copy()
    new_df["date_collected"] = TODAY

    if master_path.exists():
        old = pd.read_csv(master_path)
        log(f"Existing dataset found: {len(old):,} rows")
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df

    # If a URL is in both classes, trust the phishing label. Otherwise keep the
    # earliest row so date_collected stays the first day we saw the URL.
    combined = combined.sort_values("label", ascending=False, kind="stable")
    combined = combined.drop_duplicates("url", keep="first")
    combined = combined[["url", "label", "date_collected", "source"]].reset_index(drop=True)
    combined.to_csv(master_path, index=False)

    counts = combined["label"].value_counts().to_dict()
    log(f"Saved {master_path}: {len(combined):,} rows "
        f"(phishing={counts.get(1, 0):,}, legitimate={counts.get(0, 0):,})")

    if balance:
        n = min(counts.get(0, 0), counts.get(1, 0))
        if n == 0:
            log("Cannot balance: one class is empty.")
            return
        balanced = pd.concat([
            combined[combined.label == 1].sample(n, random_state=42),
            combined[combined.label == 0].sample(n, random_state=42),
        ]).sample(frac=1, random_state=42)
        out = PROC_DIR / "dataset_balanced.csv"
        balanced.to_csv(out, index=False)
        log(f"Saved {out}: {len(balanced):,} rows ({n:,} per class)")


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Download and label phishing-detection data.")
    ap.add_argument("--legit-domains", type=int, default=4000,
                    help="how many Tranco domains to sample (default 4000)")
    ap.add_argument("--links-per-site", type=int, default=10,
                    help="max internal links kept per site (default 10)")
    ap.add_argument("--workers", type=int, default=16, help="parallel crawl workers")
    ap.add_argument("--skip-legit-crawl", action="store_true",
                    help="use bare homepages only, no crawling (fast, less realistic)")
    ap.add_argument("--phish-only", action="store_true", help="only refresh phishing feeds")
    ap.add_argument("--balance", action="store_true",
                    help="also write dataset_balanced.csv with equal class counts")
    args = ap.parse_args()

    parts = [collect_phishing()]
    if not args.phish_only:
        parts.append(collect_legit(args.legit_domains, args.links_per_site,
                                   args.workers, args.skip_legit_crawl))
    new_df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if new_df.empty:
        log("Nothing downloaded. Check your internet connection.")
        return
    merge_and_save(new_df, args.balance)
    log("Done. NEVER open the phishing URLs in your normal browser.")


if __name__ == "__main__":
    main()