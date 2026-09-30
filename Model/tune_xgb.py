#!/usr/bin/env python3
"""
tune_xgb.py - search for better XGBoost settings, judged the way a phishing
detector should be judged: how much phishing is CAUGHT while legitimate sites are
wrongly flagged only rarely (default: at most 1% false alarms).

Reads    data/processed/train.csv and data/processed/val.csv   (never test.csv)
Tries    several XGBoost settings (the first one is your current baseline). Each
         run uses early stopping, so the number of trees is chosen automatically.
Prints   a ranked table of the runs, then a threshold table for the best one
Saves    models/tuned/best_model.joblib   the best model + the chosen threshold
         models/tuned/tuning_results.csv  every run

Must sit in the same folder as features.py and train_baseline.py.

Usage
    python tune_xgb.py                       # 15 runs, target 1% false alarms
    python tune_xgb.py --trials 5            # quick test
    python tune_xgb.py --trials 30 --target-fpr 0.005
    python tune_xgb.py --exclude             # keep all features (is_https too)

Score a URL with the tuned model:
    python train_baseline.py --model-dir models/tuned --predict "http://example.com/login"

Install
    pip install pandas numpy scikit-learn joblib xgboost
"""

import argparse
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit("xgboost is not installed. Run: pip install xgboost")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES  # noqa: E402
from train_baseline import load_split, log  # noqa: E402

# Settings to sample from (see the table in the chat for what each one does).
SPACE = {
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": [0.03, 0.05, 0.08, 0.1],
    "min_child_weight": [1, 3, 5, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.8, 1.0],
    "reg_lambda": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.5, 1],
}

# The settings train_baseline.py used, so you can see whether tuning really helps.
BASELINE = {"max_depth": 6, "learning_rate": 0.1, "min_child_weight": 1, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_lambda": 1, "gamma": 0}

TARGETS = (0.05, 0.02, 0.01, 0.005)


# --------------------------------------------------------------------------- helpers
def short(params):
    return (f"depth {params['max_depth']}, lr {params['learning_rate']}, "
            f"mcw {params['min_child_weight']}, sub {params['subsample']}, "
            f"col {params['colsample_bytree']}, lambda {params['reg_lambda']}, "
            f"gamma {params['gamma']}")


def at_fpr(y, proba, target_fpr):
    """Recall and threshold at the strictest point where false alarms stay <= target."""
    fpr, tpr, thr = roc_curve(y, proba)
    i = np.where(fpr <= target_fpr)[0][-1]
    return float(tpr[i]), float(min(thr[i], 1.0)), float(fpr[i])


def evaluate(y, proba, target_fpr):
    recall, threshold, _ = at_fpr(y, proba, target_fpr)
    pred = (proba >= 0.5).astype(int)
    negatives = max(int((y == 0).sum()), 1)
    return {
        "recall_at_fpr": recall,
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
        "fpr_at_0.5": float(((pred == 1) & (y == 0)).sum() / negatives),
    }


def make_configs(n, seed):
    rng = random.Random(seed)
    configs, seen = [dict(BASELINE)], {tuple(sorted(BASELINE.items()))}
    tries = 0
    while len(configs) < n and tries < n * 50:
        tries += 1
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Tune XGBoost for a phishing-URL detector.")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--val", default="data/processed/val.csv")
    ap.add_argument("--model-dir", default="models/tuned")
    ap.add_argument("--trials", type=int, default=15, help="how many settings to try (default 15)")
    ap.add_argument("--target-fpr", type=float, default=0.01,
                    help="highest acceptable share of legit sites flagged (default 0.01 = 1%%)")
    ap.add_argument("--exclude", nargs="*", default=["is_https"],
                    help="features to leave out (default: is_https); '--exclude' alone keeps all")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    unknown = [f for f in args.exclude if f not in FEATURE_NAMES]
    if unknown:
        sys.exit(f"Unknown feature name(s) in --exclude: {unknown}")
    cols = [f for f in FEATURE_NAMES if f not in args.exclude]

    log("Extracting features...")
    _, X_tr, y_tr = load_split(args.train, args.workers, args.max_rows, args.seed)
    _, X_va, y_va = load_split(args.val, args.workers, args.max_rows, args.seed)
    X_tr, X_va = X_tr[cols], X_va[cols]
    if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
        sys.exit("Both train and val need phishing AND legitimate rows.")
    pos, neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    log(f"  train {len(X_tr):,} rows, val {len(X_va):,} rows, {len(cols)} features")
    log(f"  goal: catch as much phishing as possible with at most "
        f"{args.target_fpr:.1%} of legitimate sites flagged\n")

    configs = make_configs(args.trials, args.seed)
    rows, models, probas = [], [], []
    for n, cfg in enumerate(configs, 1):
        t0 = time.time()
        model = XGBClassifier(
            n_estimators=2000, early_stopping_rounds=50, eval_metric="aucpr",
            scale_pos_weight=neg / max(pos, 1), n_jobs=-1, random_state=args.seed, **cfg)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        proba = model.predict_proba(X_va)[:, 1]
        res = evaluate(y_va, proba, args.target_fpr)
        res.update(trial=n, trees=int(getattr(model, "best_iteration", 0) or 0) + 1,
                   seconds=round(time.time() - t0, 1), settings=short(cfg),
                   baseline=(n == 1))
        rows.append(res)
        models.append(model)
        probas.append(proba)
        log(f"  run {n:>2}/{len(configs)}: catches {res['recall_at_fpr']:.1%} of phishing at "
            f"{args.target_fpr:.1%} false alarms  (PR-AUC {res['pr_auc']:.4f}, "
            f"{res['trees']} trees, {res['seconds']}s)" + ("  <- baseline" if n == 1 else ""))

    results = pd.DataFrame(rows)
    ranked = results.sort_values(["recall_at_fpr", "pr_auc"], ascending=False).reset_index(drop=True)
    best_row = ranked.iloc[0]
    best_idx = int(best_row["trial"]) - 1
    base = results.iloc[0]

    # ---- ranked table
    show = ranked.head(8)[["trial", "recall_at_fpr", "pr_auc", "roc_auc", "f1_at_0.5",
                           "fpr_at_0.5", "trees", "settings"]].copy()
    for c in ("recall_at_fpr", "fpr_at_0.5"):
        show[c] = show[c].map("{:.2%}".format)
    for c in ("pr_auc", "roc_auc", "f1_at_0.5"):
        show[c] = show[c].map("{:.4f}".format)
    show = show.rename(columns={"recall_at_fpr": f"catch@{args.target_fpr:.1%}fp",
                                "fpr_at_0.5": "fp_rate@0.5"})
    log("\n=== Best runs on validation (ranked by phishing caught at the false-alarm limit) ===")
    log(show.to_string(index=False))
    log(f"\nBaseline (run 1): catches {base['recall_at_fpr']:.1%}  ->  "
        f"best (run {int(best_row['trial'])}): catches {best_row['recall_at_fpr']:.1%}")
    if best_idx == 0:
        log("None of the tried settings beat the baseline. Try more runs (--trials 30), "
            "or improve the data and features instead.")

    # ---- threshold table for the best model
    log("\n=== Choosing the cut-off for the best model ===")
    log(f"{'false alarms allowed':<22}{'threshold':>10}{'phishing caught':>18}")
    for t in TARGETS:
        rec, thr, actual = at_fpr(y_va, probas[best_idx], t)
        log(f"{t:<22.1%}{thr:>10.3f}{rec:>18.1%}")

    # ---- save
    threshold = float(best_row["threshold"])
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": models[best_idx], "features": cols, "name": "XGBoost (tuned)",
                 "threshold": threshold, "excluded": list(args.exclude)},
                out / "best_model.joblib")
    results.drop(columns=["baseline"]).to_csv(out / "tuning_results.csv", index=False)
    log(f"\nSaved {out}/best_model.joblib with threshold {threshold:.3f} "
        f"(URLs scoring at or above this are flagged).")
    log("Note: the settings and the threshold were both chosen on val.csv, so the real-world "
        "numbers will be slightly worse. test.csv has NOT been used.")


if __name__ == "__main__":
    main()