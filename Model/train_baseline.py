#!/usr/bin/env python3
"""
train_baseline.py - train a first phishing-URL classifier and see how well it does.

Reads    data/processed/train.csv and data/processed/val.csv   (from balance_data.py)
Trains   Logistic Regression, Random Forest, XGBoost (or a scikit-learn substitute
         if xgboost is not installed)
Prints   a comparison table measured on the VALIDATION set, the most important
         features, and the URLs the best model got wrong
Saves    models/best_model.joblib   the winning model
         models/metrics.json        all scores
         models/val_errors.csv      every validation mistake, most confident first

It NEVER touches test.csv. Keep that file for the very last step.

Usage
    python train_baseline.py
    python train_baseline.py --max-rows 20000          # quick run on a smaller sample
    python train_baseline.py --exclude                 # keep ALL features (is_https too)
    python train_baseline.py --exclude is_https url_length
    python train_baseline.py --predict "http://paypa1-secure-login.xyz/verify.php" "https://www.wikipedia.org/"

Install
    pip install pandas numpy scikit-learn joblib xgboost
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional
    XGBClassifier = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES, build_feature_frame  # noqa: E402


# --------------------------------------------------------------------------- helpers
def log(msg=""):
    print(msg, flush=True)


def defang(url):
    """Print-safe version of a URL (http -> hxxp) so nobody clicks it by accident."""
    return re.sub(r"^http", "hxxp", str(url), flags=re.I)


def load_split(path, workers, max_rows, seed):
    if not Path(path).exists():
        sys.exit(f"File not found: {path}\nRun collect_data.py and balance_data.py first.")
    df = pd.read_csv(path).dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    return df, build_feature_frame(df["url"], workers), df["label"].to_numpy()


def score(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": auc,
        "false_pos_rate": fp / max(fp + tn, 1),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }, proba, pred


def build_models(y_train, seed):
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=seed),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=neg / pos, eval_metric="logloss",
            n_jobs=-1, random_state=seed)
    else:
        log("xgboost is not installed (pip install xgboost); using scikit-learn's "
            "HistGradientBoosting instead.")
        models["Gradient Boosting (sklearn)"] = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed)
    return models


def predict_mode(urls, model_dir):
    path = Path(model_dir) / "best_model.joblib"
    if not path.exists():
        sys.exit(f"No saved model at {path}. Train first: python train_baseline.py")
    bundle = joblib.load(path)
    X = build_feature_frame(urls)[bundle["features"]]
    proba = bundle["model"].predict_proba(X)[:, 1]
    log(f"Model: {bundle['name']}\n")
    for u, p in zip(urls, proba):
        verdict = "PHISHING  " if p >= bundle["threshold"] else "looks safe"
        log(f"{p:7.1%} phishing   {verdict}   {defang(u)}")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Train baseline phishing-URL models.")
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--val", default="data/processed/val.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--exclude", nargs="*", default=["is_https"],
                    help="features to leave out (default: is_https). Use '--exclude' "
                         "with nothing after it to keep every feature.")
    ap.add_argument("--max-rows", type=int, default=0, help="use at most this many rows per file")
    ap.add_argument("--workers", type=int, default=1, help="processes for feature extraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict", nargs="+", metavar="URL",
                    help="score URLs with the saved model instead of training")
    args = ap.parse_args()

    if args.predict:
        predict_mode(args.predict, args.model_dir)
        return

    unknown = [f for f in args.exclude if f not in FEATURE_NAMES]
    if unknown:
        sys.exit(f"Unknown feature name(s) in --exclude: {unknown}")
    cols = [f for f in FEATURE_NAMES if f not in args.exclude]

    # ---- features
    t0 = time.time()
    log("Extracting features...")
    tr_df, X_tr, y_tr = load_split(args.train, args.workers, args.max_rows, args.seed)
    va_df, X_va, y_va = load_split(args.val, args.workers, args.max_rows, args.seed)
    X_tr, X_va = X_tr[cols], X_va[cols]
    log(f"  train: {len(X_tr):,} rows (phishing {int((y_tr == 1).sum()):,}, legit {int((y_tr == 0).sum()):,})")
    log(f"  val:   {len(X_va):,} rows (phishing {int((y_va == 1).sum()):,}, legit {int((y_va == 0).sum()):,})")
    log(f"  {len(cols)} features used"
        + (f", excluded: {', '.join(args.exclude)}" if args.exclude else "")
        + f"  ({time.time() - t0:.0f}s)")
    if len(set(y_tr)) < 2 or len(set(y_va)) < 2:
        sys.exit("Both train and val need phishing AND legitimate rows.")

    # ---- train and compare
    models, results, fitted = build_models(y_tr, args.seed), {}, {}
    for name, model in models.items():
        t1 = time.time()
        log(f"Training {name}...")
        model.fit(X_tr, y_tr)
        secs = time.time() - t1
        val_scores, proba, pred = score(model, X_va, y_va)
        train_scores, _, _ = score(model, X_tr, y_tr)
        val_scores.update(train_f1=train_scores["f1"], seconds=round(secs, 1))
        results[name], fitted[name] = val_scores, (model, proba, pred)

    table = pd.DataFrame(results).T
    show = table[["accuracy", "precision", "recall", "f1", "roc_auc", "false_pos_rate", "train_f1", "seconds"]].copy()
    for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "train_f1"]:
        show[c] = show[c].astype(float).map("{:.4f}".format)
    show["false_pos_rate"] = show["false_pos_rate"].astype(float).map("{:.2%}".format)
    log("\n=== Validation results (val.csv) ===")
    log(show.to_string())
    log("\nHow to read this:"
        "\n  precision      of the URLs flagged as phishing, how many really were"
        "\n  recall         of all real phishing URLs, how many were caught"
        "\n  false_pos_rate share of legitimate sites wrongly flagged (keep this LOW)"
        "\n  train_f1       if far above the val f1, the model is memorising (overfitting)")

    best = table["f1"].astype(float).idxmax()
    model, proba, pred = fitted[best]
    r = results[best]
    log(f"\nBest model: {best}  (F1 {r['f1']:.4f})")
    log(f"Confusion matrix on val:  caught phishing {r['tp']:,} | missed phishing {r['fn']:,} | "
        f"false alarms {r['fp']:,} | correct legit {r['tn']:,}")

    # ---- feature importance
    for name in (best, "Random Forest"):
        est = fitted[name][0]
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=cols).sort_values(ascending=False)
            log(f"\nTop 15 features ({name}):")
            log(imp.head(15).round(4).to_string())
            log("\nTip: if one feature dominates, check it is a real phishing signal and not a "
                "quirk of how the data was collected (e.g. all legitimate URLs coming from one "
                "kind of source).")
            break

    # ---- errors
    errors = va_df.assign(proba=proba, pred=pred)
    errors = errors[errors["pred"] != errors["label"]].copy()
    errors["error_type"] = np.where(errors["label"] == 0, "false_alarm (legit flagged)",
                                    "missed_phishing")
    errors = pd.concat([
        errors[errors["label"] == 0].sort_values("proba", ascending=False),
        errors[errors["label"] == 1].sort_values("proba", ascending=True),
    ])
    for label, title in ((0, "Legitimate sites wrongly flagged (most confident first)"),
                         (1, "Phishing URLs the model missed (most confident miss first)")):
        sub = errors[errors["label"] == label].head(10)
        log(f"\n{title}:")
        if sub.empty:
            log("  none")
        for _, row in sub.iterrows():
            log(f"  {row['proba']:.2f}  {defang(row['url'])[:110]}")

    # ---- save
    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": cols, "name": best, "threshold": 0.5,
                 "excluded": list(args.exclude)}, out / "best_model.joblib")
    errors[["url", "label", "proba", "error_type"]].to_csv(out / "val_errors.csv", index=False)
    with open(out / "metrics.json", "w") as fh:
        json.dump({"best": best, "features": cols, "excluded": list(args.exclude),
                   "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
                   "results": {k: {m: float(v) for m, v in d.items()} for k, d in results.items()}},
                  fh, indent=2)
    log(f"\nSaved to {out}/: best_model.joblib, metrics.json, val_errors.csv")
    log("test.csv has NOT been used. Keep it for the final check.")


if __name__ == "__main__":
    main()