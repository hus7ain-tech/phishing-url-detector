import joblib, pandas as pd
from sklearn.metrics import roc_curve
from features import build_feature_frame

bundle = joblib.load("models/best_model.joblib")
val = pd.read_csv("data/processed/val.csv")
X = build_feature_frame(val["url"])[bundle["features"]]
proba = bundle["model"].predict_proba(X)[:, 1]

fpr, tpr, thr = roc_curve(val["label"], proba)
for target in (0.05, 0.02, 0.01, 0.005):
    i = (fpr <= target).nonzero()[0][-1]
    print(f"false alarms <= {target:.1%}: threshold {thr[i]:.3f}, catches {tpr[i]:.1%} of phishing")