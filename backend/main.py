#!/usr/bin/env python3
"""
main.py - phishing-URL detection API (FastAPI).

Endpoints
    POST /scan    body: {"url": "http://example.com/login"}
                  returns: {"verdict": "safe" | "suspicious" | "phishing", "score": 0.93,
                            "signals": [...], ...}
    GET  /health  model name, thresholds and allowlist size (for hosting health checks)

The server only analyses the TEXT of a URL. It never visits the address.

Folder layout (this file lives in backend/)
    backend/
      main.py
      requirements.txt
      ml/features.py            copy of Model/features.py (the version used in training)
      ml/best_model.joblib      copy of Model/models/tuned/best_model.joblib
      allowlist.txt             optional: one domain per line

Run locally
    cd backend
    pip install fastapi uvicorn pandas numpy scikit-learn joblib xgboost
    uvicorn main:app --reload
    then open http://127.0.0.1:8000/docs to try it in the browser

Settings (environment variables, all optional)
    ALLOWED_ORIGINS       comma-separated frontend addresses allowed by CORS
                          (default http://localhost:3000,http://localhost:5173; "*" allows any)
    RATE_LIMIT_PER_MIN    requests per client address per minute (default 60, 0 = off)
    TRUST_PROXY           "1" when running behind exactly ONE proxy (Render, Railway, nginx),
                          so the real client address is read from X-Forwarded-For
    PHISHING_THRESHOLD    score at or above which a URL is "phishing"
                          (default: the threshold saved with the model)
    SUSPICIOUS_THRESHOLD  score at or above which a URL is "suspicious"
                          (default: 0.6 x the phishing threshold)
    MODEL_PATH, ALLOWLIST_PATH   override the file locations
    LOG_URLS              "1" to log full URLs (default: log only the host name, for privacy)
    DISABLE_DOCS          "1" to switch off /docs and /redoc

Notes
    * Scores are per-process. If you run several workers, each keeps its own rate-limit counts.
    * The "signals" in a response are plain-language hints based on simple URL checks. They
      are NOT an exact explanation of why the model produced its score.
"""

import logging
import os
import re
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ml"
sys.path.insert(0, str(ML_DIR))
try:
    from features import build_feature_frame  # noqa: E402
except ImportError as exc:
    raise SystemExit(f"Cannot import features.py from {ML_DIR}: {exc}\n"
                     "Copy Model/features.py into backend/ml/ (and install pandas).")

log = logging.getLogger("phishing-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ------------------------------------------------------------------------------ settings
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ML_DIR / "best_model.joblib")))
ALLOWLIST_PATH = Path(os.getenv("ALLOWLIST_PATH", str(BASE_DIR / "allowlist.txt")))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if o.strip()]
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
TRUST_PROXY = os.getenv("TRUST_PROXY", "0") == "1"
LOG_URLS = os.getenv("LOG_URLS", "0") == "1"
DOCS_ON = os.getenv("DISABLE_DOCS", "0") != "1"
MAX_URL_LENGTH = 2048


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"{name} must be a number, got {raw!r}")


# ------------------------------------------------------------------------------ helpers
class RateLimiter:
    """Simple sliding-window limiter, kept in memory (one instance per process)."""

    def __init__(self, limit, window=60.0):
        self.limit, self.window = limit, window
        self.hits = {}
        self.lock = threading.Lock()

    def allow(self, key):
        """Return (allowed, seconds_to_wait)."""
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, int(self.window - (now - q[0])) + 1
            q.append(now)
            if len(self.hits) > 10_000:  # forget clients that went quiet
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.window]:
                    del self.hits[k]
            return True, 0


def client_ip(request):
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:  # the LAST entry is the one our own proxy added; earlier ones can be faked
            return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


_OTHER_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:(?!\d)", re.I)


def normalize_url(raw):
    """Validate the input. Returns (url, host) or raises ValueError with a friendly message."""
    url = raw.strip()
    if not url:
        raise ValueError("Please enter a URL.")
    if len(url) > MAX_URL_LENGTH:
        raise ValueError(f"URL is too long (limit {MAX_URL_LENGTH} characters).")
    if any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError("URL must not contain spaces or control characters.")
    if "://" not in url:
        if _OTHER_SCHEME_RE.match(url):
            raise ValueError("Only http:// and https:// URLs are supported.")
        url = "http://" + url  # the user typed 'example.com/login'
    try:
        parsed = urlparse(url)
        parsed.port  # noqa: B018 - raises ValueError if the port is invalid
    except ValueError:
        raise ValueError("That does not look like a valid URL.")
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Only http:// and https:// URLs are supported.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("The URL has no host name.")
    return url, host


def load_allowlist(path):
    """Read domains, one per line. Accepts '# comments', '*.example.com' and Tranco 'rank,domain'."""
    domains = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.split("#")[0].strip().lower()
            if "," in line:
                line = line.split(",")[-1].strip()
            line = line.lstrip("*.").strip(".")
            if line:
                domains.add(line)
    return domains


def is_allowlisted(host, allow):
    """True if the host, or any parent domain with at least two labels, is on the list."""
    if not allow:
        return False
    parts = host.split(".")
    return any(".".join(parts[i:]) in allow for i in range(len(parts) - 1))


# (feature name, minimum value that triggers it, message)
SIGNAL_RULES = [
    ("host_is_ip", 1, "The address is a raw IP number instead of a normal domain name."),
    ("has_userinfo", 1, "The URL contains '@', a trick used to hide the real destination."),
    ("has_punycode", 1, "The domain uses look-alike international characters."),
    ("non_ascii_host", 1, "The domain contains non-English characters that can imitate a real name."),
    ("brand_in_domain_not_exact", 1, "The domain imitates a well-known brand name."),
    ("brand_in_subdomain", 1, "A well-known brand name appears in the sub-domain, not in the real domain."),
    ("brand_in_path", 1, "A well-known brand name appears in the path of an unrelated site."),
    ("suspicious_tld", 1, "The domain ends in an extension that is often abused for phishing."),
    ("is_shortener", 1, "The link goes through a URL shortener that hides the real destination."),
    ("lure_token_in_host", 1, "Account or notice words (login, verify, payroll ...) appear in the domain name."),
    ("suspicious_word_in_host", 1, "Account or notice words (login, verify, payroll ...) appear in the domain name."),
    ("suspicious_word_count", 2, "Several account-related words (login, verify, secure ...) appear in the URL."),
    ("subdomain_count", 3, "The address has an unusually large number of sub-domains."),
    ("domain_hyphens", 2, "The domain name contains several hyphens."),
    ("url_length", 100, "The URL is unusually long."),
    ("last_seg_transitions", 5, "The end of the URL looks like a random tracking code."),
    ("nonstandard_port", 1, "The URL uses an unusual network port."),
]


def describe_signals(row, limit=5):
    """Turn a few feature values into short sentences a user can read."""
    out = []
    for name, minimum, message in SIGNAL_RULES:
        if row.get(name, 0) >= minimum and message not in out:
            out.append(message)
    return out[:limit]


def verdict_for(score, phishing_at, suspicious_at):
    if score >= phishing_at:
        return "phishing"
    if score >= suspicious_at:
        return "suspicious"
    return "safe"


# ---------------------------------------------------------------------------- schemas
class ScanRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=4096, description="The URL to check")


class ScanResponse(BaseModel):
    url: str
    host: str
    verdict: str
    score: Optional[float] = None
    allowlisted: bool = False
    signals: List[str] = []
    thresholds: Dict[str, float]
    model: str


# ------------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}. "
                           "Copy Model/models/tuned/best_model.joblib into backend/ml/.")
    bundle = joblib.load(MODEL_PATH)
    for key in ("model", "features"):
        if key not in bundle:
            raise RuntimeError(f"{MODEL_PATH} is not a model saved by train_baseline.py / "
                               f"tune_xgb.py (missing '{key}').")

    # Fail at start-up, not on the first request, if features.py and the model do not match.
    probe = build_feature_frame(["http://example.com/"])
    missing = [c for c in bundle["features"] if c not in probe.columns]
    if missing:
        raise RuntimeError("features.py does not produce the columns this model was trained on "
                           f"(missing e.g. {missing[:3]}). Copy features.py and best_model.joblib "
                           "from the same training run.")
    bundle["model"].predict_proba(probe[bundle["features"]])  # warm-up

    phishing_at = _env_float("PHISHING_THRESHOLD", float(bundle.get("threshold", 0.5)))
    suspicious_at = _env_float("SUSPICIOUS_THRESHOLD", round(phishing_at * 0.6, 3))
    if not 0 < suspicious_at <= phishing_at <= 1:
        raise RuntimeError("Thresholds must satisfy 0 < SUSPICIOUS <= PHISHING <= 1 "
                           f"(got {suspicious_at} and {phishing_at}).")

    app.state.bundle = bundle
    app.state.allow = load_allowlist(ALLOWLIST_PATH)
    app.state.limiter = RateLimiter(RATE_LIMIT)
    app.state.phishing_at = phishing_at
    app.state.suspicious_at = suspicious_at
    log.info("Loaded %s (%d features). suspicious >= %.3f, phishing >= %.3f, allowlist %d domains, "
             "rate limit %s/min", bundle.get("name", "model"), len(bundle["features"]),
             suspicious_at, phishing_at, len(app.state.allow), RATE_LIMIT or "off")
    yield


app = FastAPI(title="Phishing URL Detector", version="1.0.0", lifespan=lifespan,
              docs_url="/docs" if DOCS_ON else None, redoc_url="/redoc" if DOCS_ON else None,
              openapi_url="/openapi.json" if DOCS_ON else None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health(request: Request):
    st = request.app.state
    return {"status": "ok", "model": st.bundle.get("name", "model"),
            "features": len(st.bundle["features"]),
            "thresholds": {"suspicious": round(st.suspicious_at, 4), "phishing": round(st.phishing_at, 4)},
            "allowlist_size": len(st.allow)}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest, request: Request):
    st = request.app.state

    allowed, wait = st.limiter.allow(client_ip(request))
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.",
                            headers={"Retry-After": str(wait)})
    try:
        url, host = normalize_url(req.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    common = dict(url=url, host=host, model=st.bundle.get("name", "model"),
                  thresholds={"suspicious": round(st.suspicious_at, 4),
                              "phishing": round(st.phishing_at, 4)})

    if is_allowlisted(host, st.allow):
        log.info("scan %s -> safe (allowlisted)", url if LOG_URLS else host)
        return ScanResponse(verdict="safe", allowlisted=True, **common)

    try:
        frame = build_feature_frame([url])
        score = float(st.bundle["model"].predict_proba(frame[st.bundle["features"]])[0, 1])
    except Exception:
        log.exception("scoring failed for %s", host)
        raise HTTPException(status_code=500, detail="Could not score this URL.")

    verdict = verdict_for(score, st.phishing_at, st.suspicious_at)
    signals = describe_signals(frame.iloc[0].to_dict()) if verdict != "safe" else []
    if verdict != "safe" and not signals:
        signals = ["No single obvious warning sign; the score comes from a combination of URL patterns."]
    log.info("scan %s -> %s (%.3f)", url if LOG_URLS else host, verdict, score)
    return ScanResponse(verdict=verdict, score=round(score, 4), signals=signals, **common)