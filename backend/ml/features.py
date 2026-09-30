#!/usr/bin/env python3
"""
features.py - turn a URL string into a row of numbers a model can learn from.

Every feature here is computed from the URL text alone. Nothing is downloaded and
no website is visited, so it is safe and fast (tens of thousands of URLs per second).

Use it from other scripts:
    from features import extract_features, build_feature_frame
    extract_features("http://paypa1-secure-login.xyz/verify.php")   # -> dict
    build_feature_frame(list_of_urls)                               # -> pandas DataFrame

Or try it directly on a few URLs:
    python features.py "http://paypa1-secure-login.xyz/verify.php" "https://www.wikipedia.org/"

Install:
    pip install pandas
"""

import math
import re
import sys
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

# ------------------------------------------------------------------------ word lists
# Brands that phishing pages love to imitate (kept to 4+ letters to avoid false matches).
BRANDS = [
    "paypal", "apple", "icloud", "google", "gmail", "microsoft", "outlook", "office365",
    "onedrive", "amazon", "netflix", "facebook", "instagram", "whatsapp", "linkedin",
    "twitter", "dropbox", "adobe", "docusign", "dhl", "fedex", "usps", "wellsfargo",
    "bankofamerica", "citibank", "hsbc", "barclays", "coinbase", "binance", "metamask",
    "spotify", "ebay", "alibaba", "telegram", "paytm", "phonepe", "hdfc", "icici",
    "irctc", "flipkart", "airtel",
]

# Words that show up in phishing URLs far more often than in normal ones.
SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "secure", "security",
    "account", "update", "confirm", "password", "passwd", "banking", "wallet",
    "support", "billing", "invoice", "suspend", "unlock", "recover", "alert",
    "authenticate", "validate", "webscr",
]

# Cheap or free top-level domains that are heavily abused.
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "icu", "click", "buzz", "work",
    "support", "live", "site", "online", "club", "monster", "cyou", "cfd", "sbs",
    "rest", "fit", "quest", "vip", "loan", "men", "zip", "mov", "country", "stream",
    "download", "gdn", "bid",
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly",
    "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy", "lnkd.in", "s.id",
}

RISKY_EXTS = {"exe", "zip", "rar", "apk", "scr", "msi", "js", "jar", "bat", "vbs", "iso", "dmg"}

# Second-level labels used under country codes (co.uk, com.au, co.in ...).
SECOND_LEVEL = {"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go", "gob", "mil"}

# Words used in credential-phishing lures (HR notices, payroll, mailbox quotas, ...).
# They are matched as WHOLE WORDS taken from the host and path (so "hr" does not fire
# on "three"). Extend this list with words from the "candidate lure words" section of
# analyze_errors.py, then retrain.
LURE_TOKENS = {
    "hr", "payroll", "salary", "bonus", "benefit", "benefits", "notice", "notify",
    "notification", "notifications", "voicemail", "fax", "helpdesk", "webmail",
    "mailbox", "quota", "expire", "expired", "expiry", "refund", "receipt",
    "remittance", "statement", "approval", "approve", "esign", "docusign",
    "invitation", "survey", "shared", "document", "documents", "delivery", "parcel",
    "claim", "reward", "prize", "gift", "urgent", "restricted", "suspended",
    "unusual", "otp", "mfa", "reactivate", "resolve",
}
# Word beginnings that count as a lure too (notif -> notif, notifications, notify ...).
LURE_PREFIXES = ("notif", "payroll", "voicemail", "mailbox", "webmail", "helpdesk", "sharepoint")

# TLDs that get their own yes/no feature so the model can learn TLD-specific behaviour.
COMMON_TLDS = [
    "com", "net", "org", "info", "biz", "xyz", "top", "online", "site", "icu", "cc",
    "co", "io", "me", "in", "ru", "cn", "de", "uk", "br", "app", "link", "live",
    "shop", "pw", "ws", "su", "tk",
]

VOWELS = set("aeiou")

IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.I)
DECIMAL_IP_RE = re.compile(r"^\d{8,10}$")
HEX_ESCAPE_RE = re.compile(r"%[0-9a-fA-F]{2}")
DOMAIN_IN_PATH_RE = re.compile(r"[a-z0-9-]+\.(?:com|net|org|co|info|biz|gov|edu|in|uk)\b", re.I)

# Look-alike digits: paypa1 -> paypal, g00gle -> google, micr0soft -> microsoft.
LEET_L = str.maketrans("013457", "oleast")
LEET_I = str.maketrans("013457", "oieast")


# --------------------------------------------------------------------------- helpers
def entropy(text):
    """Shannon entropy: higher = more random-looking (typical of generated domains)."""
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def brand_hit(text):
    """True if a known brand name (or a digit look-alike of it) appears in `text`."""
    if not text:
        return False
    variants = (text, text.translate(LEET_L), text.translate(LEET_I))
    return any(b in v for b in BRANDS for v in variants)


def max_consonant_run(text):
    """Longest run of consonants in a row (random-looking names have long runs)."""
    best = run = 0
    for c in text:
        if c.isalpha() and c not in VOWELS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def alnum_transitions(text):
    """How many times the text switches between letters and digits (random IDs switch a lot)."""
    kinds = ["d" if c.isdigit() else "a" for c in text if c.isalnum()]
    return sum(1 for a, b in zip(kinds, kinds[1:]) if a != b)


def is_lure(token):
    return token in LURE_TOKENS or token.startswith(LURE_PREFIXES)


def split_host(host):
    """
    Split a host name into (subdomain, main_domain_label, suffix) without any
    network lookup. Example: 'login.paypal.co.uk' -> ('login', 'paypal', 'co.uk').
    This is a simple heuristic, not a full public-suffix list, which is fine for a baseline.
    """
    labels = host.split(".")
    if len(labels) == 1:
        return "", host, ""
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in SECOND_LEVEL:
        return ".".join(labels[:-3]), labels[-3], ".".join(labels[-2:])
    return ".".join(labels[:-2]), labels[-2], labels[-1]


# ------------------------------------------------------------------ the main function
def extract_features(url):
    url = str(url).strip()
    try:
        p = urlparse(url)
        host = (p.hostname or "").lower()
        try:
            port = p.port
        except ValueError:
            port = None
        netloc, path, query, scheme = p.netloc, p.path or "", p.query or "", p.scheme.lower()
    except ValueError:  # badly formed URL, e.g. a broken IPv6 bracket
        host, port, netloc, path, query, scheme = "", None, "", "", "", ""

    url_l = url.lower()
    n = max(len(url), 1)
    is_ip = bool(IPV4_RE.match(host) or HEX_IP_RE.match(host)
                 or DECIMAL_IP_RE.match(host) or ":" in host)
    sub, domain, suffix = ("", host, "") if is_ip else split_host(host)
    tld = suffix.split(".")[-1] if suffix else ""

    host_tokens = [t for t in re.split(r"[.\-]", host) if t]
    segments = [s for s in path.split("/") if s]
    last = segments[-1] if segments else ""
    ext = last.rsplit(".", 1)[-1].lower() if "." in last else ""
    digit_runs = re.findall(r"\d+", url)
    exact_brand = domain in BRANDS
    domain_letters = [c for c in domain if c.isalpha()]
    host_words = [t for t in re.split(r"[^a-z0-9]+", host) if t]
    url_words = [t for t in re.split(r"[^a-z0-9]+", (host + "/" + path).lower()) if t]

    feats = {
        # ---- sizes
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        # ---- character counts
        "dots_url": url.count("."),
        "dots_host": host.count("."),
        "hyphens_url": url.count("-"),
        "hyphens_host": host.count("-"),
        "underscores_url": url.count("_"),
        "slashes_url": url.count("/"),
        "at_symbols": url.count("@"),
        "question_marks": url.count("?"),
        "equals_signs": url.count("="),
        "ampersands": url.count("&"),
        "percent_signs": url.count("%"),
        "hex_escapes": len(HEX_ESCAPE_RE.findall(url)),
        # ---- character ratios
        "digit_ratio": sum(c.isdigit() for c in url) / n,
        "letter_ratio": sum(c.isalpha() for c in url) / n,
        "upper_ratio": sum(c.isupper() for c in url) / n,
        "host_digit_ratio": sum(c.isdigit() for c in host) / max(len(host), 1),
        "longest_digit_run": max((len(r) for r in digit_runs), default=0),
        # ---- host structure
        "host_is_ip": int(is_ip),
        "subdomain_count": len(sub.split(".")) if sub else 0,
        "domain_length": len(domain),
        "domain_hyphens": domain.count("-"),
        "domain_digits": sum(c.isdigit() for c in domain),
        "tld_length": len(tld),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "host_token_count": len(host_tokens),
        "longest_host_token": max((len(t) for t in host_tokens), default=0),
        "avg_host_token": (sum(len(t) for t in host_tokens) / len(host_tokens)) if host_tokens else 0.0,
        "is_shortener": int(host in SHORTENERS
                            or (bool(domain) and bool(suffix) and f"{domain}.{suffix}" in SHORTENERS)),
        # ---- connection details
        "is_https": int(scheme == "https"),
        "nonstandard_port": int(port not in (None, 80, 443)),
        "has_userinfo": int("@" in netloc),
        "has_punycode": int("xn--" in host),
        "non_ascii_host": int(any(ord(c) > 127 for c in host)),
        # ---- randomness
        "host_entropy": entropy(host),
        "path_entropy": entropy(path),
        "url_entropy": entropy(url),
        # ---- path and query
        "path_depth": len(segments),
        "double_slash_in_path": int("//" in path),
        "num_params": (query.count("&") + 1) if query else 0,
        "ext_php": int(ext in ("php", "php3", "php5", "phtml")),
        "ext_html": int(ext in ("html", "htm", "shtml")),
        "ext_risky": int(ext in RISKY_EXTS),
        "domain_in_path": int(bool(DOMAIN_IN_PATH_RE.search(path + "?" + query))),
        # ---- imitation signals
        "brand_in_domain_not_exact": int(brand_hit(domain) and not exact_brand),
        "brand_in_subdomain": int(brand_hit(sub)),
        "brand_in_path": int(brand_hit((path + "?" + query).lower()) and not exact_brand),
        "suspicious_word_count": sum(w in url_l for w in SUSPICIOUS_WORDS),
        "suspicious_word_in_host": int(any(w in host for w in SUSPICIOUS_WORDS)),
        # ---- randomness of the last path segment (per-victim IDs, random tokens)
        "last_seg_length": len(last),
        "last_seg_digit_ratio": sum(c.isdigit() for c in last) / max(len(last), 1),
        "last_seg_entropy": entropy(last),
        "last_seg_transitions": alnum_transitions(last),
        "last_seg_mixed": int(any(c.isdigit() for c in last) and any(c.isalpha() for c in last)),
        "longest_path_token": max((len(t) for t in re.findall(r"[A-Za-z0-9]+", path)), default=0),
        # ---- randomness of the domain name (generated domains)
        "domain_entropy": entropy(domain),
        "domain_vowel_ratio": sum(c in VOWELS for c in domain) / max(len(domain_letters), 1),
        "domain_max_consonant_run": max_consonant_run(domain),
        # ---- lure words (whole-word match) and TLD type
        "lure_token_count": sum(is_lure(t) for t in url_words),
        "lure_token_in_host": int(any(is_lure(t) for t in host_words)),
        "tld_is_country_code": int(len(tld) == 2),
    }
    for common in COMMON_TLDS:
        feats[f"tld_is_{common}"] = int(tld == common)
    return feats


FEATURE_NAMES = list(extract_features("http://example.com/").keys())


def build_feature_frame(urls, workers=1):
    """Turn many URLs into a DataFrame (one row per URL, one column per feature)."""
    urls = list(urls)
    if workers > 1 and len(urls) > 5000:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            rows = pool.map(extract_features, urls, chunksize=2000)
    else:
        rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES).fillna(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python features.py "http://example.com/login" [more URLs...]')
    frame = build_feature_frame(sys.argv[1:])
    frame.index = [u if len(u) <= 40 else u[:37] + "..." for u in sys.argv[1:]]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(frame.T.round(3))