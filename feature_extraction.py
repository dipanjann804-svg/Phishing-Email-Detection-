"""
Feature extraction for phishing email detection.

Three kinds of features are produced from each email (subject + body + sender):
  1. Textual content -> fed into a TF-IDF vectorizer (captures wording,
     phrasing, urgency language, etc.)
  2. Hand-crafted content features -> capture signals TF-IDF alone tends to
     miss (link counts, suspicious keywords, punctuation patterns,
     capitalization, etc.)
  3. Sender/domain features -> capture signals from the "From" address
     (suspicious top-level domains, mismatch between the sender's domain
     and any linked domains in the body -- a classic spoofing tell).

All feature sets are combined at training and inference time so the exact
same logic is used in both places.
"""

import re

import numpy as np

# Keywords commonly seen in phishing/social-engineering emails
SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "suspend", "suspended", "click here", "confirm",
    "password", "update your", "immediately", "act now", "limited time",
    "winner", "won", "claim", "free", "gift card", "bank", "account locked",
    "unauthorized", "security alert", "login", "expire", "final notice",
    "restricted", "unusual activity", "credentials", "ssn", "social security",
]

# TLDs disproportionately abused for throwaway phishing infrastructure.
# (Heuristic, not a blocklist -- plenty of legitimate sites use these too.)
SUSPICIOUS_TLDS = {"xyz", "top", "icu", "click", "info", "ru", "tk", "ga", "ml", "cf", "work", "support"}

URL_PATTERN = re.compile(r"https?://[^\s]+")
IP_URL_PATTERN = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
DOMAIN_PATTERN = re.compile(r"https?://([^/\s]+)")


def combine_text(subject: str, body: str) -> str:
    """Combine subject + body into one string for the TF-IDF vectorizer."""
    subject = subject or ""
    body = body or ""
    return f"{subject}\n{body}"


def _extract_domain(address_or_url: str) -> str:
    """Pull a bare domain out of an email address or URL host."""
    if not address_or_url:
        return ""
    address_or_url = address_or_url.strip().lower()
    if "@" in address_or_url:
        domain = address_or_url.split("@")[-1]
    else:
        domain = address_or_url
    domain = domain.split("/")[0].split(":")[0]
    return domain


def _tld(domain: str) -> str:
    parts = domain.split(".")
    return parts[-1] if len(parts) > 1 else ""


def extract_numeric_features(subject: str, body: str, sender: str = "") -> dict:
    """Extract hand-crafted numeric/structural features from one email."""
    subject = subject or ""
    body = body or ""
    sender = sender or ""
    text = f"{subject} {body}"
    text_lower = text.lower()

    urls = URL_PATTERN.findall(text)
    num_links = len(urls)
    num_ip_urls = len(IP_URL_PATTERN.findall(text))

    num_suspicious_words = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in text_lower
    )

    num_exclamations = text.count("!")
    num_words = max(len(text.split()), 1)

    # crude "shouting" ratio: fraction of alphabetic chars that are uppercase
    letters = [c for c in text if c.isalpha()]
    num_upper = sum(1 for c in letters if c.isupper())
    upper_ratio = num_upper / len(letters) if letters else 0.0

    has_html_tags = 1 if re.search(r"<[^>]+>", body) else 0
    has_dollar_sign = 1 if "$" in text else 0
    has_urgent_subject = 1 if re.search(
        r"urgent|action required|alert|suspend|verify", subject, re.IGNORECASE
    ) else 0

    # --- sender/domain features ---
    sender_domain = _extract_domain(sender)
    sender_suspicious_tld = 1 if _tld(sender_domain) in SUSPICIOUS_TLDS else 0

    link_domains = {_extract_domain(u) for u in DOMAIN_PATTERN.findall(text)}
    if sender_domain and link_domains:
        sender_link_domain_mismatch = 1 if sender_domain not in link_domains else 0
    else:
        sender_link_domain_mismatch = 0

    return {
        "num_links": num_links,
        "num_ip_urls": num_ip_urls,
        "num_suspicious_words": num_suspicious_words,
        "num_exclamations": num_exclamations,
        "text_length": len(text),
        "num_words": num_words,
        "upper_ratio": round(upper_ratio, 4),
        "has_html_tags": has_html_tags,
        "has_dollar_sign": has_dollar_sign,
        "has_urgent_subject": has_urgent_subject,
        "sender_suspicious_tld": sender_suspicious_tld,
        "sender_link_domain_mismatch": sender_link_domain_mismatch,
    }


NUMERIC_FEATURE_NAMES = [
    "num_links", "num_ip_urls", "num_suspicious_words", "num_exclamations",
    "text_length", "num_words", "upper_ratio", "has_html_tags",
    "has_dollar_sign", "has_urgent_subject", "sender_suspicious_tld",
    "sender_link_domain_mismatch",
]


def numeric_features_to_array(feature_dict: dict) -> np.ndarray:
    return np.array([feature_dict[name] for name in NUMERIC_FEATURE_NAMES], dtype=float)
