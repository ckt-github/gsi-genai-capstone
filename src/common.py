"""Shared helpers: paths, settings, and a polite SEC HTTP client."""
import json
import os
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW, INTERIM, PROCESSED = ROOT / "data/raw", ROOT / "data/interim", ROOT / "data/processed"
RESULTS = ROOT / "results"
for p in (RAW, INTERIM, PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

SETTINGS = yaml.safe_load((ROOT / "config/settings.yaml").read_text())

_session = None
_min_gap = 1.0 / SETTINGS["max_requests_per_second"]
_last = [0.0]


def _sec_session():
    """SEC fair-access rules require a descriptive User-Agent with a contact e-mail."""
    global _session
    if _session is None:
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise SystemExit("Set SEC_USER_AGENT, e.g. 'Your Name your.email@example.com' (SEC fair-access rule).")
        _session = requests.Session()
        _session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    return _session


def sec_get(url: str, cache: Path | None = None, as_json: bool = True, retries: int = 5):
    """GET a SEC URL with rate limiting, retries/back-off and an optional on-disk cache."""
    if cache is not None and cache.exists() and cache.stat().st_size > 0:
        text = cache.read_text(encoding="utf-8", errors="ignore")
        try:
            return json.loads(text) if as_json else text
        except json.JSONDecodeError:
            cache.unlink()  # corrupt cache from an interrupted run: fetch again
    s = _sec_session()
    for attempt in range(retries):
        wait = _min_gap - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = s.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 200:
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(r.text, encoding="utf-8")
            return r.json() if as_json else r.text
        if r.status_code == 404:
            return None
        time.sleep(2 ** attempt * 2)  # back off on 429 / 5xx
    print(f"WARNING: giving up on {url}")
    return None


def cik10(cik) -> str:
    return str(int(cik)).zfill(10)
