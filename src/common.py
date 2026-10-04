"""Shared helpers: paths, settings, and a polite SEC HTTP client."""
import json
import os
import time
from pathlib import Path

import pandas as pd
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
        if r.status_code == 403 and attempt >= 1:
            break  # access refused (User-Agent or IP block); retrying will not help
        time.sleep(2 ** attempt * 2)  # back off on 429 / 5xx
    print(f"WARNING: giving up on {url} (HTTP {r.status_code if 'r' in locals() else 'n/a'})")
    return None


def sec_preflight():
    """Fail fast with a clear message if SEC refuses this machine or User-Agent."""
    s = _sec_session()
    for url in ("https://data.sec.gov/submissions/CIK0000320193.json",
                "https://www.sec.gov/files/company_tickers.json"):
        r = s.get(url, timeout=60)
        print(f"SEC check {r.status_code} {url}")
        if r.status_code != 200:
            print(r.text[:400])
            raise SystemExit("SEC EDGAR refused the request. Check SEC_USER_AGENT ('Name email@domain') "
                             "or run the pipeline from a different network.")
    print("SEC access OK")


if __name__ == "__main__":
    sec_preflight()


def cik10(cik) -> str:
    return str(int(cik)).zfill(10)


def eligible_quarters(firm_qtrs, base_qtr, candidates, today, min_days=60, min_coverage=0.90):
    """Extension rule: which quarters after the main window can be used yet.

    firm_qtrs  : DataFrame with columns cik, cal_qtr (pandas Period) for one market
    base_qtr   : last main-window quarter; its firms are the coverage denominator
    candidates : quarters after base_qtr, in order
    A quarter is eligible when it ended at least `min_days` ago and at least `min_coverage` of the
    base-quarter firms have reported it. Quarters are added in order and the first failure stops
    the sequence, so the extension never has gaps.
    Returns (list of eligible quarters, report rows for the output file).
    """
    base = set(firm_qtrs.loc[firm_qtrs["cal_qtr"] == base_qtr, "cik"])
    ok, report, stop = [], [], False
    for q in candidates:
        age = (pd.Timestamp(today) - q.end_time.normalize()).days
        have = set(firm_qtrs.loc[firm_qtrs["cal_qtr"] == q, "cik"]) & base
        cov = len(have) / len(base) if base else 0.0
        passed = (not stop) and age >= min_days and cov >= min_coverage
        report.append({"quarter": str(q), "days_since_end": age, "firms_reported": len(have),
                       "base_firms": len(base), "coverage": round(cov, 3), "included": passed})
        if passed:
            ok.append(q)
        else:
            stop = True
    return ok, report
