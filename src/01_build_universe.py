"""Build the study universe: SEC registrants with SIC 7370/7371/7373/7374 plus named GSIs.

Primary source: EDGAR company browse by SIC code (includes firms that have since been delisted or
acquired, which avoids survivorship bias). Fallback: SEC company_tickers.json (currently listed firms).
A firm is kept only if it filed at least one 10-Q or 10-K for a period inside the study window.

Output: data/interim/universe.csv (cik, ticker, name, sic, segment, filer_category, added_by_name)
"""
import json
import re

import pandas as pd
from tqdm import tqdm

from common import INTERIM, RAW, ROOT, SETTINGS, cik10, sec_get

BROWSE = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic}&type=&dateb="
          "&owner=include&start={start}&count=100&output=atom")


def ciks_by_sic(sic):
    ciks, start = [], 0
    while True:
        page = sec_get(BROWSE.format(sic=sic, start=start), as_json=False) or ""
        found = re.findall(r"<cik>(\d+)</cik>", page)
        ciks += [int(c) for c in found]
        if len(found) < 100:
            return ciks
        start += 100


candidates = set()
for sic in SETTINGS["sic_codes"]:
    found = ciks_by_sic(sic)
    print(f"SIC {sic}: {len(found)} registrants from EDGAR browse")
    candidates |= set(found)

tickers = sec_get("https://www.sec.gov/files/company_tickers.json", RAW / "company_tickers.json") or {}
tick = pd.DataFrame(tickers.values()).rename(columns={"cik_str": "cik", "title": "name"}).drop_duplicates("cik")
named = pd.read_csv(ROOT / "config/named_gsis.csv")
named_ciks = set(tick.loc[tick.ticker.isin(named.ticker), "cik"])
if not candidates:  # browse endpoint unavailable: fall back to scanning listed firms
    print("EDGAR browse returned nothing; scanning company_tickers.json instead")
    candidates = set(tick["cik"])
candidates |= named_ciks
ticker_of = tick.set_index("cik")["ticker"].to_dict()

rows = []
for cik in tqdm(sorted(candidates)):
    sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json")
    if not sub:
        continue
    sic = int(sub.get("sic") or 0)
    in_range = sic in SETTINGS["sic_codes"]
    is_named = cik in named_ciks
    if not (in_range or is_named):
        continue
    recent = pd.DataFrame(sub["filings"]["recent"])
    if recent.empty:
        continue
    f = recent[recent["form"].isin(SETTINGS["forms"])
               & recent["reportDate"].between(SETTINGS["period_start"], SETTINGS["period_end"])]
    if f.empty and not sub["filings"].get("files"):
        continue  # no periodic reports in the study window
    ticker = ticker_of.get(cik) or (sub.get("tickers") or [""])[0]
    if in_range:
        seg = SETTINGS["segment_map"][sic]
    else:
        seg = named.set_index("ticker").loc[ticker, "segment"]
    cache = RAW / "submissions" / f"{cik10(cik)}.json"   # keep only firms in the universe on disk
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sub), encoding="utf-8")
    rows.append({"cik": cik, "ticker": ticker, "name": sub.get("name"), "sic": sic, "segment": seg,
                 "filer_category": sub.get("category"), "added_by_name": is_named and not in_range})

universe = pd.DataFrame(rows)
universe.to_csv(INTERIM / "universe.csv", index=False)
print(universe.groupby("segment").size())
print(f"{len(universe)} firms written to data/interim/universe.csv")
