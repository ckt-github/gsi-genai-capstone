"""Download the extra SEC data needed for two checks requested in the interim review.

1. XBRL vintages (first-reported values). Company facts list every filing that reported a value, with
   its filing date. 02_get_financials.py keeps only the latest (restated) value; here every reported
   value for revenue and operating income is kept, so 11_sensitivity.py can rebuild the RQ3 features
   with only the figures that were public on the prediction date.
   Output: data/interim/xbrl_vintages.csv.gz

2. AI keyword precision. data/interim/ai_precision_draws.csv holds 70 filings drawn at random with
   probability proportional to their number of AI matches (seed 42), so that one random match per
   draw is a random draw from all matches. This script downloads those filings and saves the text
   around every match. 11_sensitivity.py picks one match per draw; the first 50 are reviewed by hand.
   Output: data/interim/ai_match_contexts.csv.gz
"""
import re
import warnings

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm

from common import INTERIM, RAW, ROOT, cik10, sec_get

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "SalesRevenueServicesNet"],
    "operating_income": ["OperatingIncomeLoss"],
}
KEEP = ["start", "end", "val", "filed", "form", "accn", "fy", "fp"]

# ---------- 1. XBRL vintages ----------
universe = pd.read_csv(INTERIM / "universe.csv")
rows = []
for cik in tqdm(universe["cik"], desc="company facts"):
    facts = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json")
    if not facts:
        continue
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept, tags in CONCEPTS.items():
        for rank, tag in enumerate(tags):
            node = gaap.get(tag)
            if not node or "USD" not in node.get("units", {}):
                continue
            df = pd.DataFrame(node["units"]["USD"])
            if "start" not in df:
                continue
            df = df[df["form"].astype(str).str.startswith(("10-Q", "10-K"))].dropna(subset=["start"])
            df = df[df["end"] >= "2017-01-01"]
            if df.empty:
                continue
            df = df[[c for c in KEEP if c in df.columns]].copy()
            df["cik"], df["concept"], df["tag"], df["tag_rank"] = cik, concept, tag, rank
            rows.append(df)
vint = pd.concat(rows, ignore_index=True)
vint.to_csv(INTERIM / "xbrl_vintages.csv.gz", index=False)
print(f"xbrl vintages: {len(vint):,} facts, {vint['cik'].nunique()} firms")

# ---------- 2. AI match contexts ----------
patterns = []
for line in (ROOT / "config/ai_keywords.txt").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        patterns.append((line, re.compile(line.replace("[genai]", "").strip(), re.I)))

draws = pd.read_csv(INTERIM / "ai_precision_draws.csv", dtype={"accession": str})
docs = {}
for cik in draws["cik"].unique():
    sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json",
                  RAW / "submissions" / f"{cik10(cik)}.json")
    if not sub:
        continue
    pages = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):
        page = sec_get(f"https://data.sec.gov/submissions/{extra['name']}", RAW / "submissions" / extra["name"])
        if page:
            pages.append(pd.DataFrame(page))
    f = pd.concat(pages)
    for acc, doc in f[["accessionNumber", "primaryDocument"]].itertuples(index=False):
        docs[(int(cik), acc)] = doc

out = []
for acc in tqdm(draws["accession"].unique(), desc="filings"):
    cik = int(draws.loc[draws["accession"] == acc, "cik"].iloc[0])
    doc = docs.get((cik, acc))
    if not doc:
        print(f"WARNING: no primary document for {acc}")
        continue
    html = sec_get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}", as_json=False)
    if not html:
        continue
    text = re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" "))
    n = 0
    for raw, rx in patterns:
        for m in rx.finditer(text):
            out.append({"accession": acc, "cik": cik, "pattern": raw, "match": m.group(0), "pos": m.start(),
                        "context": text[max(0, m.start() - 300): m.end() + 300]})
            n += 1
    print(f"{acc}: {n} matches")
ctx = pd.DataFrame(out)
ctx.to_csv(INTERIM / "ai_match_contexts.csv.gz", index=False)
print(f"ai match contexts: {len(ctx):,} matches in {ctx['accession'].nunique()} filings")
