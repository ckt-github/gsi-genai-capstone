"""Score each 10-Q / 10-K for AI-disclosure intensity.

AI_INTENSITY  = AI keyword matches per 10,000 words of filing text
GENAI_MENTION = 1 if any generative-AI-specific term appears, else 0

Filing HTML is scored in memory and not stored, so the job needs little disk space. Scores are
appended to a progress file after every firm, so an interrupted run resumes where it stopped.
Filings are scored up to the 2026-extension bound in config/settings.yaml; the main panel still uses
only 2019-2025 because src/04_build_panel.py applies the study window.
Output: data/interim/ai_scores.csv (cik, accession, form, report_date, words, ai_hits, ai_intensity, genai_mention)
"""
import re
import warnings

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm

from common import INTERIM, RAW, ROOT, SETTINGS, cik10, sec_get

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

patterns, genai = [], []
for line in (ROOT / "config/ai_keywords.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    is_gen = line.startswith("[genai]")
    rx = re.compile(line.replace("[genai]", "").strip(), re.I)
    patterns.append(rx)
    if is_gen:
        genai.append(rx)


def score(text):
    words = len(text.split())
    hits = sum(len(rx.findall(text)) for rx in patterns)
    gen = int(any(rx.search(text) for rx in genai))
    return words, hits, (hits / words * 10_000 if words else 0.0), gen


LAST = max(SETTINGS["period_end"], SETTINGS.get("extension", {}).get("period_end", SETTINGS["period_end"]))
OUT = INTERIM / "ai_scores.csv"
done = pd.read_csv(OUT, dtype={"accession": str}) if OUT.exists() else pd.DataFrame(columns=["accession"])
done_acc = set(done["accession"])
universe = pd.read_csv(INTERIM / "universe.csv")

for cik in tqdm(universe["cik"]):
    sub = sec_get(f"https://data.sec.gov/submissions/CIK{cik10(cik)}.json",
                  RAW / "submissions" / f"{cik10(cik)}.json")
    if not sub:
        continue
    recent = pd.DataFrame(sub["filings"]["recent"])
    # older filings live in extra pages listed under filings.files
    for extra in sub["filings"].get("files", []):
        page = sec_get(f"https://data.sec.gov/submissions/{extra['name']}", RAW / "submissions" / extra["name"])
        if page:
            recent = pd.concat([recent, pd.DataFrame(page)])
    if recent.empty:
        continue
    f = recent[recent["form"].isin(SETTINGS["forms"])
               & recent["reportDate"].between(SETTINGS["period_start"], LAST)
               & ~recent["accessionNumber"].isin(done_acc)]
    rows = []
    for acc, doc, form, rdate in f[["accessionNumber", "primaryDocument", "form", "reportDate"]].itertuples(index=False):
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
        html = sec_get(url, as_json=False)
        if not html:
            continue
        text = BeautifulSoup(html, "lxml").get_text(" ")
        words, hits, intensity, gen = score(text)
        rows.append({"cik": cik, "accession": acc, "form": form, "report_date": rdate,
                     "words": words, "ai_hits": hits, "ai_intensity": round(intensity, 3), "genai_mention": gen})
    if rows:  # checkpoint after each firm
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        done_acc |= {r["accession"] for r in rows}

print(f"{len(done_acc):,} filings scored in total")
