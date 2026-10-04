"""Merge U.S. financials, AI-disclosure scores and the Indian comparison firms into one firm-quarter panel.

MARKET = "US" for SEC registrants, "India" for the NSE-listed comparison firms.
Revenue growth is computed in each firm's reporting currency (USD for U.S. firms, INR for Indian firms),
so exchange-rate movements do not distort Indian growth rates. REVENUE_USD and LOG_REVENUE use USD.
Output: data/processed/panel_firm_quarter.csv (variables defined in docs/data_dictionary.csv)

--extension: also add the 2026 quarters that pass the eligibility rule in config/settings.yaml and
write data/processed/panel_extension.csv instead (column SAMPLE = "main" or "extension"). The main
2019-2025 rows are built exactly as in the main panel, including winsorizing limits taken from the
main window only, so the extension can never change the main results.
"""
import sys
from datetime import date

import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, RESULTS, SETTINGS, eligible_quarters

EXT = "--extension" in sys.argv

parts = []
us_path = INTERIM / "financials_quarterly.csv"
if us_path.exists():
    us = pd.read_csv(us_path, parse_dates=["period_end"])
    # calendarize: map each fiscal quarter-end to the calendar quarter it mostly falls in
    us["cal_qtr"] = (us["period_end"] - pd.Timedelta(days=45)).dt.to_period("Q")
    uni = pd.read_csv(INTERIM / "universe.csv")
    us = us.merge(uni[["cik", "name", "sic", "segment"]], on="cik", how="left")
    us["growth_base"] = us["revenue"]
    us["MARKET"] = "US"
    parts.append(us)
else:
    print("WARNING: U.S. financials not found; building India-only panel")

in_path = INTERIM / "india_financials_quarterly.csv"
if in_path.exists():
    ind = pd.read_csv(in_path, parse_dates=["period_end"])
    ind["cal_qtr"] = pd.PeriodIndex(ind["cal_qtr"], freq="Q")
    ind["growth_base"] = ind["revenue_inr"]
    ind["sga"], ind["sic"], ind["MARKET"] = np.nan, np.nan, "India"
    parts.append(ind)

fin = pd.concat(parts, ignore_index=True)
fin["cik"] = fin["cik"].astype(str)
fin = fin.sort_values(["cik", "cal_qtr"]).drop_duplicates(["cik", "cal_qtr"], keep="last").reset_index(drop=True)

by = lambda col: fin.groupby("cik")[col]
fin["qn"] = fin["cal_qtr"].apply(lambda q: q.year * 4 + q.quarter)
consecutive = (fin["qn"] - by("qn").shift(4)) == 4          # guard against gaps in the quarter sequence
fin["base_lag4"] = by("growth_base").shift(4).where(consecutive)
fin["rev_ttm"] = by("revenue").transform(lambda s: s.rolling(4).sum())
fin["REV_GROWTH_YOY"] = (fin["growth_base"] - fin["base_lag4"]) / fin["base_lag4"].abs() * 100
fin["REV_GROWTH_LAG1"] = by("REV_GROWTH_YOY").shift(1).where((fin["qn"] - by("qn").shift(1)) == 1)
fin["DECLINE_Q"] = (fin["REV_GROWTH_YOY"] < 0).astype("Int64").mask(fin["REV_GROWTH_YOY"].isna())
fin["DECLINE_NEXT_Q"] = by("DECLINE_Q").shift(-1).where((by("qn").shift(-1) - fin["qn"]) == 1)
fin["OP_MARGIN"] = fin["operating_income"] / fin["revenue"] * 100
fin["SGA_RATIO"] = fin["sga"] / fin["revenue"] * 100
fin["LOG_REVENUE"] = np.log(fin["revenue"].where(fin["revenue"] > 0))

ai_path = INTERIM / "ai_scores.csv"
if ai_path.exists():
    ai = pd.read_csv(ai_path, parse_dates=["report_date"])
    ai["cik"] = ai["cik"].astype(str)
    ai["cal_qtr"] = (ai["report_date"] - pd.Timedelta(days=45)).dt.to_period("Q")
    ai = ai.sort_values("form").drop_duplicates(["cik", "cal_qtr"], keep="first")  # 10-K before 10-K/A
    panel = fin.merge(ai[["cik", "cal_qtr", "ai_intensity", "genai_mention"]], on=["cik", "cal_qtr"], how="left")
else:
    panel = fin.assign(ai_intensity=np.nan, genai_mention=np.nan)
panel = panel.rename(columns={"ai_intensity": "AI_INTENSITY", "genai_mention": "GENAI_MENTION",
                              "segment": "SEGMENT", "revenue": "REVENUE_USD"})

qstart = panel["cal_qtr"].dt.start_time
pre, post = SETTINGS["pre_genai"], SETTINGS["post_genai"]
panel["PERIOD"] = np.select([qstart.between(*pre), qstart.between(*post)], ["pre", "post"], "transition")
panel["SAMPLE"] = np.where(qstart > post[1], "extension", "main")
panel.loc[panel["SAMPLE"] == "extension", "PERIOD"] = "post"   # 2026 is still the GenAI era
panel["POST_GENAI"] = (panel["PERIOD"] == "post").astype(int)

# filters: study window, shells, missing core fields
last = SETTINGS["extension"]["period_end"] if EXT else post[1]
panel = panel[(qstart >= "2019-01-01") & (qstart <= last)]
panel = panel[panel["rev_ttm"] >= SETTINGS["min_annual_revenue_usd"]]
panel = panel.dropna(subset=["REV_GROWTH_YOY"]).copy()

if EXT:  # keep only the 2026 quarters that pass the eligibility rule, market by market
    cfg = SETTINGS["extension"]
    base = pd.Period(post[1], freq="Q")
    main_firms = set(panel.loc[panel["SAMPLE"] == "main", "cik"])
    panel = panel[panel["cik"].isin(main_firms)]            # same firms as the main study, no newcomers
    keep, report = panel["SAMPLE"] == "main", []
    for market, sub in panel.groupby("MARKET"):
        cands = sorted(q for q in sub["cal_qtr"].unique() if q > base)
        ok, rep = eligible_quarters(sub[["cik", "cal_qtr"]], base, cands, date.today(),
                                    cfg["min_days_after_quarter"], cfg["min_coverage"])
        report += [{"market": market, **r} for r in rep]
        keep |= (panel["MARKET"] == market) & panel["cal_qtr"].isin(ok)
        if ok:  # the quarter after the last eligible one is not usable yet, so its outcome is unknown
            panel.loc[(panel["MARKET"] == market) & (panel["cal_qtr"] == ok[-1]), "DECLINE_NEXT_Q"] = np.nan
    panel = panel[keep].copy()
    pd.DataFrame(report).to_csv(RESULTS / "extension_eligibility.csv", index=False)

# winsorize ratio variables at 1st / 99th percentiles within each market (limits from the main window)
lo, hi = SETTINGS["winsorize"]
main_rows = panel[panel["SAMPLE"] == "main"]
for c in ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "OP_MARGIN", "SGA_RATIO", "AI_INTENSITY"]:
    limits = main_rows.groupby("MARKET")[c].quantile([lo, hi]).unstack()
    panel[c] = panel[c].clip(panel["MARKET"].map(limits[lo]), panel["MARKET"].map(limits[hi]))

panel["FISCAL_QTR"] = panel["cal_qtr"].astype(str)
cols = ["cik", "ticker", "name", "MARKET", "sic", "SEGMENT", "FISCAL_QTR", "PERIOD", "POST_GENAI",
        "REVENUE_USD", "LOG_REVENUE", "REV_GROWTH_YOY", "REV_GROWTH_LAG1", "DECLINE_Q", "DECLINE_NEXT_Q",
        "OP_MARGIN", "SGA_RATIO", "AI_INTENSITY", "GENAI_MENTION"]
if EXT:
    panel[cols + ["SAMPLE"]].to_csv(PROCESSED / "panel_extension.csv", index=False)
    print(panel.groupby(["MARKET", "SAMPLE", "PERIOD"]).size())
else:
    panel[cols].to_csv(PROCESSED / "panel_firm_quarter.csv", index=False)
    print(panel.groupby(["MARKET", "PERIOD"]).size())
print(f"{panel['cik'].nunique()} firms, {len(panel):,} firm-quarters")
