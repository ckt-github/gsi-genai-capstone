"""Merge U.S. financials, AI-disclosure scores and the Indian comparison firms into one firm-quarter panel.

MARKET = "US" for SEC registrants, "India" for the NSE-listed comparison firms.
Revenue growth is computed in each firm's reporting currency (USD for U.S. firms, INR for Indian firms),
so exchange-rate movements do not distort Indian growth rates. REVENUE_USD and LOG_REVENUE use USD.
Output: data/processed/panel_firm_quarter.csv (variables defined in docs/data_dictionary.csv)
"""
import numpy as np
import pandas as pd

from common import INTERIM, PROCESSED, SETTINGS

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
panel["POST_GENAI"] = (panel["PERIOD"] == "post").astype(int)

# filters: study window, shells, missing core fields
panel = panel[(qstart >= "2019-01-01") & (qstart <= post[1])]
panel = panel[panel["rev_ttm"] >= SETTINGS["min_annual_revenue_usd"]]
panel = panel.dropna(subset=["REV_GROWTH_YOY"]).copy()

# winsorize ratio variables at 1st / 99th percentiles within each market
lo, hi = SETTINGS["winsorize"]
for c in ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "OP_MARGIN", "SGA_RATIO", "AI_INTENSITY"]:
    g = panel.groupby("MARKET")[c]
    panel[c] = panel[c].clip(g.transform(lambda s: s.quantile(lo)), g.transform(lambda s: s.quantile(hi)))

panel["FISCAL_QTR"] = panel["cal_qtr"].astype(str)
cols = ["cik", "ticker", "name", "MARKET", "sic", "SEGMENT", "FISCAL_QTR", "PERIOD", "POST_GENAI",
        "REVENUE_USD", "LOG_REVENUE", "REV_GROWTH_YOY", "REV_GROWTH_LAG1", "DECLINE_Q", "DECLINE_NEXT_Q",
        "OP_MARGIN", "SGA_RATIO", "AI_INTENSITY", "GENAI_MENTION"]
panel[cols].to_csv(PROCESSED / "panel_firm_quarter.csv", index=False)
print(panel.groupby(["MARKET", "PERIOD"]).size())
print(f"{panel['cik'].nunique()} firms, {len(panel):,} firm-quarters")
