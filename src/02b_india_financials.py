"""Quarterly financials for 15 NSE-listed Indian IT services firms (comparison market).

Default: read the committed snapshot data/raw/india/nse_quarterly_xbrl_snapshot.csv.
--refresh: try to rebuild that snapshot from the live NSE feeds first. NSE often blocks automated
requests from cloud servers, so any failure falls back to the snapshot without stopping the pipeline.

Steps
1. Keep consolidated Ind-AS results; assign each filing to the calendar quarter of its period end.
2. One row per firm-quarter (latest filing wins, so restatements replace originals).
3. OPERATING_INCOME = profit before tax + finance costs - other income (EBIT from operations).
4. Convert INR to USD with the FRED DEXINUS quarterly average (live, with snapshot fallback).

Output: data/interim/india_financials_quarterly.csv
"""
import io
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from common import INTERIM, RAW, ROOT

SNAP = RAW / "india" / "nse_quarterly_xbrl_snapshot.csv"
FX_SNAP = RAW / "india" / "fred_dexinus_quarterly_snapshot.csv"
FIRMS = pd.read_csv(ROOT / "config/india_firms.csv")
ELEMENTS = ["RevenueFromOperations", "Income", "OtherIncome", "Expenses", "EmployeeBenefitExpense",
            "FinanceCosts", "DepreciationDepletionAndAmortisationExpense", "OtherExpenses",
            "ProfitBeforeTax", "ProfitLossForPeriod"]
NSE = "https://www.nseindia.com"
HDRS = {"User-Agent": "Mozilla/5.0 (academic research; Walsh College capstone)",
        "Accept": "application/json,text/plain,*/*", "Referer": NSE + "/"}


# ---------------------------------------------------------------- live refresh (best effort)
def parse_xbrl(xml_text):
    """Return the consolidated quarterly facts from one NSE results XBRL file."""
    root = ET.fromstring(xml_text)
    local = lambda tag: tag.rsplit("}", 1)[-1]
    contexts = {}
    for ctx in root.iter():
        if local(ctx.tag) != "context":
            continue
        dates = {local(e.tag): e.text for e in ctx.iter() if local(e.tag) in ("startDate", "endDate")}
        has_dim = any(local(e.tag) == "explicitMember" for e in ctx.iter())
        if len(dates) == 2 and not has_dim:
            contexts[ctx.get("id")] = (dates["startDate"], dates["endDate"])
    facts, text = {}, {}
    for el in root:
        name = local(el.tag)
        if name in ("DateOfStartOfReportingPeriod", "DateOfEndOfReportingPeriod",
                    "NatureOfReportStandaloneConsolidated", "DescriptionOfPresentationCurrency"):
            text[name] = (el.text or "").strip()
        if name in ELEMENTS and el.text:
            facts.setdefault(name, []).append((el.get("contextRef"), el.text.strip()))
    start, end = text.get("DateOfStartOfReportingPeriod"), text.get("DateOfEndOfReportingPeriod")
    # quarter context = the 80-100 day period ending on the reporting end date ("OneD" in old files)
    qctx = {c for c, (s, e) in contexts.items()
            if e == end and 80 <= (pd.Timestamp(e) - pd.Timestamp(s)).days <= 100}
    qctx.add("OneD")
    row = {"start": start, "end": end, "nature": text.get("NatureOfReportStandaloneConsolidated"),
           "currency": text.get("DescriptionOfPresentationCurrency")}
    for name in ELEMENTS:
        vals = [v for c, v in facts.get(name, []) if c in qctx]
        row[name] = vals[0] if vals else None
    return row


def refresh_snapshot():
    s = requests.Session()
    s.headers.update(HDRS)
    s.get(NSE, timeout=20)  # sets the cookies the API requires
    rows = []
    for sym in FIRMS.nse_symbol:
        listing = []
        legacy = s.get(f"{NSE}/api/corporates-financial-results?index=equities&symbol={sym}&period=Quarterly",
                       timeout=30).json()
        listing += [("legacy", r.get("filingDate") or r.get("broadCastDate"), r.get("xbrl"))
                    for r in legacy if r.get("consolidated") == "Consolidated"]
        page = 1
        while True:
            integ = s.get(f"{NSE}/api/integrated-filing-results?symbol={sym}"
                          f"&type=Integrated%20Filing-%20Financials&page={page}&size=50", timeout=30).json()
            data = integ.get("data", [])
            listing += [("integrated", r.get("broadcast_Date"), r.get("xbrl"))
                        for r in data if r.get("consolidated") == "Consolidated"]
            if page * 50 >= int(integ.get("totalCount", 0)) or not data:
                break
            page += 1
        for src, filed, url in listing:
            if not url or not url.endswith(".xml"):
                continue
            r = s.get(url, timeout=30)
            if r.status_code != 200:
                continue
            row = parse_xbrl(r.text)
            rows.append({"sym": sym, "src": src, "filed": filed, "file": url.rsplit("/", 1)[-1], **row})
            time.sleep(0.3)
        print(f"  {sym}: {len(listing)} filings")
    new = pd.DataFrame(rows)
    if len(new) < 300:  # sanity check against a partial download
        raise RuntimeError(f"only {len(new)} filings retrieved")
    new.to_csv(SNAP, index=False)
    print(f"snapshot refreshed: {len(new)} filings")


if "--refresh" in sys.argv:
    try:
        refresh_snapshot()
    except Exception as exc:  # NSE blocks many cloud IP ranges
        print(f"NSE refresh failed ({exc.__class__.__name__}: {exc}); using committed snapshot")


# ---------------------------------------------------------------- FX
def load_fx():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS&cosd=2017-01-01&fq=Quarterly&fam=avg"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": HDRS["User-Agent"]})
        r.raise_for_status()
        fx = pd.read_csv(io.StringIO(r.text))
        assert len(fx) > 30
        fx.to_csv(FX_SNAP, index=False)
        print("FX: live FRED DEXINUS")
    except Exception as exc:
        print(f"FX: FRED unavailable ({exc.__class__.__name__}); using snapshot")
        fx = pd.read_csv(FX_SNAP)
    fx.columns = ["date", "inr_per_usd"]
    fx["cal_qtr"] = pd.to_datetime(fx["date"]).dt.to_period("Q")
    fx["inr_per_usd"] = pd.to_numeric(fx["inr_per_usd"], errors="coerce")
    return fx.dropna()[["cal_qtr", "inr_per_usd"]]


# ---------------------------------------------------------------- clean
raw = pd.read_csv(SNAP)
df = raw.dropna(subset=["end", "RevenueFromOperations"]).copy()
df = df[df["nature"].fillna("Consolidated").str.startswith("Consolidated")]
df["period_end"] = pd.to_datetime(df["end"])
df["cal_qtr"] = df["period_end"].dt.to_period("Q")   # quarter from end date (see data/raw/india/README.md)
df["filed_ts"] = pd.to_datetime(df["filed"], format="mixed", dayfirst=True, errors="coerce")
df = (df.sort_values(["sym", "cal_qtr", "filed_ts"], na_position="first")
        .drop_duplicates(["sym", "cal_qtr"], keep="last"))

df["revenue_inr"] = df["RevenueFromOperations"]
df["operating_income_inr"] = df["ProfitBeforeTax"] + df["FinanceCosts"].fillna(0) - df["OtherIncome"].fillna(0)
fx = load_fx()
df = df.merge(fx, on="cal_qtr", how="left")
missing_fx = df["inr_per_usd"].isna().sum()
if missing_fx:
    print(f"WARNING: {missing_fx} firm-quarters without an FX rate (dropped)")
    df = df.dropna(subset=["inr_per_usd"])
df["revenue"] = df["revenue_inr"] / df["inr_per_usd"]
df["operating_income"] = df["operating_income_inr"] / df["inr_per_usd"]
df = df.merge(FIRMS, left_on="sym", right_on="nse_symbol", how="left")

out = pd.DataFrame({
    "cik": "NSE:" + df["sym"], "ticker": df["sym"], "name": df["firm_name"], "segment": df["segment"],
    "period_end": df["period_end"].dt.date, "cal_qtr": df["cal_qtr"].astype(str),
    "revenue": df["revenue"].round(0), "operating_income": df["operating_income"].round(0),
    "revenue_inr": df["revenue_inr"], "operating_income_inr": df["operating_income_inr"],
    "inr_per_usd": df["inr_per_usd"], "source_file": df["file"],
})
out.to_csv(INTERIM / "india_financials_quarterly.csv", index=False)
print(f"{out['ticker'].nunique()} Indian firms, {len(out):,} firm-quarters "
      f"({out['cal_qtr'].min()} to {out['cal_qtr'].max()}) written")
