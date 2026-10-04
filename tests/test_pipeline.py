"""Unit tests for the pieces of the pipeline that do not need network access."""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("SEC_USER_AGENT", "test test@example.com")


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SRC / name)
    mod = importlib.util.module_from_spec(spec)
    return spec, mod


def test_q4_derivation():
    """Fourth quarter = fiscal-year total minus the three reported quarters (46 - 33 = 13)."""
    src = (SRC / "02_get_financials.py").read_text()
    ns = {}
    exec(src.split("universe = pd.read_csv")[0], ns)  # function definitions only
    df = pd.DataFrame({
        "start": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-01-01"]),
        "end": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]),
        "val": [10, 11, 12, 46],
    })
    df["days"] = (df["end"] - df["start"]).dt.days
    q = ns["quarterly"](df)
    assert q.loc[pd.Timestamp("2023-12-31")] == 13


def test_india_snapshot_integrity():
    snap = pd.read_csv(SRC.parent / "data/raw/india/nse_quarterly_xbrl_snapshot.csv")
    firms = pd.read_csv(SRC.parent / "config/india_firms.csv")
    assert set(snap["sym"]) == set(firms["nse_symbol"])
    ok = snap.dropna(subset=["RevenueFromOperations"])
    assert len(ok) >= 480
    assert (ok["currency"] == "INR").all()


def test_xbrl_parser_reads_quarter_context():
    src = (SRC / "02b_india_financials.py").read_text()
    ns = {"__name__": "x"}
    exec(src.split('if "--refresh" in sys.argv')[0], ns)
    xml = """<xbrl xmlns="http://www.xbrl.org/2003/instance" xmlns:in="x">
      <context id="Q"><period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period></context>
      <context id="Y"><period><startDate>2024-04-01</startDate><endDate>2024-12-31</endDate></period></context>
      <in:DateOfStartOfReportingPeriod contextRef="Q">2024-10-01</in:DateOfStartOfReportingPeriod>
      <in:DateOfEndOfReportingPeriod contextRef="Q">2024-12-31</in:DateOfEndOfReportingPeriod>
      <in:RevenueFromOperations contextRef="Y">300</in:RevenueFromOperations>
      <in:RevenueFromOperations contextRef="Q">100</in:RevenueFromOperations>
    </xbrl>"""
    row = ns["parse_xbrl"](xml)
    assert row["RevenueFromOperations"] == "100"


def test_extension_eligibility_rule():
    """2026Q1: old enough, 9 of 10 firms reported (90%) -> in. 2026Q2: 8 of 10 -> out.
    2026Q3: everyone reported but too recent, and the sequence already stopped -> out."""
    from common import eligible_quarters
    q = lambda s: pd.Period(s, freq="Q")
    rows = [(c, q("2025Q4")) for c in range(10)]
    rows += [(c, q("2026Q1")) for c in range(9)] + [(c, q("2026Q2")) for c in range(8)]
    rows += [(c, q("2026Q3")) for c in range(10)]
    firm_qtrs = pd.DataFrame(rows, columns=["cik", "cal_qtr"])
    ok, report = eligible_quarters(firm_qtrs, q("2025Q4"), [q("2026Q1"), q("2026Q2"), q("2026Q3")],
                                   "2026-10-04", min_days=60, min_coverage=0.90)
    assert ok == [q("2026Q1")]
    assert [r["included"] for r in report] == [True, False, False]
    assert report[1]["coverage"] == 0.8


def test_gsi_classification_file():
    """Each firm listed once, flag is 0/1, every firm has a reason, and the Indian GSIs are study firms."""
    root = SRC.parent
    cls = pd.read_csv(root / "config/gsi_classification.csv", dtype={"cik": str})
    assert cls["cik"].is_unique
    assert set(cls["is_gsi"]) <= {0, 1}
    assert cls["reason"].notna().all()
    india = pd.read_csv(root / "config/india_firms.csv")
    listed = set("NSE:" + india["nse_symbol"])
    assert set(cls.loc[cls.market == "India", "cik"]) <= listed
