"""Database set-up and deployment scorecard.

1. Loads the study tables into one SQLite database (data/processed/gsi_capstone.sqlite) with keys and indexes,
   so analysts can query the panel with SQL instead of opening CSV files.
2. Scores the latest reported quarter with the frozen RQ3 random forest (trained 2019-2023, threshold 0.31 fixed
   on 2024, exactly as in the main analysis) and writes the partner-risk scorecard to results/scorecard_latest.csv
   and to the database table `scorecard`.
Nothing here changes the main results; it packages them for use.
"""
import sqlite3

import numpy as np
import pandas as pd

import rq3_models as rq3m
from common import INTERIM, PROCESSED, RESULTS, SETTINGS

SEED = SETTINGS["random_seed"]
DB = PROCESSED / "gsi_capstone.sqlite"

panel = pd.read_csv(PROCESSED / "panel_firm_quarter.csv", dtype={"cik": str})
ext = pd.read_csv(PROCESSED / "panel_extension.csv", dtype={"cik": str})

# ------------------------------------------------------------------ frozen model and threshold
us = panel[panel.MARKET == "US"]
d, X = rq3m.design(us)
train, valid, test = rq3m.split(d)
rf = rq3m.make_models(SEED)["random_forest"]
thr = rq3m.fit_and_tune(rf, train, valid, X)

# ------------------------------------------------------------------ score the latest quarter
e_us = ext[ext.MARKET == "US"].copy()
latest_q = e_us.FISCAL_QTR.max()
capped, _ = rq3m.train_caps(e_us)
cur = capped[capped.FISCAL_QTR == latest_q].dropna(subset=rq3m.FEATS + ["SEGMENT"]).copy()
cur[rq3m.FEATS] = cur[rq3m.FEATS].astype(float)
for c in X:
    if c.startswith("SEG_"):
        cur[c] = (cur["SEGMENT"] == c[4:]).astype(int)
cur["risk_score"] = rf.predict_proba(cur[X])[:, 1]
cur["risk_band"] = np.select([cur.risk_score >= 0.50, cur.risk_score >= thr], ["High", "Elevated"], "Low")
cur["flagged"] = (cur.risk_score >= thr).astype(int)
card = cur[["cik", "ticker", "name", "SEGMENT", "FISCAL_QTR", "REV_GROWTH_YOY", "REV_GROWTH_LAG1", "OP_MARGIN",
            "AI_INTENSITY", "risk_score", "risk_band", "flagged"]].sort_values("risk_score", ascending=False)
card["risk_score"] = card.risk_score.round(3)
card.to_csv(RESULTS / "scorecard_latest.csv", index=False)

# ------------------------------------------------------------------ database
if DB.exists():
    DB.unlink()
con = sqlite3.connect(DB)
firms = panel.drop_duplicates("cik")[["cik", "ticker", "name", "MARKET", "sic", "SEGMENT"]]
con.executescript("""
CREATE TABLE firms (cik TEXT PRIMARY KEY, ticker TEXT, name TEXT, market TEXT, sic REAL, segment TEXT);
""")
firms.to_sql("firms", con, if_exists="append", index=False)
panel.to_sql("panel", con, index=False)
ext.to_sql("panel_extension", con, index=False)
for name, f in [("financials_quarterly", "financials_quarterly.csv"), ("ai_scores", "ai_scores.csv"),
                ("india_financials_quarterly", "india_financials_quarterly.csv")]:
    pd.read_csv(INTERIM / f, dtype={"cik": str}).to_sql(name, con, index=False)
card.to_sql("scorecard", con, index=False)
con.executescript("""
CREATE UNIQUE INDEX ix_panel_key ON panel (cik, FISCAL_QTR);
CREATE INDEX ix_panel_q ON panel (FISCAL_QTR);
CREATE UNIQUE INDEX ix_ext_key ON panel_extension (cik, FISCAL_QTR);
CREATE INDEX ix_ai_cik ON ai_scores (cik);
CREATE INDEX ix_fin_cik ON financials_quarterly (cik);
CREATE VIEW v_rq3_rows AS
  SELECT cik, FISCAL_QTR, SEGMENT, AI_INTENSITY, GENAI_MENTION, LOG_REVENUE, IS_10K, OP_MARGIN,
         REV_GROWTH_YOY, REV_GROWTH_LAG1, POST_GENAI, DECLINE_NEXT_Q,
         CASE WHEN FISCAL_QTR < '2024Q1' THEN 'train' WHEN FISCAL_QTR < '2025Q1' THEN 'validation'
              ELSE 'test' END AS split
  FROM panel WHERE MARKET = 'US' AND DECLINE_NEXT_Q IS NOT NULL;
""")
con.commit()
counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
          for t in ["firms", "panel", "panel_extension", "financials_quarterly", "ai_scores",
                    "india_financials_quarterly", "scorecard"]}
split_counts = con.execute("SELECT split, COUNT(*) FROM v_rq3_rows GROUP BY split").fetchall()
con.close()

lines = [f"Database: {DB.name}", "Rows per table: " + "; ".join(f"{k} {v:,}" for k, v in counts.items()),
         f"v_rq3_rows by split (before feature completeness): {split_counts}",
         f"\nScorecard: frozen random forest (trained 2019-2023, threshold {thr:.2f} fixed on 2024), "
         f"scoring {latest_q} features for the next quarter",
         f"firms scored {len(card)}; bands: " + "; ".join(f"{k} {v}" for k, v in card.risk_band.value_counts().items()),
         f"flagged (score >= {thr:.2f}): {card.flagged.sum()} of {len(card)} ({card.flagged.mean():.1%})"]
gsi = card[card.ticker.isin(["ACN", "CTSH", "DXC", "KD", "EPAM", "GDYN", "UNSY", "IBEX"])]
lines.append("Named GSIs in the scorecard:\n" + gsi[["ticker", "name", "risk_score", "risk_band"]].to_string(index=False))
(RESULTS / "scorecard_summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
