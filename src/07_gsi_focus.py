"""GSI focus: the study's questions for the large commercial GSIs inside the existing sample.

The firm scope and the main analysis are unchanged. This script takes the firms in the main panel
that are commercial global system integrators (config/gsi_classification.csv) and reports:
  A. which firms qualify and why (rule below, every $1B+ firm classified with a reason)
  B. revenue growth by period and by year for GSIs and for the other firms in each market
  C. RQ1 for GSIs: pre- vs post-GenAI growth, row-level and firm-level
  D. Did GSIs slow more than the other firms? (POST x GSI interaction, mean and median)
  E. a firm-by-firm table
  F. AI disclosure of U.S. GSIs (descriptive; too few firms for a regression)
  G. how the main RQ3 model scores on the GSI rows of the 2025 test year

GSI rule: annualised revenue (mean quarterly revenue 2021-2022 x 4) of at least the threshold in
config/settings.yaml, measured before the GenAI period so the selection cannot depend on the outcome,
and a business built on IT consulting, systems integration or managed IT services for commercial
clients. Government integrators, software-product firms, internet platforms, data providers and
BPO firms are not counted as GSIs.
Output: results/gsi_focus_output.txt
"""
import contextlib
import io
import traceback

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import rq3_models as rq3m
from common import PROCESSED, RESULTS, ROOT, SETTINGS

CFG = SETTINGS["gsi_focus"]
cls = pd.read_csv(ROOT / "config/gsi_classification.csv", dtype={"cik": str})
df = pd.read_csv(PROCESSED / "panel_firm_quarter.csv", dtype={"cik": str})
ext_path = PROCESSED / "panel_extension.csv"
ext = pd.read_csv(ext_path, dtype={"cik": str}) if ext_path.exists() else None

gsi_ciks = set(cls.loc[cls.is_gsi == 1, "cik"])
tick = cls.set_index("cik")["ticker"]
for d in [df] + ([ext] if ext is not None else []):   # delisted firms (e.g. Thoughtworks) have no ticker
    d["ticker"] = d["ticker"].fillna(d["cik"].map(tick))
df["GSI"] = df["cik"].isin(gsi_ciks).astype(int)
df["YEAR"] = df["FISCAL_QTR"].str[:4]
df["GROUP"] = np.where(df.GSI == 1, "GSIs", "other firms")


def annual_revenue(panel):
    lo, hi = CFG["revenue_window"]
    w = panel[panel.FISCAL_QTR.between(lo, hi)]
    return (w.groupby("cik")["REVENUE_USD"].mean() * 4 / 1e9).round(2)


def section_a():
    rev = annual_revenue(df)
    print(f"A. Who counts as a GSI (annualised {CFG['revenue_window'][0]}-{CFG['revenue_window'][1]} revenue "
          f">= ${CFG['min_annual_revenue_usd_bn']:g}B and a commercial IT-services business)")
    g = cls[cls.is_gsi == 1].copy()
    g["annual_rev_usd_bn"] = g["cik"].map(rev)
    g["rows_in_panel"] = g["cik"].map(df.groupby("cik").size()).fillna(0).astype(int)
    print(g[["market", "ticker", "name", "annual_rev_usd_bn", "rows_in_panel"]].to_string(index=False))
    below = g[g.annual_rev_usd_bn < CFG["min_annual_revenue_usd_bn"]]
    if len(below):
        print("WARNING: listed GSIs below the revenue threshold:", ", ".join(below.name))
    big = set(rev[rev >= CFG["min_annual_revenue_usd_bn"]].index)
    unclassified = sorted(big - set(cls.cik))
    print(f"\n{(cls.is_gsi == 0).sum()} other firms above the threshold are classified as not GSIs "
          "(reasons in config/gsi_classification.csv).")
    if unclassified:
        names = df.drop_duplicates("cik").set_index("cik")["name"]
        print("WARNING: firms above the threshold that are not classified yet:",
              "; ".join(f"{c} {names.get(c, '')}" for c in unclassified))
    print(f"\nGSI firm-quarters in the main panel: {df.GSI.sum()} "
          f"(US {df[(df.GSI == 1) & (df.MARKET == 'US')].shape[0]}, "
          f"India {df[(df.GSI == 1) & (df.MARKET == 'India')].shape[0]})")


def summary(s):
    s = s.dropna()
    return pd.Series({"n": len(s), "mean": s.mean(), "median": s.median(),
                      "trimmed_mean_5pct": stats.trim_mean(s, 0.05) if len(s) else np.nan}).round(2)


def section_b():
    print("B. Year-over-year revenue growth (%) by period")
    d = df[df.PERIOD != "transition"]
    t = d.groupby(["MARKET", "GROUP", "PERIOD"])["REV_GROWTH_YOY"].apply(summary).unstack()
    print(t.to_string())
    print("\nMedian growth (%) by year (2026 from the extension panel where eligible)")
    years = d[["MARKET", "GROUP", "YEAR", "REV_GROWTH_YOY", "PERIOD"]]
    if ext is not None:
        e = ext[ext.SAMPLE == "extension"].copy()
        e["GROUP"] = np.where(e.cik.isin(gsi_ciks), "GSIs", "other firms")
        e["YEAR"] = e.FISCAL_QTR.str[:4]
        years = pd.concat([years, e[["MARKET", "GROUP", "YEAR", "REV_GROWTH_YOY", "PERIOD"]]])
    print(years.pivot_table(index=["MARKET", "GROUP"], columns="YEAR", values="REV_GROWTH_YOY",
                            aggfunc="median").round(1).to_string())


def section_c():
    print("C. RQ1 for GSIs: post-GenAI (2023-2025) vs pre-GenAI (2019-2022Q3)")
    for market in ["US", "India"]:
        g = df[(df.GSI == 1) & (df.MARKET == market)]
        pre = g.loc[g.PERIOD == "pre", "REV_GROWTH_YOY"].dropna()
        post = g.loc[g.PERIOD == "post", "REV_GROWTH_YOY"].dropna()
        t, p = stats.ttest_ind(post, pre, equal_var=False)
        print(f"  [{market} GSIs] n_pre={len(pre)} n_post={len(post)}  mean {pre.mean():.2f} -> {post.mean():.2f} "
              f"(Welch p={p:.4f})  median {pre.median():.2f} -> {post.median():.2f} "
              f"(Mann-Whitney p={stats.mannwhitneyu(post, pre).pvalue:.4f})")
    # firm level: each GSI is one observation, so repeated quarters cannot inflate significance
    f = df[df.GSI == 1].groupby(["MARKET", "cik", "PERIOD"])["REV_GROWTH_YOY"].median().unstack()
    f = f.dropna(subset=["pre", "post"])
    print("\n  Firm level (each GSI's median growth pre vs post; Wilcoxon signed-rank test)")
    for label, sub in [("US", f.loc["US"]), ("India", f.loc["India"]), ("All GSIs", f)]:
        if len(sub) >= 5:
            w = stats.wilcoxon(sub["post"], sub["pre"])
            slowed = int((sub["post"] < sub["pre"]).sum())
            print(f"  [{label}] firms={len(sub)}  slowed: {slowed} of {len(sub)}  median change "
                  f"{(sub['post'] - sub['pre']).median():+.2f} pp  Wilcoxon p={w.pvalue:.4f}")
        else:
            print(f"  [{label}] firms={len(sub)} (too few for a test)")


def section_d():
    print("D. Did GSIs slow more than the other firms in their market? (POST_GENAI x GSI)")
    for market in ["US", "India"]:
        d = df[(df.MARKET == market) & df.PERIOD.isin(["pre", "post"])].dropna(subset=["REV_GROWTH_YOY"])
        if d.GSI.nunique() < 2:
            continue
        m = smf.ols("REV_GROWTH_YOY ~ POST_GENAI * GSI", data=d).fit(cov_type="cluster",
                                                                     cov_kwds={"groups": d["cik"]})
        q = smf.quantreg("REV_GROWTH_YOY ~ POST_GENAI * GSI", data=d).fit(q=0.5)
        b, p = m.params["POST_GENAI:GSI"], m.pvalues["POST_GENAI:GSI"]
        bq, pq = q.params["POST_GENAI:GSI"], q.pvalues["POST_GENAI:GSI"]
        print(f"  [{market}] GSI firms={d[d.GSI == 1].cik.nunique()}, other firms={d[d.GSI == 0].cik.nunique()}")
        print(f"     mean model   (firm-clustered): extra change for GSIs = {b:+.2f} pp, p={p:.4f}")
        print(f"     median model (quantile reg.) : extra change for GSIs = {bq:+.2f} pp, p={pq:.4f}")
    print("  A negative 'extra change' means GSIs slowed more than the other firms after GenAI.")
    print("  U.S. 'other firms' are the rest of the SIC-based sample (software, internet and smaller "
          "IT-services firms), so this compares GSIs with that whole group, not only with small IT-services firms.")


def section_e():
    print("E. Firm by firm (median year-over-year growth, %)")
    g = df[(df.GSI == 1) & (df.PERIOD != "transition")]
    t = g.pivot_table(index=["MARKET", "ticker"], columns="PERIOD", values="REV_GROWTH_YOY", aggfunc="median")
    t["change_pp"] = t["post"] - t["pre"]
    if ext is not None:
        e = ext[(ext.SAMPLE == "extension") & ext.cik.isin(gsi_ciks)]
        t = t.join(e.groupby(["MARKET", "ticker"])["REV_GROWTH_YOY"].median().rename("2026"))
    ai = g[g.PERIOD == "post"].groupby(["MARKET", "ticker"])["AI_INTENSITY"].mean().rename("AI_intensity_post")
    t = t.join(ai)
    names = cls.set_index("ticker")["name"]
    t.index = [f"{m} {tk} ({names.get(tk, tk)})" for m, tk in t.index]
    print(t.round(2).to_string())
    print("  AI intensity is only available for U.S. firms (10-Q/10-K text).")


def section_f():
    print("F. AI disclosure, U.S. firms, post-GenAI quarters (mentions per 10,000 words)")
    d = df[(df.MARKET == "US") & (df.PERIOD == "post")]
    print(d.groupby("GROUP")[["AI_INTENSITY", "GENAI_MENTION"]].agg(["mean", "median"]).round(2).to_string())
    g = d[d.GSI == 1].groupby("cik").agg(ai=("AI_INTENSITY", "mean"), growth=("REV_GROWTH_YOY", "median"))
    if len(g) >= 5:
        rho, p = stats.spearmanr(g.ai, g.growth)
        print(f"  Across the {len(g)} U.S. GSIs: Spearman rho(AI intensity, median growth) = {rho:.2f} (p={p:.3f}). "
              "Descriptive only: too few firms for the RQ2 regression.")


def section_g():
    print("G. Main RQ3 model (random forest, trained 2019-2023, tuned on 2024) on the GSI rows of 2025")
    us = df[df.MARKET == "US"]
    d, X = rq3m.design(us)
    train, valid, test = rq3m.split(d)
    model = rq3m.make_models(SETTINGS["random_seed"])["random_forest"]
    thr = rq3m.fit_and_tune(model, train, valid, X)
    gt = test[test.GSI == 1]
    if gt.empty:
        print("  No GSI rows in the 2025 test year."); return
    p = model.predict_proba(gt[X])[:, 1]
    y = gt[rq3m.TARGET].astype(int)
    flag = p >= thr
    print(f"  GSI rows: {len(gt)}; actual next-quarter declines: {y.sum()}; flagged: {flag.sum()}; "
          f"declines caught: {int((flag & (y == 1)).sum())}; false alarms: {int((flag & (y == 0)).sum())}")
    print("  Too few GSI rows for a separate AUC; read these as counts.")


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print("GSI focus: the study's questions for the large commercial GSIs inside the existing sample.\n"
          "The main analysis (results/analysis_output.txt) and the firm scope are unchanged.\n")
    for name, fn in [("A", section_a), ("B", section_b), ("C", section_c), ("D", section_d),
                     ("E", section_e), ("F", section_f), ("G", section_g)]:
        try:
            fn()
        except Exception:
            print(f"Section {name} could not be run:\n{traceback.format_exc(limit=1)}")
        print()
(RESULTS / "gsi_focus_output.txt").write_text(buf.getvalue())
print(buf.getvalue())
