"""RQ1-RQ4 analyses on the firm-quarter panel.

Primary sample: U.S. SEC registrants (MARKET == "US").
Comparison sample: 15 NSE-listed Indian IT services firms (MARKET == "India"), used in RQ1 as a
second market and in a difference-in-differences style robustness model. The Indian firms have no
10-Q/10-K text, so they do not enter the AI-disclosure analyses (RQ2, RQ3) or RQ4.
Results are printed and also saved to results/analysis_output.txt.
"""
import contextlib
import io
import traceback

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import rq3_models as rq3m
from common import PROCESSED, RESULTS, SETTINGS

SEED = SETTINGS["random_seed"]
df = pd.read_csv(PROCESSED / "panel_firm_quarter.csv")
us = df[df.MARKET == "US"]
india = df[df.MARKET == "India"]


def welch(sub, label):
    pre = sub.loc[sub.PERIOD == "pre", "REV_GROWTH_YOY"].dropna()
    post = sub.loc[sub.PERIOD == "post", "REV_GROWTH_YOY"].dropna()
    if len(pre) < 2 or len(post) < 2:
        print(f"RQ1 [{label}] not enough observations"); return
    t, p = stats.ttest_ind(post, pre, equal_var=False)
    sp = np.sqrt(((len(pre) - 1) * pre.var() + (len(post) - 1) * post.var()) / (len(pre) + len(post) - 2))
    print(f"RQ1 [{label}] n_pre={len(pre)} n_post={len(post)} mean_pre={pre.mean():.2f} mean_post={post.mean():.2f} "
          f"diff={post.mean() - pre.mean():.2f} pp  Welch t={t:.2f} p={p:.4f} d={(post.mean() - pre.mean()) / sp:.3f}")
    print(f"     Mann-Whitney p = {stats.mannwhitneyu(post, pre).pvalue:.4f}")


def rq1():
    welch(us, "US"); welch(india, "India")
    d = df[df.PERIOD.isin(["pre", "post"])].dropna(subset=["REV_GROWTH_YOY"])
    if d.MARKET.nunique() == 2:
        m = smf.ols("REV_GROWTH_YOY ~ POST_GENAI * C(MARKET, Treatment('US'))", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["cik"]})
        print("RQ1 robustness: growth ~ POST x MARKET (firm-clustered SEs)"); print(m.summary().tables[1])


def rq2():
    d2 = us[us.PERIOD == "post"].dropna(subset=["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "SGA_RATIO",
                                                "OP_MARGIN", "REV_GROWTH_LAG1", "SEGMENT"])
    m2 = smf.ols("REV_GROWTH_YOY ~ AI_INTENSITY + LOG_REVENUE + SGA_RATIO + OP_MARGIN + REV_GROWTH_LAG1 + C(SEGMENT)",
                 data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2["cik"]})
    print("RQ2  N =", int(m2.nobs)); print(m2.summary().tables[1])
    r = m2.resid
    print(f"     R2={m2.rsquared:.3f} adjR2={m2.rsquared_adj:.3f} RMSE={np.sqrt((r ** 2).mean()):.2f} MAE={r.abs().mean():.2f}")


def rq3():
    d3, X = rq3m.design(us)
    train, valid, test = rq3m.split(d3)
    print(f"RQ3  train={len(train)} valid={len(valid)} test={len(test)}  decline rate={d3.DECLINE_NEXT_Q.mean():.3f}")
    for name, model in rq3m.make_models(SEED).items():
        thr = rq3m.fit_and_tune(model, train, valid, X)
        print(f"RQ3  {name:13s} {rq3m.scores(model, test, X, thr)}")


def rq4():
    d4 = us[us.PERIOD == "post"].dropna(subset=["SEGMENT", "DECLINE_Q"])
    ct = pd.crosstab(d4["SEGMENT"], d4["DECLINE_Q"])
    chi2, p4, dof, exp = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print("RQ4 contingency table\n", ct)
    print(f"     chi2={chi2:.2f} df={dof} p={p4:.4f} Cramer's V={v:.3f}  min expected={exp.min():.1f}")
    print("     standardized residuals\n", (ct - exp) / np.sqrt(exp))


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print(f"Panel: {len(df):,} firm-quarters ({len(us):,} US, {len(india):,} India); "
          f"{df.cik.nunique()} firms\n")
    for name, fn in [("RQ1", rq1), ("RQ2", rq2), ("RQ3", rq3), ("RQ4", rq4)]:
        try:
            fn()
        except Exception:
            print(f"{name} could not be run:\n{traceback.format_exc(limit=1)}")
        print()
(RESULTS / "analysis_output.txt").write_text(buf.getvalue())
print(buf.getvalue())
