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


def firm_paired(sub, label):
    """Firm-level paired test: each firm's mean growth after minus before, one value per firm."""
    m = sub[sub.PERIOD.isin(["pre", "post"])].groupby(["cik", "PERIOD"]).REV_GROWTH_YOY.mean().unstack().dropna()
    if len(m) < 3:
        print(f"RQ1 [{label}] firm-level: not enough firms"); return
    diff = m["post"] - m["pre"]
    t, p = stats.ttest_1samp(diff, 0)
    print(f"RQ1 [{label}] firm-level paired: firms={len(m)} mean change={diff.mean():.2f} pp "
          f"median change={diff.median():.2f} pp t={t:.2f} p={p:.4f} dz={diff.mean() / diff.std(ddof=1):.3f}  "
          f"Wilcoxon p={stats.wilcoxon(diff).pvalue:.4f}")


def rq1():
    # primary: firm-quarter regression with standard errors clustered by firm (respects repeated quarters)
    d = us[us.PERIOD.isin(["pre", "post"])].dropna(subset=["REV_GROWTH_YOY"])
    m = smf.ols("REV_GROWTH_YOY ~ POST_GENAI", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
    ci = m.conf_int().loc["POST_GENAI"]
    print(f"RQ1 [US] primary: growth ~ POST_GENAI, firm-clustered SEs: N={int(m.nobs)} firms={d.cik.nunique()} "
          f"b={m.params['POST_GENAI']:.2f} pp SE={m.bse['POST_GENAI']:.2f} 95% CI {ci[0]:.2f} to {ci[1]:.2f} "
          f"p={m.pvalues['POST_GENAI']:.4f}")
    firm_paired(us, "US"); firm_paired(india, "India")
    # descriptive / robustness: the quarter-level Welch and Mann-Whitney tests from the synopsis
    welch(us, "US descriptive"); welch(india, "India descriptive")
    d = df[df.PERIOD.isin(["pre", "post"])].dropna(subset=["REV_GROWTH_YOY"])
    if d.MARKET.nunique() == 2:
        m = smf.ols("REV_GROWTH_YOY ~ POST_GENAI * C(MARKET, Treatment('US'))", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["cik"]})
        print("RQ1 robustness: growth ~ POST x MARKET (firm-clustered SEs)"); print(m.summary().tables[1])


def rq2():
    base = ["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "OP_MARGIN", "REV_GROWTH_LAG1", "IS_10K", "SEGMENT"]
    d2 = us[us.PERIOD == "post"].dropna(subset=base).copy()
    d2["IS_10K"] = d2["IS_10K"].astype(float)
    f = "REV_GROWTH_YOY ~ AI_INTENSITY + LOG_REVENUE + OP_MARGIN + REV_GROWTH_LAG1 + IS_10K + C(SEGMENT)"
    m2 = smf.ols(f, data=d2).fit(cov_type="cluster", cov_kwds={"groups": d2["cik"]})
    print(f"RQ2 primary  N = {int(m2.nobs)}  firms = {d2.cik.nunique()}"); print(m2.summary().tables[1])
    r = m2.resid
    print(f"     R2={m2.rsquared:.3f} adjR2={m2.rsquared_adj:.3f} RMSE={np.sqrt((r ** 2).mean()):.2f} MAE={r.abs().mean():.2f}")

    def line(label, model, term="AI_INTENSITY"):
        print(f"RQ2 {label}: N={int(model.nobs)} b={model.params[term]:.3f} SE={model.bse[term]:.3f} "
              f"p={model.pvalues[term]:.4f}")

    fit = lambda formula, d: smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
    line("+ quarter fixed effects", fit(f + " + C(FISCAL_QTR)", d2))
    d_sga = d2.dropna(subset=["SGA_RATIO"])
    line(f"+ SGA_RATIO (firms reporting SG&A, {d_sga.cik.nunique()} firms)", fit(f + " + SGA_RATIO", d_sga))
    d_4q = d2.dropna(subset=["AI_INTENSITY_4Q"])
    line("four-quarter average AI score", fit(f.replace("AI_INTENSITY", "AI_INTENSITY_4Q"), d_4q), "AI_INTENSITY_4Q")


def rq3():
    d3, X = rq3m.design(us)
    train, valid, test = rq3m.split(d3)
    print(f"RQ3  train={len(train)} valid={len(valid)} test={len(test)}  decline rate={d3.DECLINE_NEXT_Q.mean():.3f}")
    print(f"     test 2025: firms={test.cik.nunique()} declines={int(test.DECLINE_NEXT_Q.sum())} "
          f"non-declines={int((1 - test.DECLINE_NEXT_Q).sum())}")
    y = test[rq3m.TARGET].astype(int)
    base_p = test["DECLINE_Q"].astype(float)   # persistence baseline: "a firm declining now declines next quarter"
    print(f"RQ3  persistence baseline AUC={rq3m.roc_auc_score(y, base_p):.3f}  "
          f"prevalence-only Brier={rq3m.brier_score_loss(y, np.full(len(y), train[rq3m.TARGET].mean())):.3f}")
    for name, model in rq3m.make_models(SEED).items():
        thr = rq3m.fit_and_tune(model, train, valid, X)
        print(f"RQ3  {name:13s} {rq3m.scores(model, test, X, thr)}")
        p = model.predict_proba(test[X])[:, 1]
        lo, hi = rq3m.firm_bootstrap_auc(y, p, test["cik"], seed=SEED)
        se = rq3m.delong_se(y, p)
        a1, a2, z, pv = rq3m.delong(y, p, base_p)
        auc = rq3m.roc_auc_score(y, p)
        print(f"     DeLong 95% CI {auc - 1.96 * se:.3f} to {auc + 1.96 * se:.3f}; firm-bootstrap 95% CI {lo:.3f} to {hi:.3f}; "
              f"vs persistence baseline: diff={a1 - a2:.3f} z={z:.2f} p={pv:.4f}")


def rq4():
    d4 = us[us.PERIOD == "post"].dropna(subset=["SEGMENT", "DECLINE_Q"]).copy()
    d4["DECLINE_Q"] = d4["DECLINE_Q"].astype(int)
    ct = pd.crosstab(d4["SEGMENT"], d4["DECLINE_Q"])
    chi2, p4, dof, exp = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print("RQ4 contingency table (descriptive)\n", ct)
    print(f"     Pearson chi2={chi2:.2f} df={dof} p={p4:.4f} Cramer's V={v:.3f}  min expected={exp.min():.1f} "
          f"(treats firm-quarters as independent; descriptive only)")
    print("     standardized residuals\n", (ct - exp) / np.sqrt(exp))
    # primary: logistic regression with firm-clustered (GEE, exchangeable) errors, Wald test of segment
    import statsmodels.api as sm
    gee = smf.gee("DECLINE_Q ~ C(SEGMENT)", groups="cik", data=d4, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    terms = [t for t in gee.params.index if t.startswith("C(SEGMENT)")]
    R = np.zeros((len(terms), len(gee.params)))
    for i, t in enumerate(terms):
        R[i, list(gee.params.index).index(t)] = 1
    w = gee.wald_test(R, scalar=True)
    print(f"RQ4 primary: GEE logistic, firms={d4.cik.nunique()} firm-quarters={len(d4)}, "
          f"working correlation={gee.cov_struct.dep_params:.3f}; Wald chi2={float(w.statistic):.2f} "
          f"df={len(terms)} p={float(w.pvalue):.4f}")
    print("     odds ratios vs " + sorted(d4.SEGMENT.unique())[0] + ": " +
          "; ".join(f"{t[len('C(SEGMENT)[T.'):-1]} {np.exp(gee.params[t]):.2f}" for t in terms))
    # Rao-Scott style first-order correction: divide Pearson chi2 by the design effect
    icc = gee.cov_struct.dep_params
    m = len(d4) / d4.cik.nunique()
    deff = 1 + (m - 1) * icc
    print(f"     design-effect-adjusted chi2 = {chi2:.2f} / {deff:.2f} = {chi2 / deff:.2f}, "
          f"p={1 - stats.chi2.cdf(chi2 / deff, dof):.4f}")
    # firm-level check: each firm's share of declining quarters, compared across segments
    share = d4.groupby(["cik", "SEGMENT"]).DECLINE_Q.mean().reset_index()
    groups = [g.DECLINE_Q.values for _, g in share.groupby("SEGMENT")]
    print(f"     firm-level decline share by segment (Kruskal-Wallis on {len(share)} firms): "
          f"H={stats.kruskal(*groups).statistic:.2f} p={stats.kruskal(*groups).pvalue:.4f}; medians: " +
          "; ".join(f"{s} {g.DECLINE_Q.median():.2f}" for s, g in share.groupby("SEGMENT")))


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
