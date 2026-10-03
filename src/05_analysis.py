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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
    feats = ["AI_INTENSITY", "GENAI_MENTION", "LOG_REVENUE", "SGA_RATIO", "OP_MARGIN",
             "REV_GROWTH_YOY", "REV_GROWTH_LAG1", "POST_GENAI"]
    d3 = us.dropna(subset=feats + ["DECLINE_NEXT_Q", "SEGMENT"]).copy()
    d3 = pd.concat([d3, pd.get_dummies(d3["SEGMENT"], prefix="SEG", drop_first=True, dtype=int)], axis=1)
    X = feats + [c for c in d3.columns if c.startswith("SEG_")]
    train = d3[d3.FISCAL_QTR < "2024Q1"]
    valid = d3[d3.FISCAL_QTR.between("2024Q1", "2024Q4")]
    test = d3[d3.FISCAL_QTR >= "2025Q1"]
    print(f"RQ3  train={len(train)} valid={len(valid)} test={len(test)}  decline rate={d3.DECLINE_NEXT_Q.mean():.3f}")
    models = {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced",
                                                random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="auc", random_state=SEED),
    }
    for name, model in models.items():
        model.fit(train[X], train["DECLINE_NEXT_Q"].astype(int))
        pv = model.predict_proba(valid[X])[:, 1]
        thr = max(np.linspace(0.1, 0.9, 81), key=lambda c: f1_score(valid["DECLINE_NEXT_Q"].astype(int), pv >= c))
        pt = model.predict_proba(test[X])[:, 1]
        y, yhat = test["DECLINE_NEXT_Q"].astype(int), (pt >= thr).astype(int)
        print(f"RQ3  {name:13s} AUC={roc_auc_score(y, pt):.3f} acc={accuracy_score(y, yhat):.3f} "
              f"prec={precision_score(y, yhat, zero_division=0):.3f} rec={recall_score(y, yhat):.3f} "
              f"F1={f1_score(y, yhat):.3f} Brier={brier_score_loss(y, pt):.3f} thr={thr:.2f}")


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
