"""Model diagnostics and evaluation details for the interim report (runs after src/05_analysis.py).

Adds what the interim report needs beyond the hypothesis tests in 05:
  RQ1  primary test without 2020; growth by year
  RQ2  multicollinearity (VIF), heteroscedasticity (Breusch-Pagan), residual normality, random forest + SHAP
  RQ3  model chosen on 2024 validation AUC, confusion matrices, Wilson intervals, expected cost, acceptance
       criteria, ablation without the AI features, SHAP importance, ROC curves
  RQ4  decline rate by segment with firm-bootstrap intervals and GEE odds ratios with 95% CIs
Outputs: results/model_details.txt and results/figures/model_*.png
"""
import contextlib
import io
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

import rq3_models as rq3m
from common import PROCESSED, RESULTS, SETTINGS

SEED = SETTINGS["random_seed"]
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e7ee", "axes.axisbelow": True})
INK, MUTED = "#1b2a3a", "#5a6b7d"
MODEL_C = {"logit": "#5b9bd5", "random_forest": "#1f5a96", "xgboost": "#c95b37"}
LABEL = {"logit": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}

df = pd.read_csv(PROCESSED / "panel_firm_quarter.csv", dtype={"cik": str})
us = df[df.MARKET == "US"].copy()


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)

    # ------------------------------------------------------------ RQ1
    print("RQ1 DETAILS")
    d = us[us.PERIOD.isin(["pre", "post"])].dropna(subset=["REV_GROWTH_YOY"]).copy()
    d["YEAR"] = d.FISCAL_QTR.str[:4].astype(int)
    for label, sub in [("all quarters", d), ("excluding 2020", d[d.YEAR != 2020])]:
        m = smf.ols("REV_GROWTH_YOY ~ POST_GENAI", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["cik"]})
        ci = m.conf_int().loc["POST_GENAI"]
        print(f"   {label:15s} N={int(m.nobs)} firms={sub.cik.nunique()} b={m.params['POST_GENAI']:.2f} pp "
              f"95% CI {ci[0]:.2f} to {ci[1]:.2f} p={m.pvalues['POST_GENAI']:.4f}")
    med = smf.quantreg("REV_GROWTH_YOY ~ POST_GENAI", d).fit(q=0.5)
    print(f"   median regression: change in median growth = {med.params['POST_GENAI']:.2f} pp "
          f"(95% CI {med.conf_int().loc['POST_GENAI', 0]:.2f} to {med.conf_int().loc['POST_GENAI', 1]:.2f}; "
          f"not clustered, descriptive)")

    # ------------------------------------------------------------ RQ2
    print("\nRQ2 DETAILS")
    base = ["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "OP_MARGIN", "REV_GROWTH_LAG1", "IS_10K", "SEGMENT"]
    d2 = us[us.PERIOD == "post"].dropna(subset=base).copy()
    d2["IS_10K"] = d2["IS_10K"].astype(float)
    f = "REV_GROWTH_YOY ~ AI_INTENSITY + LOG_REVENUE + OP_MARGIN + REV_GROWTH_LAG1 + IS_10K + C(SEGMENT)"
    ols = smf.ols(f, data=d2).fit()
    X = ols.model.exog
    names = ols.model.exog_names
    print("   variance inflation factors:")
    for i, n in enumerate(names):
        if n != "Intercept":
            print(f"      {n:55s} {variance_inflation_factor(X, i):.2f}")
    print(f"   plain OLS (no clustering, baseline): AI_INTENSITY b = {ols.params['AI_INTENSITY']:.3f}, "
          f"SE = {ols.bse['AI_INTENSITY']:.3f}, p = {ols.pvalues['AI_INTENSITY']:.4f}")
    bp = het_breuschpagan(ols.resid, X)
    print(f"   Breusch-Pagan LM = {bp[0]:.1f}, p = {bp[1]:.4g}  (heteroscedasticity -> clustered/robust SEs used)")
    jb = stats.jarque_bera(ols.resid)
    print(f"   residual skewness = {stats.skew(ols.resid):.2f}, kurtosis = {stats.kurtosis(ols.resid, fisher=False):.1f}, "
          f"Jarque-Bera p = {jb.pvalue:.3g}")
    rob = smf.rlm(f, data=d2, M=sm.robust.norms.HuberT()).fit()
    print(f"   robust (Huber) regression: AI_INTENSITY b = {rob.params['AI_INTENSITY']:.3f}, "
          f"SE = {rob.bse['AI_INTENSITY']:.3f}, p = {rob.pvalues['AI_INTENSITY']:.4f}")
    dd = d2.copy()
    for c in ["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "OP_MARGIN", "REV_GROWTH_LAG1"]:
        dd[c] = (dd[c] - dd[c].mean()) / dd[c].std()
    std = smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["cik"]})
    print(f"   standardized coefficient of AI_INTENSITY = {std.params['AI_INTENSITY']:.3f} SD of growth per SD of AI")
    feats2 = ["AI_INTENSITY", "LOG_REVENUE", "OP_MARGIN", "REV_GROWTH_LAG1", "IS_10K"]
    Xr = pd.concat([d2[feats2], pd.get_dummies(d2.SEGMENT, prefix="SEG", drop_first=True, dtype=int)], axis=1)
    rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=10, oob_score=True, random_state=SEED, n_jobs=-1)
    rf.fit(Xr, d2.REV_GROWTH_YOY)
    sv = shap.TreeExplainer(rf).shap_values(Xr)
    imp = pd.Series(np.abs(sv).mean(0), index=Xr.columns).sort_values(ascending=False)
    print(f"   random forest regressor: out-of-bag R2 = {rf.oob_score_:.3f}; mean |SHAP| (pp of growth):")
    for k, v in imp.items():
        print(f"      {k:40s} {v:.2f}")
    ai_sv = pd.DataFrame({"ai": Xr.AI_INTENSITY, "shap": sv[:, list(Xr.columns).index("AI_INTENSITY")]})
    ai_sv["bin"] = pd.cut(ai_sv.ai, [-0.01, 0, 1, 3, 6, 10, 100])
    print("   average SHAP contribution of AI_INTENSITY by AI level (pp of growth):")
    print(ai_sv.groupby("bin", observed=True).shap.agg(["count", "mean"]).round(2).to_string())

    # ------------------------------------------------------------ RQ3
    print("\nRQ3 DETAILS")
    d3, Xf = rq3m.design(us)
    train, valid, test = rq3m.split(d3)
    y_te = test[rq3m.TARGET].astype(int).values
    y_va = valid[rq3m.TARGET].astype(int).values
    base_p = test["DECLINE_Q"].astype(float).values
    prev_te = y_te.mean()
    print(f"   test prevalence = {prev_te:.3f}; no-skill Brier p(1-p) = {prev_te * (1 - prev_te):.3f}")
    rows, probs, models = [], {}, {}
    for name, model in rq3m.make_models(SEED).items():
        thr = rq3m.fit_and_tune(model, train, valid, Xf)
        models[name] = (model, thr)
        pv, pt = model.predict_proba(valid[Xf])[:, 1], model.predict_proba(test[Xf])[:, 1]
        probs[name] = pt
        yhat = (pt >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
        rec_ci, prec_ci = wilson(tp, tp + fn), wilson(tp, tp + fp)
        brier = float(np.mean((pt - y_te) ** 2))
        auc = roc_auc_score(y_te, pt)
        se = rq3m.delong_se(y_te, pt)
        _, _, z, pdl = rq3m.delong(y_te, pt, base_p)
        rows.append(dict(model=name, val_auc=roc_auc_score(y_va, pv), threshold=thr, test_auc=auc,
                         auc_lo=auc - 1.96 * se, auc_hi=auc + 1.96 * se, TP=tp, FP=fp, TN=tn, FN=fn,
                         recall=tp / (tp + fn), rec_lo=rec_ci[0], rec_hi=rec_ci[1], precision=tp / (tp + fp),
                         prec_lo=prec_ci[0], prec_hi=prec_ci[1], accuracy=(tp + tn) / len(y_te),
                         f1=2 * tp / (2 * tp + fp + fn), brier=brier, cost=3 * fn + fp, delong_vs_base_p=pdl))
    tb = pd.DataFrame(rows).set_index("model")
    yb = base_p.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yb).ravel()
    print(f"   persistence baseline: AUC={roc_auc_score(y_te, base_p):.3f} TP={tp} FP={fp} TN={tn} FN={fn} "
          f"recall={tp / (tp + fn):.3f} precision={tp / (tp + fp):.3f} cost={3 * fn + fp}")
    print(tb.round(3).T.to_string())
    best = tb.val_auc.idxmax()
    print(f"   model chosen before opening the test set (highest 2024 validation AUC): {best}")
    r = tb.loc[best]
    crit = [("lower 95% DeLong bound of AUC > 0.50", r.auc_lo > 0.5),
            ("AUC >= 0.70", r.test_auc >= 0.70),
            ("AUC above persistence baseline (DeLong p < .05)", r.delong_vs_base_p < 0.05),
            ("recall >= 0.70", r.recall >= 0.70),
            ("precision >= 0.50", r.precision >= 0.50),
            ("Brier below no-skill p(1-p)", r.brier < prev_te * (1 - prev_te))]
    print("   acceptance criteria (Table 9 of the synopsis) for the chosen model:")
    for c, ok in crit:
        print(f"      {c:50s} {'met' if ok else 'NOT met'}")
    # ablation: drop the two AI features
    no_ai = [c for c in Xf if c not in ("AI_INTENSITY", "GENAI_MENTION")]
    print("   ablation (same model without AI_INTENSITY and GENAI_MENTION):")
    for name, model in rq3m.make_models(SEED).items():
        thr = rq3m.fit_and_tune(model, train, valid, no_ai)
        p0 = model.predict_proba(test[no_ai])[:, 1]
        a1, a0, z, pdl = rq3m.delong(y_te, probs[name], p0)
        print(f"      {name:13s} AUC with AI {a1:.3f}, without {a0:.3f}, difference {a1 - a0:+.3f} "
              f"(DeLong z={z:.2f}, p={pdl:.3f})")
    # sensitivity: purge the last quarter of training and validation, whose outcome falls in the next split
    tr_p = train[train.FISCAL_QTR != "2023Q4"]; va_p = valid[valid.FISCAL_QTR != "2024Q4"]
    print(f"   purged-boundary sensitivity (training without 2023Q4, validation without 2024Q4; same models, "
          f"same threshold rule; train={len(tr_p)}, valid={len(va_p)}):")
    for name, model in rq3m.make_models(SEED).items():
        thr = rq3m.fit_and_tune(model, tr_p, va_p, Xf)
        pp = model.predict_proba(test[Xf])[:, 1]
        yh = (pp >= thr).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_te, yh).ravel()
        print(f"      {name:13s} test AUC {roc_auc_score(y_te, pp):.3f} (main {tb.loc[name, 'test_auc']:.3f}); "
              f"threshold {thr:.2f}; recall {tp_ / (tp_ + fn_):.3f}; precision {tp_ / (tp_ + fp_):.3f}")
    # SHAP for the chosen model (tree models) on the test set
    mdl = models[best][0]
    if best in ("random_forest", "xgboost"):
        ex = shap.TreeExplainer(mdl)
        s = ex.shap_values(test[Xf])
        if isinstance(s, list):
            s = s[1]
        if s.ndim == 3:
            s = s[:, :, 1]
        shap_imp = pd.Series(np.abs(s).mean(0), index=Xf).sort_values(ascending=False)
        print(f"   mean |SHAP| on the 2025 test set ({best}):")
        for k, v in shap_imp.items():
            print(f"      {k:45s} {v:.4f}")

    # ------------------------------------------------------------ RQ4
    print("\nRQ4 DETAILS")
    d4 = us[us.PERIOD == "post"].dropna(subset=["SEGMENT", "DECLINE_Q"]).copy()
    d4["DECLINE_Q"] = d4.DECLINE_Q.astype(int)
    gee = smf.gee("DECLINE_Q ~ C(SEGMENT)", groups="cik", data=d4, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = gee.conf_int()
    for t in gee.params.index:
        if t.startswith("C(SEGMENT)"):
            print(f"   odds ratio {t[len('C(SEGMENT)[T.'):-1]:40s} {np.exp(gee.params[t]):.2f} "
                  f"(95% CI {np.exp(ci.loc[t, 0]):.2f} to {np.exp(ci.loc[t, 1]):.2f})")
    rng = np.random.default_rng(SEED)
    seg_rows = []
    for sname, g in d4.groupby("SEGMENT"):
        firms = g.cik.unique()
        idx = {f: g.index[g.cik == f] for f in firms}
        boots = []
        for _ in range(2000):
            pick = rng.choice(firms, len(firms), replace=True)
            rows_ = np.concatenate([idx[f] for f in pick])
            boots.append(g.loc[rows_, "DECLINE_Q"].mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        seg_rows.append((sname, g.DECLINE_Q.mean(), lo, hi, len(firms), len(g)))
        print(f"   {sname:40s} (2,000 firm-bootstrap resamples) decline share {g.DECLINE_Q.mean():.3f} (firm-bootstrap 95% CI {lo:.3f} to {hi:.3f}; "
              f"{len(firms)} firms, {len(g)} firm-quarters)")
(RESULTS / "model_details.txt").write_text(buf.getvalue())
print(buf.getvalue())

# ------------------------------------------------------------ figures
fig, ax = plt.subplots(figsize=(8.5, 7.5))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, color=MODEL_C[name], lw=2.4, label=f"{LABEL[name]} (AUC {roc_auc_score(y_te, p):.3f})")
fpr, tpr, _ = roc_curve(y_te, base_p)
ax.plot(fpr, tpr, color="#5a6b7d", lw=2, ls="--", marker="o",
        label=f"Persistence baseline (AUC {roc_auc_score(y_te, base_p):.3f})")
ax.plot([0, 1], [0, 1], color="#c3c9d2", lw=1)
ax.set_xlabel("False positive rate (1 − specificity)", color=MUTED)
ax.set_ylabel("True positive rate (recall)", color=MUTED)
ax.legend(frameon=False, loc="lower right")
ax.set_aspect("equal")
fig.tight_layout(); fig.savefig(FIG / "model_roc_test2025.png", dpi=110); plt.close(fig)

if best in ("random_forest", "xgboost"):
    top = shap_imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color="#1f5a96")
    for i, v in enumerate(top.values):
        ax.text(v + top.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=10.5, color=INK)
    ax.set_xlabel(f"Mean |SHAP value| on the 2025 test set ({LABEL[best]}, probability scale)", color=MUTED)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "model_shap_rq3.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4.8))
seg_rows.sort(key=lambda r: r[1])
for i, (sname, m, lo, hi, nf, nq) in enumerate(seg_rows):
    ax.plot([lo * 100, hi * 100], [i, i], color="#5b9bd5", lw=6, solid_capstyle="round")
    ax.plot(m * 100, i, "o", color="#1f5a96", ms=10)
    ax.text(hi * 100 + 1, i, f"{m * 100:.1f}%  ({nf} firms)", va="center", fontsize=11, color=INK)
ax.set_yticks(range(len(seg_rows))); ax.set_yticklabels([r[0] for r in seg_rows])
ax.set_xlim(0, 65)
ax.set_xlabel("Share of post-GenAI firm-quarters in decline, with firm-bootstrap 95% interval (%)", color=MUTED)
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "model_rq4_segments.png", dpi=110); plt.close(fig)
