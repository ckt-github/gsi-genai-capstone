"""Sensitivity and robustness checks added after the interim review.

  1. RQ1: how much of the post-GenAI drop is the end of the 2021 boom (confounding checks)
  2. RQ2: the AI coefficient under quarter and firm fixed effects, with confidence intervals
  3. RQ3: winsorizing limits from the training years only (now the main specification) vs. full window
  4. RQ3: features rebuilt from first-reported XBRL values (as known on each prediction date)
  5. RQ3: rolling-origin evaluation over several test years
  6. RQ3: uncertainty of the operating point (firm-bootstrap recall and precision, alert volume)
  7. Narrower "services core" sample: the large firms classified as non-services removed (SIC breadth)
  8. RQ4: confidence intervals for the differences in decline share between segments
  9. AI keyword dictionary: precision on 50 randomly drawn matched passages

Inputs: data/processed/panel_firm_quarter.csv, data/interim/panel_prewinsor.csv,
        data/interim/xbrl_vintages.csv.gz and ai_match_contexts.csv.gz (from 10_sensitivity_data.py),
        data/interim/ai_precision_labels.csv (manual review labels: label 1 = the match refers to artificial
        intelligence, 0 = false match; context_type = what the passage is about), config/gsi_classification.csv
Output: results/sensitivity_output.txt, results/ai_precision_sample.csv
"""
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix, roc_auc_score

import rq3_models as rq3m
from common import INTERIM, PROCESSED, RESULTS, ROOT, SETTINGS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e7ee", "axes.axisbelow": True})
INK, MUTED = "#1b2a3a", "#5a6b7d"
ROLL = []
SEED = SETTINGS["random_seed"]
LO, HI = SETTINGS["winsorize"]
df = pd.read_csv(PROCESSED / "panel_firm_quarter.csv", dtype={"cik": str})
us = df[df.MARKET == "US"].copy()


def clustered(formula, d, term):
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["cik"]})
    ci = m.conf_int().loc[term]
    return m, f"b={m.params[term]:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f}) p={m.pvalues[term]:.4f} N={int(m.nobs)} firms={d.cik.nunique()}"


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def fit_rf(train, valid, X):
    model = rq3m.make_models(SEED)["random_forest"]
    thr = rq3m.fit_and_tune(model, train, valid, X)
    return model, thr


# ------------------------------------------------------------------ 1. RQ1 confounding
def rq1_confounding():
    print("1. RQ1 CONFOUNDING CHECKS (U.S., firm-clustered SEs)")
    us["YEAR"] = us.FISCAL_QTR.str[:4].astype(int)
    by_year = us[us.PERIOD != "transition"].groupby("YEAR").REV_GROWTH_YOY.agg(["count", "mean", "median"])
    print("   growth by calendar year (transition quarter excluded):\n" + by_year.round(2).to_string())
    post = us[us.PERIOD == "post"]
    for label, pre_rows in [("post vs. 2019 only (pre-pandemic baseline)", us[us.YEAR == 2019]),
                            ("post vs. 2019-2020 (before the 2021 boom)", us[us.YEAR.isin([2019, 2020])]),
                            ("post vs. 2021-2022Q3 (boom years)", us[(us.YEAR >= 2021) & (us.PERIOD == "pre")])]:
        d = pd.concat([pre_rows.assign(POSTX=0), post.assign(POSTX=1)])
        print(f"   {label}: {clustered('REV_GROWTH_YOY ~ POSTX', d, 'POSTX')[1]}")
    # pre-existing slowdown: 2022Q1-Q3 against 2021, both before ChatGPT
    d = us[us.FISCAL_QTR.between("2021Q1", "2022Q3")].assign(Y2022=lambda x: (x.YEAR == 2022).astype(int))
    print(f"   slowdown already under way before ChatGPT, 2022Q1-Q3 vs. 2021: {clustered('REV_GROWTH_YOY ~ Y2022', d, 'Y2022')[1]}")
    # placebo break inside the pre-GenAI period at 2021Q1
    pre = us[us.PERIOD == "pre"].assign(LATE=lambda x: (x.FISCAL_QTR >= "2021Q1").astype(int))
    print(f"   placebo break at 2021Q1 inside the pre-GenAI period: {clustered('REV_GROWTH_YOY ~ LATE', pre, 'LATE')[1]}")
    # median quarterly growth at the end of each window
    med = us.groupby("FISCAL_QTR").REV_GROWTH_YOY.median()
    print("   median growth 2021Q4, 2022Q1-Q4, 2023Q1: " + "; ".join(f"{q} {med[q]:.1f}%" for q in
                                                                  ["2021Q4", "2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1"]))


# ------------------------------------------------------------------ 2. RQ2 fixed effects
def rq2_fixed_effects():
    print("\n2. RQ2 AI COEFFICIENT UNDER FIXED EFFECTS (post-GenAI U.S., firm-clustered SEs)")
    base = ["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "OP_MARGIN", "REV_GROWTH_LAG1", "IS_10K", "SEGMENT"]
    d2 = us[us.PERIOD == "post"].dropna(subset=base).copy()
    d2["IS_10K"] = d2.IS_10K.astype(float)
    f = "REV_GROWTH_YOY ~ AI_INTENSITY + LOG_REVENUE + OP_MARGIN + REV_GROWTH_LAG1 + IS_10K"
    for label, formula in [("primary (segment dummies)", f + " + C(SEGMENT)"),
                           ("+ quarter fixed effects", f + " + C(SEGMENT) + C(FISCAL_QTR)"),
                           ("+ year fixed effects", f + " + C(SEGMENT) + C(FISCAL_QTR.str[:4])"),
                           ("firm fixed effects (within-firm change)", f + " + C(cik)"),
                           ("firm and quarter fixed effects", f + " + C(cik) + C(FISCAL_QTR)")]:
        print(f"   {label:42s} {clustered(formula, d2, 'AI_INTENSITY')[1]}")
    fseg = f + " + C(SEGMENT)"
    d_sga = d2.dropna(subset=["SGA_RATIO"])
    print(f"   {'+ SGA_RATIO (firms reporting SG&A)':42s} {clustered(fseg + ' + SGA_RATIO', d_sga, 'AI_INTENSITY')[1]}")
    d_4q = d2.dropna(subset=["AI_INTENSITY_4Q"])
    print(f"   {'four-quarter average AI score':42s} "
          f"{clustered(fseg.replace('AI_INTENSITY', 'AI_INTENSITY_4Q'), d_4q, 'AI_INTENSITY_4Q')[1]}")
    rob = smf.rlm(fseg, data=d2, M=sm.robust.norms.HuberT()).fit()
    ci = rob.conf_int().loc["AI_INTENSITY"]
    print(f"   {'robust (Huber) regression':42s} b={rob.params['AI_INTENSITY']:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f}) "
          f"p={rob.pvalues['AI_INTENSITY']:.4f} N={int(rob.nobs)}")


# ------------------------------------------------------------------ 3. RQ3 winsorizing limits
def rq3_caps():
    print("\n3. RQ3 WINSORIZING LIMITS: TRAINING YEARS ONLY (MAIN) VS. FULL 2019-2025 WINDOW (INTERIM DRAFT)")
    _, lim = rq3m.train_caps(us)
    for c in rq3m.CAPPED:
        print(f"   {c:16s} training-only limits {lim[c][0]:8.2f} to {lim[c][1]:8.2f}; "
              f"full-window limits {us[c].min():8.2f} to {us[c].max():8.2f}")
    for label, flag in [("training-only caps (main)", True), ("full-window caps (draft)", False)]:
        d3, X = rq3m.design(us, train_only_caps=flag)
        tr, va, te = rq3m.split(d3)
        m, thr = fit_rf(tr, va, X)
        p = m.predict_proba(te[X])[:, 1]
        y = te[rq3m.TARGET].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p >= thr).ravel()
        print(f"   random forest, {label:26s} test AUC {roc_auc_score(y, p):.3f}; threshold {thr:.2f}; "
              f"recall {tp / (tp + fn):.3f}; precision {tp / (tp + fp):.3f}")


# ------------------------------------------------------------------ 4. first-reported values
def quarterly(df):
    q = df[df["days"].between(80, 100)][["end", "val"]]
    fy = df[df["days"].between(350, 380)][["start", "end", "val"]]
    derived = []
    for _, y in fy.iterrows():
        if (q["end"] == y["end"]).any():
            continue
        inside = q[(q["end"] > y["start"]) & (q["end"] < y["end"])]
        if len(inside) == 3:
            derived.append({"end": y["end"], "val": y["val"] - inside["val"].sum()})
    return pd.concat([q, pd.DataFrame(derived)]).drop_duplicates("end").set_index("end")["val"]


def snapshots(g):
    """For one firm and concept: the quarterly series as it stood after each filing date.
    Uses the same rules as 02_get_financials.py (latest filing wins, then the tag order), but only
    with facts filed on or before that date."""
    out = {}
    for d in sorted(g["filed"].unique()):
        known = g[g["filed"] <= d].sort_values(["filed", "tag_rank"], ascending=[False, True])
        s = quarterly(known.drop_duplicates(["start", "end"]))
        s.index = (pd.to_datetime(s.index) - pd.Timedelta(days=45)).to_period("Q")
        out[d] = s[~s.index.duplicated(keep="last")]
    return out


def first_reported_features():
    path = INTERIM / "xbrl_vintages.csv.gz"
    if not path.exists():
        return None
    v = pd.read_csv(path, parse_dates=["start", "end"])
    v["days"] = (v["end"] - v["start"]).dt.days
    v["cik"] = v["cik"].astype(str)
    need = set(us.cik)
    rows = []
    for cik, g in v[v.cik.isin(need)].groupby("cik"):
        rev = snapshots(g[g.concept == "revenue"]) if (g.concept == "revenue").any() else {}
        oi = snapshots(g[g.concept == "operating_income"]) if (g.concept == "operating_income").any() else {}
        if not rev:
            continue
        dates = sorted(rev)
        first_date = {}
        for d in dates:                                   # date each quarter's revenue was first known
            for q in rev[d].index:
                first_date.setdefault(q, d)
        oi_dates = sorted(oi)
        for q, d in first_date.items():
            s = rev[d]
            r = {"cik": cik, "FISCAL_QTR": str(q), "known_on": d, "rev_t": s.get(q),
                 "rev_t4": s.get(q - 4), "rev_t1": s.get(q - 1), "rev_t5": s.get(q - 5)}
            od = [x for x in oi_dates if x <= d]
            r["oi_t"] = oi[od[-1]].get(q) if od else np.nan
            rows.append(r)
    fr = pd.DataFrame(rows)
    floor = SETTINGS.get("min_prior_revenue_usd", 1_000_000)
    g = lambda a, b: np.where(fr[b] >= floor, (fr[a] - fr[b]) / fr[b].abs() * 100, np.nan)
    fr["REV_GROWTH_YOY"] = g("rev_t", "rev_t4")
    fr["REV_GROWTH_LAG1"] = g("rev_t1", "rev_t5")
    fr["OP_MARGIN"] = fr["oi_t"] / fr["rev_t"] * 100
    fr["LOG_REVENUE"] = np.log(fr["rev_t"].where(fr["rev_t"] > 0))
    return fr


def rq3_first_reported():
    print("\n4. RQ3 WITH FIRST-REPORTED XBRL VALUES (features as known on the date quarter t was first reported)")
    fr = first_reported_features()
    if fr is None:
        print("   xbrl_vintages.csv.gz not found; run src/10_sensitivity_data.py"); return
    cols = ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "OP_MARGIN", "LOG_REVENUE"]
    raw = pd.read_csv(INTERIM / "panel_prewinsor.csv", dtype={"cik": str})
    m = us.merge(raw, on=["cik", "FISCAL_QTR"], suffixes=("", "_latest_raw")).merge(
        fr[["cik", "FISCAL_QTR", "known_on"] + cols], on=["cik", "FISCAL_QTR"], how="left", suffixes=("", "_fr"))
    have = m.REV_GROWTH_YOY_fr.notna()
    print(f"   U.S. panel rows with a first-reported growth value: {have.sum():,} of {len(m):,}")
    lat = m.loc[have, "REVENUE_USD"]
    diff_rev = (np.abs(np.exp(m.loc[have, "LOG_REVENUE_fr"]) - lat) / lat.abs() > 0.005)
    diff_g = (m.loc[have, "REV_GROWTH_YOY_fr"] - m.loc[have, "REV_GROWTH_YOY_latest_raw"]).abs()
    print(f"   quarters whose revenue was later revised by more than 0.5%: {diff_rev.mean():.1%}; "
          f"growth differs by more than 1 point: {(diff_g > 1).mean():.1%} (median absolute difference "
          f"{diff_g.median():.2f} points)")
    sign = (m.loc[have, "REV_GROWTH_YOY_fr"] < 0) != (m.loc[have, "REV_GROWTH_YOY_latest_raw"] < 0)
    print(f"   quarters whose decline status (growth < 0) differs between first-reported and latest values: "
          f"{sign.sum()} ({sign.mean():.1%})")
    # build the RQ3 design with first-reported features, capped with training-year limits
    alt = m.copy()
    for c in cols:
        alt[c] = alt[c + "_fr"]
    tr_mask = alt.FISCAL_QTR < rq3m.TRAIN_END
    for c in ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "OP_MARGIN"]:
        lo, hi = alt.loc[tr_mask, c].quantile([LO, HI])
        alt[c] = alt[c].clip(lo, hi)
    alt_cap = alt.copy()
    _, lim = rq3m.train_caps(us)
    alt_cap["AI_INTENSITY"] = alt["AI_INTENSITY_latest_raw"].clip(*lim["AI_INTENSITY"])
    d_alt, X = rq3m.design(alt_cap, train_only_caps=False)
    d_main, _ = rq3m.design(us)
    keys = set(zip(d_alt.cik, d_alt.FISCAL_QTR)) & set(zip(d_main.cik, d_main.FISCAL_QTR))
    pick = lambda d: d[[k in keys for k in zip(d.cik, d.FISCAL_QTR)]].sort_values(["cik", "FISCAL_QTR"])
    d_alt, d_main = pick(d_alt), pick(d_main)
    print(f"   common rows: {len(d_main):,} (test 2025: {(d_main.FISCAL_QTR >= '2025Q1').sum()})")
    res = {}
    for label, d in [("latest values (main)", d_main), ("first-reported values", d_alt)]:
        tr, va, te = rq3m.split(d)
        y = te[rq3m.TARGET].astype(int).values
        for name, model in rq3m.make_models(SEED).items():
            thr = rq3m.fit_and_tune(model, tr, va, X)
            p = model.predict_proba(te[X])[:, 1]
            tn, fp, fn, tp = confusion_matrix(y, p >= thr).ravel()
            res[(label, name)] = p
            print(f"   {label:24s} {name:13s} test AUC {roc_auc_score(y, p):.3f}; threshold {thr:.2f}; "
                  f"recall {tp / (tp + fn):.3f}; precision {tp / (tp + fp):.3f}")
        base = te["DECLINE_Q"].astype(float).values
        print(f"   {label:24s} persistence baseline AUC {roc_auc_score(y, base):.3f}")
    for name in ["logit", "random_forest", "xgboost"]:
        a1, a2, z, p = rq3m.delong(y, res[("first-reported values", name)], res[("latest values (main)", name)])
        print(f"   {name:13s} first-reported minus latest: {a1 - a2:+.3f} (DeLong z={z:.2f}, p={p:.3f})")


# ------------------------------------------------------------------ 5. rolling origin
def rq3_rolling():
    print("\n5. RQ3 ROLLING-ORIGIN EVALUATION (random forest; train on all earlier years, tune the threshold on "
          "the year before the test year, test on one year; caps from that origin's training years)")
    for ty in [2022, 2023, 2024, 2025]:
        train_end = f"{ty - 1}Q1"
        d, X = rq3m.design(us, train_end=train_end)
        tr = d[d.FISCAL_QTR < train_end]
        va = d[d.FISCAL_QTR.str[:4] == str(ty - 1)]
        te = d[d.FISCAL_QTR.str[:4] == str(ty)]
        model, thr = fit_rf(tr, va, X)
        y = te[rq3m.TARGET].astype(int).values
        p = model.predict_proba(te[X])[:, 1]
        base = te.DECLINE_Q.astype(float).values
        auc, se = roc_auc_score(y, p), rq3m.delong_se(y, p)
        tn, fp, fn, tp = confusion_matrix(y, p >= thr).ravel()
        btn, bfp, bfn, btp = confusion_matrix(y, base.astype(int)).ravel()
        _, _, z, pdl = rq3m.delong(y, p, base)
        ROLL.append(dict(year=ty, auc=auc, lo=auc - 1.96 * se, hi=auc + 1.96 * se, base=roc_auc_score(y, base),
                         rec=tp / (tp + fn), prec=tp / (tp + fp), brec=btp / (btp + bfn), bprec=btp / (btp + bfp)))
        print(f"   test {ty}: train {tr.FISCAL_QTR.min()}-{tr.FISCAL_QTR.max()} (n={len(tr)}), valid {ty - 1} "
              f"(n={len(va)}), test n={len(te)} firms={te.cik.nunique()} declines={y.sum()} ({y.mean():.1%}); "
              f"AUC {auc:.3f} (95% CI {auc - 1.96 * se:.3f} to {auc + 1.96 * se:.3f}); persistence AUC "
              f"{roc_auc_score(y, base):.3f} (DeLong p={pdl:.4f}); threshold {thr:.2f}; recall {tp / (tp + fn):.3f}; "
              f"precision {tp / (tp + fp):.3f}; persistence recall {btp / (btp + bfn):.3f}, precision {btp / (btp + bfp):.3f}")


# ------------------------------------------------------------------ 6. operating point uncertainty
def rq3_operating():
    print("\n6. RQ3 OPERATING-POINT UNCERTAINTY (chosen random forest, 2025 test set, threshold fixed on 2024)")
    d, X = rq3m.design(us)
    tr, va, te = rq3m.split(d)
    model, thr = fit_rf(tr, va, X)
    y = te[rq3m.TARGET].astype(int).values
    p = model.predict_proba(te[X])[:, 1]
    yhat = p >= thr
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    print(f"   threshold {thr:.2f}: flagged {tp + fp} of {len(y)} firm-quarters ({(tp + fp) / len(y):.1%}); "
          f"TP {tp}, FP {fp}; recall {tp / (tp + fn):.3f} (Wilson {wilson(tp, tp + fn)[0]:.3f}-{wilson(tp, tp + fn)[1]:.3f}); "
          f"precision {tp / (tp + fp):.3f} (Wilson {wilson(tp, tp + fp)[0]:.3f}-{wilson(tp, tp + fp)[1]:.3f}); "
          f"false alarms per decline caught {fp / tp:.2f}")
    rng = np.random.default_rng(SEED)
    firms = te.cik.values
    ids = np.unique(firms)
    idx = {f: np.where(firms == f)[0] for f in ids}
    recs, precs = [], []
    for _ in range(2000):
        r = np.concatenate([idx[f] for f in rng.choice(ids, len(ids), replace=True)])
        t_ = (y[r] == 1) & yhat[r]
        recs.append(t_.sum() / max((y[r] == 1).sum(), 1))
        precs.append(t_.sum() / max(yhat[r].sum(), 1))
    precs = np.array(precs)
    print(f"   firm-bootstrap 95% CI (2,000 resamples of firms): recall {np.percentile(recs, 2.5):.3f} to "
          f"{np.percentile(recs, 97.5):.3f}; precision {np.percentile(precs, 2.5):.3f} to {np.percentile(precs, 97.5):.3f}; "
          f"share of resamples with precision below 0.50: {(precs < 0.5).mean():.1%}")
    for t in [0.30, 0.40, 0.50, 0.60]:
        yh = p >= t
        tn_, fp_, fn_, tp_ = confusion_matrix(y, yh).ravel()
        print(f"   alternative threshold {t:.2f}: recall {tp_ / (tp_ + fn_):.3f}, precision {tp_ / max(tp_ + fp_, 1):.3f}, "
              f"flagged {tp_ + fp_}")


# ------------------------------------------------------------------ 7. services core
def services_core():
    print("\n7. SERVICES-CORE SAMPLE (SIC breadth): U.S. firms classified in config/gsi_classification.csv as "
          "internet platforms, games, advertising technology, software products, data providers or other non-IT-services "
          "businesses are removed; government integrators and the business process outsourcer are kept")
    cls = pd.read_csv(ROOT / "config/gsi_classification.csv", dtype={"cik": str})
    cls = cls[cls.market == "US"]
    drop = cls[(cls.is_gsi == 0) & ~cls.reason.str.contains("Government integrator|Business process", regex=True)]
    sub = us[~us.cik.isin(drop.cik)]
    gone = us[us.cik.isin(drop.cik)]
    print(f"   removed {gone.cik.nunique()} firms ({len(gone)} firm-quarters, {gone.REVENUE_USD.sum() / us.REVENUE_USD.sum():.0%} "
          f"of U.S. panel revenue); remaining {sub.cik.nunique()} firms, {len(sub)} firm-quarters")
    d = sub[sub.PERIOD.isin(["pre", "post"])]
    print(f"   RQ1 primary: {clustered('REV_GROWTH_YOY ~ POST_GENAI', d, 'POST_GENAI')[1]}")
    base = ["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "OP_MARGIN", "REV_GROWTH_LAG1", "IS_10K", "SEGMENT"]
    d2 = sub[sub.PERIOD == "post"].dropna(subset=base).copy()
    d2["IS_10K"] = d2.IS_10K.astype(float)
    f = "REV_GROWTH_YOY ~ AI_INTENSITY + LOG_REVENUE + OP_MARGIN + REV_GROWTH_LAG1 + IS_10K + C(SEGMENT)"
    print(f"   RQ2 primary: {clustered(f, d2, 'AI_INTENSITY')[1]}")
    print(f"   RQ2 + quarter fixed effects: {clustered(f + ' + C(FISCAL_QTR)', d2, 'AI_INTENSITY')[1]}")
    d3, X = rq3m.design(sub)
    tr, va, te = rq3m.split(d3)
    model, thr = fit_rf(tr, va, X)
    y = te[rq3m.TARGET].astype(int).values
    p = model.predict_proba(te[X])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y, p >= thr).ravel()
    print(f"   RQ3 random forest (retrained): test n={len(te)}, AUC {roc_auc_score(y, p):.3f}; persistence "
          f"{roc_auc_score(y, te.DECLINE_Q.astype(float)):.3f}; recall {tp / (tp + fn):.3f}; precision {tp / (tp + fp):.3f}")
    d4 = sub[sub.PERIOD == "post"].dropna(subset=["SEGMENT", "DECLINE_Q"]).copy()
    d4["DECLINE_Q"] = d4.DECLINE_Q.astype(int)
    gee = smf.gee("DECLINE_Q ~ C(SEGMENT)", groups="cik", data=d4, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    terms = [t for t in gee.params.index if t.startswith("C(SEGMENT)")]
    R = np.zeros((len(terms), len(gee.params)))
    for i, t in enumerate(terms):
        R[i, list(gee.params.index).index(t)] = 1
    w = gee.wald_test(R, scalar=True)
    print(f"   RQ4 GEE Wald chi2={float(w.statistic):.2f}, p={float(w.pvalue):.3f}; decline shares: " +
          "; ".join(f"{s} {g.DECLINE_Q.mean():.3f}" for s, g in d4.groupby("SEGMENT")))


# ------------------------------------------------------------------ 8. RQ4 differences
def rq4_differences():
    print("\n8. RQ4 DIFFERENCES IN DECLINE SHARE BETWEEN SEGMENTS (post-GenAI U.S., firm-bootstrap 95% CI)")
    d4 = us[us.PERIOD == "post"].dropna(subset=["SEGMENT", "DECLINE_Q"]).copy()
    d4["DECLINE_Q"] = d4.DECLINE_Q.astype(int)
    segs = sorted(d4.SEGMENT.unique())
    ref = "Data processing & managed services"
    rng = np.random.default_rng(SEED)
    groups = {s: g for s, g in d4.groupby("SEGMENT")}
    idx = {s: {f: g.index[g.cik == f] for f in g.cik.unique()} for s, g in groups.items()}
    boots = {s: [] for s in segs}
    for _ in range(2000):
        for s in segs:
            firms = list(idx[s])
            rows = np.concatenate([idx[s][f] for f in rng.choice(firms, len(firms), replace=True)])
            boots[s].append(d4.loc[rows, "DECLINE_Q"].mean())
    for s in segs:
        if s == ref:
            continue
        diff = np.array(boots[s]) - np.array(boots[ref])
        point = groups[s].DECLINE_Q.mean() - groups[ref].DECLINE_Q.mean()
        print(f"   {s} minus {ref}: {point * 100:+.1f} points (95% CI {np.percentile(diff, 2.5) * 100:+.1f} to "
              f"{np.percentile(diff, 97.5) * 100:+.1f})")
    a, b = "Systems integration", "IT consulting & application services"
    diff = np.array(boots[a]) - np.array(boots[b])
    print(f"   {a} minus {b}: {(groups[a].DECLINE_Q.mean() - groups[b].DECLINE_Q.mean()) * 100:+.1f} points "
          f"(95% CI {np.percentile(diff, 2.5) * 100:+.1f} to {np.percentile(diff, 97.5) * 100:+.1f})")


# ------------------------------------------------------------------ 9. AI dictionary precision
def ai_precision():
    print("\n9. AI KEYWORD DICTIONARY: PRECISION ON 50 RANDOMLY DRAWN MATCHES")
    ctx_path = INTERIM / "ai_match_contexts.csv.gz"
    if not ctx_path.exists():
        print("   ai_match_contexts.csv.gz not found; run src/10_sensitivity_data.py"); return
    draws = pd.read_csv(INTERIM / "ai_precision_draws.csv", dtype={"accession": str})
    ctx = pd.read_csv(ctx_path, dtype={"accession": str}).sort_values(["accession", "pos", "pattern"])
    rng = np.random.default_rng(SEED)
    picked = []
    for r in draws.itertuples():
        c = ctx[ctx.accession == r.accession]
        if c.empty:
            continue
        m = c.iloc[rng.integers(len(c))]
        picked.append({"draw": r.draw, "accession": r.accession, "form": r.form, "report_date": r.report_date,
                       "pattern": m.pattern, "match": m["match"], "context": m.context})
        if len(picked) == 50:
            break
    sample = pd.DataFrame(picked)
    sample.to_csv(RESULTS / "ai_precision_sample.csv", index=False)
    print(f"   sample: {len(sample)} matches from {sample.accession.nunique()} filings "
          f"({(sample.report_date < '2022-10-01').sum()} from filings before 2022Q4); written to results/ai_precision_sample.csv")
    lab_path = INTERIM / "ai_precision_labels.csv"
    if not lab_path.exists():
        print("   labels not found (data/interim/ai_precision_labels.csv)"); return
    lab = pd.read_csv(lab_path)
    s = sample.merge(lab[["draw", "label", "context_type", "note"]], on="draw", how="left")
    k, n = int((s.label == 1).sum()), int(s.label.notna().sum())
    lo, hi = wilson(k, n)
    print(f"   reviewed {n}; true AI references {k}; precision {k / n:.3f} (Wilson 95% CI {lo:.3f} to {hi:.3f})")
    s["term"] = s.pattern.str.replace(r"^\[genai\]\s*", "", regex=True)
    print("   by dictionary term (true / reviewed):")
    for t, g in s.groupby("term"):
        print(f"      {t:45s} {int((g.label == 1).sum())} / {len(g)}")
    print("   what the passage is about (true matches):")
    s["era"] = np.where(s.report_date < "2022-10-01", "before 2022Q4", "2022Q4 onward")
    print(pd.crosstab(s.context_type, s.era, margins=True).to_string())
    for r in s[s.label == 0].itertuples():
        print(f"   false match (draw {r.draw}, '{r.match}'): {r.note}")


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for fn in [rq1_confounding, rq2_fixed_effects, rq3_caps, rq3_first_reported, rq3_rolling, rq3_operating,
               services_core, rq4_differences, ai_precision]:
        fn()
(RESULTS / "sensitivity_output.txt").write_text(buf.getvalue())

# rolling-origin chart: AUC with 95% interval against the persistence baseline, and recall / precision
if ROLL:
    r = pd.DataFrame(ROLL)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2))
    plt.rcParams.update({"font.size": 13})
    x = np.arange(len(r))
    a1.errorbar(x, r.auc, yerr=[r.auc - r.lo, r.hi - r.auc], fmt="o", color="#1f5a96", ms=9, capsize=6, lw=2,
                label="Random forest (95% DeLong CI)")
    a1.plot(x, r.base, "s", color="#5a6b7d", ms=8, label="Persistence baseline")
    for i, v in enumerate(r.auc):
        a1.text(i + 0.12, v, f"{v:.3f}", va="center", fontsize=10.5, color=INK)
    a1.set_xticks(x); a1.set_xticklabels(r.year.astype(str)); a1.set_ylim(0.70, 0.96)
    a1.set_xlabel("Test year", color=MUTED); a1.set_ylabel("Test AUC", color=MUTED)
    a1.set_title("Ranking quality", fontsize=13, color=INK, loc="left")
    a1.legend(frameon=False, loc="lower right")
    a2.plot(x, r.rec, "o-", color="#1f5a96", lw=2, ms=8, label="Recall, random forest")
    a2.plot(x, r.prec, "o-", color="#c95b37", lw=2, ms=8, label="Precision, random forest")
    a2.plot(x, r.brec, "o--", color="#1f5a96", lw=1.2, ms=6, alpha=0.6, label="Recall, persistence")
    a2.plot(x, r.bprec, "o--", color="#c95b37", lw=1.2, ms=6, alpha=0.6, label="Precision, persistence")
    a2.axhline(0.5, color="#c3c9d2", lw=1)
    a2.set_xticks(x); a2.set_xticklabels(r.year.astype(str)); a2.set_ylim(0.4, 1.0)
    a2.set_xlabel("Test year", color=MUTED); a2.set_ylabel("Share", color=MUTED)
    a2.set_title("At the threshold tuned on the year before", fontsize=13, color=INK, loc="left")
    a2.legend(frameon=False, loc="lower left", fontsize=10.5, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "sens_rolling_origin.png", dpi=110); plt.close(fig)
print(buf.getvalue())
