"""Design check: clustering, effective sample sizes and test-set precision for RQ1-RQ4.

Runs after src/04_build_panel.py. It answers the questions a sample-size reviewer asks about a
firm-quarter panel, using only design information:
  * how many independent firms and repeated quarters each analysis uses,
  * the intraclass correlation (ICC) of the outcomes, estimated from the PRE-GenAI quarters only so the
    calibration never looks at the post-period outcomes being tested,
  * the resulting design effect, effective sample size and power,
  * the number of firms and decline events in the 2025 test set and the precision that allows for AUC,
    recall and precision,
  * missing values and revenue-growth edge cases.
Output: results/design_check.txt
"""
import contextlib
import io
import math

import numpy as np
import pandas as pd
from scipy import stats

import rq3_models as rq3m
from common import INTERIM, PROCESSED, RESULTS

ALPHA, POWER = 0.05, 0.80
Z_A, Z_B = stats.norm.ppf(1 - ALPHA / 2), stats.norm.ppf(POWER)

df = pd.read_csv(PROCESSED / "panel_firm_quarter.csv", dtype={"cik": str})
us, india = df[df.MARKET == "US"], df[df.MARKET == "India"]


def icc1(data, value, group="cik"):
    """One-way ANOVA ICC(1) with the adjusted mean cluster size for unequal clusters."""
    d = data[[group, value]].dropna()
    g = d.groupby(group)[value]
    k, n_i, N = g.ngroups, g.size(), len(d)
    grand = d[value].mean()
    ssb = (n_i * (g.mean() - grand) ** 2).sum()
    ssw = ((d[value] - d.groupby(group)[value].transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    m0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    return max(0.0, (msb - msw) / (msb + (m0 - 1) * msw)), m0


def clusters(d):
    return d.cik.nunique(), len(d), len(d) / max(1, d.cik.nunique())


def deff(m, icc):
    return 1 + (m - 1) * icc


def paired_mde(n):
    """Smallest standardized paired difference dz detectable with 80% power, two-sided alpha .05."""
    lo, hi = 0.01, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        t_crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
        pw = np.nan_to_num(paired_power(n, mid), nan=1.0)   # scipy returns nan when power is essentially 1
        lo, hi = (lo, mid) if pw >= POWER else (mid, hi)
    return hi


def paired_power(n, dz):
    t_crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    nc = dz * math.sqrt(n)
    return 1 - stats.nct.cdf(t_crit, n - 1, nc) + stats.nct.cdf(-t_crit, n - 1, nc)


def f_power(n, f2, p, u=1):
    v = n - p - 1
    return 1 - stats.ncf.cdf(stats.f.ppf(1 - ALPHA, u, v), u, v, f2 * n)


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(var)


def wilson_halfwidth(p, n):
    z = Z_A
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre, half


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print("DESIGN CHECK (alpha = .05 two-sided, power = .80). ICCs use pre-GenAI quarters only.\n")

    # ---------------------------------------------------------------- panel structure
    print("1. Firms and firm-quarters")
    for label, d in [("U.S. pre-GenAI 2019Q1-2022Q3", us[us.PERIOD == "pre"]),
                     ("U.S. post-GenAI 2023Q1-2025Q4", us[us.PERIOD == "post"]),
                     ("India pre-GenAI", india[india.PERIOD == "pre"]),
                     ("India post-GenAI", india[india.PERIOD == "post"])]:
        k, n, m = clusters(d)
        print(f"   {label:32s} firms={k:4d}  firm-quarters={n:5d}  mean quarters per firm={m:5.1f}")
    both_us = set(us[us.PERIOD == "pre"].cik) & set(us[us.PERIOD == "post"].cik)
    print(f"   U.S. firms observed in both periods: {len(both_us)}; India: "
          f"{len(set(india[india.PERIOD == 'pre'].cik) & set(india[india.PERIOD == 'post'].cik))}\n")

    # ---------------------------------------------------------------- RQ1
    pre_us = us[us.PERIOD == "pre"]
    icc_g, m0 = icc1(pre_us, "REV_GROWTH_YOY")
    print("2. RQ1 (growth, pre vs post)")
    print(f"   ICC of REV_GROWTH_YOY within firms (pre-GenAI): {icc_g:.3f}")
    need = 394
    for label, d in [("pre", us[us.PERIOD == "pre"]), ("post", us[us.PERIOD == "post"])]:
        k, n, m = clusters(d)
        de = deff(m, icc_g)
        print(f"   {label}: DEFF = 1 + ({m:.1f} - 1) x {icc_g:.3f} = {de:.2f}; effective n = {n}/{de:.2f} = {n / de:.0f}; "
              f"required firm-quarters = 394 x {de:.2f} = {math.ceil(need * de)}; available = {n}")
    k_both = len(both_us)
    print(f"   Firm-level paired test on {k_both} firms: smallest detectable dz = {paired_mde(k_both):.3f}; "
          f"power at dz = 0.20: {paired_power(k_both, 0.20):.2f}; firms needed for dz = 0.20: "
          f"{next(n for n in range(10, 2000) if paired_power(n, 0.20) >= POWER)}")
    sd_pre = pre_us["REV_GROWTH_YOY"].std()
    firm_sd = (us[us.PERIOD.isin(["pre", "post"])].groupby(["cik", "PERIOD"]).REV_GROWTH_YOY.mean().unstack()
               .dropna().pipe(lambda m: (m["post"] - m["pre"]).std()))
    print(f"   pre-GenAI SD of quarterly growth (U.S.) = {sd_pre:.1f} pp, so d = 0.20 is about {0.2 * sd_pre:.1f} pp; "
          f"median growth pre = {pre_us.REV_GROWTH_YOY.median():.1f}%")
    print(f"   SD of firm-level change in mean growth (post minus pre) = {firm_sd:.1f} pp, so dz = 0.20 is about "
          f"{0.2 * firm_sd:.1f} pp and the smallest detectable dz is about {paired_mde(k_both) * firm_sd:.1f} pp")
    k_in = india.cik.nunique()
    print(f"   India firm-level paired test on {k_in} firms: smallest detectable dz = {paired_mde(k_in):.2f}\n")

    # ---------------------------------------------------------------- RQ2
    d2 = us[us.PERIOD == "post"].dropna(subset=["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "IS_10K",
                                                "OP_MARGIN", "REV_GROWTH_LAG1", "SEGMENT"])
    import statsmodels.formula.api as smf
    dp = pre_us.dropna(subset=["REV_GROWTH_YOY", "AI_INTENSITY", "LOG_REVENUE", "OP_MARGIN", "REV_GROWTH_LAG1",
                               "IS_10K", "SEGMENT"]).copy()
    dp["IS_10K"] = dp["IS_10K"].astype(float)
    f0 = "REV_GROWTH_YOY ~ LOG_REVENUE + OP_MARGIN + REV_GROWTH_LAG1 + IS_10K + C(SEGMENT)"
    r0, r1 = smf.ols(f0, dp).fit().rsquared, smf.ols(f0 + " + AI_INTENSITY", dp).fit().rsquared
    print(f"   pilot check on pre-GenAI quarters (n = {len(dp)}): partial f2 of AI_INTENSITY in the RQ2 model = "
          f"{(r1 - r0) / (1 - r1):.4f}\n")
    print("3a. RQ2 rows lost at each step (post-GenAI U.S.)")
    step = us[us.PERIOD == "post"]
    print(f"   all post-GenAI U.S. firm-quarters: {len(step)} rows, {step.cik.nunique()} firms")
    for c in ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "AI_INTENSITY", "OP_MARGIN", "LOG_REVENUE", "SEGMENT"]:
        step = step.dropna(subset=[c])
        print(f"   after requiring {c:16s}: {len(step)} rows, {step.cik.nunique()} firms")
    sga = step.dropna(subset=["SGA_RATIO"])
    print(f"   (if SGA_RATIO were also required: {len(sga)} rows, {sga.cik.nunique()} firms)")
    k2, n2, m2 = clusters(d2)
    de2 = deff(m2, icc_g)
    print("3. RQ2 (AI_INTENSITY coefficient, post-GenAI U.S. quarters with complete data)")
    print(f"   eligible firm-quarters={n2}, firms={k2}, mean quarters per firm={m2:.1f}")
    n_req = next(n for n in range(20, 5000) if f_power(n, 0.02, 7) >= POWER)
    print(f"   noncentral F: N={n_req} gives power {f_power(n_req, 0.02, 7):.3f} "
          f"(lambda = f2 x N = {0.02 * n_req:.2f}, df1 = 1, df2 = {n_req - 8})")
    print(f"   design effect with ICC {icc_g:.3f}: {de2:.2f}; effective N = {n2 / de2:.0f}; "
          f"power at effective N for f2 = 0.02: {f_power(n2 / de2, 0.02, 7):.2f}; "
          f"smallest f2 detectable with 80% power: "
          f"{next(f / 1000 for f in range(5, 500) if f_power(n2 / de2, f / 1000, 7) >= POWER):.3f}\n")

    # ---------------------------------------------------------------- RQ3
    d3, X = rq3m.design(us)
    train, valid, test = rq3m.split(d3)
    print("4. RQ3 (next-quarter decline classifier)")
    for label, d in [("train 2019-2023", train), ("validation 2024", valid), ("test 2025", test)]:
        ev = int(d[rq3m.TARGET].sum())
        print(f"   {label:16s} firm-quarters={len(d):4d} firms={d.cik.nunique():3d} decline events={ev:3d} "
              f"non-events={len(d) - ev:3d} prevalence={ev / len(d):.3f}")
    epv = int(train[rq3m.TARGET].sum()) / len(X)
    print(f"   events per candidate predictor in training: {int(train[rq3m.TARGET].sum())}/{len(X)} = {epv:.1f}")
    npos, nneg = int(test[rq3m.TARGET].sum()), int(len(test) - test[rq3m.TARGET].sum())
    for auc in (0.70, 0.80):
        se = hanley_mcneil_se(auc, npos, nneg)
        print(f"   test set: if AUC = {auc:.2f}, Hanley-McNeil SE = {se:.3f}, 95% CI = {auc - Z_A * se:.2f} to "
              f"{auc + Z_A * se:.2f} (half-width {Z_A * se:.3f})")
    se_needed = 0.10 / Z_A
    for auc in (0.70,):
        prev = 0.30   # planning prevalence
        n_total = next(n for n in range(50, 5000)
                       if hanley_mcneil_se(auc, max(2, round(n * prev)), max(2, round(n * (1 - prev)))) <= se_needed)
        print(f"   rows needed for a +/-0.10 AUC interval at AUC {auc:.2f} and planning prevalence {prev:.2f}: {n_total} "
              f"(about {round(n_total * prev)} events)")
    for rec in (0.70, 0.80):
        c, h = wilson_halfwidth(rec, npos)
        print(f"   recall {rec:.2f} on {npos} test declines: 95% Wilson CI half-width {h:.3f}")
    exp_flagged = 0.70 * npos / 0.50
    print(f"   precision 0.50 at recall 0.70 means about {exp_flagged:.0f} flagged quarters; Wilson half-width "
          f"{wilson_halfwidth(0.50, round(exp_flagged))[1]:.3f}")
    print(f"   test firms contribute {len(test) / test.cik.nunique():.1f} quarters each on average, so intervals "
          f"are also bootstrapped by firm (resampling whole firms).\n")

    # ---------------------------------------------------------------- RQ4
    d4 = us[us.PERIOD == "post"].dropna(subset=["SEGMENT", "DECLINE_Q"])
    icc_dec, _ = icc1(pre_us.dropna(subset=["DECLINE_Q"]).assign(DQ=lambda x: x.DECLINE_Q.astype(float)), "DQ")
    k4, n4, m4 = clusters(d4)
    de4 = deff(m4, icc_dec)
    print("5. RQ4 (segment x decline, post-GenAI U.S. quarters)")
    print(f"   firm-quarters={n4}, firms={k4}, mean quarters per firm={m4:.1f}; ICC of DECLINE_Q (pre-GenAI)={icc_dec:.3f}")
    pbar = d4.DECLINE_Q.astype(float).mean()
    gap = lambda w: 2 * w * math.sqrt(1.5 * pbar * (1 - pbar))   # three equal segments at pbar - g/2, pbar, pbar + g/2
    print(f"   decline share post-GenAI = {pbar:.3f}; with three equal-sized segments, w = 0.10 is a gap of "
          f"{gap(0.10) * 100:.1f} points between the highest and lowest segment and w = 0.154 a gap of {gap(0.154) * 100:.1f}")
    print(f"   DEFF = {de4:.2f}; effective N = {n4 / de4:.0f}; required = 964 x {de4:.2f} = {math.ceil(964 * de4)}")
    print("   firms per segment: " + "; ".join(f"{s}: {n}" for s, n in d4.groupby("SEGMENT").cik.nunique().items()))
    print(f"   power of the 3 x 2 test at effective N for w = 0.10: "
          f"{1 - stats.ncx2.cdf(stats.chi2.ppf(0.95, 2), 2, 0.01 * n4 / de4):.2f}; smallest w detectable with 80% power: "
          f"{next(w / 1000 for w in range(20, 800) if 1 - stats.ncx2.cdf(stats.chi2.ppf(0.95, 2), 2, (w / 1000) ** 2 * n4 / de4) >= POWER):.3f}\n")

    # ---------------------------------------------------------------- missing values and edge cases
    print("6. Missing values in the main panel (U.S.)")
    for c in ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "SGA_RATIO", "OP_MARGIN", "AI_INTENSITY", "AI_INTENSITY_4Q",
              "IS_10K", "GENAI_MENTION", "DECLINE_NEXT_Q"]:
        print(f"   {c:16s} missing {us[c].isna().mean() * 100:5.1f}%")
    # rows removed by the US$1 million prior-revenue rule, counted inside the study window and size filter
    fin = pd.read_csv(INTERIM / "financials_quarterly.csv", parse_dates=["period_end"])
    fin["cik"] = fin["cik"].astype(str)
    fin["cal_qtr"] = (fin["period_end"] - pd.Timedelta(days=45)).dt.to_period("Q")
    fin = fin.sort_values(["cik", "cal_qtr"]).drop_duplicates(["cik", "cal_qtr"], keep="last")
    fin["qn"] = fin["cal_qtr"].apply(lambda q: q.year * 4 + q.quarter)
    g = fin.groupby("cik")
    fin["lag4"] = g["revenue"].shift(4).where(fin["qn"] - g["qn"].shift(4) == 4)
    fin["ttm"] = g["revenue"].transform(lambda s: s.rolling(4).sum())
    qs = fin["cal_qtr"].dt.start_time
    scope = fin[(qs >= "2019-01-01") & (qs <= "2025-10-01") & (fin["ttm"] >= 10_000_000) & fin["lag4"].notna()
                & fin["cik"].isin(set(us.cik.astype(str)))]
    small = scope[scope["lag4"] < 1_000_000]
    print(f"   firm-quarters in the study window whose year-earlier revenue is below US$1 million: {len(small)} "
          f"(zero or negative: {int((small.lag4 <= 0).sum())}; firms: {small.cik.nunique()}); growth left blank")
    print(f"   U.S. firm-quarters with zero or negative revenue (LOG_REVENUE blank): {int((us.REVENUE_USD <= 0).sum())}")
    print(f"   AI score source in the U.S. panel: 10-K quarters {int((us.IS_10K == 1).sum())}, 10-Q quarters "
          f"{int((us.IS_10K == 0).sum())}; mean AI_INTENSITY 10-K {us.loc[us.IS_10K == 1, 'AI_INTENSITY'].mean():.2f} vs "
          f"10-Q {us.loc[us.IS_10K == 0, 'AI_INTENSITY'].mean():.2f}")
    print(f"   growth at the winsorizing caps (U.S.): low {us.REV_GROWTH_YOY.min():.1f}%, high {us.REV_GROWTH_YOY.max():.1f}%")

    # ---------------------------------------------------------------- summary: requirement vs availability
    print("\n7. Minimum sample size after the design effect, against what the panel provides")
    rows = []
    for label, d in [("pre", us[us.PERIOD == "pre"]), ("post", us[us.PERIOD == "post"])]:
        k, n, m = clusters(d)
        rows.append(("RQ1 " + label + "-GenAI period", 394, deff(m, icc_g), n, k))
    rows.append(("RQ2 post-GenAI complete cases", 395, de2, n2, k2))
    prev_tr = 0.30   # planning prevalence (pilot training prevalence shown above)
    rows.append(("RQ3 training 2019-2023 (EPV 10)", math.ceil(10 * len(X) / prev_tr), 1.0, len(train), train.cik.nunique()))
    hm_n = next(n for n in range(50, 5000)
                if hanley_mcneil_se(0.70, max(2, round(n * prev_tr)), max(2, round(n * (1 - prev_tr)))) <= se_needed)
    m_te = len(test) / test.cik.nunique()
    rows.append(("RQ3 test 2025 (AUC +/-0.10)", hm_n, deff(m_te, icc_dec), len(test), test.cik.nunique()))
    rows.append(("RQ4 post-GenAI", 964, de4, n4, k4))
    print(f"   {'analysis':34s} {'base N':>7s} {'DEFF':>5s} {'required':>8s} {'available':>9s} {'firms':>5s}  met")
    for label, base_n, de, avail, firms in rows:
        req = math.ceil(base_n * de)
        print(f"   {label:34s} {base_n:7d} {de:5.2f} {req:8d} {avail:9d} {firms:5d}  {'yes' if avail >= req else 'NO'}")
    print(f"   RQ3 planning prevalence = {prev_tr:.2f}; test-set DEFF uses ICC {icc_dec:.3f} and "
          f"{m_te:.1f} quarters per firm; required test events = {math.ceil(hm_n * deff(m_te, icc_dec) * prev_tr)}")
    print(f"   RQ3 recall 0.70 +/-0.10 would need {next(n for n in range(10, 2000) if wilson_halfwidth(0.70, n)[1] <= 0.10)} "
          f"declines before the design effect")
    print(f"   RQ4 pre-GenAI observed Cohen's w (segment x decline) = "
          f"{math.sqrt(stats.chi2_contingency(pd.crosstab(pre_us.SEGMENT, pre_us.DECLINE_Q), correction=False)[0] / pre_us.dropna(subset=['SEGMENT', 'DECLINE_Q']).shape[0]):.3f}")
(RESULTS / "design_check.txt").write_text(buf.getvalue())
print(buf.getvalue())
