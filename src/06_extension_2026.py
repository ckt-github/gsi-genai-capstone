"""2026 extension: a robustness check reported separately from the main 2019-2025 analysis.

Two questions:
  1. Does the RQ1 slowdown hold in 2026?  Mean growth by year, and 2026 vs the pre-GenAI period.
  2. How well does the frozen RQ3 early-warning model score on 2026?  The models are trained on
     2019-2023 and tuned on 2024 exactly as in the main study, then scored on 2026 quarters they
     have never seen. No refitting or re-tuning on 2026 data.

Quarters enter only when they pass the eligibility rule in config/settings.yaml (see
results/extension_eligibility.csv). Input: data/processed/panel_extension.csv (built by
src/04_build_panel.py --extension). Output: results/extension_2026_output.txt
"""
import contextlib
import io
import traceback
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

import rq3_models as rq3m
from common import PROCESSED, RESULTS, SETTINGS

SEED = SETTINGS["random_seed"]
df = pd.read_csv(PROCESSED / "panel_extension.csv")
df["YEAR"] = df["FISCAL_QTR"].str[:4]
ext = df[df.SAMPLE == "extension"]


def eligibility():
    rep = pd.read_csv(RESULTS / "extension_eligibility.csv")
    print("Which 2026 quarters are in (rule: >= {min_days_after_quarter} days old and >= {min_coverage:.0%} of "
          "2025Q4 firms reported)".format(**SETTINGS["extension"]))
    print(rep.to_string(index=False))
    if ext.empty:
        print("\nNo 2026 quarter is eligible yet; nothing else to report.")


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    mw = stats.mannwhitneyu(a, b).pvalue
    return (f"mean diff={a.mean() - b.mean():+.2f} pp (Welch p={p:.4f}, d={(a.mean() - b.mean()) / sp:.3f}); "
            f"median {a.median():.2f} vs {b.median():.2f} (Mann-Whitney p={mw:.4f})")


def rq1_extension():
    print("RQ1 extension: year-over-year revenue growth (%) by period")
    print("Note: a few very fast-growing small firms pull the mean up; read the median and trimmed mean too.")
    for market, sub in df.groupby("MARKET", sort=False):
        sub = sub[sub.PERIOD != "transition"]
        label = np.where(sub.PERIOD == "pre", "pre-GenAI 2019-22", sub.YEAR)
        tab = sub.groupby(label)["REV_GROWTH_YOY"].agg(
            n="count", mean="mean", median="median",
            trimmed_mean_5pct=lambda s: stats.trim_mean(s.dropna(), 0.05)).round(2)
        print(f"\n[{market}]\n{tab.to_string()}")
        pre = sub.loc[sub.PERIOD == "pre", "REV_GROWTH_YOY"].dropna()
        y26 = sub.loc[sub.YEAR == "2026", "REV_GROWTH_YOY"].dropna()
        q26 = sorted(sub.loc[sub.YEAR == "2026", "FISCAL_QTR"].unique())
        q25 = [q.replace("2026", "2025") for q in q26]
        y25 = sub.loc[sub.FISCAL_QTR.isin(q25), "REV_GROWTH_YOY"].dropna()
        if len(y26) >= 2:
            print(f"  2026 vs pre-GenAI:                   {welch(y26, pre)}")
            print(f"  2026 vs same quarters of 2025 ({', '.join(q25)}): {welch(y26, y25)}")


def rq3_frozen():
    us = df[df.MARKET == "US"]
    d, X = rq3m.design(us)
    train, valid, test = rq3m.split(d)
    new = d[d.SAMPLE == "extension"]
    print(f"RQ3 frozen model: trained 2019-2023 (n={len(train)}), tuned on 2024 (n={len(valid)}); "
          f"2025 test n={len(test)}; 2026 rows with a known outcome n={len(new)}")
    if len(new) < 20 or new[rq3m.TARGET].nunique() < 2:
        print("  Not enough 2026 rows with a known next-quarter outcome yet. Each 2026 quarter's outcome is the "
              "following quarter, so scores appear once two consecutive 2026 quarters are eligible.")
        return
    print(f"  2026 decline rate = {new[rq3m.TARGET].mean():.3f} (2025 test: {test[rq3m.TARGET].mean():.3f})")
    for name, model in rq3m.make_models(SEED).items():
        thr = rq3m.fit_and_tune(model, train, valid, X)
        print(f"  {name:13s} 2025 test: {rq3m.scores(model, test, X, thr)}")
        print(f"  {'':13s} 2026 new : {rq3m.scores(model, new, X, thr)}")


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print(f"2026 extension (robustness check), built {date.today():%d %b %Y}. "
          f"Main 2019-2025 results are in results/analysis_output.txt and are not affected.\n")
    eligibility()
    print()
    if not ext.empty:
        for name, fn in [("RQ1 extension", rq1_extension), ("RQ3 frozen model", rq3_frozen)]:
            try:
                fn()
            except Exception:
                print(f"{name} could not be run:\n{traceback.format_exc(limit=1)}")
            print()
(RESULTS / "extension_2026_output.txt").write_text(buf.getvalue())
print(buf.getvalue())
