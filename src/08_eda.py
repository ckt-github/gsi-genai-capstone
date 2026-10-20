"""Exploratory data analysis for the interim report.

Reads the firm-quarter panel built by src/04_build_panel.py and writes
  results/eda_output.txt          summary statistics, missing values, outliers, trends, correlations
  results/figures/eda_*.png       charts used in the interim report
Nothing here changes the panel or the hypothesis tests in src/05_analysis.py.
"""
import contextlib
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import PROCESSED, RESULTS

FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e7ee", "axes.axisbelow": True})
US_C, IN_C, LIGHT = "#1f5a96", "#c95b37", "#5b9bd5"
SEG_C = {"Data processing & managed services": LIGHT, "IT consulting & application services": US_C,
         "Systems integration": IN_C}
INK, MUTED = "#1b2a3a", "#5a6b7d"

df = pd.read_csv(PROCESSED / "panel_firm_quarter.csv", dtype={"cik": str})
df["Q"] = pd.PeriodIndex(df["FISCAL_QTR"], freq="Q")
df["YEAR"] = df["Q"].dt.year
us, india = df[df.MARKET == "US"], df[df.MARKET == "India"]
CONT = ["REV_GROWTH_YOY", "AI_INTENSITY", "OP_MARGIN", "LOG_REVENUE", "SGA_RATIO"]


def describe(d, cols):
    t = d[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    t["missing_%"] = d[cols].isna().mean() * 100
    return t[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_%"]].round(2)


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print("EXPLORATORY DATA ANALYSIS (panel_firm_quarter.csv)\n")
    print("1. Size of the panel")
    print(df.groupby(["MARKET", "PERIOD"]).agg(firm_quarters=("cik", "size"), firms=("cik", "nunique")))
    print("\nU.S. firms and firm-quarters by segment")
    print(us.groupby("SEGMENT").agg(firm_quarters=("cik", "size"), firms=("cik", "nunique")))
    big = us.groupby("name").REVENUE_USD.max().sort_values(ascending=False).head(5) / 1e6
    print("\nLargest U.S. firms by maximum quarterly revenue (US$ millions): " +
          "; ".join(f"{n} {v:,.0f}" for n, v in big.items()))
    miss_ai = us[us.AI_INTENSITY.isna()]
    print("U.S. firm-quarters without an AI score, by firm: " +
          "; ".join(f"{n} {k}" for n, k in miss_ai.groupby("name").size().sort_values(ascending=False).items()))
    print(f"\nRevenue (US$ millions) of U.S. firm-quarters: median {us.REVENUE_USD.median() / 1e6:.1f}, "
          f"mean {us.REVENUE_USD.mean() / 1e6:.1f}, largest {us.REVENUE_USD.max() / 1e6:,.0f}")

    print("\n2. Descriptive statistics, U.S. firm-quarters, by period")
    for per in ["pre", "post"]:
        print(f"\n[{per}-GenAI]")
        print(describe(us[us.PERIOD == per], CONT))
    print("\n[India, all periods] (no 10-Q/10-K text, no SG&A line)")
    print(describe(india, ["REV_GROWTH_YOY", "OP_MARGIN", "LOG_REVENUE"]))

    print("\n3. Binary variables (share of firm-quarters = 1), U.S.")
    print(us.groupby("PERIOD")[["DECLINE_Q", "DECLINE_NEXT_Q", "GENAI_MENTION", "IS_10K"]].mean().round(3))
    print("\nIndia decline share by period:", india.groupby("PERIOD").DECLINE_Q.mean().round(3).to_dict())

    print("\n4. Skewness and extreme values (after winsorizing)")
    for c in ["REV_GROWTH_YOY", "AI_INTENSITY", "OP_MARGIN"]:
        s = us[c].dropna()
        print(f"   {c:15s} skew={s.skew():.2f}  mean={s.mean():.2f}  median={s.median():.2f}  "
              f"share above 100 = {(s > 100).mean() * 100:.1f}%")
    top = (us[us.PERIOD == "post"].groupby(["cik", "name"]).REV_GROWTH_YOY.median()
           .sort_values(ascending=False).head(5).round(1))
    print("   fastest-growing U.S. firms after GenAI (median growth %):")
    for (cik, name), v in top.items():
        print(f"      {name}: {v}")
    print(f"   U.S. firm-quarters with zero or negative revenue: {int((us.REVENUE_USD <= 0).sum())}")

    print("\n5. Trend by year (U.S. and India): median and mean growth, decline share")
    yr = df.groupby(["MARKET", "YEAR"]).agg(n=("cik", "size"), median_growth=("REV_GROWTH_YOY", "median"),
                                            mean_growth=("REV_GROWTH_YOY", "mean"),
                                            decline_share=("DECLINE_Q", "mean")).round(2)
    print(yr)
    print("\nU.S. growth quartiles by year (P25, median, P75)")
    print(us.groupby("YEAR").REV_GROWTH_YOY.quantile([0.25, 0.5, 0.75]).unstack().round(1))
    print("\nMedian growth by calendar quarter, U.S. and India")
    print(df.groupby(["Q", "MARKET"]).REV_GROWTH_YOY.median().unstack().round(1).to_string())
    lr = us.LOG_REVENUE.dropna()
    print(f"\nLOG_REVENUE range {lr.min():.2f} to {lr.max():.2f} = {(lr.max() - lr.min()) / np.log(10):.1f} orders of magnitude")
    print("\nU.S. AI disclosure by year: median AI_INTENSITY, mean AI_INTENSITY, share with a GenAI term")
    print(us.groupby("YEAR").agg(median_ai=("AI_INTENSITY", "median"), mean_ai=("AI_INTENSITY", "mean"),
                                 genai_share=("GENAI_MENTION", "mean")).round(3))

    print("\n6. Decline share by segment and period (U.S.)")
    seg = us[us.PERIOD.isin(["pre", "post"])].groupby(["SEGMENT", "PERIOD"]).agg(
        firm_quarters=("cik", "size"), firms=("cik", "nunique"), decline_share=("DECLINE_Q", "mean"),
        median_growth=("REV_GROWTH_YOY", "median")).round(3)
    print(seg)

    print("\n7. Spearman correlations, U.S. post-GenAI firm-quarters")
    corr_cols = ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "AI_INTENSITY", "GENAI_MENTION", "LOG_REVENUE",
                 "OP_MARGIN", "IS_10K", "DECLINE_NEXT_Q"]
    post = us[us.PERIOD == "post"]
    corr = post[corr_cols].astype(float).corr(method="spearman").round(2)
    print(corr)

    print("\n8. Growth by AI-disclosure group, U.S. post-GenAI (no AI terms, then quartiles of the positive values)")
    p = post.dropna(subset=["AI_INTENSITY"]).copy()
    pos = p.AI_INTENSITY > 0
    p["AI_Q"] = "None (0)"
    p.loc[pos, "AI_Q"] = pd.qcut(p.loc[pos, "AI_INTENSITY"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]).astype(str)
    p["AI_Q"] = pd.Categorical(p["AI_Q"], ["None (0)", "Q1 (low)", "Q2", "Q3", "Q4 (high)"], ordered=True)
    aiq = p.groupby("AI_Q", observed=True).agg(n=("cik", "size"), ai_median=("AI_INTENSITY", "median"),
                                               growth_median=("REV_GROWTH_YOY", "median"),
                                               growth_mean=("REV_GROWTH_YOY", "mean"),
                                               decline_share=("DECLINE_Q", "mean"),
                                               log_rev_median=("LOG_REVENUE", "median")).round(2)
    print(aiq)

    print("\n9. Quarters per firm (U.S.)")
    qpf = us.groupby("cik").size()
    print(f"   mean {qpf.mean():.1f}, median {qpf.median():.0f}, min {qpf.min()}, max {qpf.max()}; "
          f"firms with all 28 quarters: {(qpf == 28).sum()}")
(RESULTS / "eda_output.txt").write_text(buf.getvalue())
print(buf.getvalue())

# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=110)
    plt.close(fig)


# A. median growth by quarter, US vs India
q = df.groupby(["MARKET", "Q"]).REV_GROWTH_YOY.median().unstack(0)
fig, ax = plt.subplots(figsize=(13, 5.6))
x = q.index.to_timestamp()
ax.plot(x, q["US"], color=US_C, lw=2.2, marker="o", ms=4)
ax.plot(x, q["India"], color=IN_C, lw=2.2, marker="o", ms=4)
ax.axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-12-31"), color="#d9dde3", alpha=0.7, lw=0)
ax.text(pd.Timestamp("2022-11-15"), ax.get_ylim()[1] * 0.97, "ChatGPT\nrelease", ha="center", va="top",
        fontsize=11, color=MUTED)
ax.text(x[-1] + pd.Timedelta(days=25), q["US"].iloc[-1], "U.S.", color=US_C, fontsize=12, va="center", weight="bold")
ax.text(x[-1] + pd.Timedelta(days=25), q["India"].iloc[-1], "India", color=IN_C, fontsize=12, va="center",
        weight="bold")
ax.axhline(0, color="#9aa7b5", lw=1)
ax.set_ylabel("Median year-over-year revenue growth (%)", color=MUTED)
save(fig, "eda_growth_by_quarter.png")

# B. distribution of growth by year (U.S.): box = middle half, whiskers = 5th to 95th percentile
fig, ax = plt.subplots(figsize=(13, 5.4))
years = sorted(us.YEAR.unique())
data = [us.loc[us.YEAR == y, "REV_GROWTH_YOY"].values for y in years]
bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.55,
                medianprops=dict(color="white", lw=2.2))
for patch, y in zip(bp["boxes"], years):
    patch.set_facecolor(LIGHT if y <= 2022 else US_C); patch.set_edgecolor(LIGHT if y <= 2022 else US_C)
for k, y in enumerate(years):
    med = np.median(data[k])
    ax.text(k + 1.33, med, f"{med:.1f}", va="center", fontsize=10.5, color=INK)
ax.set_xticklabels([f"{y}\nn = {len(d)}" for y, d in zip(years, data)])
ax.axhline(0, color="#9aa7b5", lw=1)
ax.set_ylabel("Year-over-year revenue growth (%)", color=MUTED)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=LIGHT, label="2019–2022 (pre-GenAI and transition)"),
                   Patch(color=US_C, label="2023–2025 (post-GenAI)")], frameon=False, loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "eda_growth_by_year.png")

# C. AI disclosure over time (US): median intensity and GenAI mention share, two small panels
ai = us.groupby("Q").agg(med=("AI_INTENSITY", "median"), genai=("GENAI_MENTION", "mean"))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
xx = ai.index.to_timestamp()
axes[0].plot(xx, ai["med"], color=US_C, lw=2.2, marker="o", ms=4)
axes[0].set_title("Median AI mentions per 10,000 words", fontsize=13, color=INK, loc="left")
axes[1].plot(xx, ai["genai"] * 100, color=IN_C, lw=2.2, marker="o", ms=4)
axes[1].set_title("Firm-quarters with a generative-AI term (%)", fontsize=13, color=INK, loc="left")
for a in axes:
    a.axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-12-31"), color="#d9dde3", alpha=0.7, lw=0)
    a.tick_params(axis="x", labelrotation=0)
save(fig, "eda_ai_disclosure_trend.png")

# D. decline share by segment, pre vs post (US)
sd = us[us.PERIOD.isin(["pre", "post"])].groupby(["SEGMENT", "PERIOD"]).DECLINE_Q.mean().unstack()[["pre", "post"]]
fig, ax = plt.subplots(figsize=(13, 5))
ypos = np.arange(len(sd))
h = 0.36
ax.barh(ypos + h / 2, sd["pre"] * 100, h, color=LIGHT, label="Pre-GenAI")
ax.barh(ypos - h / 2, sd["post"] * 100, h, color=US_C, label="Post-GenAI")
for i, (a, b) in enumerate(zip(sd["pre"], sd["post"])):
    ax.text(a * 100 + 0.6, i + h / 2, f"{a * 100:.1f}%", va="center", fontsize=11, color=INK)
    ax.text(b * 100 + 0.6, i - h / 2, f"{b * 100:.1f}%", va="center", fontsize=11, color=INK)
ax.set_yticks(ypos); ax.set_yticklabels(sd.index)
ax.set_xlabel("Share of firm-quarters with a year-over-year revenue decline (%)", color=MUTED)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.set_xlim(0, 42)
ax.grid(axis="y", visible=False)
save(fig, "eda_decline_by_segment.png")

# E. growth by AI-disclosure quintile (post, US)
fig, ax = plt.subplots(figsize=(13, 5))
xs = np.arange(len(aiq))
ax.bar(xs, aiq["growth_median"], color=US_C, width=0.6)
for i, v in enumerate(aiq["growth_median"]):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=11, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels([f"{lab}\nn = {n}, AI median {m:.1f}" for lab, n, m in zip(aiq.index, aiq["n"], aiq["ai_median"])])
ax.set_ylabel("Median year-over-year growth (%)", color=MUTED)
ax.set_xlabel("AI_INTENSITY group: firm-quarters with no AI terms, then quartiles of positive values (post-GenAI, U.S.)", color=MUTED)
ax.grid(axis="x", visible=False)
save(fig, "eda_growth_by_ai_group.png")

# F. Spearman correlation heatmap (post, US)
fig, ax = plt.subplots(figsize=(10.5, 8.5))
m = corr.values
im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=40, ha="right", fontsize=10.5)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=10.5)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(m[i, j]) > 0.5 else INK)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Spearman correlation", color=MUTED)
save(fig, "eda_correlations.png")
print("figures written to", FIG)
