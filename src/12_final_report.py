"""Final-report outputs: separated training / validation / test metrics for RQ3, confusion matrices, and every
report figure redrawn at its printed size (6.5 in wide or less) so labels stay readable in the document.

Reads the processed panel and the text results written by 05, 09 and 11; trains the RQ3 models again with the
same settings and seed (rq3_models.py), so the test figures match results/analysis_output.txt.
Writes results/final_report_metrics.txt and results/figures/report/*.png.
"""
import contextlib
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import rq3_models as rq3m
from common import PROCESSED, RESULTS, SETTINGS

SEED = SETTINGS["random_seed"]
FIG = RESULTS / "figures" / "report"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e7ee", "axes.axisbelow": True, "axes.titlelocation": "left"})
US_C, IN_C, LIGHT, GREY = "#1f5a96", "#c95b37", "#5b9bd5", "#5a6b7d"
INK, MUTED = "#1b2a3a", "#4a5b6d"
MODEL_C = {"logit": LIGHT, "random_forest": US_C, "xgboost": IN_C}
LABEL = {"logit": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}
W = 6.5


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)


df = pd.read_csv(PROCESSED / "panel_firm_quarter.csv", dtype={"cik": str})
df["Q"] = pd.PeriodIndex(df["FISCAL_QTR"], freq="Q")
us = df[df.MARKET == "US"]
post = us[us.PERIOD == "post"]

# ------------------------------------------------------------------ RQ3: separated evaluation
d, X = rq3m.design(us)
train, valid, test = rq3m.split(d)
buf = io.StringIO()
rows, probs, cms = [], {}, {}
with contextlib.redirect_stdout(buf):
    print("RQ3 SEPARATED EVALUATION (training 2019-2023, validation 2024, test 2025)")
    print(f"rows: train {len(train)} ({train[rq3m.TARGET].sum():.0f} declines), valid {len(valid)} "
          f"({valid[rq3m.TARGET].sum():.0f}), test {len(test)} ({test[rq3m.TARGET].sum():.0f})\n")
    for name, model in rq3m.make_models(SEED).items():
        thr = rq3m.fit_and_tune(model, train, valid, X)
        for split_name, part in [("train", train), ("validation", valid), ("test", test)]:
            y = part[rq3m.TARGET].astype(int).values
            p = model.predict_proba(part[X])[:, 1]
            yhat = (p >= thr).astype(int)
            rows.append(dict(model=name, split=split_name, n=len(part), auc=roc_auc_score(y, p),
                             accuracy=accuracy_score(y, yhat), precision=precision_score(y, yhat),
                             recall=recall_score(y, yhat), f1=f1_score(y, yhat), brier=brier_score_loss(y, p),
                             thr=thr))
            if split_name == "test":
                probs[name] = p
                cms[name] = confusion_matrix(y, yhat, labels=[0, 1])
    for split_name, part in [("train", train), ("validation", valid), ("test", test)]:
        y = part[rq3m.TARGET].astype(int).values
        yb = (part["REV_GROWTH_YOY"] < 0).astype(int).values
        rows.append(dict(model="persistence", split=split_name, n=len(part), auc=roc_auc_score(y, yb),
                         accuracy=accuracy_score(y, yb), precision=precision_score(y, yb),
                         recall=recall_score(y, yb), f1=f1_score(y, yb), brier=float("nan"), thr=float("nan")))
        if split_name == "test":
            cms["persistence"] = confusion_matrix(y, yb, labels=[0, 1])
            base_p = yb
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print("\nConfusion matrices on the 2025 test set [[TN, FP], [FN, TP]]")
    for k, v in cms.items():
        print(f"   {k:14s} {v.tolist()}")
(RESULTS / "final_report_metrics.txt").write_text(buf.getvalue())
print(buf.getvalue())
y_te = test[rq3m.TARGET].astype(int).values

# ------------------------------------------------------------------ Figure: architecture / workflow
fig, ax = plt.subplots(figsize=(W, 4.6))
ax.set_xlim(0, 100); ax.set_ylim(-6, 62); ax.axis("off")
stages = [
    ("1  Data", "SEC EDGAR APIs\n(XBRL facts, filing text)\nNSE XBRL results\nFRED DEXINUS", "#dbe7f3"),
    ("2  Preprocessing", "Tag order, Q4 derivation\ncalendar mapping\nde-duplication, filters\nwinsorizing (1st/99th)", "#dbe7f3"),
    ("3  EDA", "Summaries, trends\nsegment and AI groups\nSpearman correlations\nanomaly checks", "#dbe7f3"),
    ("4  Feature\n    engineering", "AI_INTENSITY, IS_10K\nlog revenue, lags\none-hot segment\ntraining-year caps", "#fbe3d9"),
    ("5  Model", "Clustered OLS (RQ1-2)\nLogit / RF / XGBoost (RQ3)\nGEE logit (RQ4)\ntime split 2019-23/24/25", "#fbe3d9"),
    ("6  Evaluation", "AUC, DeLong, bootstrap\nrecall, precision, Brier\nSHAP, ablation\nrolling origin", "#fbe3d9"),
    ("7  Deployment", "Quarterly batch job\n(GitHub Actions)\npartner-risk scorecard\nCSV + dashboard", "#e2efe0"),
]
# lay the seven stages on two rows: 1-3 left to right on top, 4-6 right to left below, 7 at far right
coords = [(1, 33), (26, 33), (51, 33), (76, 33), (76, 2), (51, 2), (26, 2)]
bw, bh = 23, 26
for (title, body, col), (x, y) in zip(stages, coords):
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.4,rounding_size=1.5", fc=col, ec=US_C, lw=0.9))
    ax.text(x + 1.2, y + bh - 2.2, title, fontsize=8.6, weight="bold", color=INK, va="top")
    ax.text(x + 1.2, y + bh - 8.6, body, fontsize=7.2, color=INK, va="top", linespacing=1.35)
arrow = dict(arrowstyle="-|>", color=GREY, lw=1.2, mutation_scale=10)
for (x0, y0), (x1, y1) in zip(coords[:3], coords[1:4]):
    ax.annotate("", xy=(x1 - 0.6, y1 + bh / 2), xytext=(x0 + bw + 0.6, y0 + bh / 2), arrowprops=arrow)
ax.annotate("", xy=(76 + bw / 2, 2 + bh + 0.7), xytext=(76 + bw / 2, 33 - 0.7), arrowprops=arrow)
for (x0, y0), (x1, y1) in zip(coords[4:6], coords[5:7]):
    ax.annotate("", xy=(x1 + bw + 0.6, y1 + bh / 2), xytext=(x0 - 0.6, y0 + bh / 2), arrowprops=arrow)
ax.text(1.5, 15, "Outputs\nresults/*.txt\nresults/figures\nscorecard.csv", fontsize=7.2, color=MUTED, va="center",
        linespacing=1.35)
ax.annotate("", xy=(13, 15), xytext=(26 - 0.6, 15), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1, ls="--",
                                                                    mutation_scale=9))
ax.legend(handles=[Patch(fc="#dbe7f3", ec=US_C, label="Data pipeline"),
                   Patch(fc="#fbe3d9", ec=US_C, label="Analytics and models"),
                   Patch(fc="#e2efe0", ec=US_C, label="Deployment")],
          loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=7.5, handlelength=1.2)
save(fig, "fig_architecture.png")

# ------------------------------------------------------------------ EDA figures
q = df.groupby(["MARKET", "Q"]).REV_GROWTH_YOY.median().unstack(0)
x = q.index.to_timestamp()
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-12-31"), color="#d9dde3", alpha=0.8, lw=0)
ax.plot(x, q["US"], color=US_C, lw=1.8, marker="o", ms=3, label="U.S. (201 firms)")
ax.plot(x, q["India"], color=IN_C, lw=1.8, marker="o", ms=3, label="India (15 firms)")
ax.text(pd.Timestamp("2022-11-15"), 26.5, "2022Q4\n(ChatGPT)", ha="center", va="top", fontsize=7.5, color=MUTED)
ax.axhline(0, color="#9aa7b5", lw=0.8)
ax.set_ylim(-2, 27)
ax.set_ylabel("Median YoY revenue growth (%)")
ax.set_xlabel("Calendar quarter")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_growth_by_quarter.png")

ai = us.groupby("Q").agg(med=("AI_INTENSITY", "median"), genai=("GENAI_MENTION", "mean"))
xx = ai.index.to_timestamp()
fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
axes[0].plot(xx, ai["med"], color=US_C, lw=1.6, marker="o", ms=2.5)
axes[0].set_title("(a) Median AI mentions per 10,000 words")
axes[0].set_ylabel("Mentions per 10,000 words")
axes[1].plot(xx, ai["genai"] * 100, color=IN_C, lw=1.6, marker="o", ms=2.5)
axes[1].set_title("(b) Firm-quarters with a GenAI term")
axes[1].set_ylabel("Share of firm-quarters (%)")
for a in axes:
    a.axvspan(pd.Timestamp("2022-10-01"), pd.Timestamp("2022-12-31"), color="#d9dde3", alpha=0.8, lw=0)
    a.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
save(fig, "fig_ai_disclosure.png")

sd = us[us.PERIOD.isin(["pre", "post"])].groupby(["SEGMENT", "PERIOD"]).DECLINE_Q.mean().unstack()[["pre", "post"]]
short = {"Data processing & managed services": "Data processing &\nmanaged services",
         "IT consulting & application services": "IT consulting &\napplication services",
         "Systems integration": "Systems\nintegration"}
fig, ax = plt.subplots(figsize=(W, 2.6))
ypos, h = np.arange(len(sd)), 0.36
ax.barh(ypos + h / 2, sd["pre"] * 100, h, color=LIGHT, label="Pre-GenAI (2019Q1-2022Q3)")
ax.barh(ypos - h / 2, sd["post"] * 100, h, color=US_C, label="Post-GenAI (2023Q1-2025Q4)")
for i, (a, b) in enumerate(zip(sd["pre"], sd["post"])):
    ax.text(a * 100 + 0.5, i + h / 2, f"{a * 100:.1f}%", va="center", fontsize=8, color=INK)
    ax.text(b * 100 + 0.5, i - h / 2, f"{b * 100:.1f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(ypos); ax.set_yticklabels([short[s] for s in sd.index])
ax.set_xlabel("Share of firm-quarters with a YoY revenue decline (%)")
ax.set_xlim(0, 45)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.grid(axis="y", visible=False)
save(fig, "fig_decline_by_segment.png")

p = post.dropna(subset=["AI_INTENSITY"]).copy()
posm = p.AI_INTENSITY > 0
p["AI_Q"] = "None (0)"
p.loc[posm, "AI_Q"] = pd.qcut(p.loc[posm, "AI_INTENSITY"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]).astype(str)
p["AI_Q"] = pd.Categorical(p["AI_Q"], ["None (0)", "Q1 (low)", "Q2", "Q3", "Q4 (high)"], ordered=True)
aiq = p.groupby("AI_Q", observed=True).agg(n=("cik", "size"), ai=("AI_INTENSITY", "median"),
                                           g=("REV_GROWTH_YOY", "median"))
fig, ax = plt.subplots(figsize=(W, 2.8))
xs = np.arange(len(aiq))
ax.bar(xs, aiq["g"], color=US_C, width=0.6)
for i, v in enumerate(aiq["g"]):
    ax.text(i, v + 0.25, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels([f"{lab}\nn = {n}\nmedian AI {m:.1f}" for lab, n, m in zip(aiq.index, aiq["n"], aiq["ai"])],
                   fontsize=8)
ax.set_ylabel("Median YoY growth (%)")
ax.set_ylim(0, 12)
ax.grid(axis="x", visible=False)
save(fig, "fig_growth_by_ai_group.png")

corr_cols = ["REV_GROWTH_YOY", "REV_GROWTH_LAG1", "AI_INTENSITY", "GENAI_MENTION", "LOG_REVENUE", "OP_MARGIN",
             "IS_10K", "DECLINE_NEXT_Q"]
m = post[corr_cols].astype(float).corr(method="spearman").values
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(8)); ax.set_xticklabels(corr_cols, rotation=40, ha="right", fontsize=7.8)
ax.set_yticks(range(8)); ax.set_yticklabels(corr_cols, fontsize=7.8)
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.4,
                color="white" if abs(m[i, j]) > 0.5 else INK)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("Spearman correlation"); cb.ax.tick_params(labelsize=7.5)
save(fig, "fig_correlations.png")

# ------------------------------------------------------------------ RQ3 figures
fig, ax = plt.subplots(figsize=(4.4, 4.2))
for name, pt in probs.items():
    fpr, tpr, _ = roc_curve(y_te, pt)
    ax.plot(fpr, tpr, color=MODEL_C[name], lw=1.8, label=f"{LABEL[name]} (AUC {roc_auc_score(y_te, pt):.3f})")
fpr, tpr, _ = roc_curve(y_te, base_p)
ax.plot(fpr, tpr, color=GREY, lw=1.4, ls="--", marker="o", ms=4,
        label=f"Persistence baseline (AUC {roc_auc_score(y_te, base_p):.3f})")
ax.plot([0, 1], [0, 1], color="#c3c9d2", lw=0.8, label="Chance (AUC 0.500)")
ax.set_xlabel("False positive rate (1 - specificity)")
ax.set_ylabel("True positive rate (recall)")
ax.legend(frameon=False, loc="lower right", fontsize=7.8)
ax.set_aspect("equal")
save(fig, "fig_roc.png")

fig, axes = plt.subplots(1, 4, figsize=(W, 2.2))
for a, name in zip(axes, ["persistence", "logit", "random_forest", "xgboost"]):
    cm = cms[name]
    a.imshow(cm, cmap="Blues", vmin=0, vmax=320)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{['TN', 'FP', 'FN', 'TP'][2 * i + j]}\n{cm[i, j]}", ha="center", va="center",
                   fontsize=8, color="white" if cm[i, j] > 160 else INK)
    a.set_xticks([0, 1]); a.set_xticklabels(["No", "Yes"], fontsize=7.8)
    a.set_yticks([0, 1]); a.set_yticklabels(["No", "Yes"], fontsize=7.8)
    a.set_xlabel("Predicted decline", fontsize=7.8)
    a.set_title(("Persistence" if name == "persistence" else LABEL[name]).replace("Logistic regression", "Logistic"),
                fontsize=8.5, weight="bold" if name == "random_forest" else "normal")
    a.grid(False)
axes[0].set_ylabel("Actual decline", fontsize=7.8)
save(fig, "fig_confusion.png")

auc_tab = res[res.model != "persistence"].pivot(index="model", columns="split", values="auc")
auc_tab = auc_tab.loc[["logit", "random_forest", "xgboost"], ["train", "validation", "test"]]
fig, ax = plt.subplots(figsize=(W, 2.6))
xs, wbar = np.arange(3), 0.25
for k, (sp, col) in enumerate(zip(["train", "validation", "test"], ["#c9d6e3", LIGHT, US_C])):
    ax.bar(xs + (k - 1) * wbar, auc_tab[sp], wbar, color=col, label={"train": "Training 2019-2023",
           "validation": "Validation 2024", "test": "Test 2025"}[sp])
    for i, v in enumerate(auc_tab[sp]):
        ax.text(i + (k - 1) * wbar, v + 0.005, f"{v:.3f}", ha="center", fontsize=7.4, color=INK)
base_test = res[(res.model == "persistence") & (res.split == "test")].auc.iloc[0]
ax.axhline(base_test, color=GREY, ls="--", lw=1, label=f"Persistence baseline, test ({base_test:.3f})")
ax.set_xticks(xs); ax.set_xticklabels([LABEL[k] for k in auc_tab.index])
ax.set_ylim(0.7, 1.04); ax.set_ylabel("AUC")
ax.set_ylim(0.7, 1.10)
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.6)
ax.grid(axis="x", visible=False)
save(fig, "fig_train_test_auc.png")

rf = rq3m.make_models(SEED)["random_forest"]
rq3m.fit_and_tune(rf, train, valid, X)
sv = shap.TreeExplainer(rf).shap_values(test[X])
sv = sv[:, :, 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
imp = pd.Series(np.abs(sv).mean(0), index=X).sort_values()
imp.index = [i.replace("SEG_", "Segment: ").replace("Systems integration", "Systems integr.")
             .replace("IT consulting & application services", "IT consulting") for i in imp.index]
fig, ax = plt.subplots(figsize=(W, 3.0))
ax.barh(imp.index, imp.values, color=US_C, height=0.65)
for i, v in enumerate(imp.values):
    ax.text(v + imp.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=7.8, color=INK)
ax.set_xlabel("Mean |SHAP value| on the 2025 test set (change in predicted probability)")
ax.tick_params(axis="y", labelsize=8)
ax.set_xlim(0, imp.max() * 1.15)
ax.grid(axis="y", visible=False)
save(fig, "fig_shap.png")

txt = (RESULTS / "sensitivity_output.txt").read_text()
roll = [dict(year=int(a), auc=float(b), lo=float(c), hi=float(e), base=float(f), rec=float(g), prec=float(h),
             brec=float(i), bprec=float(j)) for a, b, c, e, f, g, h, i, j in re.findall(
    r"test (\d{4}): .*?AUC ([\d.]+) \(95% CI ([\d.]+) to ([\d.]+)\); persistence AUC ([\d.]+).*?recall ([\d.]+); "
    r"precision ([\d.]+); persistence recall ([\d.]+), precision ([\d.]+)", txt)]
r = pd.DataFrame(roll)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.8))
xr = np.arange(len(r))
a1.errorbar(xr, r.auc, yerr=[r.auc - r.lo, r.hi - r.auc], fmt="o", color=US_C, ms=5, capsize=4, lw=1.4,
            label="Random forest (95% CI)")
a1.plot(xr, r.base, "s", color=GREY, ms=5, label="Persistence")
for i, v in enumerate(r.auc):
    a1.text(i + 0.14, v, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
a1.set_xticks(xr); a1.set_xticklabels(r.year.astype(str)); a1.set_ylim(0.70, 0.97)
a1.set_xlim(-0.4, len(r) - 0.3)
a1.set_xlabel("Test year"); a1.set_ylabel("Test AUC"); a1.set_title("(a) Ranking quality")
a1.legend(frameon=False, loc="lower right", fontsize=7.5)
a2.plot(xr, r.rec, "o-", color=US_C, lw=1.5, ms=4, label="Recall, RF")
a2.plot(xr, r.prec, "o-", color=IN_C, lw=1.5, ms=4, label="Precision, RF")
a2.plot(xr, r.brec, "o--", color=US_C, lw=1, ms=3, alpha=0.6, label="Recall, persistence")
a2.plot(xr, r.bprec, "o--", color=IN_C, lw=1, ms=3, alpha=0.6, label="Precision, persistence")
a2.set_xticks(xr); a2.set_xticklabels(r.year.astype(str)); a2.set_ylim(0.3, 1.0)
a2.set_xlabel("Test year"); a2.set_ylabel("Share"); a2.set_title("(b) At the tuned threshold")
a2.legend(frameon=False, loc="lower right", fontsize=7, ncol=2)
save(fig, "fig_rolling_origin.png")

seg = []
md = (RESULTS / "model_details.txt").read_text()
for name, share, lo, hi, nf in re.findall(r"   (\S.*?)\s+\(2,000 firm-bootstrap resamples\) decline share ([\d.]+) "
                                         r"\(firm-bootstrap 95% CI ([\d.]+) to ([\d.]+); (\d+) firms", md):
    seg.append((name.strip(), float(share), float(lo), float(hi), int(nf)))
seg.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(W, 2.3))
for i, (sname, mm, lo, hi, nf) in enumerate(seg):
    ax.plot([lo * 100, hi * 100], [i, i], color=LIGHT, lw=5, solid_capstyle="round")
    ax.plot(mm * 100, i, "o", color=US_C, ms=7)
    ax.text(hi * 100 + 1.2, i, f"{mm * 100:.1f}%  ({nf} firms)", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(seg))); ax.set_yticklabels([short[s[0]] for s in seg])
ax.set_xlim(0, 65)
ax.set_xlabel("Post-GenAI decline share with firm-bootstrap 95% interval (%)")
ax.grid(axis="y", visible=False)
save(fig, "fig_rq4_segments.png")
print("report figures written to", FIG)
