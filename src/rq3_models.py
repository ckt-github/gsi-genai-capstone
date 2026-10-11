"""RQ3 model definitions, shared by the main analysis (05) and the 2026 extension (06).

Keeping them in one place guarantees the extension scores 2026 with exactly the model that was
trained and tuned for the main study (same features, splits, settings and random seed).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# SGA_RATIO was dropped as a feature after the synopsis review: two-thirds of U.S. firm-quarters do not
# report SG&A, so requiring it removed most firms from RQ3. IS_10K takes its place (see the change log).
FEATS = ["AI_INTENSITY", "GENAI_MENTION", "LOG_REVENUE", "IS_10K", "OP_MARGIN",
         "REV_GROWTH_YOY", "REV_GROWTH_LAG1", "POST_GENAI"]
TARGET = "DECLINE_NEXT_Q"


def design(us):
    """Rows with complete features and outcome, plus segment dummies. Returns (data, feature list)."""
    d = us.dropna(subset=FEATS + [TARGET, "SEGMENT"]).copy()
    d[FEATS] = d[FEATS].astype(float)
    d = pd.concat([d, pd.get_dummies(d["SEGMENT"], prefix="SEG", drop_first=True, dtype=int)], axis=1)
    return d, FEATS + [c for c in d.columns if c.startswith("SEG_")]


def split(d):
    """Time-ordered split: train 2019-2023, validate 2024, test 2025."""
    return (d[d.FISCAL_QTR < "2024Q1"], d[d.FISCAL_QTR.between("2024Q1", "2024Q4")],
            d[d.FISCAL_QTR.between("2025Q1", "2025Q4")])


def make_models(seed):
    return {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced",
                                                random_state=seed),
        "xgboost": XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="auc", random_state=seed),
    }


COST_FN, COST_FP = 3, 1   # a missed decline is treated as three times as costly as a false alarm


def fit_and_tune(model, train, valid, X):
    """Fit on the training years, then pick the decision threshold with the lowest expected cost on 2024
    (3 x missed declines + 1 x false alarms). 2025 is never used here."""
    model.fit(train[X], train[TARGET].astype(int))
    pv = model.predict_proba(valid[X])[:, 1]
    y = valid[TARGET].astype(int).values

    def cost(c):
        yhat = pv >= c
        return COST_FN * ((y == 1) & ~yhat).sum() + COST_FP * ((y == 0) & yhat).sum()
    return min(np.linspace(0.1, 0.9, 81), key=cost)


def scores(model, rows, X, thr):
    y, p = rows[TARGET].astype(int), model.predict_proba(rows[X])[:, 1]
    yhat = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if y.nunique() == 2 else float("nan")
    return (f"AUC={auc:.3f} acc={accuracy_score(y, yhat):.3f} prec={precision_score(y, yhat, zero_division=0):.3f} "
            f"rec={recall_score(y, yhat, zero_division=0):.3f} F1={f1_score(y, yhat, zero_division=0):.3f} "
            f"Brier={brier_score_loss(y, p):.3f} thr={thr:.2f}")


def firm_bootstrap_auc(y, p, firms, n_boot=2000, seed=0):
    """95% interval for AUC that resamples whole firms, so repeated quarters of one firm stay together."""
    rng = np.random.default_rng(seed)
    y, p, firms = np.asarray(y), np.asarray(p), np.asarray(firms)
    ids = np.unique(firms)
    idx = {f: np.where(firms == f)[0] for f in ids}
    aucs = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[f] for f in rng.choice(ids, len(ids), replace=True)])
        if len(np.unique(y[rows])) == 2:
            aucs.append(roc_auc_score(y[rows], p[rows]))
    return np.percentile(aucs, [2.5, 97.5])


def delong(y, p1, p2):
    """DeLong et al. (1988) test. Returns (auc1, auc2, z, p) for the difference auc1 - auc2 on the same rows."""
    from scipy import stats
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]

    def comps(p):
        p = np.asarray(p, float)
        x, z = p[pos][:, None], p[neg][None, :]
        psi = (x > z) + 0.5 * (x == z)
        return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)

    a1, v1, w1 = comps(p1)
    a2, v2, w2 = comps(p2)
    s10, s01 = np.cov(np.vstack([v1, v2])), np.cov(np.vstack([w1, w2]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(pos) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(neg)
    zval = (a1 - a2) / np.sqrt(var)
    return a1, a2, zval, 2 * (1 - stats.norm.cdf(abs(zval)))


def delong_se(y, p):
    """DeLong standard error of a single AUC."""
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    p = np.asarray(p, float)
    psi = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
    return np.sqrt(psi.mean(axis=1).var(ddof=1) / len(pos) + psi.mean(axis=0).var(ddof=1) / len(neg))
