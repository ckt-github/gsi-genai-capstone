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

FEATS = ["AI_INTENSITY", "GENAI_MENTION", "LOG_REVENUE", "SGA_RATIO", "OP_MARGIN",
         "REV_GROWTH_YOY", "REV_GROWTH_LAG1", "POST_GENAI"]
TARGET = "DECLINE_NEXT_Q"


def design(us):
    """Rows with complete features and outcome, plus segment dummies. Returns (data, feature list)."""
    d = us.dropna(subset=FEATS + [TARGET, "SEGMENT"]).copy()
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


def fit_and_tune(model, train, valid, X):
    """Fit on the training years, then pick the decision threshold that maximises F1 on 2024."""
    model.fit(train[X], train[TARGET].astype(int))
    pv = model.predict_proba(valid[X])[:, 1]
    return max(np.linspace(0.1, 0.9, 81), key=lambda c: f1_score(valid[TARGET].astype(int), pv >= c))


def scores(model, rows, X, thr):
    y, p = rows[TARGET].astype(int), model.predict_proba(rows[X])[:, 1]
    yhat = (p >= thr).astype(int)
    auc = roc_auc_score(y, p) if y.nunique() == 2 else float("nan")
    return (f"AUC={auc:.3f} acc={accuracy_score(y, yhat):.3f} prec={precision_score(y, yhat, zero_division=0):.3f} "
            f"rec={recall_score(y, yhat, zero_division=0):.3f} F1={f1_score(y, yhat, zero_division=0):.3f} "
            f"Brier={brier_score_loss(y, p):.3f} thr={thr:.2f}")
