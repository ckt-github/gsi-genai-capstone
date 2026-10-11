"""Minimum sample size for RQ1-RQ4 before the design effect (alpha = .05, power = .80, small effects).

Runs offline; no SEC access needed. These are the base requirements for independent observations.
src/00b_design_check.py then inflates them by the design effect for repeated quarters of the same firm
and compares them with what the panel actually provides.
"""
import math

from scipy import stats
from statsmodels.stats.power import TTestIndPower

ALPHA, POWER = 0.05, 0.80

# RQ1: Welch two-sample t-test, pre vs post GenAI, Cohen's d = 0.2 (small)
n_rq1_group = TTestIndPower().solve_power(effect_size=0.2, alpha=ALPHA, power=POWER, ratio=1.0)
n_rq1 = 2 * int(-(-n_rq1_group // 1))


# RQ2: multiple regression, test of 1 coefficient among p = 7 predictors, f^2 = 0.02 (small)
def power_f(n, f2=0.02, u=1, p=7):
    v = n - p - 1
    return 1 - stats.ncf.cdf(stats.f.ppf(1 - ALPHA, u, v), u, v, f2 * n)


n_rq2 = 10
while power_f(n_rq2) < POWER:
    n_rq2 += 1

# RQ3: precision of the test-set ROC-AUC (Hanley & McNeil, 1982). The 95% CI for an AUC of 0.70 should
# be no wider than +/-0.10, at the planning prevalence of 30% next-quarter declines.
# Training data: at least 10 decline events per candidate predictor (Peduzzi et al., 1996), 10 predictors.
z = stats.norm.ppf(1 - ALPHA / 2)
PREV, AUC, HALF = 0.30, 0.70, 0.10


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))


n_rq3_test = next(n for n in range(20, 5000)
                  if z * hanley_mcneil_se(AUC, round(n * PREV), round(n * (1 - PREV))) <= HALF)
n_rq3_train = math.ceil(10 * 10 / PREV)
n_rq3 = n_rq3_test + n_rq3_train


# RQ4: chi-square test of independence, 3 segments x 2 outcomes (df = 2), Cohen's w = 0.1 (small)
def power_chi(n, w=0.1, df=2):
    return 1 - stats.ncx2.cdf(stats.chi2.ppf(1 - ALPHA, df), df, n * w * w)


n_rq4 = 10
while power_chi(n_rq4) < POWER:
    n_rq4 += 1

results = {"RQ1": n_rq1, "RQ2": n_rq2, "RQ3": n_rq3, "RQ4": n_rq4}
for k, v in results.items():
    print(f"{k}: minimum N = {v:,}")
print(f"RQ1 per group = {int(-(-n_rq1_group // 1))}")
print(f"RQ3 test set = {n_rq3_test} (about {round(n_rq3_test * PREV)} declines; Hanley-McNeil SE = "
      f"{hanley_mcneil_se(AUC, round(n_rq3_test * PREV), round(n_rq3_test * (1 - PREV))):.4f}); "
      f"training = {n_rq3_train} (100 declines)")
print(f"FINAL N = max = {max(results.values()):,} firm-quarters (before the design effect; see 00b_design_check.py)")
