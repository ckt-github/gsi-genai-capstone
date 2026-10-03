"""Minimum sample size for RQ1-RQ4 (alpha = .05, power = .80, Cohen's small effects).

Runs offline; no SEC access needed.
"""
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

# RQ3: confidence-interval method for the proportion of decline events, p = 0.5, e = 0.03, 95%
z = stats.norm.ppf(1 - ALPHA / 2)
n_rq3 = int(-(-(z ** 2 * 0.5 * 0.5 / 0.03 ** 2) // 1))
epv_check = int(-(-(10 * 10 / 0.30) // 1))  # 10 events per variable, 10 predictors, 30% prevalence


# RQ4: chi-square test of independence, 3 segments x 2 outcomes (df = 2), Cohen's w = 0.1 (small)
def power_chi(n, w=0.1, df=2):
    return 1 - stats.ncx2.cdf(stats.chi2.ppf(1 - ALPHA, df), df, n * w * w)


n_rq4 = 10
while power_chi(n_rq4) < POWER:
    n_rq4 += 1

results = {"RQ1": n_rq1, "RQ2": n_rq2, "RQ3": n_rq3, "RQ4": n_rq4}
for k, v in results.items():
    print(f"{k}: minimum N = {v:,}")
print(f"RQ1 per group = {int(-(-n_rq1_group // 1))}; RQ3 events-per-variable cross-check = {epv_check}")
print(f"FINAL N = max = {max(results.values()):,} firm-quarters")
