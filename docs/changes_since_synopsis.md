# Changes since the synopsis

The synopsis (`docs/synopsis.pdf`) is the submitted version and is not edited. This log records every
change made to the study afterwards, so later submissions (interim and final reports) can describe them.
Newest entries at the top.

## 20 October 2026 (interim report)

### Added: exploratory analysis (`src/08_eda.py`)
- Summary statistics by period, missing values, skewness, trends by quarter and year, decline share by
  segment, growth by AI-disclosure group, and Spearman correlations. Output: `results/eda_output.txt` and
  charts in `results/figures/eda_*.png`.

### Added: model details (`src/09_model_details.py`)
- RQ1 without 2020 and a median regression; RQ2 VIF, Breusch–Pagan, residual checks, Huber regression and a
  random forest with SHAP; RQ3 model choice on 2024 validation AUC, confusion matrices, Wilson intervals,
  expected cost, the acceptance-criteria check, an ablation without the AI features, and SHAP importance;
  RQ4 odds ratios with confidence intervals and firm-bootstrap decline shares. Output:
  `results/model_details.txt` and `results/figures/model_*.png`.

### Added: preprocessing log (`src/04_build_panel.py`)
- Writes `results/preprocessing_log.txt` with the rows removed or changed at each step (duplicates,
  amendments, shells, growth not computable, winsorized values). The panel itself is unchanged.

## 11 October 2026 (after mentor review of the synopsis)

The mentor's review asked for sample sizes that allow for repeated quarters of the same firm, a
better-specified RQ3 test design, and clearer data-construction rules. These changes implement the
revised synopsis.

### Added: design check (`src/00b_design_check.py`)
- Counts firms and firm-quarters for every analysis, estimates the intraclass correlation (ICC) from
  pre-GenAI quarters only, and inflates each base sample size by the design effect
  DEFF = 1 + (m − 1) × ICC. Also reports the 2025 test set (firms, declines), the precision of AUC, recall
  and precision on it, missing values and revenue edge cases. Output: `results/design_check.txt`.

### Changed: RQ3 sample-size method (`src/00_sample_size.py`)
- The proportion-based figure (1,068) is replaced by the precision of the test-set ROC-AUC
  (Hanley–McNeil): 139 test firm-quarters before the design effect, plus 334 training firm-quarters
  (10 events per predictor).

### Changed: analysis (`src/05_analysis.py`)
- RQ1 primary test is now a firm-clustered regression of growth on POST_GENAI, with a firm-level paired
  test as confirmation; the Welch and Mann–Whitney tests are kept as descriptive checks.
- RQ2 drops SGA_RATIO from the main model (two-thirds of U.S. rows have no SG&A, which cut the sample to
  52 firms) and adds IS_10K. SG&A, quarter fixed effects and the four-quarter AI average are sensitivity
  checks.
- RQ3 adds a persistence baseline (current decline status), DeLong and firm-bootstrap confidence
  intervals, and a DeLong comparison with the baseline. SGA_RATIO is replaced by IS_10K as a feature.
  The decision threshold is now chosen on 2024 by expected cost (a missed decline counts three times a
  false alarm) instead of by F1. Planned, not yet coded: winsorizing limits for RQ3 taken from the
  training years only, and a rerun with first-reported (not restated) XBRL values.
- RQ4 primary test is now a GEE logistic regression clustered by firm (Wald test); the Pearson
  chi-square is descriptive, with a design-effect-adjusted version and a firm-level check.

### Changed: panel (`src/04_build_panel.py`)
- Growth is left blank when revenue in the same quarter a year earlier was below US$1 million or
  zero/negative (33 U.S. firm-quarters in the study window).
- New variables IS_10K (AI score taken from a 10-K) and AI_INTENSITY_4Q (four-quarter average).
- `docs/data_dictionary.csv` gains these variables and a column with valid values and missing-value rules;
  a new unit test checks that the dictionary and the panel have the same columns.

## 4 October 2026

### Added: GSI focus (`src/07_gsi_focus.py`)
- Repeats the analysis for the large commercial GSIs already inside the sample. The firm scope and the
  main analysis are unchanged.
- Rule: annualised 2021–2022 revenue of at least $1 billion (measured before the GenAI period) and a
  commercial IT-consulting, systems-integration or managed-IT-services business. Every $1B+ firm is
  classified, with reasons, in `config/gsi_classification.csv`.
- GSIs: Accenture, Cognizant, Kyndryl, DXC, EPAM, Rackspace, Unisys, Thoughtworks (U.S.); TCS, Infosys,
  HCLTech, Wipro, Tech Mahindra, LTIMindtree, Mphasis (India). Government integrators are not counted.
- Output: `results/gsi_focus_output.txt`.

### Added: 2026 extension (`src/06_extension_2026.py`)
- A robustness check kept separate from the 2019–2025 analysis. A 2026 quarter is used only when it is at
  least 60 days old and at least 90% of the 2025Q4 firms have reported it (per market).
- RQ1 by year (mean, median, trimmed mean); the RQ3 models trained and tuned as in the main study are
  scored on 2026 without refitting.
- Output: `results/extension_2026_output.txt`, `results/extension_eligibility.csv`.

### Fixed: AI scores for fiscal quarters ending in January 2026
- Quarters ending in January 2026 belong to calendar 2025Q4, but filings were only scored up to
  31 December 2025, so eight firm-quarters (for example Workday, Leidos, SAIC, Zoom) had no AI score and
  dropped out of RQ2 and RQ3. Scoring now covers them.
- Effect on main results: RQ2 N 452 → 454, AI_INTENSITY coefficient 0.96 → 0.94 (p = .015 unchanged);
  RQ3 test set 135 → 137, random-forest AUC 0.895 → 0.886. RQ1 and RQ4 unchanged.

### Changed: package versions pinned (`requirements.txt`)
- Exact versions used by the GitHub pipeline, so every rerun gives identical numbers on any computer.

### Changed: RQ3 model definitions shared (`src/rq3_models.py`)
- Used by the main analysis and the extension; main RQ3 output verified identical after the move.

### Security: SEC contact moved to a repository secret
- The User-Agent contact is masked in public run logs. Logs of earlier runs that showed it were deleted.

## Known issues to address in the interim report
- **Outliers in mean growth.** A few very fast-growing small firms (for example quantum-computing and
  AI-infrastructure firms) pull U.S. mean growth up; report medians and Mann-Whitney tests alongside means.
- **SIC codes are broad.** The SIC-based sample includes internet platforms and software-product firms
  (already noted as a limitation in the synopsis); the GSI focus addresses this.
- **AI_INTENSITY is higher in 10-K quarters.** Annual reports mention AI about twice as often per 10,000
  words as 10-Qs. Addressed on 11 October 2026 with IS_10K and AI_INTENSITY_4Q.
- **India snapshot gaps.** Several Indian firms (for example Wipro, Mphasis, Persistent) have no
  quarterly results from mid-2022 to mid-2023 in the NSE snapshot, which removes some rows and
  year-over-year comparisons.
- **Kyndryl has no operating margin.** Its filings do not use the `OperatingIncomeLoss` XBRL tag that
  `src/02_get_financials.py` reads, so OP_MARGIN is missing and Kyndryl drops out of RQ2 and RQ3.
  Fix: read an alternative operating-income line for firms without that tag.
- **RQ4 power.** Even with every eligible firm, RQ4 can only detect a segment effect of about w = 0.15
  (an 18-point gap in decline share) once repeated quarters are allowed for; to be reported with
  confidence intervals rather than by widening the sample.
