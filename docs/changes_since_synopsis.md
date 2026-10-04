# Changes since the synopsis

The synopsis (`docs/synopsis.pdf`) is the submitted version and is not edited. This log records every
change made to the study afterwards, so later submissions (interim and final reports) can describe them.
Newest entries at the top.

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
  words as 10-Qs, so a firm's score jumps in the quarter it files its 10-K. Options: a 10-K indicator as a
  control, or a four-quarter rolling average.
- **India snapshot gaps.** Several Indian firms (for example Wipro, Mphasis, Persistent) have no
  quarterly results from mid-2022 to mid-2023 in the NSE snapshot, which removes some rows and
  year-over-year comparisons.
- **Kyndryl has no operating margin.** Its filings do not use the `OperatingIncomeLoss` XBRL tag that
  `src/02_get_financials.py` reads, so OP_MARGIN is missing and Kyndryl drops out of RQ2 and RQ3.
  Fix: read an alternative operating-income line for firms without that tag.
- **RQ3 sample.** The minimum of 1,068 is met overall, but the RQ3 rows with complete features are a
  few short of it; to be reported.
