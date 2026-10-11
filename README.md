# Generative AI and the Pivot of Global System Integrators

**QM640 Data Analytics Capstone, Walsh College · Chandan Tiwari**

[![Checks](https://github.com/ckt-github/gsi-genai-capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/ckt-github/gsi-genai-capstone/actions/workflows/ci.yml)
[![Data pipeline](https://github.com/ckt-github/gsi-genai-capstone/actions/workflows/pipeline.yml/badge.svg)](https://github.com/ckt-github/gsi-genai-capstone/actions/workflows/pipeline.yml)

This repository holds the data pipeline, data dictionary and analysis code for the capstone study
*Generative AI and the Pivot of Global System Integrators: Evidence From U.S.- and India-Listed IT Services Firms, 2019–2025*. The main sample is U.S.-listed IT services firms that file with the SEC; 15 firms listed on the
National Stock Exchange of India (NSE) form a comparison group.

All data are free and public: SEC EDGAR filings for U.S. registrants, NSE XBRL results filings for the
Indian firms, and the Federal Reserve's INR/USD exchange rate. No Kaggle, synthetic, personal or
confidential data are used.

## Research questions

| RQ | Question | Method | Sample |
|----|----------|--------|--------|
| RQ1 | Did mean YoY revenue growth differ between the pre-GenAI (2019Q1–2022Q3) and post-GenAI (2023Q1–2025Q4) periods? | Regression on POST_GENAI with firm-clustered SEs; firm-level paired test; POST × MARKET regression | U.S. (primary) and India (comparison) |
| RQ2 | In the post-GenAI period, is AI-disclosure intensity associated with YoY revenue growth, after controls? | Multiple linear regression (firm-clustered SEs) | U.S. |
| RQ3 | Can financial and AI-disclosure features predict next-quarter revenue decline better than chance? | Logistic regression, random forest, XGBoost (ROC-AUC vs persistence baseline) | U.S. |
| RQ4 | Is post-GenAI revenue-decline status associated with service segment? | GEE logistic regression clustered by firm; chi-square as descriptive | U.S. |

Sample size: base requirements in `src/00_sample_size.py`; the design effect for repeated quarters and
the comparison with the available panel in `src/00b_design_check.py` (`results/design_check.txt`).

## Indian comparison firms

TCS, Infosys, HCLTech, Wipro, Tech Mahindra, LTIMindtree, Mphasis, Coforge, Persistent Systems,
Birlasoft, Zensar, Mastek, Sonata Software, Cyient and L&T Technology Services (`config/india_firms.csv`).
Their consolidated quarterly results come from the XBRL files each company files with NSE. Growth is
measured in rupees (so currency moves do not distort it); size is converted to USD with the FRED DEXINUS
quarterly average. Indian firms do not file 10-Q/10-K reports, so they are not part of the AI-disclosure
questions (RQ2, RQ3).

The synopsis in `docs/synopsis.pdf` is the submitted version. Everything changed or added since then is
logged in [`docs/changes_since_synopsis.md`](docs/changes_since_synopsis.md).

## Repository structure

```
gsi-genai-capstone/
├── .github/workflows/
│   ├── ci.yml                 # checks on every push (compile, tests, sample size, India pipeline)
│   └── pipeline.yml           # full pipeline in the cloud; commits outputs back to the repo
├── config/
│   ├── settings.yaml          # study window, SIC codes, filters
│   ├── ai_keywords.txt        # AI / GenAI term dictionary
│   ├── named_gsis.csv         # U.S.-listed GSIs added by name (SIC outside range)
│   ├── gsi_classification.csv # every $1B+ firm: GSI or not, with the reason
│   └── india_firms.csv        # NSE comparison firms and segment
├── data/
│   ├── raw/india/             # committed NSE and FX snapshots (see README there)
│   ├── interim/               # universe, financials, AI scores (written by the pipeline)
│   └── processed/
│       ├── panel_firm_quarter.csv   # main study, 2019–2025
│       └── panel_extension.csv      # main rows + eligible 2026 quarters
├── docs/
│   ├── changes_since_synopsis.md   # log of everything added after the synopsis
│   ├── data_dictionary.csv
│   └── synopsis.pdf
├── results/                   # analysis output written by the pipeline
├── src/
│   ├── common.py              # paths, settings, rate-limited SEC client
│   ├── 00_sample_size.py      # RQ1–RQ4 base sample sizes (independent observations)
│   ├── 00b_design_check.py    # design effect, effective n, test-set events, missing values
│   ├── 01_build_universe.py   # SIC 7370/7371/7373/7374 registrants (EDGAR browse by SIC)
│   ├── 02_get_financials.py   # XBRL company facts → quarterly revenue, operating income, SG&A
│   ├── 02b_india_financials.py# NSE XBRL results → quarterly financials, INR→USD
│   ├── 03_ai_intensity.py     # 10-Q/10-K text → AI mentions per 10,000 words (resumable)
│   ├── 04_build_panel.py      # merge, calendarize, derive variables (--extension adds 2026)
│   ├── 05_analysis.py         # RQ1–RQ4 tests and models
│   ├── rq3_models.py          # RQ3 features, time splits and models (shared by 05 and 06)
│   ├── 06_extension_2026.py   # 2026 robustness check: RQ1 by year, frozen RQ3 model
│   └── 07_gsi_focus.py        # the study's questions for the large commercial GSIs
└── tests/test_pipeline.py
```

## How to reproduce

**Option 1 – in the cloud (no installation).** Open the **Actions** tab, choose **Data pipeline**, and
click **Run workflow**. The job runs every script in order and commits the outputs to `data/interim`,
`data/processed` and `results`. It also runs automatically on the 2nd of each month. If a run stops part
way, the AI-scoring step keeps what it finished and the next run continues from there.

**Option 2 – on your own computer.**

```bash
git clone https://github.com/ckt-github/gsi-genai-capstone.git
cd gsi-genai-capstone
pip install -r requirements.txt
# SEC fair-access rule: identify yourself with a name and contact e-mail
export SEC_USER_AGENT="Your Name your.email@example.com"      # PowerShell: $env:SEC_USER_AGENT="..."
python src/00_sample_size.py
python src/01_build_universe.py
python src/02_get_financials.py
python src/02b_india_financials.py          # add --refresh to try the live NSE feeds
python src/03_ai_intensity.py
python src/04_build_panel.py
python src/05_analysis.py
python src/00b_design_check.py              # needs the panel from step 04
python src/04_build_panel.py --extension    # optional: 2026 extension
python src/06_extension_2026.py
python src/07_gsi_focus.py                  # GSI focus
```

The analysis file is `data/processed/panel_firm_quarter.csv`; every variable is defined in
`docs/data_dictionary.csv`. The SEC steps respect the 10 requests/second limit and take about one to
two hours in total.

## What each script writes

| Script | Output | What to check |
|--------|--------|---------------|
| 00_sample_size.py | `results/sample_size.txt` | Base N per RQ (788, 395, 473, 964) |
| 01_build_universe.py | `data/interim/universe.csv` | One row per firm with SIC and segment |
| 02_get_financials.py | `data/interim/financials_quarterly.csv` | Quarterly revenue, operating income, SG&A |
| 02b_india_financials.py | `data/interim/india_financials_quarterly.csv` | 15 firms, revenue in INR and USD |
| 03_ai_intensity.py | `data/interim/ai_scores.csv` | One row per 10-Q/10-K with AI score and form |
| 04_build_panel.py | `data/processed/panel_firm_quarter.csv`, `results/panel_summary.txt` | Firm-quarters by market and period |
| 05_analysis.py | `results/analysis_output.txt` | RQ1–RQ4 results |
| 00b_design_check.py | `results/design_check.txt` | Required vs available N, ICC, test-set events |
| 06_extension_2026.py | `results/extension_2026_output.txt`, `results/extension_eligibility.csv` | 2026 robustness check |
| 07_gsi_focus.py | `results/gsi_focus_output.txt` | GSI-only view |

**Data versions.** SEC data are downloaded fresh on each run (the run date is in the commit message of
the output commit). The NSE and FRED files are fixed snapshots captured on 3 October 2026
(`data/raw/india/README.md`). Nothing is corrected by hand: every rule (tag priority, derived Q4,
restatements, growth edge cases, winsorizing) is in the scripts or `config/settings.yaml`, and no file
outside the repository is needed.

## 2026 extension (robustness check)

The main analysis is fixed at 2019–2025, as set out in the synopsis. As 2026 results come in, a
separate run checks whether the findings hold on newer data, without changing the main results:

- **Which quarters count.** A 2026 quarter is used only when it ended at least 60 days ago and at least
  90% of the firms in the last main quarter (2025Q4) have reported it, checked separately for the U.S.
  and India. Quarters are added in order, so there are no gaps. The check is written to
  `results/extension_eligibility.csv` on every run (rule in `config/settings.yaml`).
- **Same firms.** Only firms already in the main study are used, and winsorizing limits come from the
  main window, so the 2019–2025 rows are identical in both panels.
- **RQ1:** growth by year (mean, median and 5% trimmed mean) and 2026 compared with the pre-GenAI years
  and with the same quarters of 2025.
- **RQ3:** the models are trained on 2019–2023 and tuned on 2024 exactly as in the main study, then
  scored on 2026 quarters they have never seen. Nothing is refitted on 2026.

Results: `results/extension_2026_output.txt`. Package versions are pinned in `requirements.txt` so
reruns reproduce the same numbers.

## GSI focus

The SIC-based sample answers the questions for U.S. IT services as a whole, but SIC codes are coarse:
they also include internet platforms and software-product firms. The study's interest is the large
commercial GSIs, whose revenue depends most on billable people and who cannot change business model as
quickly as smaller firms. `src/07_gsi_focus.py` therefore repeats the analysis for the GSIs inside the
existing sample, without changing the firm scope or the main results.

- **Rule.** Annualised revenue of at least $1 billion (mean quarterly revenue in 2021–2022 × 4, measured
  before the GenAI period so selection cannot depend on the outcome) and a business built on IT consulting,
  systems integration or managed IT services for commercial clients. Every firm above $1 billion is listed
  in `config/gsi_classification.csv` as GSI or not, with the reason; the script warns about any firm above
  the threshold that has not been classified.
- **GSIs.** United States: Accenture, Cognizant, Kyndryl, DXC Technology, EPAM, Rackspace, Unisys and
  Thoughtworks. India: TCS, Infosys, HCLTech, Wipro, Tech Mahindra, LTIMindtree and Mphasis.
  Government integrators (Leidos, SAIC, CACI, Parsons) are not counted.
- **Reported.** Growth by period and year (mean, median, trimmed mean), RQ1 at row and firm level,
  whether GSIs slowed more than the other firms in their market, a firm-by-firm table, AI disclosure of
  U.S. GSIs, and how the main RQ3 model scores on GSI rows.
- **Caution.** Eight U.S. and seven Indian GSIs are a small number of firms, so firm-level tests are
  reported alongside row-level ones and RQ2-style regressions are not run for this group.

Results: `results/gsi_focus_output.txt`.

## Data sources

- SEC EDGAR APIs (submissions, XBRL company facts): https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- EDGAR company search by SIC code and filing archives: https://www.sec.gov/cgi-bin/browse-edgar
- SEC fair-access rules: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- NSE corporate financial results and integrated filings (XBRL): https://www.nseindia.com/companies-listing/corporate-filings-financial-results
- FRED, Indian Rupees to U.S. Dollar Spot Exchange Rate (DEXINUS): https://fred.stlouisfed.org/series/DEXINUS

## Ethics

Firm-level public disclosures only; no personal data. The author works with several GSIs professionally.
To avoid conflicts of interest, no proprietary or partner-confidential information is used, and every firm
is processed by the same rules.
