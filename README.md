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
| RQ1 | Did mean YoY revenue growth differ between the pre-GenAI (2019Q1–2022Q3) and post-GenAI (2023Q1–2025Q4) periods? | Welch two-sample t-test; POST × MARKET regression as robustness check | U.S. (primary) and India (comparison) |
| RQ2 | In the post-GenAI period, is AI-disclosure intensity associated with YoY revenue growth, after controls? | Multiple linear regression (firm-clustered SEs) | U.S. |
| RQ3 | Can financial and AI-disclosure features predict next-quarter revenue decline better than chance? | Logistic regression, random forest, XGBoost (ROC-AUC) | U.S. |
| RQ4 | Is post-GenAI revenue-decline status associated with service segment? | Chi-square test of independence (Cramér's V) | U.S. |

Minimum sample size (maximum across RQ1–RQ4): **N = 1,068 firm-quarters** (see `src/00_sample_size.py`).

## Indian comparison firms

TCS, Infosys, HCLTech, Wipro, Tech Mahindra, LTIMindtree, Mphasis, Coforge, Persistent Systems,
Birlasoft, Zensar, Mastek, Sonata Software, Cyient and L&T Technology Services (`config/india_firms.csv`).
Their consolidated quarterly results come from the XBRL files each company files with NSE. Growth is
measured in rupees (so currency moves do not distort it); size is converted to USD with the FRED DEXINUS
quarterly average. Indian firms do not file 10-Q/10-K reports, so they are not part of the AI-disclosure
questions (RQ2, RQ3).

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
│   └── india_firms.csv        # NSE comparison firms and segment
├── data/
│   ├── raw/india/             # committed NSE and FX snapshots (see README there)
│   ├── interim/               # universe, financials, AI scores (written by the pipeline)
│   └── processed/
│       └── panel_firm_quarter.csv
├── docs/
│   ├── data_dictionary.csv
│   └── synopsis.pdf
├── results/                   # analysis output written by the pipeline
├── src/
│   ├── common.py              # paths, settings, rate-limited SEC client
│   ├── 00_sample_size.py      # RQ1–RQ4 sample-size calculations
│   ├── 01_build_universe.py   # SIC 7370/7371/7373/7374 registrants (EDGAR browse by SIC)
│   ├── 02_get_financials.py   # XBRL company facts → quarterly revenue, operating income, SG&A
│   ├── 02b_india_financials.py# NSE XBRL results → quarterly financials, INR→USD
│   ├── 03_ai_intensity.py     # 10-Q/10-K text → AI mentions per 10,000 words (resumable)
│   ├── 04_build_panel.py      # merge, calendarize, derive variables
│   └── 05_analysis.py         # RQ1–RQ4 tests and models
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
```

The analysis file is `data/processed/panel_firm_quarter.csv`; every variable is defined in
`docs/data_dictionary.csv`. The SEC steps respect the 10 requests/second limit and take about one to
two hours in total.

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
