# India snapshot data

These two files are committed so that the India part of the study can always be reproduced, even
when the National Stock Exchange of India (NSE) blocks automated requests from cloud servers.

| File | Content | Source | Captured |
|------|---------|--------|----------|
| `nse_quarterly_xbrl_snapshot.csv` | One row per quarterly results filing (consolidated, Ind-AS) for 15 NSE-listed IT services firms, 2018–2026. Amounts in Indian rupees (INR). | NSE corporate filings, XBRL files at `nsearchives.nseindia.com/corporate/xbrl/` (legacy results feed to Dec 2024; integrated-filing feed from 2025) | 3 Oct 2026 |
| `fred_dexinus_quarterly_snapshot.csv` | Quarterly average INR per USD | Federal Reserve Bank of St. Louis, FRED series DEXINUS | 3 Oct 2026 |

Column notes for the NSE file:

- `src` = `legacy` (financial-results feed) or `integrated` (integrated-filing feed, 2025 onward).
- `start`, `end` = reporting period in the XBRL file. A few integrated filings carry a six-month start date
  but quarterly values; `src/02b_india_financials.py` therefore assigns the quarter from `end`.
- One HCLTech filing (quarter ending Mar 2019) contains no financial facts and is left blank.
- Element names follow the Ind-AS XBRL taxonomy (for example `RevenueFromOperations`,
  `ProfitBeforeTax`, `FinanceCosts`, `OtherIncome`).

`src/02b_india_financials.py --refresh` tries to rebuild the NSE file from the live NSE feeds and falls
back to this snapshot if the site cannot be reached.
