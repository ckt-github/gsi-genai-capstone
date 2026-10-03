"""Extract quarterly revenue, operating income and SG&A from XBRL company facts.

Discrete quarters (about 90 days) are taken directly; a missing fourth quarter is derived as
fiscal-year total minus the three reported quarters.
Output: data/interim/financials_quarterly.csv
"""
import pandas as pd
from tqdm import tqdm

from common import INTERIM, cik10, sec_get

CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "SalesRevenueServicesNet"],
    "operating_income": ["OperatingIncomeLoss"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
}


def facts_frame(facts, tags):
    """Return all USD duration facts for the first tag list entry that exists, deduplicated."""
    frames = []
    for tag in tags:
        node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
        if node and "USD" in node["units"]:
            df = pd.DataFrame(node["units"]["USD"])
            df = df[df["form"].str.startswith(("10-Q", "10-K"))].dropna(subset=["start"])
            df["tag"] = tag
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df["start"], df["end"] = pd.to_datetime(df["start"]), pd.to_datetime(df["end"])
    df["days"] = (df["end"] - df["start"]).dt.days
    # latest filing wins for restated periods; earlier tag in list wins on ties
    df["tag_rank"] = df["tag"].map({t: i for i, t in enumerate(tags)})
    return df.sort_values(["filed", "tag_rank"], ascending=[False, True]).drop_duplicates(["start", "end"])


def quarterly(df):
    q = df[df["days"].between(80, 100)][["end", "val"]]
    fy = df[df["days"].between(350, 380)][["start", "end", "val"]]
    derived = []
    for _, y in fy.iterrows():
        if (q["end"] == y["end"]).any():
            continue
        inside = q[(q["end"] > y["start"]) & (q["end"] < y["end"])]
        if len(inside) == 3:
            derived.append({"end": y["end"], "val": y["val"] - inside["val"].sum()})
    return pd.concat([q, pd.DataFrame(derived)]).drop_duplicates("end").set_index("end")["val"]


universe = pd.read_csv(INTERIM / "universe.csv")
out = []
for cik, ticker in tqdm(universe[["cik", "ticker"]].itertuples(index=False), total=len(universe)):
    # company-facts files are large, so they are not cached on disk
    facts = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json")
    if not facts:
        continue
    series = {}
    for name, tags in CONCEPTS.items():
        df = facts_frame(facts, tags)
        if not df.empty:
            series[name] = quarterly(df)
    if "revenue" not in series:
        continue
    panel = pd.DataFrame(series)
    panel["cik"], panel["ticker"] = cik, ticker
    out.append(panel.reset_index().rename(columns={"index": "period_end", "end": "period_end"}))

fin = pd.concat(out, ignore_index=True)
fin.to_csv(INTERIM / "financials_quarterly.csv", index=False)
print(f"{fin['cik'].nunique()} firms, {len(fin):,} firm-quarters written")
