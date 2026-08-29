"""
13_download_cfpb.py

Download a real sample of CFPB Consumer Complaint Database records
for the Customer Experience Intelligence Platform.
"""

from pathlib import Path
import requests
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "cx"
OUTPUT_FILE = OUTPUT_DIR / "cfpb_complaints.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# We intentionally start with a manageable dataset.
TARGET_RECORDS = 20_000

# CFPB public complaint API
API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"


# ============================================================
# DOWNLOAD
# ============================================================

def download_complaints():

    print("=" * 70)
    print("CFPB CONSUMER COMPLAINT DATA DOWNLOAD")
    print("=" * 70)

    print(f"\nTarget records: {TARGET_RECORDS:,}")
    print("Downloading real public CFPB complaint data...\n")

    params = {
        "size": TARGET_RECORDS,
        "from": 0,
        "sort": "created_date_desc",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    # CFPB API returns complaint records inside hits
    records = data.get("hits", [])

    if not records:
        raise RuntimeError(
            "The CFPB API returned no complaint records."
        )

    rows = []

    for item in records:

        # API responses may wrap the actual record in _source.
        source = item.get("_source", item)

        rows.append(source)

    df = pd.DataFrame(rows)

    print(f"Downloaded records: {len(df):,}")
    print(f"Columns received:  {len(df.columns)}")

    # --------------------------------------------------------
    # Keep only useful CX/NLP fields when available
    # --------------------------------------------------------

    preferred_columns = [
        "complaint_id",
        "date_received",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "state",
        "submitted_via",
        "date_sent_to_company",
        "company_response",
        "timely_response",
        "complaint_what_happened",
        "company_public_response",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    if "complaint_what_happened" in df.columns:

        df = df[
            df["complaint_what_happened"]
            .fillna("")
            .str.strip()
            .ne("")
        ].copy()

        print(
            f"Records with complaint narratives: "
            f"{len(df):,}"
        )

    if not available_columns:
        raise RuntimeError(
            "Expected CFPB fields were not found. "
            "Inspect the API response before continuing."
        )

    df = df[available_columns].copy()

    # Remove duplicate complaint IDs if present.
    if "complaint_id" in df.columns:

        df = df.drop_duplicates(
            subset=["complaint_id"]
        )

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    print(f"\nFinal records: {len(df):,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    download_complaints()