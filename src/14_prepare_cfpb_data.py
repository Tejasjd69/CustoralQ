from pathlib import Path
import zipfile
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "cx"
    / "complaints.csv.zip"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cx"
    / "cfpb_complaints.csv"
)

TARGET_RECORDS = 30_000
CHUNK_SIZE = 10_000


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_cfpb_data():

    print("=" * 70)
    print("CFPB CUSTOMER EXPERIENCE DATA PREPARATION")
    print("=" * 70)

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"ZIP file not found:\n{ZIP_FILE}"
        )

    print(f"\nSource ZIP:")
    print(ZIP_FILE)

    print(f"\nTarget usable complaints: {TARGET_RECORDS:,}")
    print(f"Processing in chunks of: {CHUNK_SIZE:,}")

    collected = []
    total_scanned = 0
    total_with_narrative = 0

    with zipfile.ZipFile(ZIP_FILE, "r") as z:

        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            raise RuntimeError(
                "No CSV file found inside the ZIP."
            )

        csv_name = csv_files[0]

        print(f"\nReading: {csv_name}")

        with z.open(csv_name) as csv_file:

            for chunk in pd.read_csv(
                csv_file,
                chunksize=CHUNK_SIZE,
                low_memory=False
            ):

                total_scanned += len(chunk)

                # ------------------------------------------------
                # Find complaint narrative column
                # ------------------------------------------------

                narrative_candidates = [
                 "Consumer complaint narrative",
                 "complaint_what_happened",
                 "Complaint what happened",
                ]
                narrative_column = None

                for column in narrative_candidates:

                    if column in chunk.columns:
                        narrative_column = column
                        break

                if narrative_column is None:
                    raise RuntimeError(
                        "Could not find the complaint narrative "
                        "column.\n\nAvailable columns:\n"
                        + "\n".join(chunk.columns)
                    )

                # ------------------------------------------------
                # Keep records with actual narratives
                # ------------------------------------------------

                chunk[narrative_column] = (
                    chunk[narrative_column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                usable = chunk[
                    chunk[narrative_column].ne("")
                ].copy()

                total_with_narrative += len(usable)

                if len(usable) > 0:
                    collected.append(usable)

                print(
                    f"Scanned: {total_scanned:,} | "
                    f"With narrative: "
                    f"{total_with_narrative:,}"
                )

                # ------------------------------------------------
                # Stop once we have enough
                # ------------------------------------------------

                current_total = sum(
                    len(x)
                    for x in collected
                )

                if current_total >= TARGET_RECORDS:
                    break

    if not collected:
        raise RuntimeError(
            "No complaint narratives were found."
        )

    df = pd.concat(
        collected,
        ignore_index=True
    )

    # Limit to target
    df = df.head(TARGET_RECORDS).copy()

    # Remove duplicate complaint IDs if available
    id_candidates = [
        "complaint_id",
        "Complaint ID",
    ]

    for id_column in id_candidates:

        if id_column in df.columns:

            df = df.drop_duplicates(
                subset=[id_column]
            )

            break

    # ------------------------------------------------------------
    # Rename important columns into clean snake_case names
    # ------------------------------------------------------------

    rename_map = {

        "Complaint ID":
            "complaint_id",

        "Date received":
            "date_received",

        "Product":
            "product",

        "Sub-product":
            "sub_product",

        "Issue":
            "issue",

        "Sub-issue":
            "sub_issue",

        "Company":
            "company",

        "State":
            "state",

        "Submitted via":
            "submitted_via",

        "Date sent to company":
            "date_sent_to_company",

        "Company response to consumer":
            "company_response",

        "Timely response?":
            "timely_response",

        "Consumer complaint narrative":
            "complaint_what_happened",

        "Company public response":
            "company_public_response",

    }

    df = df.rename(
        columns=rename_map
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal records: {len(df):,}"
    )

    print(
        f"Final columns: {len(df.columns)}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prepare_cfpb_data()