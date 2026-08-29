from pathlib import Path
import zipfile
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_FILE = PROJECT_ROOT / "data" / "cx" / "complaints.csv.zip"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cx"
    / "cfpb_channel_sample.csv"
)

CHUNK_SIZE = 20_000
TARGET_PER_CHANNEL = 5_000


def main():

    print("=" * 70)
    print("CFPB CHANNEL-STRATIFIED SAMPLE")
    print("=" * 70)

    samples = {}
    channel_counts = {}

    with zipfile.ZipFile(ZIP_FILE, "r") as z:

        csv_name = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ][0]

        print(f"\nReading: {csv_name}")
        print("\nScanning complete CFPB file...\n")

        with z.open(csv_name) as csv_file:

            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    csv_file,
                    chunksize=CHUNK_SIZE,
                    low_memory=False
                ),
                start=1
            ):

                # ------------------------------------------------
                # Keep only complaints with narratives
                # ------------------------------------------------

                narrative = (
                    chunk["Consumer complaint narrative"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )

                chunk = chunk[narrative.ne("")].copy()

                if chunk.empty:
                    continue

                # ------------------------------------------------
                # Count every channel
                # ------------------------------------------------

                counts = (
                    chunk["Submitted via"]
                    .fillna("Unknown")
                    .value_counts()
                )

                for channel, count in counts.items():

                    channel_counts[channel] = (
                        channel_counts.get(channel, 0)
                        + int(count)
                    )

                # ------------------------------------------------
                # Collect up to target for each channel
                # ------------------------------------------------

                for channel in chunk[
                    "Submitted via"
                ].dropna().unique():

                    existing = sum(
                        len(part)
                        for part in samples.get(
                            channel,
                            []
                        )
                    )

                    if existing >= TARGET_PER_CHANNEL:
                        continue

                    channel_rows = chunk[
                        chunk["Submitted via"] == channel
                    ]

                    remaining = (
                        TARGET_PER_CHANNEL
                        - existing
                    )

                    selected = channel_rows.head(
                        remaining
                    )

                    if channel not in samples:
                        samples[channel] = []

                    samples[channel].append(
                        selected
                    )

                if chunk_number % 10 == 0:

                    collected = sum(
                        sum(len(x) for x in parts)
                        for parts in samples.values()
                    )

                    print(
                        f"Chunks scanned: {chunk_number:,} | "
                        f"Records collected: {collected:,}"
                    )

    # ============================================================
    # CHANNEL DISTRIBUTION
    # ============================================================

    print("\n" + "=" * 70)
    print("CHANNEL DISTRIBUTION IN FULL DATASET")
    print("=" * 70)

    for channel, count in sorted(
        channel_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{channel}: {count:,}"
        )

    # ============================================================
    # COMBINE SAMPLE
    # ============================================================

    output_parts = []

    for channel, parts in samples.items():

        if parts:

            output_parts.append(
                pd.concat(
                    parts,
                    ignore_index=True
                )
            )

    if not output_parts:

        raise RuntimeError(
            "No complaint narratives were collected."
        )

    df = pd.concat(
        output_parts,
        ignore_index=True
    )

    # Remove duplicates
    if "Complaint ID" in df.columns:

        df = df.drop_duplicates(
            subset=["Complaint ID"]
        )

    # ============================================================
    # CLEAN COLUMN NAMES
    # ============================================================

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

        "Consumer complaint narrative":
            "complaint_what_happened",

        "Company public response":
            "company_public_response",

        "Company":
            "company",

        "State":
            "state",

        "ZIP code":
            "zip_code",

        "Tags":
            "tags",

        "Submitted via":
            "submitted_via",

        "Date sent to company":
            "date_sent_to_company",

        "Company response to consumer":
            "company_response",

        "Timely response?":
            "timely_response",
    }

    df = df.rename(
        columns=rename_map
    )

    # ============================================================
    # SAVE
    # ============================================================

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("CHANNEL SAMPLE COMPLETE")
    print("=" * 70)

    print(
        f"\nFinal records: {len(df):,}"
    )

    print("\nFinal channel distribution:")

    print(
        df["submitted_via"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()