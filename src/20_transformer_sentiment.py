from pathlib import Path
import pandas as pd
import torch
from transformers import pipeline


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cx"
    / "cfpb_complaints.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "cx_nlp"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "transformer_sentiment_test_2000.csv"
)

MODEL_NAME = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

TEST_ROWS = 2000

BATCH_SIZE = 8

MAX_CHARS = 800


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("OPTIMIZED TRANSFORMER SENTIMENT TEST")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    df["complaint_what_happened"] = (
        df["complaint_what_happened"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["complaint_what_happened"].str.len() > 0
    ].copy()

    # Deterministic sample
    df = df.head(TEST_ROWS).copy()

    # Limit text length for CPU performance
    df["analysis_text"] = (
        df["complaint_what_happened"]
        .str.slice(0, MAX_CHARS)
    )

    print(
        f"\nComplaints selected: {len(df):,}"
    )

    print(
        f"Maximum text length: {MAX_CHARS} characters"
    )

    return df


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading model...")

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Device: CPU"
    )

    sentiment_model = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1,
        truncation=True,
        max_length=256,
    )

    print(
        "Model loaded."
    )

    return sentiment_model


# ============================================================
# RUN SENTIMENT
# ============================================================

def analyze(df, sentiment_model):

    print("\n" + "=" * 70)
    print("RUNNING SENTIMENT TEST")
    print("=" * 70)

    texts = (
        df["analysis_text"]
        .tolist()
    )

    labels = []
    scores = []

    total = len(texts)

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total
        )

        batch = texts[start:end]

        results = sentiment_model(
            batch,
            batch_size=BATCH_SIZE
        )

        for result in results:

            labels.append(
                result["label"]
            )

            scores.append(
                result["score"]
            )

        processed = end

        if (
            processed % 100 == 0
            or processed == total
        ):

            print(
                f"Processed "
                f"{processed:,}/{total:,} "
                f"({processed / total * 100:.1f}%)"
            )

    df["transformer_sentiment"] = labels

    df["transformer_confidence"] = scores

    df["sentiment"] = (
        df["transformer_sentiment"]
        .map({
            "POSITIVE": "Positive",
            "NEGATIVE": "Negative",
        })
        .fillna("Neutral")
    )

    return df


# ============================================================
# SUMMARY
# ============================================================

def summary(df):

    print("\n" + "=" * 70)
    print("SENTIMENT RESULTS")
    print("=" * 70)

    result = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(
            name="complaints"
        )
    )

    result["percentage"] = (
        result["complaints"]
        / len(df)
        * 100
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\nAverage confidence:",
        f"{df['transformer_confidence'].mean():.4f}"
    )

    return result


# ============================================================
# SAVE
# ============================================================

def save(df):

    columns = [
        "complaint_id",
        "product",
        "issue",
        "submitted_via",
        "complaint_what_happened",
        "transformer_sentiment",
        "transformer_confidence",
        "sentiment",
    ]

    available = [
        c for c in columns
        if c in df.columns
    ]

    df[available].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nSaved test results to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    model = load_model()

    df = analyze(
        df,
        model
    )

    summary(df)

    save(df)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()