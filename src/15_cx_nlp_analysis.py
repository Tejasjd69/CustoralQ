"""
15_cx_nlp_analysis.py

Customer Voice / NLP Analytics
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
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


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# SENTIMENT LEXICON
# ============================================================

POSITIVE_WORDS = {
    "helpful",
    "resolved",
    "resolution",
    "satisfied",
    "excellent",
    "good",
    "great",
    "thank",
    "thanks",
    "appreciate",
    "successful",
    "easy",
    "quick",
    "prompt",
}

NEGATIVE_WORDS = {
    "fraud",
    "scam",
    "wrong",
    "error",
    "problem",
    "issue",
    "complaint",
    "angry",
    "terrible",
    "horrible",
    "bad",
    "unfair",
    "failed",
    "failure",
    "charged",
    "charge",
    "refund",
    "delay",
    "delayed",
    "denied",
    "refused",
    "cancel",
    "cancellation",
    "dispute",
    "stolen",
    "unauthorized",
    "misleading",
    "incorrect",
    "difficult",
    "unable",
}


def calculate_sentiment(text):

    words = text.split()

    if not words:
        return "Neutral", 0

    positive = sum(
        word in POSITIVE_WORDS
        for word in words
    )

    negative = sum(
        word in NEGATIVE_WORDS
        for word in words
    )

    score = positive - negative

    if score > 0:
        sentiment = "Positive"
    elif score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, score


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_statistics(df):

    print("\n" + "=" * 70)
    print("1. TEXT QUALITY ANALYSIS")
    print("=" * 70)

    df["text_length"] = (
        df["complaint_what_happened"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["word_count"] = (
        df["clean_text"]
        .str.split()
        .str.len()
    )

    print(
        f"\nComplaints: {len(df):,}"
    )

    print(
        f"Average words: "
        f"{df['word_count'].mean():.1f}"
    )

    print(
        f"Median words: "
        f"{df['word_count'].median():.1f}"
    )

    return df


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def sentiment_analysis(df):

    print("\n" + "=" * 70)
    print("2. SENTIMENT ANALYSIS")
    print("=" * 70)

    sentiment_results = (
        df["clean_text"]
        .apply(calculate_sentiment)
    )

    df["sentiment"] = sentiment_results.apply(
        lambda result: result[0]
    )

    df["sentiment_score"] = sentiment_results.apply(
        lambda result: result[1]
    )

    sentiment_counts = (
        df["sentiment"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(
            name="complaints"
        )
    )

    sentiment_counts["percentage"] = (
        sentiment_counts["complaints"]
        / len(df)
        * 100
    )

    print("\nOverall sentiment:")

    print(
        sentiment_counts.to_string(
            index=False
        )
    )

    sentiment_counts.to_csv(
        OUTPUT_DIR
        / "sentiment_distribution.csv",
        index=False
    )

    return df


# ============================================================
# TF-IDF
# ============================================================

def tfidf_analysis(df):

    print("\n" + "=" * 70)
    print("3. TF-IDF CUSTOMER THEMES")
    print("=" * 70)

    texts = (
        df["clean_text"]
        .fillna("")
        .tolist()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.90,
        max_features=5000,
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    scores = np.asarray(
        matrix.mean(axis=0)
    ).ravel()

    terms = np.array(
        vectorizer.get_feature_names_out()
    )

    top_indices = scores.argsort()[::-1][:100]

    result = pd.DataFrame({
        "term": terms[top_indices],
        "tfidf_score": scores[top_indices],
    })

    print("\nTop terms:")

    print(
        result.head(30).to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "top_tfidf_terms.csv",
        index=False
    )

    return result


# ============================================================
# NEGATIVE COMPLAINT THEMES
# ============================================================

def negative_tfidf_analysis(df):

    print("\n" + "=" * 70)
    print("4. NEGATIVE CUSTOMER THEMES")
    print("=" * 70)

    negative = df[
        df["sentiment"] == "Negative"
    ].copy()

    print(
        f"\nNegative complaints: "
        f"{len(negative):,}"
    )

    if len(negative) < 10:
        print(
            "Not enough negative complaints "
            "for TF-IDF analysis."
        )
        return pd.DataFrame()

    texts = (
        negative["clean_text"]
        .fillna("")
        .tolist()
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=5000,
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    scores = np.asarray(
        matrix.mean(axis=0)
    ).ravel()

    terms = np.array(
        vectorizer.get_feature_names_out()
    )

    top_indices = scores.argsort()[::-1][:100]

    result = pd.DataFrame({
        "term": terms[top_indices],
        "tfidf_score": scores[top_indices],
    })

    print("\nTop negative themes:")

    print(
        result.head(30).to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "negative_complaint_themes.csv",
        index=False
    )

    return result


# ============================================================
# PRODUCT ANALYSIS
# ============================================================

def sentiment_by_product(df):

    print("\n" + "=" * 70)
    print("5. SENTIMENT BY PRODUCT")
    print("=" * 70)

    result = (
        df.groupby(
            "product",
            dropna=False
        )
        .agg(
            complaints=(
                "complaint_id",
                "count"
            ),
            negative_complaints=(
                "sentiment",
                lambda x:
                (x == "Negative").sum()
            ),
            positive_complaints=(
                "sentiment",
                lambda x:
                (x == "Positive").sum()
            ),
        )
        .reset_index()
    )

    result["negative_rate"] = (
        result["negative_complaints"]
        / result["complaints"]
        * 100
    )

    result = result.sort_values(
        "complaints",
        ascending=False
    )

    print(
        result.head(20).to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "sentiment_by_product.csv",
        index=False
    )

    return result


# ============================================================
# CHANNEL ANALYSIS
# ============================================================

def sentiment_by_channel(df):

    print("\n" + "=" * 70)
    print("6. SENTIMENT BY SERVICING CHANNEL")
    print("=" * 70)

    result = (
        df.groupby(
            "submitted_via",
            dropna=False
        )
        .agg(
            complaints=(
                "complaint_id",
                "count"
            ),
            negative_complaints=(
                "sentiment",
                lambda x:
                (x == "Negative").sum()
            ),
        )
        .reset_index()
    )

    result["negative_rate"] = (
        result["negative_complaints"]
        / result["complaints"]
        * 100
    )

    result = result.sort_values(
        "complaints",
        ascending=False
    )

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "sentiment_by_channel.csv",
        index=False
    )

    return result


# ============================================================
# COMPLAINT CATEGORY ANALYSIS
# ============================================================

def complaint_categories(df):

    print("\n" + "=" * 70)
    print("7. COMPLAINT CATEGORIES")
    print("=" * 70)

    result = (
        df.groupby(
            ["product", "issue"],
            dropna=False
        )
        .agg(
            complaints=(
                "complaint_id",
                "count"
            ),
            negative_complaints=(
                "sentiment",
                lambda x:
                (x == "Negative").sum()
            ),
        )
        .reset_index()
    )

    result["negative_rate"] = (
        result["negative_complaints"]
        / result["complaints"]
        * 100
    )

    result = result.sort_values(
        "complaints",
        ascending=False
    )

    print(
        result.head(30).to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "complaint_categories.csv",
        index=False
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOMER VOICE NLP ANALYSIS")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded {len(df):,} complaints."
    )

    df["complaint_what_happened"] = (
        df["complaint_what_happened"]
        .fillna("")
        .astype(str)
    )

    df["clean_text"] = (
        df["complaint_what_happened"]
        .apply(clean_text)
    )

    df = df[
        df["clean_text"].str.len() > 0
    ].copy()

    df = text_statistics(df)

    df = sentiment_analysis(df)

    tfidf_analysis(df)

    negative_tfidf_analysis(df)

    sentiment_by_product(df)

    sentiment_by_channel(df)

    complaint_categories(df)

    output_dataset = (
        OUTPUT_DIR
        / "cfpb_complaints_nlp.csv"
    )

    df.to_csv(
        output_dataset,
        index=False
    )

    print("\n" + "=" * 70)
    print("NLP ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nEnriched dataset:\n"
        f"{output_dataset}"
    )

    print("\nReports generated:")

    for file in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            f"  - {file.name}"
        )


if __name__ == "__main__":
    main()