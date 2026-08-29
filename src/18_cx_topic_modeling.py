from pathlib import Path
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


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

N_TOPICS = 8
TOP_WORDS = 12


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove CFPB anonymization artifacts
    text = re.sub(
        r"\bx+\b",
        " ",
        text
    )

    text = re.sub(
        r"\b[x]{2,}\b",
        " ",
        text
    )

    # Keep alphabetic text
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("CX TOPIC MODELING")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded complaints: {len(df):,}"
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
        df["clean_text"].str.len() >= 20
    ].copy()

    print(
        f"Usable complaints: {len(df):,}"
    )

    return df


# ============================================================
# TF-IDF
# ============================================================

def build_tfidf(df):

    print("\nBuilding TF-IDF matrix...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=15,
        max_df=0.90,
        max_features=10_000,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        df["clean_text"]
    )

    print(
        f"Documents: {matrix.shape[0]:,}"
    )

    print(
        f"Features: {matrix.shape[1]:,}"
    )

    return vectorizer, matrix


# ============================================================
# TRAIN NMF
# ============================================================

def train_topic_model(matrix):

    print(
        f"\nTraining NMF with "
        f"{N_TOPICS} topics..."
    )

    model = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        random_state=42,
        max_iter=400
    )

    topic_matrix = model.fit_transform(
        matrix
    )

    print(
        "Topic model training complete."
    )

    return model, topic_matrix


# ============================================================
# DISPLAY TOPIC WORDS
# ============================================================

def extract_topics(
    model,
    vectorizer
):

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    rows = []

    print("\n" + "=" * 70)
    print("DISCOVERED CUSTOMER EXPERIENCE TOPICS")
    print("=" * 70)

    for topic_number, topic in enumerate(
        model.components_,
        start=1
    ):

        top_indices = topic.argsort()[
            ::-1
        ][:TOP_WORDS]

        words = [
            feature_names[index]
            for index in top_indices
        ]

        print(
            f"\nTOPIC {topic_number}"
        )

        print(
            "  " + ", ".join(words)
        )

        for rank, index in enumerate(
            top_indices,
            start=1
        ):

            rows.append({
                "topic_id": topic_number,
                "rank": rank,
                "term": feature_names[index],
                "weight": topic[index]
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_DIR
        / "topic_terms.csv",
        index=False
    )

    return result


# ============================================================
# ASSIGN DOMINANT TOPIC
# ============================================================

def assign_topics(
    df,
    topic_matrix
):

    topic_numbers = (
        np.argmax(
            topic_matrix,
            axis=1
        ) + 1
    )

    topic_strength = (
        np.max(
            topic_matrix,
            axis=1
        )
    )

    df["dominant_topic"] = (
        topic_numbers
    )

    df["topic_strength"] = (
        topic_strength
    )

    return df


# ============================================================
# TOPIC DISTRIBUTION
# ============================================================

def topic_distribution(df):

    print("\n" + "=" * 70)
    print("TOPIC DISTRIBUTION")
    print("=" * 70)

    result = (
        df["dominant_topic"]
        .value_counts()
        .sort_index()
        .rename_axis("topic_id")
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

    result.to_csv(
        OUTPUT_DIR
        / "topic_distribution.csv",
        index=False
    )

    return result


# ============================================================
# TOPIC × PRODUCT
# ============================================================

def topic_by_product(df):

    print("\n" + "=" * 70)
    print("TOPIC BY PRODUCT")
    print("=" * 70)

    result = (
        df.groupby(
            [
                "product",
                "dominant_topic"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="complaints"
        )
    )

    result["product_total"] = (
        result.groupby("product")[
            "complaints"
        ]
        .transform("sum")
    )

    result["topic_share_pct"] = (
        result["complaints"]
        / result["product_total"]
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
        / "topic_by_product.csv",
        index=False
    )

    return result


# ============================================================
# TOPIC × ISSUE
# ============================================================

def topic_by_issue(df):

    print("\n" + "=" * 70)
    print("TOPIC BY ISSUE")
    print("=" * 70)

    result = (
        df.groupby(
            [
                "issue",
                "dominant_topic"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="complaints"
        )
    )

    result = result.sort_values(
        "complaints",
        ascending=False
    )

    print(
        result.head(40).to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "topic_by_issue.csv",
        index=False
    )

    return result


# ============================================================
# TOPIC × RESPONSE
# ============================================================

def topic_by_response(df):

    print("\n" + "=" * 70)
    print("TOPIC × TIMELY RESPONSE")
    print("=" * 70)

    if "timely_response" not in df.columns:

        print(
            "timely_response column unavailable."
        )

        return pd.DataFrame()

    result = (
        df.groupby(
            [
                "dominant_topic",
                "timely_response"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="complaints"
        )
    )

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "topic_by_timely_response.csv",
        index=False
    )

    return result


# ============================================================
# SAVE ENRICHED DATASET
# ============================================================

def save_dataset(df):

    output_file = (
        OUTPUT_DIR
        / "cfpb_complaints_topics.csv"
    )

    # Don't need to save clean text.
    output_df = df.drop(
        columns=["clean_text"],
        errors="ignore"
    )

    output_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nEnriched dataset saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    vectorizer, matrix = (
        build_tfidf(df)
    )

    model, topic_matrix = (
        train_topic_model(matrix)
    )

    extract_topics(
        model,
        vectorizer
    )

    df = assign_topics(
        df,
        topic_matrix
    )

    topic_distribution(df)

    topic_by_product(df)

    topic_by_issue(df)

    topic_by_response(df)

    save_dataset(df)

    print("\n" + "=" * 70)
    print("TOPIC MODELING COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    for file in sorted(
        OUTPUT_DIR.glob("topic*.csv")
    ):

        print(
            f"  - {file.name}"
        )


if __name__ == "__main__":
    main()