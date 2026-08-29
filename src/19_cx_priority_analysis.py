from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "cx_nlp"
    / "cfpb_complaints_topics.csv"
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
# BUSINESS THEME NAMES
# ============================================================

TOPIC_NAMES = {
    1: "Debt verification / FDCPA",
    2: "Transactions / funds disputes",
    3: "False credit reporting",
    4: "Credit-report accuracy / rights",
    5: "Debt collection / validation",
    6: "Debt legitimacy / authorization",
    7: "Banking / account servicing",
    8: "Credit reporting / identity",
}


# ============================================================
# SENTIMENT WORDS
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


# ============================================================
# SENTIMENT FUNCTION
# ============================================================

def get_sentiment(text):

    if pd.isna(text):
        return "Neutral"

    words = str(text).lower().split()

    positive = sum(
        word in POSITIVE_WORDS
        for word in words
    )

    negative = sum(
        word in NEGATIVE_WORDS
        for word in words
    )

    if negative > positive:
        return "Negative"

    if positive > negative:
        return "Positive"

    return "Neutral"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("CX PRIORITY ANALYSIS")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded complaints: {len(df):,}"
    )

    required_columns = [
        "dominant_topic",
        "complaint_what_happened",
        "timely_response",
        "product",
        "issue",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df["theme"] = (
        pd.to_numeric(
            df["dominant_topic"],
            errors="coerce"
        )
        .map(TOPIC_NAMES)
        .fillna("Other")
    )

    df["sentiment"] = (
        df["complaint_what_happened"]
        .apply(get_sentiment)
    )

    df["timely_yes"] = (
        df["timely_response"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    )

    return df


# ============================================================
# THEME METRICS
# ============================================================

def calculate_theme_metrics(df):

    print(
        "\nCalculating theme-level metrics..."
    )

    result = (
        df.groupby(
            "theme",
            dropna=False
        )
        .agg(
            complaints=(
                "theme",
                "size"
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
            timely_responses=(
                "timely_yes",
                "sum"
            ),
        )
        .reset_index()
    )

    result["negative_rate"] = (
        result["negative_complaints"]
        / result["complaints"]
        * 100
    )

    result["positive_rate"] = (
        result["positive_complaints"]
        / result["complaints"]
        * 100
    )

    result["timely_response_rate"] = (
        result["timely_responses"]
        / result["complaints"]
        * 100
    )

    result["volume_share"] = (
        result["complaints"]
        / len(df)
        * 100
    )

    return result


# ============================================================
# MIN-MAX NORMALIZATION
# ============================================================

def min_max(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:

        return pd.Series(
            0.5,
            index=series.index
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


# ============================================================
# PRIORITY SCORE
# ============================================================

def calculate_priority_score(result):

    print(
        "\nCalculating CX priority scores..."
    )

    result["volume_score"] = min_max(
        result["complaints"]
    )

    result["negative_score"] = min_max(
        result["negative_rate"]
    )

    response_gap = (
        100
        - result["timely_response_rate"]
    )

    result["response_gap_score"] = min_max(
        response_gap
    )

    # Transparent scoring framework:
    #
    # 40% complaint volume
    # 40% negative sentiment
    # 20% response gap

    result["cx_priority_score"] = (
        (
            0.40 * result["volume_score"]
            + 0.40 * result["negative_score"]
            + 0.20 * result["response_gap_score"]
        )
        * 100
    )

    result["priority_rank"] = (
        result["cx_priority_score"]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    result["priority_tier"] = pd.cut(
        result["cx_priority_score"],
        bins=[
            -np.inf,
            33,
            66,
            np.inf,
        ],
        labels=[
            "Lower Priority",
            "Medium Priority",
            "High Priority",
        ],
    )

    result = result.sort_values(
        "cx_priority_score",
        ascending=False
    )

    return result


# ============================================================
# PRODUCT × THEME
# ============================================================

def product_theme_analysis(df):

    print(
        "\nCalculating product × theme analysis..."
    )

    result = (
        df.groupby(
            [
                "product",
                "theme",
            ],
            dropna=False
        )
        .agg(
            complaints=(
                "theme",
                "size"
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

    output_file = (
        OUTPUT_DIR
        / "product_theme_analysis.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nTop product/theme combinations:"
    )

    print(
        result.head(30).to_string(
            index=False
        )
    )

    return result


# ============================================================
# ISSUE × THEME
# ============================================================

def issue_theme_analysis(df):

    print(
        "\nCalculating issue × theme analysis..."
    )

    result = (
        df.groupby(
            [
                "issue",
                "theme",
            ],
            dropna=False
        )
        .agg(
            complaints=(
                "theme",
                "size"
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

    output_file = (
        OUTPUT_DIR
        / "issue_theme_analysis.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nTop issue/theme combinations:"
    )

    print(
        result.head(40).to_string(
            index=False
        )
    )

    return result


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

def executive_summary(result):

    print("\n" + "=" * 70)
    print("EXECUTIVE CX PRIORITIES")
    print("=" * 70)

    for _, row in result.iterrows():

        print(
            f"\n#{int(row['priority_rank'])} "
            f"{row['theme']}"
        )

        print(
            f"  Complaints: "
            f"{int(row['complaints']):,}"
        )

        print(
            f"  Volume share: "
            f"{row['volume_share']:.1f}%"
        )

        print(
            f"  Negative rate: "
            f"{row['negative_rate']:.1f}%"
        )

        print(
            f"  Timely response: "
            f"{row['timely_response_rate']:.1f}%"
        )

        print(
            f"  CX priority score: "
            f"{row['cx_priority_score']:.1f}"
        )

        print(
            f"  Priority tier: "
            f"{row['priority_tier']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    df = prepare_data(df)

    result = calculate_theme_metrics(
        df
    )

    result = calculate_priority_score(
        result
    )

    priority_file = (
        OUTPUT_DIR
        / "cx_priority_scores.csv"
    )

    result.to_csv(
        priority_file,
        index=False
    )

    product_theme_analysis(
        df
    )

    issue_theme_analysis(
        df
    )

    executive_summary(
        result
    )

    print("\n" + "=" * 70)
    print("CX PRIORITY ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nCreated:\n{priority_file}"
    )

    print(
        "\nOther generated files:"
    )

    print(
        "  - product_theme_analysis.csv"
    )

    print(
        "  - issue_theme_analysis.csv"
    )


if __name__ == "__main__":
    main()