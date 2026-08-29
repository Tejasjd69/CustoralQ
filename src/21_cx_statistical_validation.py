from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


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
    / "cx_statistics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("CX STATISTICAL VALIDATION")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded records: {len(df):,}"
    )

    required = [
        "dominant_topic",
        "timely_response",
        "product",
        "issue",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    df["topic_id"] = pd.to_numeric(
        df["dominant_topic"],
        errors="coerce"
    )

    df["theme"] = (
        df["topic_id"]
        .map(TOPIC_NAMES)
        .fillna("Other")
    )

    df["timely_yes"] = (
        df["timely_response"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
    )

    df["not_timely"] = (
        ~df["timely_yes"]
    )

    return df


# ============================================================
# CRAMER'S V
# ============================================================

def cramers_v(table):

    chi2, _, _, _ = chi2_contingency(
        table,
        correction=False
    )

    n = table.to_numpy().sum()

    rows, cols = table.shape

    phi2 = chi2 / n

    correction = (
        (phi2 - ((cols - 1) * (rows - 1)) / (n - 1))
    )

    correction = max(
        0,
        correction
    )

    denominator = min(
        cols - 1,
        rows - 1
    )

    if denominator == 0:
        return 0.0

    return np.sqrt(
        correction / denominator
    )


# ============================================================
# CHI-SQUARE: TOPIC × TIMELY RESPONSE
# ============================================================

def topic_timeliness_test(df):

    print("\n" + "=" * 70)
    print("ANALYSIS 1: CX THEME VS RESPONSE TIMELINESS")
    print("=" * 70)

    table = pd.crosstab(
        df["theme"],
        df["timely_yes"]
    )

    table.columns = [
        "Not Timely" if x is False
        else "Timely"
        for x in table.columns
    ]

    chi2, p_value, dof, expected = (
        chi2_contingency(table)
    )

    v = cramers_v(table)

    print("\nContingency table:")
    print(table)

    print(
        f"\nChi-square statistic: "
        f"{chi2:.4f}"
    )

    print(
        f"Degrees of freedom: "
        f"{dof}"
    )

    print(
        f"P-value: "
        f"{p_value:.10f}"
    )

    print(
        f"Cramer's V: "
        f"{v:.4f}"
    )

    significant = p_value < 0.05

    print(
        f"\nStatistically significant: "
        f"{significant}"
    )

    if significant:

        print(
            "Conclusion: Response timeliness "
            "differs significantly across CX themes."
        )

    else:

        print(
            "Conclusion: No statistically "
            "significant difference in response "
            "timeliness across CX themes."
        )

    output = table.reset_index()

    output.to_csv(
        OUTPUT_DIR
        / "theme_timeliness_contingency.csv",
        index=False
    )

    test_result = pd.DataFrame([
        {
            "analysis": "CX theme vs response timeliness",
            "chi_square": chi2,
            "degrees_of_freedom": dof,
            "p_value": p_value,
            "cramers_v": v,
            "statistically_significant": significant,
        }
    ])

    test_result.to_csv(
        OUTPUT_DIR
        / "theme_timeliness_chi_square.csv",
        index=False
    )

    return table, test_result


# ============================================================
# PRODUCT × TOPIC ASSOCIATION
# ============================================================

def product_topic_test(df):

    print("\n" + "=" * 70)
    print("ANALYSIS 2: PRODUCT VS CX THEME")
    print("=" * 70)

    table = pd.crosstab(
        df["product"],
        df["theme"]
    )

    chi2, p_value, dof, expected = (
        chi2_contingency(table)
    )

    v = cramers_v(table)

    print(
        f"\nProducts: {table.shape[0]}"
    )

    print(
        f"CX themes: {table.shape[1]}"
    )

    print(
        f"Chi-square statistic: "
        f"{chi2:.4f}"
    )

    print(
        f"Degrees of freedom: "
        f"{dof}"
    )

    print(
        f"P-value: "
        f"{p_value:.10f}"
    )

    print(
        f"Cramer's V: "
        f"{v:.4f}"
    )

    significant = p_value < 0.05

    print(
        f"\nStatistically significant: "
        f"{significant}"
    )

    if significant:

        print(
            "Conclusion: Product and CX theme "
            "are statistically associated."
        )

    else:

        print(
            "Conclusion: No statistically "
            "significant association between "
            "product and CX theme."
        )

    test_result = pd.DataFrame([
        {
            "analysis": "Product vs CX theme",
            "chi_square": chi2,
            "degrees_of_freedom": dof,
            "p_value": p_value,
            "cramers_v": v,
            "statistically_significant": significant,
        }
    ])

    test_result.to_csv(
        OUTPUT_DIR
        / "product_theme_chi_square.csv",
        index=False
    )

    return table, test_result


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def wilson_interval(
    successes,
    total,
    confidence=0.95
):

    if total == 0:

        return (
            np.nan,
            np.nan
        )

    z = 1.959963984540054

    p = successes / total

    denominator = (
        1
        + z ** 2 / total
    )

    center = (
        p
        + z ** 2 / (2 * total)
    ) / denominator

    margin = (
        z
        * np.sqrt(
            (
                p * (1 - p)
                + z ** 2 / (4 * total)
            )
            / total
        )
        / denominator
    )

    return (
        max(0, center - margin),
        min(1, center + margin)
    )


# ============================================================
# THEME RESPONSE RATES + CI
# ============================================================

def theme_response_rates(df):

    print("\n" + "=" * 70)
    print("ANALYSIS 3: RESPONSE RATE CONFIDENCE INTERVALS")
    print("=" * 70)

    rows = []

    for theme, group in df.groupby(
        "theme",
        dropna=False
    ):

        total = len(group)

        not_timely = int(
            group["not_timely"].sum()
        )

        timely = total - not_timely

        rate = (
            not_timely
            / total
            * 100
        )

        lower, upper = wilson_interval(
            not_timely,
            total
        )

        rows.append(
            {
                "theme": theme,
                "complaints": total,
                "not_timely": not_timely,
                "timely": timely,
                "not_timely_rate_pct": rate,
                "ci_95_lower_pct": lower * 100,
                "ci_95_upper_pct": upper * 100,
            }
        )

    result = pd.DataFrame(
        rows
    ).sort_values(
        "not_timely_rate_pct",
        ascending=False
    )

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "theme_response_rates_ci.csv",
        index=False
    )

    return result


# ============================================================
# PRODUCT RESPONSE RATES
# ============================================================

def product_response_rates(df):

    print("\n" + "=" * 70)
    print("ANALYSIS 4: PRODUCT RESPONSE PERFORMANCE")
    print("=" * 70)

    result = (
        df.groupby(
            "product",
            dropna=False
        )
        .agg(
            complaints=(
                "timely_yes",
                "size"
            ),
            timely_responses=(
                "timely_yes",
                "sum"
            ),
        )
        .reset_index()
    )

    result["timely_rate_pct"] = (
        result["timely_responses"]
        / result["complaints"]
        * 100
    )

    result["not_timely_rate_pct"] = (
        100
        - result["timely_rate_pct"]
    )

    result = result.sort_values(
        "not_timely_rate_pct",
        ascending=False
    )

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        OUTPUT_DIR
        / "product_response_performance.csv",
        index=False
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    topic_timeliness_test(
        df
    )

    product_topic_test(
        df
    )

    theme_response_rates(
        df
    )

    product_response_rates(
        df
    )

    print("\n" + "=" * 70)
    print("STATISTICAL VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        "  - theme_timeliness_contingency.csv"
    )

    print(
        "  - theme_timeliness_chi_square.csv"
    )

    print(
        "  - product_theme_chi_square.csv"
    )

    print(
        "  - theme_response_rates_ci.csv"
    )

    print(
        "  - product_response_performance.csv"
    )


if __name__ == "__main__":
    main()