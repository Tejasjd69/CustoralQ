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


# ============================================================
# THEME NAMES
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
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("CX PRODUCT × THEME RESIDUAL ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded records: {len(df):,}"
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

    return df


# ============================================================
# STANDARDIZED RESIDUALS
# ============================================================

def calculate_residuals(table):

    chi2, p_value, dof, expected = (
        chi2_contingency(table)
    )

    expected_df = pd.DataFrame(
        expected,
        index=table.index,
        columns=table.columns
    )

    # Pearson standardized residual
    residuals = (
        table - expected_df
    ) / np.sqrt(expected_df)

    return (
        chi2,
        p_value,
        dof,
        expected_df,
        residuals
    )


# ============================================================
# PRODUCT × THEME ANALYSIS
# ============================================================

def product_theme_residuals(df):

    print("\n" + "=" * 70)
    print("PRODUCT × CX THEME RESIDUALS")
    print("=" * 70)

    table = pd.crosstab(
        df["product"],
        df["theme"]
    )

    (
        chi2,
        p_value,
        dof,
        expected,
        residuals
    ) = calculate_residuals(
        table
    )

    print(
        f"\nChi-square: {chi2:.4f}"
    )

    print(
        f"Degrees of freedom: {dof}"
    )

    print(
        f"P-value: {p_value:.10f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Positive residual = more complaints than expected."
    )

    print(
        "Negative residual = fewer complaints than expected."
    )

    # --------------------------------------------------------
    # Convert residual matrix into long format
    # --------------------------------------------------------

    rows = []

    for product in residuals.index:

        for theme in residuals.columns:

            observed = table.loc[
                product,
                theme
            ]

            expected_value = expected.loc[
                product,
                theme
            ]

            residual = residuals.loc[
                product,
                theme
            ]

            rows.append(
                {
                    "product": product,
                    "theme": theme,
                    "observed": int(observed),
                    "expected": round(
                        float(expected_value),
                        2
                    ),
                    "residual": round(
                        float(residual),
                        4
                    ),
                    "abs_residual": round(
                        abs(float(residual)),
                        4
                    ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Flag meaningful deviations
    #
    # Approximate rule:
    # |residual| >= 2 → notable
    # |residual| >= 3 → strong
    # --------------------------------------------------------

    result["association_strength"] = np.select(
        [
            result["abs_residual"] >= 3,
            result["abs_residual"] >= 2,
        ],
        [
            "Strong",
            "Notable",
        ],
        default="Normal",
    )

    result["direction"] = np.where(
        result["residual"] > 0,
        "Higher than expected",
        np.where(
            result["residual"] < 0,
            "Lower than expected",
            "Expected"
        )
    )

    # Sort strongest positive deviations first
    positive = (
        result[
            result["residual"] > 0
        ]
        .sort_values(
            "residual",
            ascending=False
        )
    )

    output_file = (
        OUTPUT_DIR
        / "product_theme_residuals.csv"
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(
        "\nTop product/theme combinations "
        "higher than expected:"
    )

    print(
        positive.head(30).to_string(
            index=False
        )
    )

    return result


# ============================================================
# STRONGEST FRICTION POINTS
# ============================================================

def strongest_friction_points(result):

    print("\n" + "=" * 70)
    print("STRONGEST PRODUCT-SPECIFIC CX FRICTION POINTS")
    print("=" * 70)

    strong = (
        result[
            (result["residual"] >= 2)
        ]
        .sort_values(
            "residual",
            ascending=False
        )
    )

    if len(strong) == 0:

        print(
            "\nNo product/theme combination "
            "has a residual >= 2."
        )

        return

    print(
        strong.head(20).to_string(
            index=False
        )
    )

    strong.to_csv(
        OUTPUT_DIR
        / "strong_product_theme_friction_points.csv",
        index=False
    )


# ============================================================
# THEME CONCENTRATION BY PRODUCT
# ============================================================

def theme_concentration(df):

    print("\n" + "=" * 70)
    print("THEME CONCENTRATION")
    print("=" * 70)

    counts = (
        df.groupby(
            [
                "product",
                "theme",
            ]
        )
        .size()
        .reset_index(
            name="complaints"
        )
    )

    product_totals = (
        counts.groupby(
            "product"
        )["complaints"]
        .transform("sum")
    )

    counts["theme_share_pct"] = (
        counts["complaints"]
        / product_totals
        * 100
    )

    counts = counts.sort_values(
        "theme_share_pct",
        ascending=False
    )

    counts.to_csv(
        OUTPUT_DIR
        / "product_theme_concentration.csv",
        index=False
    )

    print(
        counts.head(30).to_string(
            index=False
        )
    )

    return counts


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    residuals = product_theme_residuals(
        df
    )

    strongest_friction_points(
        residuals
    )

    theme_concentration(
        df
    )

    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        "  - product_theme_residuals.csv"
    )

    print(
        "  - strong_product_theme_friction_points.csv"
    )

    print(
        "  - product_theme_concentration.csv"
    )


if __name__ == "__main__":
    main()