"""
13_cx_statistical_analysis_v2.py

Customer Experience Statistical Analysis
-----------------------------------------
Builds a statistically defensible analytics layer on top of the
existing Customer 360 dataset.

Includes:
1. Descriptive statistics
2. 95% confidence intervals
3. Returns vs health hypothesis test
4. Cohen's d effect size
5. Churn-risk group comparison
6. Spearman correlation analysis
7. Logistic regression for actual churn
8. Odds ratios
9. Basic data-quality checks

Important:
Statistical association does not imply causation.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_360.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "cx_statistics_v2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_customer_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Customer dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print("=" * 70)
    print("CUSTOMER EXPERIENCE STATISTICAL ANALYSIS V2")
    print("=" * 70)

    print(f"\nCustomers: {len(df):,}")
    print(f"Features:  {len(df.columns)}")

    return df


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def confidence_interval_mean(
    series,
    confidence=0.95
):

    values = (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    n = len(values)

    if n < 2:
        return np.nan, np.nan

    mean = values.mean()

    standard_error = stats.sem(values)

    margin = (
        standard_error
        * stats.t.ppf(
            (1 + confidence) / 2,
            n - 1
        )
    )

    return (
        mean - margin,
        mean + margin
    )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(df):

    print("\n" + "=" * 70)
    print("1. DESCRIPTIVE STATISTICS")
    print("=" * 70)

    metrics = [
        "customer_health_score",
        "churn_probability",
        "total_revenue",
        "total_invoices",
        "recency_days",
        "purchase_frequency_per_month",
        "return_rate",
    ]

    results = []

    for metric in metrics:

        if metric not in df.columns:
            continue

        values = (
            pd.to_numeric(
                df[metric],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(values) < 2:
            continue

        lower, upper = confidence_interval_mean(
            values
        )

        result = {
            "metric": metric,
            "n": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "ci_95_lower": lower,
            "ci_95_upper": upper,
        }

        results.append(result)

        print(
            f"\n{metric}"
            f"\n  N:       {len(values):,}"
            f"\n  Mean:    {values.mean():.2f}"
            f"\n  Median:  {values.median():.2f}"
            f"\n  95% CI:  [{lower:.2f}, {upper:.2f}]"
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_DIR
        / "descriptive_statistics.csv",
        index=False
    )

    return result_df


# ============================================================
# COHEN'S D
# ============================================================

def cohens_d(group1, group2):

    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    n1 = len(group1)
    n2 = len(group2)

    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(
        group1,
        ddof=1
    )

    var2 = np.var(
        group2,
        ddof=1
    )

    pooled_sd = np.sqrt(
        (
            (n1 - 1) * var1
            + (n2 - 1) * var2
        )
        / (n1 + n2 - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (
        np.mean(group1)
        - np.mean(group2)
    ) / pooled_sd


# ============================================================
# RETURNS VS HEALTH
# ============================================================

def returns_vs_health(df):

    print("\n" + "=" * 70)
    print("2. RETURNS VS CUSTOMER HEALTH")
    print("=" * 70)

    returned = (
        df.loc[
            df["has_returned"] == 1,
            "customer_health_score"
        ]
        .dropna()
    )

    non_returned = (
        df.loc[
            df["has_returned"] == 0,
            "customer_health_score"
        ]
        .dropna()
    )

    t_stat, p_value = stats.ttest_ind(
        returned,
        non_returned,
        equal_var=False
    )

    effect_size = cohens_d(
        returned,
        non_returned
    )

    difference = (
        returned.mean()
        - non_returned.mean()
    )

    print(
        f"\nReturned customers:     {len(returned):,}"
        f"\nNon-returned customers: {len(non_returned):,}"
    )

    print(
        f"\nReturned mean health:     "
        f"{returned.mean():.2f}"
    )

    print(
        f"Non-returned mean health: "
        f"{non_returned.mean():.2f}"
    )

    print(
        f"\nMean difference: {difference:.2f}"
        f"\nt-statistic:     {t_stat:.4f}"
        f"\np-value:         {p_value:.6g}"
        f"\nCohen's d:       {effect_size:.4f}"
    )

    if abs(effect_size) < 0.2:
        effect_interpretation = "Negligible"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "Small"
    elif abs(effect_size) < 0.8:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"

    print(
        f"\nEffect size interpretation: "
        f"{effect_interpretation}"
    )

    result = pd.DataFrame([{

        "analysis":
            "Returns vs Customer Health",

        "returned_customers":
            len(returned),

        "non_returned_customers":
            len(non_returned),

        "returned_mean_health":
            returned.mean(),

        "non_returned_mean_health":
            non_returned.mean(),

        "mean_difference":
            difference,

        "t_statistic":
            t_stat,

        "p_value":
            p_value,

        "cohens_d":
            effect_size,

        "effect_interpretation":
            effect_interpretation,

        "statistically_significant":
            p_value < 0.05,

    }])

    result.to_csv(
        OUTPUT_DIR
        / "returns_vs_health_test.csv",
        index=False
    )

    return result


# ============================================================
# CHURN RISK GROUP COMPARISON
# ============================================================

def churn_risk_comparison(df):

    print("\n" + "=" * 70)
    print("3. CHURN RISK VS CUSTOMER BEHAVIOR")
    print("=" * 70)

    high_risk = df[
        df["churn_risk_tier"].isin(
            [
                "High Risk",
                "Critical Risk"
            ]
        )
    ].copy()

    low_risk = df[
        df["churn_risk_tier"].isin(
            [
                "Low Risk",
                "Medium Risk"
            ]
        )
    ].copy()

    print(
        f"\nHigh/Critical risk: "
        f"{len(high_risk):,}"
    )

    print(
        f"Low/Medium risk:    "
        f"{len(low_risk):,}"
    )

    metrics = [
        "total_revenue",
        "total_invoices",
        "recency_days",
        "purchase_frequency_per_month",
        "unique_purchase_days",
    ]

    results = []

    for metric in metrics:

        high = (
            pd.to_numeric(
                high_risk[metric],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        low = (
            pd.to_numeric(
                low_risk[metric],
                errors="coerce"
            )
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna()
        )

        if len(high) < 2 or len(low) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(
            high,
            low,
            equal_var=False
        )

        effect_size = cohens_d(
            high,
            low
        )

        results.append({

            "metric":
                metric,

            "high_critical_mean":
                high.mean(),

            "low_medium_mean":
                low.mean(),

            "difference":
                high.mean() - low.mean(),

            "t_statistic":
                t_stat,

            "p_value":
                p_value,

            "cohens_d":
                effect_size,

            "statistically_significant":
                p_value < 0.05,

        })

        print(
            f"\n{metric}"
            f"\n  High/Critical: "
            f"{high.mean():.2f}"
            f"\n  Low/Medium:    "
            f"{low.mean():.2f}"
            f"\n  Difference:    "
            f"{high.mean() - low.mean():.2f}"
            f"\n  p-value:       "
            f"{p_value:.6g}"
            f"\n  Cohen's d:     "
            f"{effect_size:.4f}"
        )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_DIR
        / "churn_risk_group_tests.csv",
        index=False
    )

    return result_df


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    print("\n" + "=" * 70)
    print("4. SPEARMAN CORRELATION ANALYSIS")
    print("=" * 70)

    metrics = [
        "recency_days",
        "purchase_frequency_per_month",
        "total_revenue",
        "total_invoices",
        "return_rate",
        "customer_health_score",
        "churn_probability",
    ]

    available = [
        col
        for col in metrics
        if col in df.columns
    ]

    correlation = (
        df[available]
        .corr(method="spearman")
    )

    print("\nCorrelation matrix:")
    print(correlation.round(3))

    correlation.to_csv(
        OUTPUT_DIR
        / "correlation_matrix.csv"
    )

    return correlation


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def churn_logistic_regression(df):

    print("\n" + "=" * 70)
    print("5. LOGISTIC REGRESSION — CHURN")
    print("=" * 70)

    target = "actual_churned"

    features = [
        "recency_days",
        "total_invoices",
        "total_revenue",
        "unique_products",
        "return_rate",
        "purchase_frequency_per_month",
    ]

    required = [target] + features

    model_df = df[
        required
    ].copy()

    # Convert everything to numeric
    for column in required:

        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce"
        )

    model_df = (
        model_df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    # Ensure binary target
    model_df = model_df[
        model_df[target].isin([0, 1])
    ]

    print(
        f"\nModel observations: "
        f"{len(model_df):,}"
    )

    print(
        f"Churn rate: "
        f"{model_df[target].mean():.2%}"
    )

    X = model_df[features]
    y = model_df[target]

    X = sm.add_constant(X)

    model = sm.Logit(
        y,
        X
    ).fit(
        disp=False
    )

    print("\nModel summary:\n")

    print(model.summary())

    # Odds ratios
    odds_ratios = np.exp(
        model.params
    )

    confidence_intervals = (
        np.exp(
            model.conf_int()
        )
    )

    result = pd.DataFrame({

        "feature":
            model.params.index,

        "coefficient":
            model.params.values,

        "odds_ratio":
            odds_ratios.values,

        "ci_lower":
            confidence_intervals[0].values,

        "ci_upper":
            confidence_intervals[1].values,

        "p_value":
            model.pvalues.values,

        "significant_at_5pct":
            model.pvalues.values < 0.05,

    })

    result.to_csv(
        OUTPUT_DIR
        / "churn_logistic_regression.csv",
        index=False
    )

    print("\nOdds ratios:")

    print(
        result.round(4).to_string(
            index=False
        )
    )

    return model, result


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def data_quality_checks(df):

    print("\n" + "=" * 70)
    print("6. DATA QUALITY CHECKS")
    print("=" * 70)

    checks = []

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for column, count in missing.items():

        if count > 0:

            checks.append({

                "check":
                    "Missing values",

                "column":
                    column,

                "count":
                    int(count),

                "status":
                    "Review"

            })

    # --------------------------------------------------------
    # Return rate > 100%
    # --------------------------------------------------------

    if "return_rate" in df.columns:

        invalid_return_rate = (
            df["return_rate"] > 1
        ).sum()

        print(
            f"\nCustomers with return_rate > 1: "
            f"{invalid_return_rate:,}"
        )

        checks.append({

            "check":
                "Return rate above 100%",

            "column":
                "return_rate",

            "count":
                int(invalid_return_rate),

            "status":
                "Review metric definition"

        })

    # --------------------------------------------------------
    # Churn scoring
    # --------------------------------------------------------

    if "churn_risk_tier" in df.columns:

        not_scored = (
            df["churn_risk_tier"]
            == "Not Scored"
        ).sum()

        print(
            f"Not-scored customers: "
            f"{not_scored:,}"
        )

        checks.append({

            "check":
                "Customers without churn score",

            "column":
                "churn_risk_tier",

            "count":
                int(not_scored),

            "status":
                "Expected / Excluded"

        })

    result = pd.DataFrame(checks)

    result.to_csv(
        OUTPUT_DIR
        / "data_quality_checks.csv",
        index=False
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_customer_data()

    descriptive_statistics(df)

    returns_vs_health(df)

    churn_risk_comparison(df)

    correlation_analysis(df)

    churn_logistic_regression(df)

    data_quality_checks(df)

    print("\n" + "=" * 70)
    print("V2 STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nReports saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print("\nGenerated files:")

    for file in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):

        print(
            f"  - {file.name}"
        )


if __name__ == "__main__":
    main()