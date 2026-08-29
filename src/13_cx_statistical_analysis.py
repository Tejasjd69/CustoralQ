"""
13_cx_statistical_analysis.py

Customer Experience Statistical Analysis
-----------------------------------------
Uses the existing Customer 360 dataset to answer CX/business questions
with descriptive statistics, confidence intervals, hypothesis testing,
correlation analysis, and regression.

This module does NOT modify the existing project data.
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
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_360.parquet"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "cx_statistics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_customer_data():
    """Load the existing customer-level dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Customer dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print("=" * 70)
    print("CUSTOMER EXPERIENCE STATISTICAL ANALYSIS")
    print("=" * 70)

    print(f"\nDataset: {DATA_PATH}")
    print(f"Customers: {len(df):,}")
    print(f"Features: {len(df.columns)}")

    return df


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def confidence_interval_mean(series, confidence=0.95):
    """
    Calculate a confidence interval for the population mean
    based on the observed sample.
    """

    values = pd.to_numeric(series, errors="coerce").dropna()

    n = len(values)

    if n < 2:
        return np.nan, np.nan

    mean = values.mean()
    standard_error = stats.sem(values)

    margin = standard_error * stats.t.ppf(
        (1 + confidence) / 2,
        n - 1
    )

    return mean - margin, mean + margin


def describe_metric(df, metric):
    """Return descriptive statistics for a metric."""

    values = pd.to_numeric(df[metric], errors="coerce").dropna()

    lower, upper = confidence_interval_mean(values)

    return {
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


# ============================================================
# ANALYSIS 1
# RETURNS VS CUSTOMER HEALTH
# ============================================================

def analyze_returns_vs_health(df):
    """
    Question:
    Do customers who have returned products have different
    customer health scores from customers who have not?
    """

    print("\n" + "=" * 70)
    print("ANALYSIS 1: RETURNS VS CUSTOMER HEALTH")
    print("=" * 70)

    returned = df.loc[
        df["has_returned"] == 1,
        "customer_health_score"
    ].dropna()

    non_returned = df.loc[
        df["has_returned"] == 0,
        "customer_health_score"
    ].dropna()

    print(f"\nReturned customers:     {len(returned):,}")
    print(f"Non-returned customers: {len(non_returned):,}")

    print(f"\nReturned mean health:     {returned.mean():.2f}")
    print(f"Non-returned mean health: {non_returned.mean():.2f}")

    # Welch's independent two-sample t-test
    t_stat, p_value = stats.ttest_ind(
        returned,
        non_returned,
        equal_var=False
    )

    difference = returned.mean() - non_returned.mean()

    print(f"\nMean difference: {difference:.2f}")
    print(f"t-statistic:     {t_stat:.4f}")
    print(f"p-value:         {p_value:.6f}")

    if p_value < 0.05:
        conclusion = (
            "There is statistically significant evidence that "
            "customer health differs between customers with and "
            "without returns."
        )
    else:
        conclusion = (
            "There is not sufficient statistical evidence to conclude "
            "that customer health differs between the two groups."
        )

    print(f"\nConclusion:\n{conclusion}")

    result = pd.DataFrame([{
        "analysis": "Returns vs Customer Health",
        "returned_customers": len(returned),
        "non_returned_customers": len(non_returned),
        "returned_mean_health": returned.mean(),
        "non_returned_mean_health": non_returned.mean(),
        "mean_difference": difference,
        "t_statistic": t_stat,
        "p_value": p_value,
        "statistically_significant": p_value < 0.05,
        "conclusion": conclusion,
    }])

    return result


# ============================================================
# ANALYSIS 2
# HIGH-RISK VS LOW-RISK CUSTOMERS
# ============================================================

def analyze_churn_risk_behavior(df):
    """
    Compare behavioral metrics across high-risk and low-risk customers.
    """

    print("\n" + "=" * 70)
    print("ANALYSIS 2: CHURN RISK VS CUSTOMER BEHAVIOR")
    print("=" * 70)

    # Use the actual risk categories in the dataset.
    high_risk = df[
        df["churn_risk_tier"].isin(["High Risk", "Critical Risk"])
    ].copy()

    low_risk = df[
        df["churn_risk_tier"].isin(["Low Risk", "Medium Risk"])
    ].copy()

    print(f"\nHigh/Critical risk customers: {len(high_risk):,}")
    print(f"Low/Medium risk customers:    {len(low_risk):,}")

    metrics = [
        "total_revenue",
        "total_invoices",
        "recency_days",
        "purchase_frequency_per_month",
        "unique_purchase_days",
    ]

    results = []

    for metric in metrics:

        high = pd.to_numeric(
            high_risk[metric],
            errors="coerce"
        ).dropna()

        low = pd.to_numeric(
            low_risk[metric],
            errors="coerce"
        ).dropna()

        if len(high) < 2 or len(low) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(
            high,
            low,
            equal_var=False
        )

        results.append({
            "metric": metric,
            "high_critical_mean": high.mean(),
            "low_medium_mean": low.mean(),
            "difference": high.mean() - low.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "statistically_significant": p_value < 0.05,
        })

        print(
            f"\n{metric}"
            f"\n  High/Critical: {high.mean():.2f}"
            f"\n  Low/Medium:    {low.mean():.2f}"
            f"\n  p-value:       {p_value:.6f}"
        )

    return pd.DataFrame(results)
# ============================================================
# ANALYSIS 3
# CORRELATION ANALYSIS
# ============================================================

def analyze_correlations(df):
    """
    Analyze relationships between customer behavior,
    health and churn metrics.

    Correlation indicates association, NOT causation.
    """

    print("\n" + "=" * 70)
    print("ANALYSIS 3: CUSTOMER BEHAVIOR CORRELATIONS")
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
        column for column in metrics
        if column in df.columns
    ]

    correlation_matrix = df[available].corr(method="spearman")

    print("\nSpearman correlation matrix:")
    print(correlation_matrix.round(3))

    correlation_matrix.to_csv(
        OUTPUT_DIR / "correlation_matrix.csv"
    )

    return correlation_matrix


# ============================================================
# ANALYSIS 4
# REGRESSION
# ============================================================

def run_health_regression(df):
    """
    Explain variation in customer health score using
    interpretable customer behavior variables.

    This identifies statistical associations, not causal effects.
    """

    print("\n" + "=" * 70)
    print("ANALYSIS 4: CUSTOMER HEALTH REGRESSION")
    print("=" * 70)

    target = "customer_health_score"

    features = [
        "recency_days",
        "purchase_frequency_per_month",
        "total_revenue",
        "total_invoices",
        "return_rate",
        "unique_products",
    ]

    required = [target] + features

    model_df = df[required].copy()

    for column in required:
        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce"
        )

    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    X = model_df[features]
    y = model_df[target]

    # Add intercept
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    print("\nRegression summary:\n")
    print(model.summary())

    coefficients = pd.DataFrame({
        "feature": model.params.index,
        "coefficient": model.params.values,
        "p_value": model.pvalues.values,
        "significant_at_5pct": model.pvalues.values < 0.05,
    })

    coefficients.to_csv(
        OUTPUT_DIR / "health_regression_coefficients.csv",
        index=False
    )

    print(
        f"\nR-squared: {model.rsquared:.4f}"
    )

    return model, coefficients


# ============================================================
# ANALYSIS 5
# DESCRIPTIVE STATISTICS
# ============================================================

def generate_descriptive_statistics(df):
    """Generate descriptive statistics for major CX metrics."""

    print("\n" + "=" * 70)
    print("ANALYSIS 5: DESCRIPTIVE STATISTICS")
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

        result = describe_metric(df, metric)
        results.append(result)

        print(
            f"\n{metric}"
            f"\n  Mean:   {result['mean']:.2f}"
            f"\n  Median: {result['median']:.2f}"
            f"\n  95% CI: [{result['ci_95_lower']:.2f}, "
            f"{result['ci_95_upper']:.2f}]"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "descriptive_statistics.csv",
        index=False
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_customer_data()

    # 1. Descriptive statistics
    descriptive = generate_descriptive_statistics(df)

    # 2. Returns vs health
    returns_analysis = analyze_returns_vs_health(df)

    # 3. Churn risk behavior
    risk_analysis = analyze_churn_risk_behavior(df)

    # 4. Correlations
    correlations = analyze_correlations(df)

    # 5. Regression
    regression_model, regression_coefficients = (
        run_health_regression(df)
    )

    # Save test results
    returns_analysis.to_csv(
        OUTPUT_DIR / "returns_vs_health_test.csv",
        index=False
    )

    risk_analysis.to_csv(
        OUTPUT_DIR / "churn_risk_group_tests.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nReports saved to:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()