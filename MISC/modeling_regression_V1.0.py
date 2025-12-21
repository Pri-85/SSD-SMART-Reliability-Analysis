"""
modeling_regression.py

This module implements a suite of linear modeling techniques used to analyze
SSD reliability and performance data. Models are built on the cleaned,
normalized dataset and focus on latency, usage, and reliability metrics.

Included:
- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- One-Way ANOVA
- Two-Way (Factorial) ANOVA
- ANCOVA
- Regularized Linear Models (Ridge, Lasso)
- Generalized Linear Models (GLM) for count data

All formulas use variables present in the processed dataset.
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 0. HELPER: LOAD DATA AND PRINT COLUMNS
# ---------------------------------------------------------------------------

def load_data():
    # Use raw string for Windows path
    csv_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\Data\Processed_V1.1\SSD_dataset_Cleaned_Normalized.csv"
    df = pd.read_csv(csv_path)

    print("\n=== Loaded dataset ===")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

    return df


# ---------------------------------------------------------------------------
# 1. SIMPLE LINEAR REGRESSION
# ---------------------------------------------------------------------------

def simple_linear_regression(df):
    model = smf.ols("read_latency_ms ~ power_on_hours", data=df).fit()
    print("\n=== Simple Linear Regression: read_latency_ms ~ power_on_hours ===")
    print(model.summary())
    return model


# ---------------------------------------------------------------------------
# 2. MULTIPLE LINEAR REGRESSION
# ---------------------------------------------------------------------------

def multiple_linear_regression(df):
    model = smf.ols(
        "write_latency_ms ~ power_on_hours + host_write_commands + temperature_current + percentage_used",
        data=df
    ).fit()
    print("\n=== Multiple Linear Regression: write_latency_ms with multiple predictors ===")
    print(model.summary())
    return model



# ---------------------------------------------------------------------------
# 3. ONE-WAY ANOVA
# ---------------------------------------------------------------------------

def one_way_anova(df):
    model = smf.ols("write_latency_ms ~ C(system_manufacturer)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== One-Way ANOVA: write_latency_ms ~ system_manufacturer ===")
    print(anova_table)
    return anova_table



# ---------------------------------------------------------------------------
# 4. TWO-WAY (FACTORIAL) ANOVA
# ---------------------------------------------------------------------------

def two_way_anova(df):
    model = smf.ols(
        "write_latency_ms ~ C(system_manufacturer) * C(capacity_sku)",
        data=df
    ).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== Two-Way ANOVA: write_latency_ms ~ manufacturer * capacity_sku ===")
    print(anova_table)
    return anova_table



# ---------------------------------------------------------------------------
# 5. ANCOVA (Analysis of Covariance)
# ---------------------------------------------------------------------------

def ancova(df):
    model = smf.ols(
        "read_latency_ms ~ C(system_manufacturer) + power_on_hours",
        data=df
    ).fit()
    print("\n=== ANCOVA: read_latency_ms ~ manufacturer + power_on_hours ===")
    print(model.summary())
    return model



# ---------------------------------------------------------------------------
# 6. REGULARIZED LINEAR MODELS (Ridge, Lasso)
# ---------------------------------------------------------------------------

def regularized_models(df):
    target = "write_latency_ms"
    numeric_df = df.select_dtypes(include="number").dropna()

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    lasso = Lasso(alpha=0.001).fit(X_train, y_train)

    print("\n=== Ridge Regression Coefficients ===")
    print(ridge.coef_)

    print("\n=== Lasso Regression Coefficients ===")
    print(lasso.coef_)

    return ridge, lasso



# ---------------------------------------------------------------------------
# 7. GENERALIZED LINEAR MODELS (GLM) FOR COUNT DATA
# ---------------------------------------------------------------------------

def glm_count_model(df):
    model = smf.glm(
        "unsafe_shutdowns ~ power_on_hours + power_cycles + temperature_current",
        data=df,
        family=sm.families.Poisson()
    ).fit()

    print("\n=== GLM Poisson: unsafe_shutdowns ===")
    print(model.summary())
    return model

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()

    # Run models
    simple_linear_regression(df)
    multiple_linear_regression(df)
    one_way_anova(df)
    two_way_anova(df)
    ancova(df)
    regularized_models(df)
    glm_count_model(df)
