'''

# =============================================================================
# SSD ANALYTIC MODELING PIPELINE – V1.2 
# Author: Priya Pooja Hariharan
# Script Version: 1.2
# =============================================================================


modeling_regression.py

SSD Reliability Analysis – Linear Modeling Suite

This module implements a suite of linear modeling techniques for SSD telemetry data.
It covers classical regression, ANOVA, ANCOVA, regularized regression, and GLMs for
count data. All formulas have been corrected to use valid dataset columns, and
unsafe column names have been renamed for Patsy compatibility.

Models included:
- Simple Linear Regression (SLR)
- Multiple Linear Regression (MLR)
- One-Way ANOVA
- Two-Way (Factorial) ANOVA
- ANCOVA
- Regularized Linear Models (Ridge, Lasso)
- Generalized Linear Models (GLM) for count data

'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DATA = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\EDA_analysis_results.V1.3\cleaned_SSD_dataset.csv"
OUTPUT_DIR = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\Analytic_Model_results_V1.3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(model_name, plot_name):
    filename = f"{model_name}_{plot_name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    df = pd.read_csv(INPUT_DATA)
    print("\n=== Loaded dataset ===")
    print(df.shape)
    print(df.columns.tolist())
    return df


# =============================================================================
# UNIVERSAL DIAGNOSTIC PLOT HELPERS
# =============================================================================

def plot_histogram(series, model_name, label):
    plt.figure(figsize=(8,5))
    sns.histplot(series, kde=True, bins=30, color="steelblue")
    plt.title(f"Histogram of {label}")
    plt.xlabel(label)
    save_plot(model_name, f"histogram_{label}")

def plot_qq(series, model_name, label):
    plt.figure(figsize=(6,6))
    sm.qqplot(series.dropna(), line="45", fit=True)
    plt.title(f"Q–Q Plot of {label}")
    save_plot(model_name, f"qqplot_{label}")

def plot_residuals_vs_fitted(fitted, residuals, model_name):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    save_plot(model_name, "residuals_vs_fitted")

def plot_boxplot(df, x, y, model_name, title):
    plt.figure(figsize=(10,5))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    save_plot(model_name, f"boxplot_{x}_{y}")

def plot_parallel_slopes(df, x, y, group, model_name):
    plt.figure(figsize=(10,6))
    for g in df[group].unique():
        subset = df[df[group] == g]
        sns.regplot(
            x=subset[x], y=subset[y],
            label=g, scatter_kws={"alpha":0.3},
            line_kws={"linewidth":2}
        )
    plt.title(f"Parallel Slopes: {y} ~ {x} by {group}")
    plt.legend(title=group)
    save_plot(model_name, "parallel_slopes")

# =============================================================================
# 1. LATENCY MODEL – Predict IOPS
# =============================================================================

def latency_model(df):
    model_name = "latency_model"

    # Fit model
    model = smf.ols(
        "iops ~ avg_queue_depth + bandwidth_read_gbps + bandwidth_write_gbps + "
        "data_units_read + data_units_written + composite_temperature_c + percentage_used",
        data=df
    ).fit()

    print("\n=== LATENCY MODEL (Predicting IOPS) ===")
    print(model.summary())

    # Save summary
    save_model_summary(model, model_name)

    # -----------------------------
    # DEFINE fitted and residuals
    # -----------------------------
    fitted = model.fittedvalues
    residuals = model.resid

    # -----------------------------
    # DIAGNOSTIC PLOTS
    # -----------------------------

    # Histogram of target
    plot_histogram(df["iops"], model_name, "iops")

    # Q–Q plot of target
    plot_qq(df["iops"], model_name, "iops")

    # Residual vs fitted
    plot_residuals_vs_fitted(fitted, residuals, model_name)

    # Scatterplot of main predictor
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["avg_queue_depth"], y=df["iops"], alpha=0.4)
    plt.title("IOPS vs Queue Depth")
    save_plot(model_name, "scatter_qd")

    return model


# =============================================================================
# 2. WEAR MODEL – Predict Percentage Used
# =============================================================================

def wear_model(df):
    model_name = "wear_model"

    model = smf.ols(
        "percentage_used ~ power_on_hours + data_units_written + wear_level_avg + "
        "wear_level_max + bad_block_count_grown",
        data=df
    ).fit()

    print("\n=== WEAR MODEL (Predicting Percentage Used) ===")
    print(model.summary())

    
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["power_on_hours"], y=df["percentage_used"], alpha=0.4)
    plt.title("Percentage Used vs Power-On Hours")
    save_plot(model_name, "scatter_power_on")

    return model

# =============================================================================
# 3. RELIABILITY MODEL – Predict PCIe Correctable Errors
# =============================================================================

def reliability_model(df):
    model_name = "reliability_model"

    model = smf.ols(
        "pcie_correctable_errors ~ avg_queue_depth + percentage_used + media_errors + "
        "bad_block_count_grown + C(workload_type)",
        data=df
    ).fit()

    print("\n=== RELIABILITY MODEL (Predicting PCIe Correctable Errors) ===")
    print(model.summary())

    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["percentage_used"], y=df["pcie_correctable_errors"], alpha=0.4)
    plt.title("PCIe Correctable Errors vs Percentage Used")
    save_plot(model_name, "scatter_percentage_used")

    return model

# =============================================================================
# 4. PERFORMANCE MODEL – Predict Bandwidth Write
# =============================================================================

def performance_model(df):
    model_name = "performance_model"

    model = smf.ols(
        "bandwidth_write_gbps ~ iops + avg_queue_depth + workload_block_size_kb + data_units_written",
        data=df
    ).fit()

    print("\n=== PERFORMANCE MODEL (Predicting Write Bandwidth) ===")
    print(model.summary())

    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df["iops"], y=df["bandwidth_write_gbps"], alpha=0.4)
    plt.title("Write Bandwidth vs IOPS")
    save_plot(model_name, "scatter_iops")

    return model
# =============================================================================
# 5. ONE-WAY ANOVA – Wear by Form Factor
# =============================================================================

def one_way_anova(df):
    model_name = "one_way_anova"

    model = smf.ols("wear_level_avg ~ C(ff)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== ONE-WAY ANOVA: wear_level_avg ~ form_factor ===")
    print(anova_table)

    # Save summary
    save_model_summary(model, model_name)

    # Boxplot
    plot_boxplot(df, "ff", "wear_level_avg", model_name, "Wear Level Avg by Form Factor")

    # Histogram
    plot_histogram(df["wear_level_avg"], model_name, "wear_level_avg")

    # Q–Q plot of residuals
    plot_qq(model.resid, model_name, "anova_residuals")
    fitted = model.fittedvalues
    residuals = model.resid

    # Residual vs fitted
    plot_residuals_vs_fitted(model.fittedvalues, model.resid, model_name)

    return anova_table

# =============================================================================
# 6. TWO-WAY ANOVA – NAND × Queue Depth
# =============================================================================

def two_way_anova_nand(df):
    model_name = "two_way_anova_nand"

    df["qd_bin"] = pd.qcut(df["avg_queue_depth"], q=4, labels=["Q1","Q2","Q3","Q4"])

    model = smf.ols("pcie_correctable_errors ~ C(nand_type) * C(qd_bin)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== TWO-WAY ANOVA: NAND Type × Queue Depth ===")
    print(anova_table)

    save_model_summary(model, model_name)

    # Boxplots
    plot_boxplot(df, "nand_type", "pcie_correctable_errors", model_name,
                 "PCIe Correctable Errors by NAND Type")
    plot_boxplot(df, "qd_bin", "pcie_correctable_errors", model_name,
                 "PCIe Correctable Errors by Queue Depth")

    # Histogram
    plot_histogram(df["pcie_correctable_errors"], model_name, "pcie_correctable_errors")

    # Q–Q plot
    plot_qq(model.resid, model_name, "anova_residuals")
    fitted = model.fittedvalues
    residuals = model.resid

    # Residual vs fitted
    plot_residuals_vs_fitted(model.fittedvalues, model.resid, model_name)

    return anova_table

def save_model_summary(model, model_name):
    summary_path = os.path.join(OUTPUT_DIR, f"{model_name}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(model.summary().as_text())

# =============================================================================
# 7. TWO-WAY ANOVA – Form Factor × Queue Depth
# =============================================================================

def two_way_anova_ff(df):
    model_name = "two_way_anova_ff"

    df["qd_bin"] = pd.qcut(df["avg_queue_depth"], q=4, labels=["Q1","Q2","Q3","Q4"])

    model = smf.ols("pcie_correctable_errors ~ C(ff) * C(qd_bin)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== TWO-WAY ANOVA: Form Factor × Queue Depth ===")
    print(anova_table)

    save_model_summary(model, model_name)

    # Boxplots
    plot_boxplot(df, "ff", "pcie_correctable_errors", model_name,
                 "PCIe Correctable Errors by Form Factor")
    plot_boxplot(df, "qd_bin", "pcie_correctable_errors", model_name,
                 "PCIe Correctable Errors by Queue Depth")

    # Histogram
    plot_histogram(df["pcie_correctable_errors"], model_name, "pcie_correctable_errors")

    # Q–Q plot
    plot_qq(model.resid, model_name, "anova_residuals")
    fitted = model.fittedvalues
    residuals = model.resid

    # Residual vs fitted
    plot_residuals_vs_fitted(model.fittedvalues, model.resid, model_name)

    return anova_table
# =============================================================================
# 8. REGULARIZED MODELS – Ridge & Lasso (Predict IOPS)
# =============================================================================

def regularized_models(df):
    """
    Predict iops using all numeric predictors with regularization.
    """

    model_name = "regularized_models"

    # Select ONLY numeric columns
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # Replace inf/-inf with NaN
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    # Impute missing values with medians (numeric only)
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Remove constant columns
    constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() <= 1]
    if constant_cols:
        print("\n[INFO] Dropping constant columns:", constant_cols)
        numeric_df = numeric_df.drop(columns=constant_cols)

    # Ensure target exists
    target = "iops"
    if target not in numeric_df.columns:
        print(f"[ERROR] Target '{target}' not found in numeric columns.")
        return None

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Fit models
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    lasso = Lasso(alpha=0.001).fit(X_train, y_train)

    # -----------------------------
    # PREDICTIONS & RESIDUALS
    # -----------------------------
    y_pred = ridge.predict(X_test)
    residuals = y_test - y_pred

    # -----------------------------
    # DIAGNOSTIC PLOTS
    # -----------------------------

    # Histogram of target
    plot_histogram(y, model_name, "iops")

    # Q–Q plot of residuals
    plot_qq(pd.Series(residuals), model_name, "ridge_residuals")

    # Residual vs fitted
    plot_residuals_vs_fitted(y_pred, residuals, model_name)

    # Coefficient plots
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "ridge_coef": ridge.coef_,
        "lasso_coef": lasso.coef_
    })

    plt.figure(figsize=(12, 6))
    sns.barplot(x="ridge_coef", y="feature", data=coef_df.sort_values("ridge_coef"))
    plt.title("Ridge Coefficients")
    save_plot(model_name, "ridge_coefficients")

    plt.figure(figsize=(12, 6))
    sns.barplot(x="lasso_coef", y="feature", data=coef_df.sort_values("lasso_coef"))
    plt.title("Lasso Coefficients")
    save_plot(model_name, "lasso_coefficients")

    # -----------------------------
    # PRINT PERFORMANCE
    # -----------------------------
    print("\n=== Ridge R² Test ===", ridge.score(X_test, y_test))
    print("=== Lasso R² Test ===", lasso.score(X_test, y_test))

    return ridge, lasso

# ============================================================
# MAIN MODELING FUNCTION
# ============================================================
def run_statistical_models(df):

    # Output directory
    output_dir = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\Analytic_Model_results_V1.3"
    os.makedirs(output_dir, exist_ok=True)

    print("\nBeginning GLM and ANCOVA modeling...")

    # Ensure categorical variables
    categorical_vars = [
        "workload_type", "cpu_name", "ff",
        "system_manufacturer", "nand_type"
    ]

    for col in categorical_vars:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # ============================================================
    # GLM COUNT MODEL: POWER CYCLES
    # ============================================================
    if "power_cycles" in df.columns:

        glm_formula = """
            power_cycles ~ workload_type + data_units_written +
            composite_temperature_c + percentage_used +
            cpu_name + ff + system_manufacturer
        """

        glm_model = smf.glm(
            formula=glm_formula,
            data=df,
            family=sm.families.Poisson()
        ).fit()

        # Save summary
        with open(os.path.join(output_dir, "GLM_Poisson_summary.txt"), "w") as f:
            f.write(glm_model.summary().as_text())

        print("\nSaved Poisson GLM summary.")

        # Diagnostics using your existing plot defs
        resid = glm_model.resid_deviance

        plot_histogram(resid, "GLM_Poisson", "deviance_residuals")
        plot_qq(resid, "GLM_Poisson", "deviance_residuals")
        plot_residuals_vs_fitted(glm_model.fittedvalues, resid, "GLM_Poisson")

        # Overdispersion check
        dispersion = glm_model.pearson_chi2 / glm_model.df_resid
        with open(os.path.join(output_dir, "GLM_dispersion.txt"), "w") as f:
            f.write(f"Dispersion statistic: {dispersion:.3f}\n")

        # If overdispersed → Negative Binomial
        if dispersion > 1.5:
            glm_nb_model = smf.glm(
                formula=glm_formula,
                data=df,
                family=sm.families.NegativeBinomial()
            ).fit()

            with open(os.path.join(output_dir, "GLM_NegBin_summary.txt"), "w") as f:
                f.write(glm_nb_model.summary().as_text())

            # Diagnostics
            nb_resid = glm_nb_model.resid_deviance

            plot_histogram(nb_resid, "GLM_NegBin", "deviance_residuals")
            plot_qq(nb_resid, "GLM_NegBin", "deviance_residuals")
            plot_residuals_vs_fitted(glm_nb_model.fittedvalues, nb_resid, "GLM_NegBin")

            print("\nSaved Negative Binomial GLM summary and plots.")

    # ============================================================
    # ANCOVA: LATENCY
    # ============================================================
    latency_col = next((c for c in ["io_completion_time_ms", "latency"] if c in df.columns), None)

    if latency_col:

        ancova_formula = f"""
            {latency_col} ~ C(cpu_name) + C(ff) + C(system_manufacturer)
            + data_units_written + composite_temperature_c + power_on_hours
        """

        ancova_model = smf.ols(ancova_formula, data=df).fit()

        # Save summary
        with open(os.path.join(output_dir, "ANCOVA_summary.txt"), "w") as f:
            f.write(ancova_model.summary().as_text())

        print("\nSaved ANCOVA summary.")

        # Diagnostics
        plot_histogram(ancova_model.resid, "ANCOVA", "residuals")
        plot_qq(ancova_model.resid, "ANCOVA", "residuals")
        plot_residuals_vs_fitted(ancova_model.fittedvalues, ancova_model.resid, "ANCOVA")

        # Boxplots for categorical factors
        for g in ["cpu_name", "ff", "system_manufacturer"]:
            if g in df.columns:
                plot_boxplot(
                    df, x=g, y=latency_col,
                    model_name="ANCOVA",
                    title=f"{latency_col} by {g}",
                    #output_dir=output_dir
                )

        # Optional: parallel slopes (if you want)
        # plot_parallel_slopes(df, x="data_units_written", y=latency_col, group="cpu_name",
        #                      model_name="ANCOVA"=output_dir)

        print("\nSaved ANCOVA diagnostic plots and boxplots.")

    print("\nModeling complete. All results saved.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    df = load_data()

    #latency_model(df)
    #wear_model(df)
    #reliability_model(df)
    #performance_model(df)

    #one_way_anova(df)
    #two_way_anova_nand(df)
    #two_way_anova_ff(df)

    #regularized_models(df)
    
    print("\n=== ALL ANALYTIC MODELS COMPLETED SUCCESSFULLY ===")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nSmart SSD utilization metric computed (host_read_cmds_per_power_cycle)") 
    run_statistical_models(df) 
    print("\nPipeline complete.")