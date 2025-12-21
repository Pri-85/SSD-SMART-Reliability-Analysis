"""
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

Author: Priya Pooja Hariharan (MIS581_SSD_SMART Data_Reliability_Capstone_Research)
"""
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import statsmodels.api as sm 
import statsmodels.formula.api as smf
from statsmodels.graphics.factorplots import interaction_plot 
from sklearn.linear_model import Ridge, Lasso 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
# ------------------------------------------------------------ 
# GLOBAL PLOT DIRECTORY 
# ------------------------------------------------------------ 
#BASE_PLOT_DIR = "plots" os.makedirs(BASE_PLOT_DIR, exist_ok=True) 
BASE_PLOT_DIR = "plots"
os.makedirs(BASE_PLOT_DIR, exist_ok=True)

def save_plot(model_name, plot_name): 
     """ 
     Save the current matplotlib figure into: plots/<model_name>/<plot_name>.png 
     """ 
     folder = os.path.join(BASE_PLOT_DIR, model_name) 
     os.makedirs(folder, exist_ok=True) 
     filepath = os.path.join(folder, f"{plot_name}.png") 
     plt.savefig(filepath, dpi=300, bbox_inches="tight") 
     plt.close()

# ---------------------------------------------------------------------------
# 0. HELPER: LOAD DATA AND FIX COLUMN NAMES
# ---------------------------------------------------------------------------

def load_data():
    """
    Load the cleaned SSD dataset and rename unsafe columns
    so they can be parsed by Patsy formulas.
    """
    #csv_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\EDA_analysis_results_V1.1\processed_smart_dataset.csv"
    csv_path = r"C:\Users\venki\SSD-SMART-Reliability-Analysis\TestResults\EDA_analysis_results_V1.1\synthetic_smart_data.csv"
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
    """
    Predict values.
    Formula: host_read_commands ~ data_units_written
    
    Includes:
    - Histogram of host_read_commands
    - Q–Q normality plot
    - Scatterplot with regression line
    - Residual vs fitted plot
    - Residual distribution plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm

    # REQUIRED for saving plots
    model_name = "simple_linear_regression"


    # Fit model
    model = smf.ols("host_read_commands ~ data_units_written", data=df).fit()
    print("\n=== Simple Linear Regression: host_read_commands ~ data_units_written ===")
    print(model.summary())

    # -----------------------------
    # 1. Histogram + KDE
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["host_read_commands"], kde=True, bins=30, color="steelblue")
    plt.title("Histogram of host_read_commands")
    plt.xlabel("host_read_commands")
    plt.tight_layout()
    save_plot(model_name, "histogram")

    # -----------------------------
    # 2. Q–Q Plot
    # -----------------------------
    plt.figure(figsize=(6, 6))
    sm.qqplot(df["host_read_commands"].dropna(), line="45", fit=True)
    plt.title("Q–Q Plot of host_read_commands")
    plt.tight_layout()
    save_plot(model_name, "qqplot")

    # -----------------------------
    # 3. Scatterplot with regression line
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.regplot(
        x=df["data_units_written"],
        y=df["host_read_commands"],
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"}
    )
    plt.title("host_read_commands vs data_units_written")
    plt.xlabel("data_units_written")
    plt.ylabel("host_read_commands")
    plt.tight_layout()
    save_plot(model_name, "scatterplot")

    # -----------------------------
    # 4. Residual vs Fitted
    # -----------------------------
    fitted = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residuals_vs_fitted")

    # -----------------------------
    # 5. Residual distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30, color="purple")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residual_distribution")

    return model
# ---------------------------------------------------------------------------
# 2. MULTIPLE LINEAR REGRESSION
# ---------------------------------------------------------------------------

def multiple_linear_regression(df):
    """
    Predict write latency using multiple predictors.
    Formula:
        data_units_written ~ data_units_written + wear_level_max + gc_active_time_pct

    Includes:
    - Histogram of data_units_written
    - Q–Q normality plot
    - Scatterplots for each predictor
    - Residual vs fitted plot
    - Residual distribution plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm

    # REQUIRED for saving plots
    model_name = "multiple_linear_regression"

    # Fit model
    model = smf.ols(
        "data_units_written ~ data_units_written + wear_level_max + gc_active_time_pct",
        data=df
    ).fit()

    print("\n=== Multiple Linear Regression: data_units_written with multiple predictors ===")
    print(model.summary())

    # -----------------------------
    # 1. Histogram + KDE
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["data_units_written"], kde=True, bins=30, color="darkgreen")
    plt.title("Histogram of data_units_written")
    plt.xlabel("data_units_written")
    plt.tight_layout()
    save_plot(model_name, "histogram")

    # -----------------------------
    # 2. Q–Q Plot
    # -----------------------------
    plt.figure(figsize=(6, 6))
    sm.qqplot(df["data_units_written"].dropna(), line="45", fit=True)
    plt.title("Q–Q Plot of data_units_written")
    plt.tight_layout()
    save_plot(model_name, "qqplot")

    # -----------------------------
    # 3. Scatterplots for each predictor
    # -----------------------------
    predictors = ["data_units_written", "data_units_written", "wear_level_max", "gc_active_time_pct"]

    for col in predictors:
        plt.figure(figsize=(8, 5))
        sns.regplot(
            x=df[col],
            y=df["data_units_written"],
            scatter_kws={"alpha": 0.4},
            line_kws={"color": "red"}
        )
        plt.title(f"data_units_written vs {col}")
        plt.xlabel(col)
        plt.ylabel("data_units_written")
        plt.tight_layout()
        save_plot(model_name, f"scatter_{col}")

    # -----------------------------
    # 4. Residual vs Fitted
    # -----------------------------
    fitted = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residuals_vs_fitted")

    # -----------------------------
    # 5. Residual distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30, color="purple")
    plt.title("Residual Distribution")
    plt.xlabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residual_distribution")

    return model


# ---------------------------------------------------------------------------
# 3. ONE-WAY ANOVA
# ---------------------------------------------------------------------------
def one_way_anova(df):
    """
    Test whether write latency differs by system manufacturer.
    Formula: wear_level_avg ~ form_factor

    Includes:
    - Boxplot of wear_level_avg by form_factor
    - Violin plot
    - Histogram of wear_level_avg
    - Q–Q plot of residuals
    - Residual vs fitted plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf  # ✅ Added missing import

    # REQUIRED for saving plots
    model_name = "one_way_anova"


    # Fit ANOVA model
    model = smf.ols("wear_level_avg ~ C(form_factor)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== One-Way ANOVA: wear_level_avg ~ form_factor ===")
    print(anova_table)

    # -----------------------------
    # 1. Boxplot by manufacturer
    # -----------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="form_factor", y="wear_level_avg", data=df)
    plt.title("wear_level_avg by Model Number")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(model_name, "boxplot")

    # -----------------------------
    # 2. Violin plot
    # -----------------------------
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="form_factor", y="wear_level_avg", data=df)
    plt.title("wear_level_avg Distribution by Model Number")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(model_name, "violin_plot")

    # -----------------------------
    # 3. Histogram of wear_level_avg
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["wear_level_avg"], kde=True, bins=30, color="darkblue")
    plt.title("Histogram of wear_level_avg")
    plt.xlabel("wear_level_avg")
    plt.tight_layout()
    save_plot(model_name, "histogram")

    # -----------------------------
    # 4. Q–Q plot of residuals
    # -----------------------------
    plt.figure(figsize=(6, 6))
    sm.qqplot(model.resid.dropna(), line="45", fit=True)
    plt.title("Q–Q Plot of ANOVA Residuals")
    plt.tight_layout()
    save_plot(model_name, "qqplot")

    # -----------------------------
    # 5. Residual vs fitted
    # -----------------------------
    fitted = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted (ANOVA)")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residuals_vs_fitted")

# ---------------------------------------------------------------------------
# 4. TWO-WAY (FACTORIAL) ANOVA
# ---------------------------------------------------------------------------

def two_way_anova(df):
    """
    Test interaction between manufacturer and avg_queue_depth on write latency.
    Formula: pcie_correctable_errors ~ form_factor * avg_queue_depth

    Includes:
    - Boxplot by manufacturer
    - Boxplot by avg_queue_depth
    - Interaction plot
    - Histogram of pcie_correctable_errors
    - Q–Q plot of residuals
    - Residual vs fitted plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.graphics.factorplots import interaction_plot

    # REQUIRED for saving plots
    model_name = "two_way_anova"

    # Fit model
    df["avg_qd_bin"] = pd.qcut(df["avg_queue_depth"], q=4, labels=["Q1","Q2","Q3","Q4"])
    model = smf.ols(
        "pcie_correctable_errors ~ C(form_factor) * avg_qd_bin",
        data=df
    ).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== Two-Way ANOVA: pcie_correctable_errors ~ Nand type * avg_qd_bin ===")
    print(anova_table)

    # -----------------------------
    # 1. Boxplot by Nand rype
    # -----------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="form_factor", y="pcie_correctable_errors", data=df)
    plt.title("PCIe correctible errors by Nand Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(model_name, "boxplot_nand type")

    # -----------------------------
    # 2. Boxplot by avg_queue_depth
    # -----------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="avg_qd_bin", y="pcie_correctable_errors", data=df)
    plt.title("Average QD by Capacity SKU")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(model_name, "boxplot_avg_queue_depth")

    # -----------------------------
    # 3. Interaction Plot 
    # ----------------------------- 
    manufacturers = df["form_factor"].unique() 
    num_levels = len(manufacturers) 
    # Generate markers automatically 
    default_markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"] 
    markers = default_markers[:num_levels]
    plt.figure(figsize=(10, 6))
    interaction_plot(
    df["avg_queue_depth"],
    df["form_factor"],
    df["pcie_correctable_errors"],
    #colors=["red", "blue", "green", "purple", "orange"],
    #markers=["o", "s", "D", "^", "v"],   # <-- still hard‑coded
    markers=None,
    ms=8)

    plt.title("Interaction Plot: Nand type × Capacity SKU")
    plt.xlabel("Capacity SKU")
    plt.ylabel("Nand Type")
    plt.tight_layout()
    save_plot(model_name, "interaction_plot")

    # -----------------------------
    # 4. Histogram of pcie_correctable_errors
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["pcie_correctable_errors"], kde=True, bins=30, color="darkblue")
    plt.title("Histogram of pcie_correctable_errors")
    plt.xlabel("pcie_correctable_errors")
    plt.tight_layout()
    save_plot(model_name, "histogram")

    # -----------------------------
    # 5. Q–Q plot of residuals
    # -----------------------------
    plt.figure(figsize=(6, 6))
    sm.qqplot(model.resid.dropna(), line="45", fit=True)
    plt.title("Q–Q Plot of ANOVA Residuals")
    plt.tight_layout()
    save_plot(model_name, "qqplot")

    # -----------------------------
    # 6. Residual vs fitted
    # -----------------------------
    fitted = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted (Two-Way ANOVA)")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residuals_vs_fitted")

    return anova_table


# ---------------------------------------------------------------------------
# 5. ANCOVA (Analysis of Covariance)
# ---------------------------------------------------------------------------

def ancova(df):
    """
    Compare manufacturers on read latency while adjusting for data_units_written.
    Formula:
        host_read_commands ~ form_factor + data_units_written

    Includes:
    - Boxplot of host_read_commands by manufacturer
    - Scatterplot of data_units_read vs data_units_written (colored by manufacturer)
    - Parallel slopes diagnostic plot
    - Histogram of data_units_read
    - Q–Q plot of residuals
    - Residual vs fitted plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm

    # REQUIRED for saving plots
    model_name = "ancova"

    # Fit ANCOVA model
    model = smf.ols(
        "data_units_written ~ C(form_factor) + data_units_read",
        data=df
    ).fit()

    print("\n=== ANCOVA: data_units_read ~ form_factor + data_units_written ===")
    print(model.summary())

    # -----------------------------
    # 1. Boxplot by manufacturer
    # -----------------------------
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="form_factor", y="data_units_read", data=df)
    plt.title("Read Latency by System Manufacturer")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(model_name, "boxplot_manufacturer")

    # -----------------------------
    # 2. Scatterplot with regression lines per manufacturer
    # -----------------------------
    sns.lmplot(
        x="data_units_written",
        y="data_units_read",
        hue="form_factor",
        data=df,
        height=6,
        aspect=1.3,
        scatter_kws={"alpha": 0.4}
    )
    plt.title("Read Latency vs Power-On Hours by Manufacturer")
    plt.tight_layout()
    save_plot(model_name, "scatterplot_by_manufacturer")

    # -----------------------------
    # 3. Parallel slopes diagnostic
    # -----------------------------
    plt.figure(figsize=(10, 6))
    for m in df["form_factor"].unique():
        subset = df[df["form_factor"] == m]
        sns.regplot(
            x=subset["data_units_written"],
            y=subset["data_units_read"],
            label=m,
            scatter_kws={"alpha": 0.3},
            line_kws={"linewidth": 2}
        )
    plt.title("Parallel Slopes Check (ANCOVA Assumption)")
    plt.xlabel("data_units_written")
    plt.ylabel("data_units_read")
    plt.legend(title="Manufacturer")
    plt.tight_layout()
    save_plot(model_name, "parallel_slopes")

    # -----------------------------
    # 4. Histogram of data_units_read
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["data_units_read"], kde=True, bins=30, color="darkblue")
    plt.title("Histogram of data_units_read")
    plt.xlabel("data_units_read")
    plt.tight_layout()
    save_plot(model_name, "histogram")

    # -----------------------------
    # 5. Q–Q plot of residuals
    # -----------------------------
    plt.figure(figsize=(6, 6))
    sm.qqplot(model.resid.dropna(), line="45", fit=True)
    plt.title("Q–Q Plot of ANCOVA Residuals")
    plt.tight_layout()
    save_plot(model_name, "qqplot")

    # -----------------------------
    # 6. Residual vs fitted
    # -----------------------------
    fitted = model.fittedvalues
    residuals = model.resid

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted (ANCOVA)")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residuals_vs_fitted")

    return model

# ---------------------------------------------------------------------------
# 6. REGULARIZED LINEAR MODELS (Ridge, Lasso)
# ---------------------------------------------------------------------------
def regularized_models(df):
    """
    Predict data_units_written using all numeric predictors with regularization.
    Handles NaN, inf, constant columns, and scaling issues safely.

    Includes:
    - Histogram of data_units_written
    - Correlation heatmap
    - Ridge & Lasso coefficient plots
    - Residual distribution
    - Residual vs fitted plot
    - Predicted vs actual scatterplot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import numpy as np

    # REQUIRED for saving plots
    model_name = "regularized_models"

    target = "iops"

    # Select numeric columns
    numeric_df = df.select_dtypes(include="number")

    # Replace inf/-inf with NaN
    numeric_df = numeric_df.replace([float("inf"), float("-inf")], pd.NA)

    # Impute missing values with medians
    numeric_df = numeric_df.fillna(numeric_df.median())

    # Remove constant columns
    constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() <= 1]
    if constant_cols:
        print("\n[INFO] Dropping constant columns:", constant_cols)
        numeric_df = numeric_df.drop(columns=constant_cols)

    # Remove extremely low-variance columns
    low_variance_cols = numeric_df.columns[numeric_df.var() < 1e-6].tolist()
    if low_variance_cols:
        print("\n[INFO] Dropping low-variance columns:", low_variance_cols)
        numeric_df = numeric_df.drop(columns=low_variance_cols)

    # Ensure target exists
    if target not in numeric_df.columns:
        print(f"[WARN] {target} not found in numeric columns.")
        return None, None

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    # -----------------------------
    # 1. Histogram of target
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(y, kde=True, bins=30, color="darkblue")
    plt.title("Histogram of data_units_written")
    plt.xlabel("data_units_written")
    plt.tight_layout()
    save_plot(model_name, "histogram")

    # -----------------------------
    # 2. Correlation heatmap
    # -----------------------------
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Predictors")
    plt.tight_layout()
    save_plot(model_name, "correlation_heatmap")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check for NaN after scaling
    if pd.isna(X_scaled).any():
        print("[ERROR] NaN detected after scaling. Some columns may still be invalid.")
        return None, None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Fit models
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    lasso = Lasso(alpha=0.001).fit(X_train, y_train)

    # -----------------------------
    # 3. Coefficient plots
    # -----------------------------
    coef_df = pd.DataFrame({
        "feature": X.columns,
        "ridge_coef": ridge.coef_,
        "lasso_coef": lasso.coef_
    })

    plt.figure(figsize=(12, 6))
    sns.barplot(x="ridge_coef", y="feature", data=coef_df.sort_values("ridge_coef"))
    plt.title("Ridge Coefficients")
    plt.tight_layout()
    save_plot(model_name, "ridge_coefficients")

    plt.figure(figsize=(12, 6))
    sns.barplot(x="lasso_coef", y="feature", data=coef_df.sort_values("lasso_coef"))
    plt.title("Lasso Coefficients")
    plt.tight_layout()
    save_plot(model_name, "lasso_coefficients")

    # -----------------------------
    # 4. Predictions & residuals
    # -----------------------------
    y_pred = ridge.predict(X_test)
    residuals = y_test - y_pred

    # Residual distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30, color="purple")
    plt.title("Residual Distribution (Ridge)")
    plt.xlabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residual_distribution")

    # Residual vs fitted
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted (Ridge)")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residuals_vs_fitted")

    # Predicted vs actual
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title("Predicted vs Actual (Ridge)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    save_plot(model_name, "predicted_vs_actual")

    # -----------------------------
    # Print model performance
    # -----------------------------
    print("\n=== Ridge Regression Coefficients ===")
    print(ridge.coef_)

    print("\n=== Lasso Regression Coefficients ===")
    print(lasso.coef_)

    print(f"\nRidge R^2 (train): {ridge.score(X_train, y_train):.4f}")
    print(f"Ridge R^2 (test):  {ridge.score(X_test, y_test):.4f}")
    print(f"Lasso R^2 (train): {lasso.score(X_train, y_train):.4f}")
    print(f"Lasso R^2 (test):  {lasso.score(X_test, y_test):.4f}")

    return ridge, lasso

# ---------------------------------------------------------------------------
# 7. GENERALIZED LINEAR MODELS (GLM) FOR COUNT DATA
# ---------------------------------------------------------------------------
def glm_count_model(df):
    """
    Predict bandwidth_write_gbps (count data) using Poisson regression.
    Formula:
        bandwidth_write_gbps ~ iops + percentage_used + workload_type

    Includes:
    - Histogram of bandwidth_write_gbps
    - Scatterplots of predictors vs bandwidth_write_gbps
    - Predicted vs actual plot
    - Residual distribution
    - Residual vs fitted plot
    - Q–Q plot of deviance residuals
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    import numpy as np

    # REQUIRED for saving plots
    model_name = "glm_count_model"

    # Fit Poisson GLM
    model = smf.glm(
        "bandwidth_write_gbps ~ iops + percentage_used + workload_type",
        data=df,
        family=sm.families.Poisson()
    ).fit()

    print("\n=== GLM Poisson: bandwidth_write_gbps ===")
    print(model.summary())

    # -----------------------------
    # 1. Histogram of count variable
    # -----------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["bandwidth_write_gbps"], bins=30, kde=False, color="darkred")
    plt.title("Histogram of bandwidth_write_gbps")
    plt.xlabel("bandwidth_write_gbps")
    plt.tight_layout()
    save_plot(model_name, "histogram")

    # -----------------------------
    # 2. Scatterplots of predictors
    # -----------------------------
    predictors = ["iops", "percentage_used", "workload_type"]

    for col in predictors:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=df[col], y=df["bandwidth_write_gbps"], alpha=0.5)
        plt.title(f"bandwidth_write_gbps vs {col}")
        plt.xlabel(col)
        plt.ylabel("bandwidth_write_gbps")
        plt.tight_layout()
        save_plot(model_name, f"scatter_{col}")

    # -----------------------------
    # 3. Predicted vs actual
    # -----------------------------
    y_pred = model.predict(df)
    y_true = df["bandwidth_write_gbps"]

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--")
    plt.title("Predicted vs Actual (Poisson GLM)")
    plt.xlabel("Actual bandwidth_write_gbps")
    plt.ylabel("Predicted bandwidth_write_gbps")
    plt.tight_layout()
    save_plot(model_name, "predicted_vs_actual")

    # -----------------------------
    # 4. Residual distribution
    # -----------------------------
    residuals = model.resid_deviance

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30, color="purple")
    plt.title("Residual Distribution (Deviance Residuals)")
    plt.xlabel("Residuals")
    plt.tight_layout()
    save_plot(model_name, "residual_distribution")

    # -----------------------------
    # 5. Residual vs fitted
    # -----------------------------
    fitted = model.fittedvalues

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Fitted (Poisson GLM)")
    plt.xlabel("Fitted Values")
    plt.ylabel("Deviance Residuals")
    plt.tight_layout()
    save_plot(model_name, "residuals_vs_fitted")

    # -----------------------------
    # 6. Q–Q plot of deviance residuals
    # -----------------------------
    plt.figure(figsize=(6, 6))
    sm.qqplot(residuals, line="45", fit=True)
    plt.title("Q–Q Plot of Deviance Residuals")
    plt.tight_layout()
    save_plot(model_name, "qqplot")

    return model


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data()

    # Run models sequentially
    simple_linear_regression(df)
    multiple_linear_regression(df)
    one_way_anova(df)
    two_way_anova(df)
    ancova(df)
    regularized_models(df)
    glm_count_model(df)
