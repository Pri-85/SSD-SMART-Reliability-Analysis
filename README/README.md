# SSD-SMART-Reliability-Analysis
Capstone project for SSD benchmarking and reliability analysis using SMART data.

Capstone Project Checklist (Aligned to Your Repo)
Data Acquisition & Preparation
Collect raw SMART telemetry
Validate schema, missing values, and vendor inconsistencies
Run pre-process_V1.1
Generate cleaned dataset → cleaned_SSD_dataset.xlsx
Normalize cross‑vendor attributes → Hyp1_cross_vendor_normalization_experiment_V1.1
Save processed dataset → Step1-processed_smart_dataset_V1.1.xlsx
B. Synthetic Data & Injection
Run SSD_DataInjectionAndInspection_V1.3
Generate synthetic SMART data → Step3-synthetic_smart_data_V1.1.xlsx
Validate injected anomalies
Document injection logic
C. Exploratory Data Analysis
Run EDA script → EDA_analysis_results.V1.2
Generate distribution plots, correlations, heatmaps
Identify weak signals & outliers
Save results to TestResults/EDA_analysis_results.V1.2/
D. Hypothesis Testing
Hypothesis 1: Cross‑vendor normalization
Hypothesis 2: Model sensitivity
Hypothesis 3: Usage–failure correlation
Hypothesis 4: Bayesian reliability modeling
Hypothesis 5: SVR anomaly detection
Save results to TestResults/hypothesisX_results/
E. Modeling & Evaluation
Run regression models → modeling_regression_V1.1
Run Bayesian model → Hyp4_Bayesian_Model_experiment_V1.1
Run SVR anomaly detection → Hyp5_SVR_Anomaly_Detection_Pipeline_V1.1
Evaluate using RMSE, MAE, ROC/AUC, posterior diagnostics
Save results to Analytic_Model_results_V1.1/
F. Logging & Reproducibility
Generate logs → log_generator_V1.1
Version datasets (V1.1, V1.2…)
Commit scripts + results to GitHub
Tag releases for each experiment
G. Final Deliverables
Final cleaned + synthetic datasets
Final models + evaluation metrics
