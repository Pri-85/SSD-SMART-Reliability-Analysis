# ---------------------------------------------------------
# FIX: Disable PyArrow so pandas doesn't try to import it
# ---------------------------------------------------------
import os
os.environ["PANDAS_IGNORE_ARROW"] = "1"

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Create a dataframe for recommendations
# ---------------------------------------------------------
data = {
    "recommendation": [
        "Source/Check 3rd Party Indicators",
        "Enhance Automation & Workflow",
        "Minor Hygiene Improvements",
        "Expand Data Coverage & Infrastructure"
    ],
    "impact": [2, 2, 1, 1],   # High = 2, Low = 1
    "effort": [1, 2, 1, 2]    # High = 2, Low = 1
}

df_matrix = pd.DataFrame(data)

print("\nImpact vs Effort Data:")
print(df_matrix)

# ---------------------------------------------------------
# Plot the Impact vs Effort Matrix
# ---------------------------------------------------------
plt.figure(figsize=(12, 10))

# Quadrant shading for visual clarity
plt.axhspan(1.5, 2.5, 0, 0.5, facecolor='#d4f4dd', alpha=0.4)  # High Impact, Low Effort
plt.axhspan(1.5, 2.5, 0.5, 1.0, facecolor='#dce6ff', alpha=0.4) # High Impact, High Effort
plt.axhspan(0.5, 1.5, 0, 0.5, facecolor='#f9f9c5', alpha=0.4)  # Low Impact, Low Effort
plt.axhspan(0.5, 1.5, 0.5, 1.0, facecolor='#ffe0cc', alpha=0.4) # Low Impact, High Effort

# Draw quadrant lines
plt.axhline(1.5, color='gray', linestyle='--', linewidth=1)
plt.axvline(1.5, color='gray', linestyle='--', linewidth=1)

# Plot each recommendation
for _, row in df_matrix.iterrows():
    plt.scatter(row['effort'], row['impact'], s=300, color='royalblue')

    # Smart label offset to avoid overlap
    plt.text(
        row['effort'] + 0.05,
        row['impact'] + 0.05,
        row['recommendation'],
        fontsize=11,
        weight='bold'
    )

# Axis labels
plt.xticks([1, 2], ["Low Effort", "High Effort"], fontsize=12)
plt.yticks([1, 2], ["Low Impact", "High Impact"], fontsize=12)

plt.xlabel("Effort", fontsize=14)
plt.ylabel("Impact", fontsize=14)
plt.title("Impact vs. Effort Matrix", fontsize=18, weight='bold')

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
