# ============================================
# Top Predictive Features Bar Chart (Python)
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# 1. Load or create sample data
# ------------------------------------------------
# Replace this with your real dataset
np.random.seed(42)
df = pd.DataFrame({
    "wear_leveling": np.random.rand(100),
    "read_latency": np.random.rand(100),
    "write_amplification": np.random.rand(100),
    "temperature": np.random.rand(100),
    "power_cycles": np.random.rand(100),
    "target": np.random.randint(0, 2, 100)
})

X = df.drop("target", axis=1)
y = df["target"]

# ------------------------------------------------
# 2. Train a model (Random Forest example)
# ------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# ------------------------------------------------
# 3. Extract feature importance
# ------------------------------------------------
importances = model.feature_importances_
feature_names = X.columns

df_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

# ------------------------------------------------
# 4. Plot Top Predictive Features (Horizontal Bar Chart)
# ------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_imp,
    x="importance",
    y="feature",
    palette="viridis"
)

plt.title("Top Predictive Features", fontsize=16)
plt.xlabel("Feature Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.show()
