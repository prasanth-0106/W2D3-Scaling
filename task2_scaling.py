"""
Week 2 - Day 3 | Task 2: Scaling

Run:
    py -m pip install -r requirements.txt
    py task2_scaling.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


numeric_df = pd.DataFrame({
    "age": [25, 35, 40, 35, 35],
    "income": [30000, 45000, 45000, 80000, 52000],
})

standard_scaler = StandardScaler()
standard_values = standard_scaler.fit_transform(numeric_df)
standard_df = pd.DataFrame(
    standard_values, columns=["age", "income"]
)

print("1. Original age and income:")
print(numeric_df.to_string(index=False))

print("\n2. StandardScaler result:")
print(standard_df.round(4).to_string(index=False))
print("Means (approximately 0):", np.round(standard_df.mean().values, 6))
print("Standard deviations (population, approximately 1):",
      np.round(standard_df.std(ddof=0).values, 6))

minmax_scaler = MinMaxScaler()
minmax_values = minmax_scaler.fit_transform(numeric_df)
minmax_df = pd.DataFrame(minmax_values, columns=["age", "income"])

print("\n3. MinMaxScaler result:")
print(minmax_df.round(4).to_string(index=False))
print("Minimum values:", np.round(minmax_df.min().values, 6))
print("Maximum values:", np.round(minmax_df.max().values, 6))


data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.2,
    random_state=42,
    stratify=data.target,
)


knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
pred_unscaled = knn_unscaled.predict(X_test)
accuracy_unscaled = accuracy_score(y_test, pred_unscaled)


knn_scaler = StandardScaler()
X_train_scaled = knn_scaler.fit_transform(X_train)
X_test_scaled = knn_scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, pred_scaled)

print("\n4. KNeighborsClassifier accuracy on breast cancer dataset:")
print(f"KNN without scaling: {accuracy_unscaled:.2f}")
print(f"KNN with scaling:    {accuracy_scaled:.2f}")


tree_unscaled = DecisionTreeClassifier(random_state=42)
tree_unscaled.fit(X_train, y_train)
tree_pred_unscaled = tree_unscaled.predict(X_test)
tree_accuracy_unscaled = accuracy_score(y_test, tree_pred_unscaled)

tree_scaler = StandardScaler()
X_train_tree_scaled = tree_scaler.fit_transform(X_train)
X_test_tree_scaled = tree_scaler.transform(X_test)

tree_scaled = DecisionTreeClassifier(random_state=42)
tree_scaled.fit(X_train_tree_scaled, y_train)
tree_pred_scaled = tree_scaled.predict(X_test_tree_scaled)
tree_accuracy_scaled = accuracy_score(y_test, tree_pred_scaled)

print("\n5. Decision Tree accuracy:")
print(f"Decision tree without scaling: {tree_accuracy_unscaled:.2f}")
print(f"Decision tree with scaling:    {tree_accuracy_scaled:.2f}")
print("Decision trees are generally unaffected by feature scaling because "
      "their splits depend on feature order/thresholds, not distance.")


outlier_data = pd.DataFrame({"value": [10, 12, 13, 14, 100]})
robust_scaler = RobustScaler()
outlier_data["robust_scaled"] = robust_scaler.fit_transform(outlier_data[["value"]])

print("\n6. Bonus - RobustScaler with an outlier:")
print(outlier_data.to_string(index=False))
