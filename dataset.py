# dataset.py

from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseValue'] = housing.target

print("Dataset shape:", df.shape)
print(df.head())

# 2. Split features and target
X = df.drop('MedHouseValue', axis=1)
y = df['MedHouseValue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature scaling (important for Linear Regression & XGBoost)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train models
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 5. Evaluate models
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} → RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return rmse, r2

results = []
results.append(["Linear Regression", *evaluate_model("Linear Regression", y_test, y_pred_lin)])
results.append(["Random Forest", *evaluate_model("Random Forest", y_test, y_pred_rf)])
results.append(["XGBoost", *evaluate_model("XGBoost", y_test, y_pred_xgb)])

# 6. Visualization
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_rf, alpha=0.3, label="Random Forest")
plt.scatter(y_test, y_pred_xgb, alpha=0.3, label="XGBoost", color="red")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Predictions vs Actuals")
plt.legend()
plt.show()

# 7. Feature Importance
rf_importances = rf.feature_importances_
sorted_idx = np.argsort(rf_importances)[::-1]
plt.figure(figsize=(8,5))
plt.barh([X.columns[i] for i in sorted_idx], rf_importances[sorted_idx])
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()

xgb_importances = xgb.feature_importances_
sorted_idx_xgb = np.argsort(xgb_importances)[::-1]
plt.figure(figsize=(8,5))
plt.barh([X.columns[i] for i in sorted_idx_xgb], xgb_importances[sorted_idx_xgb], color="red")
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# 8. Comparison Table
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R²"])
print("\nModel Comparison:\n", results_df)

# 9. Cross-validation
cv_scores_rf = cross_val_score(RandomForestRegressor(random_state=42), X, y, cv=5, scoring='r2')
print("\nRandom Forest CV R² scores:", cv_scores_rf)
print("Average R²:", cv_scores_rf.mean())

cv_scores_xgb = cross_val_score(XGBRegressor(random_state=42), X, y, cv=5, scoring='r2')
print("\nXGBoost CV R² scores:", cv_scores_xgb)
print("Average R²:", cv_scores_xgb.mean())
