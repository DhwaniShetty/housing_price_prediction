# 🏠 Housing Price Prediction

## 📌 Problem
Predict median house values in California using features such as:
- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

## 📊 Dataset
California Housing dataset from scikit-learn:
[`fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

## ⚙️ Techniques Used
- Linear Regression (baseline)
- Random Forest Regressor (ensemble)
- XGBoost (boosting)

## 🛠️ Skills Covered
- Data Cleaning & Preprocessing
- Feature Scaling
- Regression Metrics (RMSE, R²)
- Hyperparameter Tuning (GridSearchCV)
- Cross-Validation
- Feature Importance Analysis

## 📈 Results
| Model             | RMSE  | R²   |
|-------------------|-------|------|
| Linear Regression | 0.746 | 0.576 |
| Random Forest     | 0.506 | 0.805 |
| XGBoost           | 0.456 | 0.841 |

✅ **XGBoost performed best**, showing that boosting methods capture complex relationships better than linear or ensemble models.

## 🔍 Insights
- **Median Income** is the strongest predictor of house prices.
- **Latitude & Longitude** (location) also play a major role.
- Ensemble and boosting methods significantly outperform linear regression.

## 📊 Visualizations
- Predictions vs Actuals scatter plot
- Feature importance bar charts (Random Forest & XGBoost)

## 🚀 Conclusion
This project demonstrates an end-to-end ML workflow:
- Data preprocessing
- Model training
- Evaluation & comparison
- Insights & visualization

It’s a strong showcase of regression modeling and can be extended with:
- More advanced hyperparameter tuning
- Additional feature engineering
- Deployment as a web app
