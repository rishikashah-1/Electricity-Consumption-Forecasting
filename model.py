import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
 
 
DATA_FILE = "hourly_clean.csv"
 
data = pd.read_csv(DATA_FILE)
 
print("First 5 rows of your data :")
print(data.head())
 
FEATURE_COLUMNS = ["Global_reactive_power", "Voltage", "Global_intensity",
                    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
TARGET_COLUMN = "Global_active_power"
 
X = data[FEATURE_COLUMNS]
y = data[TARGET_COLUMN]
 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
 
# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
 
# XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
 
 
def show_scores(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"\n{name}")
    print(f"  MAE  (average error size)   : {mae:.3f}")
    print(f"  RMSE (punishes big misses)  : {rmse:.3f}")
    return mae, rmse
 
 
print("\n===== MODEL COMPARISON =====")
lr_mae, lr_rmse = show_scores("Linear Regression", y_test, lr_predictions)
rf_mae, rf_rmse = show_scores("Random Forest", y_test, rf_predictions)
xgb_mae, xgb_rmse = show_scores("XGBoost", y_test, xgb_predictions)
 
 
# ---- Results table ----
comparison_table = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "MAE": [lr_mae, rf_mae, xgb_mae],
    "RMSE": [lr_rmse, rf_rmse, xgb_rmse],
})
 
print("\n===== RESULTS TABLE =====")
print(comparison_table.to_string(index=False))
 
comparison_table.to_csv("model_comparison.csv", index=False)
print("Saved table as model_comparison.csv")
 
 
candidates = [
    (lr_rmse, "Linear Regression", lr_model, lr_predictions),
    (rf_rmse, "Random Forest", rf_model, rf_predictions),
    (xgb_rmse, "XGBoost", xgb_model, xgb_predictions),
]
 
best_rmse, best_name, best_model, best_predictions = min(candidates, key=lambda c: c[0])
 
print(f"\nWinner: {best_name} (it had the lowest RMSE: {best_rmse:.3f})")
 
 
joblib.dump(best_model, "trained_model.pkl")
print("Saved the winning model as trained_model.pkl")
 
 
# ---- Bar chart comparing MAE and RMSE across models ----
models = comparison_table["Model"]
x_positions = np.arange(len(models))
bar_width = 0.35
 
plt.figure(figsize=(8, 5))
plt.bar(x_positions - bar_width/2, comparison_table["MAE"], width=bar_width, label="MAE")
plt.bar(x_positions + bar_width/2, comparison_table["RMSE"], width=bar_width, label="RMSE")
plt.xticks(x_positions, models)
plt.ylabel("Error")
plt.title("Model Comparison: MAE and RMSE")
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison_chart.png")
print("Saved bar chart as model_comparison_chart.png")
 
print("\nAll done! Check this folder for:")
print("  - trained_model.pkl (the saved model)")
print("  - model_comparison.csv (results table)")
print("  - model_comparison_chart.png (bar chart)")
 
