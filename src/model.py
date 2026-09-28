import pandas as pd

df = pd.read_csv("data/features.csv")

y = df["Global_active_power"].shift(-1)

print(y.tail())

X = df.drop(columns=["Global_active_power", "Datetime"])

print(X.columns)
print(X.shape)

valid = y.notna()
X = X[valid]
y = y[valid]

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

baseline_pred = X_test["Lag_1"]

print(baseline_pred.head())

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

baseline_pred = X_test["Lag_1"]

baseline_mae = mean_absolute_error(y_test, baseline_pred)

baseline_rmse = mean_squared_error(
    y_test,
    baseline_pred
) ** 0.5

baseline_r2 = r2_score(y_test, baseline_pred)

print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)
print("Baseline R2:", baseline_r2)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

print(linear_pred[:5])

linear_mae = mean_absolute_error(y_test, linear_pred)

linear_rmse = mean_squared_error(
    y_test,
    linear_pred
) ** 0.5

linear_r2 = r2_score(y_test, linear_pred)

print("Linear Regression MAE:", linear_mae)
print("Linear Regression RMSE:", linear_rmse)
print("Linear Regression R2:", linear_r2)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(rf_pred[:5])
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5
rf_r2 = r2_score(y_test, rf_pred)

print("Random Forest MAE:", rf_mae)
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R2:", rf_r2)

from xgboost import XGBRegressor

