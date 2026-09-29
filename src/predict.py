import pandas as pd
import joblib

model=joblib.load("models/xgb_model.pkl")
df=pd.read_csv("data/features.csv")

latest_data = df.iloc[[-1]]

X_latest = latest_data.drop(columns=["Global_active_power", "Datetime"])

print(X_latest.columns)
print(X_latest.shape)
model.predict(X_latest)