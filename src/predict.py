import pandas as pd
import joblib
from datetime import timedelta
import numpy as np

model=joblib.load("models/xgb_model.pkl")
df=pd.read_csv("data/features.csv")

latest_data = df.iloc[[-1]]

X_latest = latest_data.drop(columns=["Global_active_power", "Datetime"])

print(X_latest.columns)
print(X_latest.shape)
prediction = model.predict(X_latest)
latest_time = latest_data["Datetime"].iloc[0]
next_hour = pd.to_datetime(latest_time) + timedelta(hours=1)

print("Latest available time:", latest_time)
print("Prediction time:", next_hour)
print("Predicted next-hour consumption:", prediction[0], "kW")

history = df["Global_active_power"].tolist()

feature_columns = X_latest.columns.tolist()

future_predictions = []

for i in range(24):

    future_time = next_hour + timedelta(hours=i)

    future_row = {
        "Global_reactive_power": latest_data["Global_reactive_power"].iloc[0],
        "Voltage": latest_data["Voltage"].iloc[0],
        "Global_intensity": latest_data["Global_intensity"].iloc[0],
        "Sub_metering_1": latest_data["Sub_metering_1"].iloc[0],
        "Sub_metering_2": latest_data["Sub_metering_2"].iloc[0],
        "Sub_metering_3": latest_data["Sub_metering_3"].iloc[0],

        "Lag_1": history[-1],
        "Lag_24": history[-24],
        "Lag_168": history[-168],
        "Lag_2": history[-2],
        "Lag_3": history[-3],
        "Lag_48": history[-48],

        "Rolling_Mean_3": sum(history[-3:]) / 3,
        "Rolling_Mean_6": sum(history[-6:]) / 6,
        "Rolling_Mean_24": sum(history[-24:]) / 24,

        "Hour": future_time.hour,
        "Day_of_Week": future_time.dayofweek,
        "Month": future_time.month,
        "Is_Weekend": int(future_time.dayofweek in [5, 6]),

        "Hour_Sin": np.sin(2 * np.pi * future_time.hour / 24),
        "Hour_Cos": np.cos(2 * np.pi * future_time.hour / 24),
        "DOW_Sin": np.sin(2 * np.pi * future_time.dayofweek / 7),
        "DOW_Cos": np.cos(2 * np.pi * future_time.dayofweek / 7),

        "Diff_1": history[-1] - history[-2],
        "Diff_24": history[-24] - history[-48]
    }

    future_X = pd.DataFrame([future_row])[feature_columns]

    prediction = model.predict(future_X)[0]

    history.append(prediction)

    future_predictions.append({
        "Datetime": future_time,
        "Predicted_Consumption": prediction
    })

print("\nNext 24-hour forecast:")

for row in future_predictions:
    print(row["Datetime"], "->", round(row["Predicted_Consumption"], 3), "kW")

forecast_df = pd.DataFrame(future_predictions)

forecast_df.to_csv("data/forecast_24h.csv", index=False)

print("\n24-hour forecast saved successfully.")