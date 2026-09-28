import matplotlib.pyplot as plt
import pandas as pd

hourly_clean = pd.read_csv("data/hourly_clean.csv")
hourly_clean["Datetime"] = pd.to_datetime(hourly_clean["Datetime"])
hourly_clean.index = hourly_clean["Datetime"]
hourly_clean["Hour"] = hourly_clean["Datetime"].dt.hour
hourly_clean.groupby("Hour")["Global_active_power"].mean()
hourly_clean["Day_of_Week"] = hourly_clean["Datetime"].dt.dayofweek
hourly_clean["Day_Type"] = hourly_clean["Day_of_Week"].isin([5, 6])
hourly_clean["Day_Type"] = hourly_clean["Day_Type"].map(
    {True: "Weekend", False: "Weekday"}
)

hourly_clean.groupby("Day_Type")["Global_active_power"].mean()
hourly_clean["Month"] = hourly_clean["Datetime"].dt.month

hourly_clean.groupby("Month")["Global_active_power"].mean()
hourly_clean["Lag_1"] = hourly_clean["Global_active_power"].shift(1)
hourly_clean[["Global_active_power", "Lag_1"]].head(10)
print(hourly_clean["Global_active_power"].corr(hourly_clean["Lag_1"]))
hourly_clean["Lag_24"] = hourly_clean["Global_active_power"].shift(24)
hourly_clean[["Global_active_power", "Lag_24"]].head(10)
print(hourly_clean["Global_active_power"].corr(hourly_clean["Lag_24"]))
hourly_clean["Lag_168"] = hourly_clean["Global_active_power"].shift(168)
hourly_clean[["Global_active_power", "Lag_168"]].head(10)
print(hourly_clean["Global_active_power"].corr(hourly_clean["Lag_168"]))
