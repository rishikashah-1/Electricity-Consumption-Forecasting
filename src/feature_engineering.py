import pandas as pd
import numpy as np

hourly_clean = pd.read_csv("data/hourly_clean.csv")
hourly_clean["Lag_1"] = hourly_clean["Global_active_power"].shift(1)
hourly_clean["Lag_24"] = hourly_clean["Global_active_power"].shift(24)
hourly_clean["Lag_168"] = hourly_clean["Global_active_power"].shift(168)
hourly_clean["Lag_2"] = hourly_clean["Global_active_power"].shift(2)
hourly_clean["Lag_3"] = hourly_clean["Global_active_power"].shift(3)
hourly_clean["Lag_48"] = hourly_clean["Global_active_power"].shift(48)
hourly_clean["Rolling_Mean_3"] = (
    hourly_clean["Global_active_power"].shift(1).rolling(3).mean()
)
hourly_clean["Rolling_Mean_6"] = (
    hourly_clean["Global_active_power"].shift(1).rolling(6).mean()
)

hourly_clean["Rolling_Mean_24"] = (
    hourly_clean["Global_active_power"].shift(1).rolling(24).mean()
)
hourly_clean["Hour"] = pd.to_datetime(hourly_clean["Datetime"]).dt.hour
hourly_clean["Day_of_Week"] = pd.to_datetime(hourly_clean["Datetime"]).dt.dayofweek
hourly_clean["Month"] = pd.to_datetime(hourly_clean["Datetime"]).dt.month

hourly_clean["Is_Weekend"] = hourly_clean["Day_of_Week"].isin([5, 6]).astype(int)

hourly_clean["Hour_Sin"] = np.sin(2 * np.pi * hourly_clean["Hour"] / 24)

hourly_clean["Hour_Cos"] = np.cos(2 * np.pi * hourly_clean["Hour"] / 24)

hourly_clean["DOW_Sin"] = np.sin(2 * np.pi * hourly_clean["Day_of_Week"] / 7)

hourly_clean["DOW_Cos"] = np.cos(2 * np.pi * hourly_clean["Day_of_Week"] / 7)

hourly_clean["Diff_1"] = hourly_clean["Global_active_power"].shift(1) - hourly_clean[
    "Global_active_power"
].shift(2)

hourly_clean["Diff_24"] = hourly_clean["Global_active_power"].shift(24) - hourly_clean[
    "Global_active_power"
].shift(48)

hourly_clean = hourly_clean.iloc[168:].copy()
hourly_clean.to_csv("data/features.csv", index=False)
