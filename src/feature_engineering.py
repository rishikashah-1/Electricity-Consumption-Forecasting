import pandas as pd

# Load cleaned hourly electricity data
data = pd.read_csv("../data/hourly_clean.csv")

# Convert Datetime column into proper datetime format
data["Datetime"] = pd.to_datetime(data["Datetime"])

# Sort data chronologically
data = data.sort_values("Datetime")

# Create time-based features
data["hour"] = data["Datetime"].dt.hour
data["day_of_week"] = data["Datetime"].dt.dayofweek

print("Feature engineering started successfully!")
print(data[["Datetime", "Global_active_power", "hour", "day_of_week"]].head(10))