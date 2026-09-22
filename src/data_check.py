import pandas as  pd

df=pd.read_csv("../data/household_power_consumption.txt",sep=";",low_memory=False)

print(df.head())
print(df.shape)
print(df.columns)
df.info()
print((df=="?").sum())
df=df.replace("?", float("nan"))
print(df.isna().sum())
print(df.isna().sum() / len(df)* 100)
print(df[df["Global_active_power"].isna()].head(20))
missing= df["Global_active_power"].isna()
groups=missing.ne(missing.shift()).cumsum()
missing_lengths=missing.groupby(groups).sum()
print(missing_lengths.max())
print((missing_lengths<=60).sum())
print((missing_lengths > 60).sum())
df["Datetime"]=df["Date"]+" "+df["Time"]
df["Datetime"] = pd.to_datetime(df["Datetime"],dayfirst=True)
print(df["Datetime"].dtype)
df[["Global_active_power","Global_reactive_power","Voltage",
    "Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]]=df[["Global_active_power","Global_reactive_power","Voltage",
                           "Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]].apply(pd.to_numeric)
df.index = df["Datetime"]
hourly=df[["Global_active_power","Global_reactive_power","Voltage",
           "Global_intensity","Sub_metering_1","Sub_metering_2","Sub_metering_3"]].resample("1h").mean()
print(hourly.head())
print(hourly.shape)
print(hourly.isna().sum())
print(hourly["Global_active_power"].notna().sum())
hourly_clean=hourly[hourly["Global_active_power"].notna()]
print(hourly_clean.isna().sum())
print(hourly_clean.index.is_monotonic_increasing)
hourly_clean.to_csv("../data/hourly_clean.csv", index=True)