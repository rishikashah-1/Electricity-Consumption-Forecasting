import pandas as  pd

df=pd.read_csv("../data/household_power_consumption.txt",sep=";",low_memory=False)

# print(df.head())
# print(df.shape)
# print(df.columns)
# df.info()
print((df=="?").sum())
df=df.replace("?", float("nan"))
print(df.isna().sum())
print(df.isna().sum() / len(df)* 100)
print(df[df["Global_active_power"].isna()].head(20))
missing= df["Global_active_power"].isna()
groups=missing.ne(missing.shift()).cumsum()
missing_lengths=missing.groupby(groups).sum()
print(missing_lengths.max())