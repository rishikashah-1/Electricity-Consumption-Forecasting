import pandas as  pd

df=pd.read_csv("../data/household_power_consumption.txt",sep=";",low_memory=False)

print(df.head())