import matplotlib.pyplot as plt
import pandas as pd

hourly_clean=pd.read_csv("data/hourly_clean.csv")
hourly_clean["Datetime"]=pd.to_datetime(hourly_clean["Datetime"])
hourly_clean.index=hourly_clean["Datetime"]
plt.plot(hourly_clean["Datetime"],hourly_clean["Global_active_power"])
plt.show()