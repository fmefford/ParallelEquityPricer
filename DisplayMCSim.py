from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

with open("forecasts.csv", "r") as f:
    startdate, periods = f.readline().split(",")
startdate = datetime.strptime(startdate, "%Y%m%d")

forecasts = pd.read_csv("forecasts.csv", header=None, skiprows=1)

for index, row in forecasts.iterrows():
    plt.plot(pd.bdate_range(startdate, startdate + pd.tseries.offsets.BDay(int(periods)), inclusive="left"), row)
plt.show()