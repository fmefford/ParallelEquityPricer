from datetime import datetime

import numpy as np
from pandas.tseries.offsets import BDay
import yfinance as yf
from scipy.stats import kstest

years_lookback = 1
backtest_days = 7
enddate = datetime.today() - BDay(backtest_days)
startdate = datetime.today() - BDay(260)
data = yf.download("SPY", startdate.strftime("%Y-%m-%d"), enddate.strftime("%Y-%m-%d"))
data = data.reset_index()
hist_prices = data["Adj Close"]
iterations = 100000

log_returns = np.log(hist_prices).diff().dropna().to_numpy()
print(kstest(log_returns, "norm"))

s0 = hist_prices.iloc[-1]
mu = log_returns.mean()
sigma = log_returns.std()
increments = 10
dt = 1
np.savetxt("log_returns.csv", np.concatenate(([s0, mu, sigma, int(enddate.strftime("%Y%m%d")), iterations, increments, dt, log_returns.size], log_returns)), delimiter=",")

