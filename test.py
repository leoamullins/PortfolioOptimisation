import yfinance as yf
import pandas as pd
import numpy as np
from EfficientFrontier import ef
from BlackLitterman import blacklitterman, implied_prior_returns, implied_risk_aversion

mcaps = {
    "GOOG": 927e9,
    "AAPL": 1.19e12,
    "FB": 574e9,
    "BABA": 533e9,
    "AMZN": 867e9,
    "GE": 96e9,
    "AMD": 43e9,
    "WMT": 339e9,
    "BAC": 301e9,
    "GM": 51e9,
    "T": 61e9,
    "UAA": 78e9,
    "SHLD": 0,
    "XOM": 295e9,
    "RRC": 1e9,
    "BBY": 22e9,
    "MA": 288e9,
    "PFE": 212e9,
    "JPM": 422e9,
    "SBUX": 102e9,
}

market_caps = pd.Series(mcaps)
tickers = market_caps.index.tolist()

prices = yf.download(tickers, start = '2015-01-01', auto_adjust=False)['Adj Close']

ef = ef(prices)
val, data = ef.max_sharpe(prices)

print(data)

spy_data = yf.download('SPY', start = '2015-01-01', auto_adjust=False)['Adj Close']



cov = ef.covariancematrix(prices)
delta = implied_risk_aversion(spy_data)
pi = implied_prior_returns(mcaps, delta, cov)
tau = 0.05

# 1. SBUX will drop by 20%
# 2. GOOG outperforms FB by 10%
# 3. BAC and JPM will outperform T and GE by 15%

bl = blacklitterman(prices, tau, pi)
Q = np.array([-0.20, 0.10, 0.15]).reshape(-1, 1)
P = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    ]
)

weights = bl.blacklitterman_port(delta, Q, P)

ret = bl.postExpectedReturns(Q, P)

print(weights)


