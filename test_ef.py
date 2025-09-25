import yfinance as yf
from EfficientFrontier import ef
from BacktestFramework import backtester, max_sharpe_wrapper

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "PG", "V"]
prices = yf.download(tickers, start = '2012-01-01', auto_adjust=False)['Adj Close']

ef1 = ef(prices) # Initialising as Efficient Frontier

val, weights = ef1.max_sharpe(prices)

backtester1 = backtester(prices, max_sharpe_wrapper, rebalance_freq=21)
backtester1.prepare_data()
history = backtester1.run()

print(history)
print(weights)