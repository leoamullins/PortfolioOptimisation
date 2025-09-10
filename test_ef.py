import yfinance as yf
from EfficientFrontier import ef

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "XOM", "PG", "V"]
prices = yf.download(tickers, start = '2015-01-01', auto_adjust=False)['Adj Close']

ef1 = ef(prices) # Initialising as Efficient Frontier

val, weights = ef1.max_sharpe(prices)
ef1.plotting(prices = prices , maxsharpe=True)