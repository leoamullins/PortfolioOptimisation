import numpy as np
import pandas as pd
import yfinance as yf

class Portfolio():
    def __init__(self, tickers, start_date, end_date):
    # Checking ticker input type, and creating attributes 
        if not isinstance(tickers, list):
            if isinstance(tickers, str):
                self.tickers = pd.read_csv(tickers).columns.tolist() # ticker data comes from csv, will calculate mcaps using yf
                                                                        # ticker data should be adjusted close prices
                self.prices = pd.read_csv(tickers)
                self.ticker_list = self.tickers
            else:
                raise ValueError("Tickers must be a list or a string (filename)")
        else:
            self.tickers = tickers
            self.prices = yf.download(tickers, start = start_date, end = end_date, auto_adjust=False)['Adj Close']
            self.ticker_list = tickers
            

        self.n_assets = len(self.ticker_list)

    def meanexpectedreturns(self, days = 252, geometric = False):
        prices = self.prices
        returns = prices.pct_change(fill_method=None).dropna(how = 'all')
        if geometric:
            return (1 + returns).prod() ** (days / returns.count()) - 1
        else:
            return returns.mean() * days
        
    def cov_matrix(self, days = 252):
        prices = self.prices
        returns = prices.pct_change(fill_method=None).dropna(how = 'all')

        return returns.cov() * days
    
    def weights_clean(self, weights, rounddp = 6):

        weightsclean = np.where(weights < 1e-6, 0, weights)

        rounded = weightsclean.round(rounddp)
        return pd.Series(rounded, self.tickers)
    
    def dict_to_absviews(self, views):
        if not isinstance(views, dict):
            raise TypeError("views must be given as a dictionary")
        
        views_df = pd.Series(views)

        Q = views_df.to_numpy.reshape(1,-1)

        return Q
    
    def calc_market_weights(self, mcaps):
        if not isinstance(mcaps, (pd.Series, pd.DataFrame, dict)):
            raise TypeError("mcaps needs pd.Series or dict type")
        elif isinstance(mcaps, dict):
            mcaps = pd.Series(mcaps)

        mweights = mcaps / mcaps.sum()

        return mweights