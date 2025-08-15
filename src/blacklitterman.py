import numpy as np
import pandas as pd
import yfinance as yf
from portfolioclass import Portfolio

def implied_prior_returns(market_weights, risk_aversion, cov_matrix, r_f = 0):
        return risk_aversion * cov_matrix.dot(market_weights) + r_f

def implied_risk_aversion(market_prices, days = 252, r_f = 0):

    market_prices = market_prices.squeeze() # Turning it into a CHECK THIS!!
    rets = market_prices.pct_change(fill_method=None).dropna(how = 'all')
    r = rets.mean() * days
    var = rets.var() * days
    delta = (r - r_f) / var
    return delta


class bl(Portfolio):
    def __init__(self, tickers, start_date = '2015-01-01', end_date = '2025-08-15', abs_views = None, tau = 0.02, Q = None, P = None,  r_f = 0.00, days = 252):
        #Checking ticker  type.
        if not isinstance(tickers, list):
            if isinstance(tickers, str):
                self.tickers = pd.read_csv(tickers).columns.tolist() # ticker data comes from csv, will calculate mcaps using yf
                                                                        # ticker data should be adjusted close prices
                tickers = self.tickers
                self.prices = pd.read_csv(tickers)
                super().__init__(tickers, start_date, end_date)
            else:
                raise ValueError("Tickers must be a list or a string (filename)")
        else:
            tickers = self.tickers
            super().__init__(tickers, start_date, end_date)

        if not isinstance(abs_views, None):
            if not isinstance(abs_views, dict):
                if not isinstance(abs_views, list):
                    raise TypeError("abs_views need to be a dictionary or in a list")
                viewsarr = np.array(abs_views)
                views = viewsarr.reshape(1,-1)
                self.abs_views = views
            else:
                views = self.dict_to_absviews(abs_views)
                self.abs_views = views
       
        self.r_f = r_f

        mcaps = {}
        for ticker in self.ticker_list:
            stock = yf.Ticker(ticker)
            info = stock.info()['marketCap']
            mcaps[ticker] = info
        self.mcaps = pd.Series(mcaps)

        mcaps = self.mcaps

        self.market_weights = self.calc_market_weights(mcaps)
        

        cov_matrix = self.cov_matrix()
        self.rawcovmat = cov_matrix # save original

        self.tau = tau
        self.days = days

        if not isinstance(Q, None):
            if isinstance(Q, (pd.Series, pd.DataFrame)):
                self.Q = Q.to_numpy.reshape(-1,1) 
            elif isinstance(Q, np.ndarray):
                self.Q = Q.reshape(-1, 1)
            else:
                raise TypeError("Q must be a numpy array, pandas series or dataframe")
        if not isinstance(P, None):
            if isinstance(P, pd.DataFrame):
                self.P = P.values
            elif isinstance(P, np.ndarray):
                self.P = P
            else:
                raise TypeError("P must be an array or dataframe")
        
    def omega(self, P, tau = None):
        if isinstance(P, pd.DataFrame):
            self.P = P.values
        elif isinstance(P, np.ndarray):
            self.P = P
        else:
            raise TypeError("P should be a numpy array or pandas dataframe")
        
        if isinstance(tau, None):
            tau = self.tau
        
        sigma = self.cov_matrix
        omega = tau * P @ sigma @ P.T
        self.omega = omega
    
    def posterior_expected_returns(self, Q, P, pi = None, tau = None, omega = None):
        if not isinstance(P, pd.DataFrame):
            P = P.values
        elif isinstance(P, np.ndarray):
            P = P
        else:
            raise TypeError("P should be a numpy array or pandas dataframe")

        if isinstance(Q, (pd.Series, pd.DataFrame)):
                Q = Q.to_numpy.reshape(-1,1) 
        elif isinstance(Q, np.ndarray):
                Q = Q.reshape(-1, 1)
        else:
            raise TypeError("Q must be a numpy array, pandas series or dataframe")
    
        if isinstance(tau, None):
            tau = self.tau

        if isinstance(omega, None):
            omega = self.omega(P, tau)

        sigma = self.cov_matrix
        
        c = P.T @ np.linalg.inv(omega)
        A = np.linalg.inv(tau * sigma) + c @ P 
        b = np.linalg.inv(tau * sigma) @ pi + c @ Q

        soln = np.linalg.solve(A,b)

        self.post_exp_ret = soln
    
    def posterior_covariance(self, P):
        if isinstance(P, pd.DataFrame):
            self.P = P.values
        elif isinstance(P, np.ndarray):
            self.P = P
        else:
            raise TypeError("P should be a numpy array or pandas dataframe")
        
        omega = self.omega(P, self.tau)

        soln = self.cov_matrix + np.linalg.inv(np.linalg.inv(self.tau * self.cov_matrix) + P.T @ np.linalg.inv(omega) @ P)
        self.post_cov_matrix = soln
    
    def bl_port(self, delta, P):
        sigma = self.cov_matrix
        tau = self.tau
        inv_omega = np.linalg.inv(self.omega(P, tau))

        sigma_hat = sigma + np.linalg.inv(np.linalg.inv(tau * sigma) + P.T @ inv_omega @ P)

        return sigma_hat