from ast import UnaryOp
from typing import Type
import numpy as np
import pandas as pd
from PortHelper import PortHelp

def implied_prior_returns(market_caps, risk_aversion, cov_matrix, r_f = 0):
    mcaps = pd.Series(market_caps)
    market_weights = mcaps / mcaps.sum()

    return risk_aversion * cov_matrix.dot(market_weights) + r_f

def implied_risk_aversion(market_prices, days = 252, r_f = 0):
    market_prices = market_prices.squeeze()
    rets = market_prices.pct_change(fill_method=None).dropna(how = 'all')
    r = rets.mean() * days
    var = rets.var() * days
    delta = (r - r_f) / var
    return delta

class blacklitterman(PortHelp):
    def __init__(self, prices, tau, pi):
        if not isinstance(prices, pd.DataFrame):
            raise TypeError('Prices must be a pd.DataFrame')

        super().__init__(prices)

        self.tau = tau
        self.prices = prices
        self.tickers = prices.columns.tolist()
        self.sigma = self.covariancematrix(prices)

        if isinstance(pi, (pd.Series, pd.DataFrame)):
            self.pi = pi.to_numpy().reshape(-1, 1)
        elif isinstance(pi, np.ndarray):
            self.pi = pi.reshape(-1, 1)
        else:
            raise TypeError("pi must be a numpy array, pandas series, or dataframe")
    
    def omega(self, P, tau = None):
        if tau is None:
            tau = self.tau
        if isinstance(P, pd.DataFrame):
            self.P = P.values
        elif isinstance(P, np.ndarray):
            self.P = P
        else:
            raise TypeError('P should be a numpy array or pd.DataFrame.')

        prices = self.prices
        sigma = self.covariancematrix(prices)
        omega = tau * P @ sigma @ P.T
        return omega
    
    def postExpectedReturns(self, Q, P, tau = None, omega = None):
        if isinstance(P, pd.DataFrame):
            P = P.values
        elif not isinstance(P, np.ndarray):
            raise TypeError("P should be a numpy array or pandas dataframe.")

        if isinstance(Q, (pd.Series, pd.DataFrame)):
                Q = Q.to_numpy().reshape(-1,1) 
        elif isinstance(Q, np.ndarray):
                Q = Q.reshape(-1, 1)
        else:
            raise TypeError("Q must be a numpy array, pandas series or dataframe")

        if tau is None:
            tau = self.tau
        if omega is None:
            omega = self.omega(P, tau)
        
        A = np.linalg.inv(tau * self.sigma)
        middle = P.T @ np.linalg.inv(omega) @ P
        rhs = A @ self.pi + P.T @ np.linalg.inv(omega) @ Q

        post_mean = np.linalg.solve(A + middle, rhs)

        self.postexpret = post_mean
        return post_mean

    def posteriorCovariance(self, P, tau = None, omega = None):
        if isinstance(P, pd.DataFrame):
            self.P = P.values
        elif not isinstance(P, np.ndarray):
            raise TypeError("P should be a numpy array or pandas dataframe")

        if tau is None: 
            tau = self.tau

        if omega is None:
            omega = self.omega(P, tau)

        soln = np.linalg.inv(np.linalg.inv(tau * self.sigma) + P.T @ np.linalg.inv(omega) @ P)
        self.posteriorCovMat = soln
        return soln

    def blacklitterman_port(self, delta, Q, P):
        post_returns = self.postExpectedReturns(Q, P)

        unnormalised = np.linalg.inv(delta * self.sigma) @ post_returns
        normalised = unnormalised / np.sum(unnormalised)

        res = normalised.squeeze()
        cleaned = self.weights_clean(res)

        return pd.Series(cleaned, self.tickers)
