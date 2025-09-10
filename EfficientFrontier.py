import pandas as pd
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
import matplotlib.pyplot as plt
from PortHelper import PortHelp

class ef(PortHelp):
    def __init__(self, prices):
        if not isinstance(prices, pd.DataFrame):
            raise TypeError('Format prices as a pd.DataFrame')

        super().__init__(prices)
        
        self.prices = prices
        self.tickers = prices.columns

    def meanexpectedreturns(self, prices, geometric = True, days = 252):
        if not isinstance(prices, pd.DataFrame):
            raise TypeError('Format prices as a pd.DataFrame')

        returns = prices.pct_change(fill_method = None).dropna(how = 'all')

        if geometric:
            return (1 + returns).prod() ** (days / returns.count()) - 1
        else:
            return returns.mean() * days
    
    def minimumvariance(self, prices, target_return):
        if not isinstance(prices, (pd.Series, pd.DataFrame)):
            raise TypeError("Prices should be a pd.DataFrame")
        n = len(prices.columns)
        bounds = Bounds(0,1)
        linear_constraint = LinearConstraint(np.ones(n), 1, 1)

        w0 = np.ones(n) / n

        expected_returns = self.meanexpectedreturns(prices)

        cov_matrix = self.covariancematrix(prices)

        target_constraint = {'type': 'eq', 'fun': lambda w: expected_returns.T @ w - target_return}

        variance = lambda w: w.T @ cov_matrix @ w

        res1 = minimize(
            variance, 
            w0, 
            args = (), 
            method = 'SLSQP', 
            bounds = bounds, 
            constraints=[linear_constraint, target_constraint]
            )
        
        return res1
    
    def efficientfrontier(self, prices, data_points = 250):
        if not isinstance(prices, (pd.Series, pd.DataFrame)):
            raise TypeError("Prices should be a pd.DataFrame")
        
        expected_returns = self.meanexpectedreturns(prices)

        cov_matrix = self.covariancematrix(prices)

        n = len(prices.columns)

        bounds = Bounds(0,1)
        linear_constraint = LinearConstraint(np.ones(n), 1, 1)
        w0 = np.ones(n) / n

        ret = lambda w: -expected_returns.T @ w
        max_ret_port = minimize(ret, 
                                w0, 
                                args = (),
                                method = 'SLSQP',
                                constraints = linear_constraint,
                                bounds = bounds
                                )
        
        var = lambda w: w.T @ cov_matrix @ w
        min_var_port = minimize(var,
                                w0,
                                args = (),
                                method = 'SLSQP',
                                constraints = linear_constraint,
                                bounds = bounds
                                )
        
        min_return = expected_returns.T @ min_var_port.x
        max_return = expected_returns.T @ max_ret_port.x

        trs = np.linspace(min_return, max_return, data_points, True)
        ef_points = []
        weight_list = []

        for tr in trs:
            target_constraint = {'type': 'eq', 'fun': lambda w, target=tr: expected_returns.T @ w - target}

            obj = lambda w: 1/2 * w.T @ cov_matrix @ w
            res1 = minimize(obj,
                            w0,
                            args = (),
                            method = 'SLSQP', 
                            bounds=bounds, 
                            constraints=(linear_constraint, target_constraint)
                            )
            
            if res1.success:
                x = res1.x 
                returns = expected_returns.T @ x
                risk = np.sqrt(x.T @ cov_matrix @ x)
                ef_points.append((risk, returns))
                weight_list.append(x)
        return ef_points, weight_list
    
    def max_sharpe(self, prices, r_f = 0, kmax = 1000, k0 = 1):
        cov_matrix = self.covariancematrix(prices)
        expret = self.meanexpectedreturns(prices)
        n = len(prices.columns)

        # Creating the object we are optimising by adding kappa as a variable to the weight array
        x0 = np.ones(n) / n 
        w0 = np.append(x0, k0)

        # Defining the objective function
        def obj(w):
            w = w[:n] # Not including kappa in the Variance calculation.
            return w.T @ cov_matrix @ w

        constraints = (
            {'type': 'eq', 'fun': lambda w: (expret - r_f).T @ w[:n] - 1}, 
            {'type': 'eq', 'fun': lambda w: np.sum(w[:n]) - w[-1]}
        ) # Explained in documentation.

        lb = [0] * n + [1e-6] # Using 1e-6 as a minimum kappa value, not including shorting.
        ub = [kmax] * n + [kmax]

        bounds = Bounds(lb=lb, ub=ub)

        wopt = minimize(obj, w0, args= (), method = 'SLSQP', bounds = bounds, constraints=constraints)

        invertedw = wopt.x[:n] / wopt.x[-1] # Inverting weight transformation done with kappa

        cleaned = self.weights_clean(invertedw)
        # Calculating Sharpe ratio
        value = (expret.T @ invertedw - r_f) / np.sqrt(invertedw.T @ cov_matrix @ invertedw)

        return value, pd.Series(cleaned, self.tickers)
    
    def plotting(self, prices, maxsharpe = False):

        ef = self.efficientfrontier(prices)
        exp_ret = self.meanexpectedreturns(prices)
        cov_matrix = self.covariancematrix(prices)
        ms = self.max_sharpe(prices)

        if maxsharpe:
            value, weights = ms
            returns = exp_ret.T @ weights.values
            risk = np.sqrt(weights.values.T @ cov_matrix @ weights.values)
            plt.scatter(risk, returns, marker ='d', c='r', s = 100, label=f"Max Sharpe: {value:.2f}")

        ef_points, _ = ef
        x, y = zip(*ef_points)
        plt.plot(x,y, label = 'Efficient Frontier')
        plt.title("Efficient Frontier")
        plt.xlabel("Risk")
        plt.ylabel("Expected Return")
        plt.legend()
        plt.show()