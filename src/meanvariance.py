import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from portfolioclass import Portfolio
import matplotlib.pyplot as plt

class ef(Portfolio):
    def __init__(self, tickers, start_date, end_date):
       if not isinstance(tickers, list):
           raise TypeError("Tickers must be stored in a list")
       super().__init__(tickers, start_date, end_date)

    def minimumvariance(self, target_return):
        prices = self.prices

        n= len(prices.columns)
        bounds = Bounds(0,1)
        linear_constraint = LinearConstraint(np.ones(n), 1, 1)

        w0 = np.ones(n) / n

        exp_ret = self.meanexpectedreturns(prices)

        cov_matrix = self.covariancematrix(prices)

        target_constraint = {'type': 'eq', 'fun': lambda w: exp_ret.T @ w - target_return}

        variance = lambda w: w.T @ cov_matrix @ w

        res1 = minimize(variance, w0, args = (), method = 'SLSQP', bounds=bounds, constraints=[linear_constraint, target_constraint])

        self.minvarport = res1
    
    def efficientfrontier(self, data_points = 250):
        prices = self.prices
        exp_ret = self.meanexpectedreturns()
        cov_matrix = self.cov_matrix()
        tickers = self.ticker_list
        n = prices.columns

        # Constraints and Initial Guess for Problem
        bounds = Bounds(0,1)
        linear_constraint = LinearConstraint(np.ones(len(n)), 1, 1)
        w0 = np.ones(len(tickers)) / len(tickers)

        # Creating the Optimisation Function to find Max. Return Portfolio
        ret = lambda w: -exp_ret.T @ w  # negative so we can find maximum
        max_ret_port = minimize(ret, w0, args=(), method = 'SLSQP', constraints=linear_constraint, bounds=bounds) 

        # Function to find max. variance portfolio
        var = lambda w : w.T @ cov_matrix @ w
        min_var_port = minimize(var, w0, args = (), method = "SLSQP", constraints=linear_constraint, bounds=bounds)

        # Calculating min and max target returns for the EF.
        min_return = exp_ret.T @ min_var_port.x
        max_return = exp_ret.T @ max_ret_port.x 

        trs = np.linspace(min_return, max_return, data_points, True)
        ef_points = []
        weight_list = []

        for tr in trs:
            targ_constr = {'type': 'eq', 'fun': lambda w, target=tr: exp_ret.T @ w - target}

            obj = lambda w: 1/2 * w.T @ cov_matrix @ w
            res1 = minimize(obj, w0, args = (), method = 'SLSQP', bounds=bounds, constraints=(linear_constraint, targ_constr))

            if res1.success:  #  Avoid insuccessful optimisations
                x = res1.x
                ret = exp_ret.T @ x
                risk = np.sqrt(x.T @ cov_matrix @ x)
                ef_points.append((risk, ret))
                weight_list.append(x)
        
        
        self.ef_datapoints = (ef_points, weight_list)
    
    def max_sharpe(self, r_f = 0.0, kmax = 1000, k0 = 1):
        prices = self.prices
        cov_matrix = self.covariancematrix(prices)
        expret = self.meanexpectedreturns(prices)
        n = len(self.ticker_list)

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

        # Calculating Sharpe ratio
        value = (expret.T @ invertedw - r_f) / np.sqrt(invertedw.T @ cov_matrix @ invertedw)
        
        self.maxsharpevalue, self. maxsharpewport = value, pd.Series(invertedw, self.tickers_list)
    
    def plotting(self, maxsharpe = False):
        prices = self.prices
        ef = self.efficientfrontier()
        exp_ret = self.meanexpectedreturns()
        cov_matrix = self.cov_matrix()
        ms = self.max_sharpe()

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