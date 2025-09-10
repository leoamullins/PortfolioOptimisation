import pandas as pd
import numpy as np



class PortfolioBacktest():

    def __init__(self, price_data: pd.DataFrame, risk_free_rate: float = 0.02):
        self.price_data = price_data.copy()
        self.returns = price_data.pct_change(fill_method = None).dropna(how = 'all')
        self.risk_free_rate = risk_free_rate
        self.results = {}

    def rolling_optimisation(self, optimisation_method: callable, 
                             lookback_window: int = 252,
                             rebalance_freq: int = 63,
                              **opt_kwargs):
        weights_list = []
        dates_list = []

        start_idx = lookback_window
        end_idx = len(self.returns)

        for i in range(start_idx, end_idx, rebalance_freq):
            hist_returns = self.returns.iloc[i-lookback_window:i]
            current_date = self.returns.index[i]

            try:
                weights = optimisation_method(hist_returns, **opt_kwargs)

                if isinstance(weights, dict):
                    weights = pd.Series(weights)

                weights = weights / weights.sum()
                weights_list.append(weights)
                dates_list.append(current_date)
            
            except Exception as e:
                print(f'Optimisation failed as {current_date}: {e}')
                
                equal_weights = pd.Series(1/len(self.returns.columns), 
                                        index=self.returns.columns)
                weights_list.append(equal_weights)
                dates_list.append(current_date)
            
        weights_df = pd.DataFrame(weights_list, index=dates_list)
        weights_df = weights_df.reindex(columns=self.returns.columns, fill_value=0)
        
        # Forward fill weights between rebalancing dates
        full_weights = weights_df.reindex(self.returns.index[lookback_window:]).fillna(method='ffill')
        
        return full_weights