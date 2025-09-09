import pandas as pd
import numpy as np

class PortHelp():
    def __init__(self, prices):
        if not isinstance(prices, pd.DataFrame):
            raise TypeError('Format prices as a pd.DataFrame')
        
        self.tickers = prices.columns

    def weights_clean(self, weights, rounddp = 6):
        weightsclean = np.where(weights < 1e-6, 0, weights)

        rounded = weightsclean.round(rounddp)
        return pd.Series(rounded, self.tickers)

    def calc_market_weights(self, mcaps):
        if not isinstance(mcaps, (pd.Series, pd.DataFrame, dict)):
            raise TypeError("mcaps needs pd.Series or dict type")
        elif isinstance(mcaps, dict):
            mcaps = pd.Series(mcaps)

        mweights = mcaps / mcaps.sum()

        return mweights
    
    