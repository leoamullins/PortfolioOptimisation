from hmac import new
from PortHelper import PortHelp
import pandas as pd
from EfficientFrontier import ef

def max_sharpe_wrapper(prices, r_f = 0, kmax = 1000, k0 = 1):
    efficient_frontier = ef(prices)
    value, weights = efficient_frontier.max_sharpe(prices, r_f, kmax, k0)
    return weights

class backtester(PortHelp):
    def __init__(self, prices, optimizer, lookback=252, rebalance_freq=21,
                 initial_capital=1_000_000, transaction_cost=0.001):

        if not isinstance(prices, pd.DataFrame):
            raise TypeError('Format prices as a pd.DataFrame')

        super().__init__(prices)
        self.prices = prices
        self.optimizer = optimizer
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def prepare_data(self):
        """
        Prepare data for backtesting:
        - Convert prices to returns
        - Handle missing data by filtering assets with low coverage
        - Store processed returns as instance variable
        """
        # Convert prices to returns
        returns = self.prices.pct_change(fill_method=None).dropna(how='all')
        
        # Filter out assets with less than 80% data coverage
        coverage = returns.notna().mean()
        returns = returns.loc[:, coverage > 0.8]
        
        # Store processed returns
        self.returns = returns
    
    def run(self):
        """
        Main backtest loop with improved error handling and logging
        """
        # Ensure data is prepared
        if not hasattr(self, 'returns') or self.returns is None:
            self.prepare_data()
            
        returns = self.returns
        portfolio_value = self.initial_capital
        current_weights = pd.Series(0, index=returns.columns)
        history = []
        
        print(f"Starting backtest with {len(returns.columns)} assets")
        print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
        
        rebalance_dates = []
        
        for t in range(self.lookback, len(returns), self.rebalance_freq):
            try:
                # Get price window for optimization
                price_window = self.prices.iloc[t - self.lookback:t]
                
                # Clean the price window - remove assets with any NaN in the window
                price_window = price_window.dropna(axis=1, how='any')
                available_assets = price_window.columns

                if len(available_assets) < 2:
                    print(f"Skipping rebalance at {returns.index[t]}: insufficient assets")
                    continue

                # Optimize using price data
                new_weights_array = self.optimizer(price_window)
                new_weights = pd.Series(new_weights_array, index=available_assets)
                
                # Reindex to match full universe and fill missing with 0
                new_weights = new_weights.reindex(returns.columns).fillna(0)
                
                # Normalize weights to sum to 1
                if new_weights.sum() > 0:
                    new_weights = new_weights / new_weights.sum()

                # Calculate turnover and transaction costs
                turnover = (new_weights - current_weights).abs().sum()
                cost = turnover * self.transaction_cost * portfolio_value
                portfolio_value -= cost

                # Update current weights
                current_weights = new_weights.copy()
                rebalance_dates.append(returns.index[t])
                
                print(f"Rebalanced on {returns.index[t]}: Turnover={turnover:.3f}, Cost=${cost:,.0f}")

            except Exception as e:
                print(f"Error during rebalancing at {returns.index[t]}: {e}")
                continue

            # Record performance for each day in the rebalancing period
            for step in range(self.rebalance_freq):
                if t + step >= len(returns):
                    break

                try:
                    # Calculate daily return
                    step_return = (current_weights * returns.iloc[t + step]).sum()
                    
                    # Handle NaN returns
                    if pd.isna(step_return):
                        step_return = 0
                    
                    portfolio_value *= (1 + step_return)

                    # Create history record
                    record = {
                        'date': returns.index[t + step],
                        'portfolio_value': portfolio_value,
                        'returns': step_return,
                        'turnover': turnover if step == 0 else 0,
                        'transaction_cost': cost if step == 0 else 0,
                    }
                    
                    # Add individual asset weights
                    for asset in returns.columns:
                        record[f'weight_{asset}'] = current_weights.get(asset, 0)
                    
                    history.append(record)
                    
                except Exception as e:
                    print(f"Error recording step {step} at {returns.index[t + step]}: {e}")
                    continue
        
        # Convert to DataFrame
        self.history = pd.DataFrame(history).set_index('date')
        self.rebalance_dates = rebalance_dates
        
        print(f"Backtest completed: {len(rebalance_dates)} rebalances")
        print(f"Final portfolio value: ${self.history['portfolio_value'].iloc[-1]:,.0f}")
        
        return self.history
                
