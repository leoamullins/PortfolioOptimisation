# Portfolio Optimisation

This repository contains a small collection of utilities for working with classical
portfolio construction techniques. It focuses on two core workflows:

* **Mean-variance optimisation** implemented in `EfficientFrontier.py`.
* **Black–Litterman blending of investor views with market equilibrium** implemented in `blacklitterman.py`.

Both workflows are built on top of the helper routines provided in `PortHelper.py` for
cleaning weights and deriving covariances from price data.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The key third-party packages are:

* `numpy` and `pandas` for vectorised data manipulation.
* `scipy` for constrained optimisation.
* `matplotlib` for plotting efficient frontiers.
* `yfinance` for retrieving example market data as demonstrated in `test_ef.py`.

## Usage

The modules are written to operate on historical price data supplied as a
`pandas.DataFrame` of adjusted closing prices. The snippet below demonstrates
how to download prices with `yfinance`, initialise the efficient frontier helper,
and retrieve the maximum Sharpe ratio portfolio.

```python
import yfinance as yf
from EfficientFrontier import ef

# Download daily prices
prices = yf.download(["AAPL", "MSFT"], start="2015-01-01")["Adj Close"].dropna()

# Optimise
optimizer = ef(prices)
sharpe, weights = optimizer.max_sharpe(prices)
print(f"Max Sharpe ratio: {sharpe:.2f}")
print(weights)
```

For a more detailed derivation of the Black–Litterman routine, refer to the notes in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Tests

The repository does not currently contain automated tests. When making changes,
please run your own smoke tests to ensure the optimisation routines work as expected
with your data.
requirements.txt
New
+5
-0

numpy>=1.23
pandas>=1.5
scipy>=1.10
matplotlib>=3.7
yfinance>=0.2
