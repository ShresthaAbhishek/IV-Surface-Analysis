# Implied Volatility Analysis for Options Pricing

Hi, I'm Abhishek. This project explores options pricing by computing and analyzing implied volatility (IV) for SPY ETF options using the Black-Scholes model. It fetches real-time option chain data, calculates IV with a bisection solver, compares it to market IV, and visualizes results with 2D and 3D plots to reveal volatility surface patterns. The stock can be chaged as desired by chainging the ticker symbol.

## What is Implied Volatility?

Implied volatility (IV) reflects the market's expectation of a stock's future volatility, derived from option prices. It's key in models like Black-Scholes, indicating uncertainty in options contracts. Higher IV suggests larger price swings, while lower IV implies stability.

## How the Black-Scholes Model Works

The Black-Scholes model prices European options, assuming:
- Log-normal stock price distribution
- Constant risk-free rate and volatility
- No dividends (or adjusted for dividends)
- No arbitrage opportunities

Call option: \( C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2) \)

Put option: \( P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1) \)

Where:
- \( S \): Current stock price
- \( K \): Strike price
- \( T \): Time to expiration (years)
- \( r \): Risk-free rate
- \( q \): Dividend yield
- \( \sigma \): Volatility (IV when solved inversely)
- \( N \): Standard normal CDF
- \( d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma \sqrt{T}} \)
- \( d_2 = d_1 - \sigma \sqrt{T} \)

This project solves for IV given market option prices.

## What This Project Does

- **Fetches Option Data**: Uses `yfinance` to retrieve QQQ option chains.
- **Computes Implied Volatility**: Applies a bisection solver to calculate IV.
- **Compares IVs**: Evaluates computed vs. market IVs, calculating errors.
- **Visualizes Results**:
  - 2D scatter plot of computed vs. market IVs across strikes.
  - 3D volatility surfaces for computed IV, market IV, and differences, plotted against moneyness and time to expiration.
- **Evaluates Accuracy**: Computes Mean Absolute Error (MAE), Maximum Absolute Error (MaxAE), and Root Mean Squared Error (RMSE) for IV and prices, for all options and ATM options (±15% moneyness).

## Example Output
===== Accuracy Metrics (All Traded Options) =====
MAE_IV: 0.0321
MaxAE_IV: 0.1876
RMSE_IV: 0.0452
MAE_Price: 0.0213
MaxAE_Price: 0.0987
RMSE_Price: 0.0289

===== Accuracy Metrics (ATM ±15% Moneyness) =====
MAE_IV: 0.0254
MaxAE_IV: 0.1123
RMSE_IV: 0.0345
MAE_Price: 0.0156
MaxAE_Price: 0.0765
RMSE_Price: 0.0212


---

## 📊 Visualizations

- **2D Plot:** Computed vs. market IVs across strike prices  
- **3D Surfaces:** IV as a function of moneyness (\( S/K \)) and time to expiration  
- **Difference Plot:** Highlights discrepancies between computed and market IVs

---

## 🧑‍💻 Implementation in Python

### 🛠️ Key Dependencies

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
```