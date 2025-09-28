# Implied Volatility Analysis for Options Pricing

Hi, I'm Abhishek. This project explores options pricing by computing and analyzing implied volatility (IV) for SPY ETF options using the Black-Scholes model. It fetches real-time option chain data, calculates IV with a bisection solver, compares it to market IV, and visualizes results with 2D and 3D plots to reveal volatility surface patterns. The stock can be chaged as desired by chainging the ticker symbol.

---
## What is the Black-Scholes Model?

The **Black-Scholes model** is a cornerstone of quantitative finance used to price European-style options. It assumes:

* Constant volatility and risk-free interest rate
* Log-normal distribution of the underlying asset
* No arbitrage opportunities

It produces a **theoretical option price** given inputs like spot price, strike, time to expiry, risk-free rate, and volatility.

---

## Black-Scholes Formula

For a call option, the price is given by:

$$
C = S e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2)
$$

For a put option:

$$
P = K e^{-rT} \Phi(-d_2) - S e^{-qT} \Phi(-d_1)
$$

Where:

$$
d_1 = \frac{\ln(S/K) + (r-q+0.5\sigma^2)T}{\sigma \sqrt{T}}, \quad 
d_2 = d_1 - \sigma \sqrt{T}
$$


- $$S$$ = current stock price  
- $$K$$ = strike price  
- $$T$$ = time to expiration (in years)  
- $$r$$ = risk-free rate  
- $$q$$ = dividend yield  
- $$\sigma$$ = volatility  
- $$\Phi$$ = cumulative distribution function of standard normal
 

This formula calculates a theoretical option price based on market inputs.

---

## What This Project Does

* Fetches **option chains** for a given stock using `yfinance`  
* Computes **implied volatilities** using a **bisection solver**  
* Compares computed IVs to market IVs  
* Generates **accuracy metrics** (MAE, RMSE, Max Error)  
* Visualizes **2D and 3D implied volatility surfaces**  
* Highlights the **ATM ±15% moneyness range** for deeper analysis  

This workflow demonstrates practical option pricing, calibration, and visual analysis used in quantitative trading.

---

## Step 1: Import Libraries

The project uses Python libraries for:

* **Data handling:** `pandas`, `numpy`  
* **Market data:** `yfinance`  
* **Mathematics & statistics:** `jax`, `scipy`  
* **Visualization:** `matplotlib`, `plotly`  

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from jax import grad
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
```
Step 2: Black-Scholes Pricing Function
--------------------------------------

We define a vectorized function to calculate **call and put option prices**:
 ```python
  def solve_for_iv_bisection(S, K, T, r, price, otype="call", tol=1e-6, max_iter=100):
    sigma_low, sigma_high = 0.01, 5.0
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = float(black_scholes(S, K, T, r, sigma_mid, otype=otype))
        if abs(price_mid - price) < tol:
            return sigma_mid
        if price_mid > price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return sigma_mid

 ```
 Step 4: Fetch Market Option Data
--------------------------------

We use `yfinance` to collect the option chain:

-   Select stock (`TSLA`)

-   Get **calls and puts** for the first 5 expirations

-   Calculate **time to expiry** in years

```python
ticker_symbol = "TSLA"
ticker = yf.Ticker(ticker_symbol)
S = ticker.history(period="1d")["Close"].iloc[-1]
r = 0.05

expirations = ticker.options[:5]
all_options = []

for exp in expirations:
    chain = ticker.option_chain(exp)
    for df, otype in [(chain.calls, "call"), (chain.puts, "put")]:
        temp = df.copy()
        temp["type"] = otype
        temp["expirationDate"] = pd.to_datetime(exp)
        all_options.append(temp)

options_df = pd.concat(all_options)
today = datetime.today()
options_df["T"] = (options_df["expirationDate"] - today).dt.days / 365
```

Step 5: Compute Implied Volatilities
------------------------------------

We calculate IVs for all traded options and compare them to market-reported IVs.

* * * * *

Step 6: Create Comparison Table
-------------------------------

Columns include:

-   Strike, Expiration, Option Type

-   Market Price, Computed IV, Market IV

-   Theoretical Price, Price Error, IV Error

```python
comparison_df = pd.DataFrame({
    "Strike": strikes,
    "Expiration": expiries,
    "OptionType": option_types,
    "MarketPrice": market_prices,
    "ComputedIV": ivs,
    "MarketIV": market_IVs
})
```

Step 7: Accuracy Metrics
------------------------

Compute **MAE, RMSE, Max Error** for:

-   All traded options

-   ATM ±15% moneyness range

```python
def calc_metrics(df):
    IV_Error = df["IV_Error"]
    Price_Error = df["PriceError"]
    return {
        "MAE_IV": IV_Error.abs().mean(),
        "MaxAE_IV": IV_Error.abs().max(),
        "RMSE_IV": np.sqrt((IV_Error**2).mean()),
        "MAE_Price": Price_Error.mean(),
        "MaxAE_Price": Price_Error.max(),
        "RMSE_Price": np.sqrt((Price_Error**2).mean())
    }
```
Step 8: 2D IV Comparison Plot
-----------------------------

Visualize computed IVs against market IVs:

-   Blue = computed

-   Red = market

```python
plt.scatter(strikes, ivs, label="Computed IV")
plt.scatter(strikes, market_IVs, label="Market IV")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()
```
Step 9: 3D IV Surfaces
----------------------

-   Use **moneyness** and **time to expiry** as axes

-   Generate **smoothed IV surfaces** using Gaussian filtering

-   Visualize **computed, market, and difference surfaces**

* * * * *

Step 10: 3D ATM ±15% IV Surfaces
--------------------------------

-   Focus on near-the-money options for a more meaningful analysis

-   Overlay **current stock price**

-   Compare computed IV, market IV, and differences

This helps identify **pricing biases** and **model accuracy** near the most liquid strikes.

* * * * *

Why This Matters
----------------

-   Provides a **practical workflow** for option pricing and implied volatility computation

-   Highlights **gaps between theoretical models and market prices**

-   Teaches **numerical techniques** (bisection method, interpolation, smoothing)

-   Essential for roles in **quant trading, risk management, and derivatives research**

* * * * *

Skills Demonstrated
-------------------

-   Option pricing with **Black-Scholes model**

-   Implied volatility computation

-   Data analysis using **Python, pandas, NumPy**

-   Visualization with **Matplotlib** and **Plotly**

-   Accuracy evaluation and surface interpolation

-   Handling **ATM and near-the-money options**

* * * * *
