# Implied Volatility Analysis for Options Pricing

Hi, I'm Abhishek. This project explores options pricing by computing and analyzing implied volatility (IV) for SPY ETF options using the Black-Scholes model. It fetches real-time option chain data, calculates IV with a bisection solver, compares it to market IV, and visualizes results with 2D and 3D plots to reveal volatility surface patterns. The stock ticker can be changed as desired.

---

## Purpose of the Project

The main goal of this project is to **bridge theoretical option pricing models with real market data**. Specifically, it aims to:

* Calculate theoretical option prices using the **Black-Scholes model**  
* Compute **implied volatility** from market-traded options  
* Compare model outputs with **market-reported IVs**  
* Visualize the **volatility surface** across strikes and expiration dates  

This helps traders, quants, and researchers understand the **dynamics of option prices**, uncover pricing discrepancies, and evaluate the behavior of implied volatility across moneyness and time to expiry.

---

## Black-Scholes Model: Idea & Formulas

The **Black-Scholes model** is a foundational model for European option pricing. It assumes:

* The underlying stock follows a **log-normal stochastic process**  
* Constant risk-free interest rate and volatility  
* No arbitrage opportunities  

It provides a **closed-form formula** for option prices:

**Call Option:**

\[
C = S e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2)
\]

**Put Option:**

\[
P = K e^{-rT} \Phi(-d_2) - S e^{-qT} \Phi(-d_1)
\]

Where:

\[
d_1 = \frac{\ln(S/K) + (r-q+0.5\sigma^2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
\]

- \(S\) = current stock price  
- \(K\) = strike price  
- \(T\) = time to expiration (in years)  
- \(r\) = risk-free rate  
- \(q\) = dividend yield  
- \(\sigma\) = volatility  
- \(\Phi\) = cumulative distribution function of standard normal  

**Intuition:**  

* \(d_1\) captures how far “in-the-money” an option is, adjusted for drift and volatility  
* \(d_2\) accounts for the remaining uncertainty until expiration  
* The formula essentially computes the **expected discounted payoff** under risk-neutral probabilities  

This theoretical framework forms the basis for **computing implied volatility**, which is the volatility value that equates the Black-Scholes price to the observed market price.

---

## Key Ideas Used

This project combines several fundamental quantitative finance and computational concepts:

* **Black-Scholes Model:** Pricing European options with closed-form formulas  
* **Implied Volatility (IV):** IV is inferred from market prices using a **numerical bisection solver**  
* **Data Handling:** Real-time option chain data is fetched using `yfinance`  
* **Visualization:** 2D and 3D plots illustrate the volatility surface and ATM ±15% range  
* **Numerical Methods:** Bisection root-finding, grid interpolation, and Gaussian smoothing are applied for accurate and smooth IV surfaces  

---

## Why This Matters

Understanding implied volatility is critical in options trading and risk management:

* Provides a **practical workflow** for assessing option prices against theoretical models  
* Highlights **differences between model predictions and market behavior**  
* Visualizes **risk and uncertainty** in the options market through volatility surfaces  
* Develops familiarity with **numerical techniques and data visualization** used in quantitative finance  

---

## Skills Demonstrated

* Option pricing with the **Black-Scholes model**  
* Implied volatility computation using **numerical root-finding**  
* Real-world data handling with **Python, pandas, and yfinance**  
* Visualization using **Matplotlib and Plotly**  
* Evaluating model accuracy and identifying **ATM and near-the-money option patterns**  
* Working with **numerical interpolation and smoothing** for visualization  

---

## Example Outputs

### 1. 2D Implied Volatility Comparison

Comparison of **computed IVs** from the Black-Scholes model against **market IVs** across various strikes.  

![2D IV Comparison](images/iv_comparison.png)

---

### 2. 3D Implied Volatility Surface

3D visualization of the implied volatility surface plotted against **moneyness** (strike/spot ratio) and **time to expiry**.  

![3D IV Surface](images/iv_surface.png)

---

### 3. ATM ±15% IV Surface Focus

Zoomed-in IV surface for **ATM ±15%** strikes, where most trading activity occurs.  

![ATM Surface](images/atm_surface.png)

---

### 4. Option Pricing Error Distribution

Histogram showing the distribution of pricing errors, highlighting where the model under- or over-prices options.  

![Error Distribution](images/price_error_hist.png)

---

## High-Level Workflow

1. Fetch real-time option chain data using `yfinance`.  
2. Calculate theoretical option prices with the **Black-Scholes formula**.  
3. Solve for implied volatility (IV) numerically via a **bisection solver**.  
4. Compare computed IV to market IVs and generate accuracy metrics.  
5. Visualize IVs across strikes and expirations using 2D scatter plots and 3D surfaces.  
6. Focus on **ATM ±15% strikes** for detailed insights into high-liquidity options.  

---

## Resources & References

* [Black-Scholes: Investopedia](https://www.investopedia.com/terms/b/blackscholes.asp)  
* [Option Greeks and IV](https://www.optionsplaybook.com/)  
* [Python Finance Tutorials: QuantStart](https://www.quantstart.com/articles/)  

---
