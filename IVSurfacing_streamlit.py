import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

st.set_page_config(layout='wide', page_title='IV Surfacing Dashboard')

# =============================
# Black-Scholes Function
# =============================
def black_scholes(S, K, T, r, sigma, q=0, otype='call'):
    d1 = (jnp.log(S/K) + (r-q+0.5*sigma**2)*T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    if otype == 'call':
        return S*jnp.exp(-q*T)*jnorm.cdf(d1) - K*jnp.exp(-r*T)*jnorm.cdf(d2)
    else:
        return K*jnp.exp(-r*T)*jnorm.cdf(-d2) - S*jnp.exp(-q*T)*jnorm.cdf(-d1)

# =============================
# IV Solver
# =============================
def solve_iv(S, K, T, r, price, otype='call', tol=1e-6, max_iter=100):
    if T <= 0:
        return np.nan
    sigma_low, sigma_high = 0.01, 5.0
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high)/2
        price_mid = float(black_scholes(S, K, T, r, sigma_mid, otype=otype))
        if abs(price_mid - price) < tol:
            return sigma_mid
        if price_mid > price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return sigma_mid

# =============================
# Fetch option data
# =============================
@st.cache_data
def fetch_options(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    S = ticker.history(period='1d')['Close'].iloc[-1]
    expirations = ticker.options[:5]
    all_options = []
    for exp in expirations:
        chain = ticker.option_chain(exp)
        for df, otype in [(chain.calls, 'call'), (chain.puts, 'put')]:
            temp = df.copy()
            temp['type'] = otype
            temp['expirationDate'] = pd.to_datetime(exp)
            all_options.append(temp)
    options_df = pd.concat(all_options)
    today = datetime.today()
    options_df['T'] = (options_df['expirationDate'] - today).dt.days / 365
    return options_df, S

ticker_symbol = st.sidebar.text_input('Ticker Symbol', value='QQQ')
r = st.sidebar.number_input('Risk-free Rate', value=0.05, step=0.01, format="%.4f")

options_df, S = fetch_options(ticker_symbol)
st.write(f"### Current Stock Price: ${S:.2f}")

# Filter strikes ±55 from current price
strike_lower = S - 55
strike_upper = S + 55
options_df = options_df[(options_df['strike'] >= strike_lower) & (options_df['strike'] <= strike_upper)]

# =============================
# Compute IVs
# =============================
@st.cache_data
def compute_ivs(options_df, S, r):
    ivs, strikes, expiries, market_IVs, types = [], [], [], [], []  # Added types list
    today = datetime.today()
    for idx, row in options_df.iterrows():
        K = row['strike']
        T = row['T']
        price = row['lastPrice']
        otype = row['type']
        market_iv = row.get('impliedVolatility', np.nan)
        own_iv = solve_iv(S, K, T, r, price, otype=otype)
        if not np.isnan(own_iv):
            ivs.append(own_iv)
            strikes.append(K)
            expiries.append(row['expirationDate'])
            market_IVs.append(market_iv)
            types.append(otype)  # Append the option type
    df = pd.DataFrame({
        'Strike': strikes,
        'Expiration': expiries,
        'ComputedIV': ivs,
        'MarketIV': market_IVs,
        'Type': types  # Add Type column
    })
    return df

comparison_df = compute_ivs(options_df, S, r)
st.write("### Options IV Comparison (Strikes ±55)")
st.dataframe(comparison_df)

# =============================
# 2D Conversion Plot (Strike vs Market IV and Computed IV)
# =============================
fig2d = go.Figure()
fig2d.add_trace(go.Scatter(x=comparison_df['Strike'], y=comparison_df['ComputedIV'],
                           mode='markers', name='Computed IV', marker=dict(color='blue', size=6)))
fig2d.add_trace(go.Scatter(x=comparison_df['Strike'], y=comparison_df['MarketIV'],
                           mode='markers', name='Market IV', marker=dict(color='red', size=6)))
fig2d.update_layout(title='2D Conversion Plot (Strike vs Market IV and Computed IV)',
                    xaxis_title='Strike', yaxis_title='Implied Volatility')
st.plotly_chart(fig2d, use_container_width=True)

# =============================
# IV Skew for Different Expirations
# =============================
st.write("### IV Skew for Different Expirations")

fig_skew = go.Figure()

for exp in comparison_df['Expiration'].unique():
    df_exp = comparison_df[comparison_df['Expiration'] == exp]
    df_exp = df_exp.groupby('Strike').agg({'ComputedIV': 'mean', 'MarketIV': 'mean'}).reset_index()
    df_exp = df_exp.sort_values('Strike')
    fig_skew.add_trace(go.Scatter(x=df_exp['Strike'], y=df_exp['MarketIV'],
                                  mode='lines+markers', name=f'Market {exp.date()}'))
    fig_skew.add_trace(go.Scatter(x=df_exp['Strike'], y=df_exp['ComputedIV'],
                                  mode='lines+markers', name=f'Computed {exp.date()}', line=dict(dash='dash')))

fig_skew.update_layout(title='IV Skew for Different Expirations',
                       xaxis_title='Strike', yaxis_title='Implied Volatility')
st.plotly_chart(fig_skew, use_container_width=True)

# =============================
# 3D IV Surfaces
# =============================
moneyness = S / comparison_df['Strike'].values
dtes = np.array([(exp - datetime.today()).days / 365 for exp in comparison_df['Expiration']])

m_grid = np.linspace(moneyness.min(), moneyness.max(), 50)
dte_grid = np.linspace(dtes.min(), dtes.max(), 50)
M, D = np.meshgrid(m_grid, dte_grid)

IV_computed_grid = gaussian_filter(griddata((moneyness, dtes), comparison_df['ComputedIV'].values, (M,D), method='linear'), sigma=1)
IV_market_grid = gaussian_filter(griddata((moneyness, dtes), comparison_df['MarketIV'].values, (M,D), method='linear'), sigma=1)

# Computed IV Surface
fig_computed = go.Figure()
fig_computed.add_trace(go.Surface(x=M, y=D, z=IV_computed_grid, colorscale='Viridis', name='Computed IV'))
fig_computed.update_layout(title='Calculated IV Surface',
                           scene=dict(xaxis_title='Moneyness', yaxis_title='Time to Expiry', zaxis_title='IV'))
st.plotly_chart(fig_computed, use_container_width=True)

# Market IV Surface
fig_market = go.Figure()
fig_market.add_trace(go.Surface(x=M, y=D, z=IV_market_grid, colorscale='Plasma', name='Market IV'))
fig_market.update_layout(title='Market IV Surface',
                         scene=dict(xaxis_title='Moneyness', yaxis_title='Time to Expiry', zaxis_title='IV'))
st.plotly_chart(fig_market, use_container_width=True)

# Combined IV Surface (Computed and Market IV in same space)
fig_combined = go.Figure()
fig_combined.add_trace(go.Surface(x=M, y=D, z=IV_computed_grid, colorscale='Viridis', name='Computed IV', opacity=0.5))
fig_combined.add_trace(go.Surface(x=M, y=D, z=IV_market_grid, colorscale='Plasma', name='Market IV', opacity=0.5))
fig_combined.update_layout(title='Combined IV Surface (Computed and Market IV)',
                           scene=dict(xaxis_title='Moneyness', yaxis_title='Time to Expiry', zaxis_title='IV'))
st.plotly_chart(fig_combined, use_container_width=True)