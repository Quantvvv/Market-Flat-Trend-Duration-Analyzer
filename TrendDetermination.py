import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Regime Analyzer")

# --- DATA LOADING ---
@st.cache_data(ttl=300)
def get_binance_data(symbol, timeframe, limit=1000):
    """Fetch historical candle data from Binance"""
    exchange = ccxt.binance()
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_market_correlation(base_symbol='BTC/USDT', alts=['ETH/USDT', 'SOL/USDT', 'BNB/USDT'], timeframe='4h'):
    """Calculate how much the market is moving together (Correlation)"""
    exchange = ccxt.binance()
    btc_df = get_binance_data(base_symbol, timeframe, limit=100)
    if btc_df.empty: return 0
    
    closes = pd.DataFrame({'BTC': btc_df['close']})
    for alt in alts:
        try:
            d = get_binance_data(alt, timeframe, limit=100)
            if not d.empty: closes[alt.split('/')[0]] = d['close']
        except: continue
        
    recent = closes.tail(30)
    corr = recent.corr()
    if 'BTC' not in corr: return 0
    return corr['BTC'].drop('BTC').mean()

# --- MATHEMATICS (MEAN REVERSION SPEED) ---
def calculate_half_life(series):
    """
    Calculates the 'Half-Life' of a price series.
    It shows how fast the price returns to its average.
    Low value = Price is bouncing in a range. High value = Price is trending.
    """
    series_lag = series.shift(1)
    series_diff = series - series_lag
    valid = pd.concat([series_lag, series_diff], axis=1).dropna()
    valid.columns = ['lag', 'diff']
    if valid.empty: return np.inf
    try:
        slope, intercept = np.polyfit(valid['lag'], valid['diff'], 1)
        if slope >= 0: return np.inf
        return -np.log(2) / slope
    except: return np.inf

# --- INDICATOR CALCULATIONS ---
def calculate_indicators(df, ma_len, ma_lookback, adx_len, atr_len, range_lookback):
    # 1. Moving Average & its Slope (Angle)
    df['MA'] = ta.sma(df['close'], length=ma_len)
    df['ma_slope_pct'] = (df['MA'] - df['MA'].shift(ma_lookback)) / df['MA'].shift(ma_lookback) * 100
    df['ma_slope_per_bar'] = df['ma_slope_pct'] / ma_lookback
    
    # 2. ADX (Trend Strength Indicator)
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=adx_len)
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df['ADX_14'] = 0

    # 3. Range Ratio (Size of the range vs Volatility)
    df['HH'] = df['high'].rolling(window=range_lookback).max()
    df['LL'] = df['low'].rolling(window=range_lookback).min()
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
    df['range_ratio'] = (df['HH'] - df['LL']) / df['ATR']
    
    return df

# --- HISTORICAL CONSOLIDATION ANALYSIS ---
def analyze_historical_flats(df, slope_th, adx_th, range_th):
    """
    Scans history to find periods where the market was moving sideways.
    Returns statistics on how long these periods typically last.
    """
    cond_slope = df['ma_slope_per_bar'].abs() < slope_th
    cond_adx = df['ADX_14'] < adx_th
    cond_range = df['range_ratio'] < range_th
    
    # Market is "Flat" if slope is small AND (ADX is low OR the price range is tight)
    df['is_flat_hist'] = cond_slope & (cond_adx | cond_range)
    
    # Group consecutive "Flat" periods
    df['group'] = (df['is_flat_hist'] != df['is_flat_hist'].shift()).cumsum()
    flats = df[df['is_flat_hist'] == True].groupby('group').size()
    
    # Filter out noise (only periods longer than 5 bars)
    valid_flats = flats[flats >= 5]
    
    if valid_flats.empty:
        return None, df
    
    stats = {
        'count': len(valid_flats),
        'avg_len': valid_flats.mean(),
        'median_len': valid_flats.median(),
        'max_len': valid_flats.max(),
        'last_flat_len': 0,
        'is_currently_flat': False
    }
    
    if df['is_flat_hist'].iloc[-1]:
        stats['is_currently_flat'] = True
        last_group_id = df['group'].iloc[-1]
        if last_group_id in flats:
            stats['last_flat_len'] = flats[last_group_id]
            
    return stats, df

# --- USER INTERFACE ---

st.title("⏳ Market Regime & Consolidation Analyzer")
st.markdown("This tool estimates how much time is left before a **breakout** based on historical patterns.")

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input(
        "Ticker", 
        "BTC/USDT", 
        help="Enter the trading pair symbol from Binance (e.g., BTC/USDT or ETH/USDT)."
    )
    tf = st.selectbox(
        "Timeframe", 
        ["4h", "1d", "1h"],
        help="Choose the candle interval. Higher timeframes (1d, 4h) provide more reliable trend signals."
    )
    
    st.divider()
    st.caption("Flat Identification Criteria")
    
    ma_len = st.number_input(
        "MA Period", 100, 300, 100,
        help="Moving Average period used as the trend baseline. Longer periods represent a more significant trend."
    )
    slope_thresh = st.number_input(
        "Slope Threshold (%)", 0.01, 0.1, 0.05, step=0.01,
        help="The maximum allowable angle (slope) of the MA to consider the market 'flat'. Lower values mean a stricter definition of a sideways market."
    )
    adx_thresh = st.number_input(
        "ADX Threshold", 15, 60, 45,
        help="Average Directional Index. Values below this threshold suggest a weak trend, indicating a consolidation phase."
    )
    range_thresh = st.number_input(
        "Range Ratio", 4.0, 15.0, 8.0,
        help="The ratio of the price range (High-Low) to the ATR. A low ratio indicates that the price is tightly compressed."
    )
    
    st.divider()
    hl_thresh = st.number_input(
        "Half-Life Threshold", 10, 100, 30,
        help="Statistically measures the speed of mean reversion. Lower values indicate a stronger tendency for the price to return to its average."
    )


if st.button("Run Historical Analysis", type="primary"):
    with st.spinner('Scanning 1000 candles of history...'):
        df = get_binance_data(symbol, tf, limit=1000)
        if df.empty or len(df) < ma_len:
            st.error("Insufficient data available.")
            st.stop()
            
        df = calculate_indicators(df, ma_len, 20, 14, 14, 50)
        hist_stats, df = analyze_historical_flats(df, slope_thresh, adx_thresh, range_thresh)
        
        last_bar = df.iloc[-1]
        half_life = calculate_half_life(df['close'].tail(200))
        btc_corr = get_market_correlation(timeframe=tf)

        # --- BLOCK 1: TIME DECAY ANALYSIS ---
        st.subheader("⏱️ Consolidation Duration (Time Decay)")
        
        if hist_stats:
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.metric("Total Flats Found", f"{hist_stats['count']}")
            with c2:
                avg_bars = int(hist_stats['avg_len'])
                st.metric("Average Duration", f"{avg_bars} bars")
            with c3:
                st.metric("Max Historical Duration", f"{hist_stats['max_len']} bars")
            with c4:
                curr_len = hist_stats['last_flat_len']
                if hist_stats['is_currently_flat']:
                    avg = hist_stats['avg_len']
                    risk_msg = "🟢 Cycle Start"
                    state_color = "normal"
                    if curr_len > avg * 1.3:
                        state_color = "inverse"
                        risk_msg = "🔥 HIGH BREAKOUT RISK"
                    elif curr_len > avg * 0.8:
                        state_color = "off"
                        risk_msg = "⚠️ Mature Flat"
                    st.metric("CURRENT FLAT", f"{curr_len} bars", delta=risk_msg, delta_color=state_color)
                else:
                    st.metric("CURRENT FLAT", "NONE (Trending)", delta="Waiting for setup", delta_color="off")
            
            # Progress bar for "Exhaustion"
            if hist_stats['is_currently_flat']:
                pct = min(int((curr_len / hist_stats['avg_len']) * 100), 100)
                st.write(f"Consolidation Maturity (vs Historical Average): **{pct}%**")
                st.progress(pct)
                if curr_len > hist_stats['avg_len']:
                    st.warning(f"Warning: Current flat duration ({curr_len}) exceeds historical average ({int(hist_stats['avg_len'])}). A volatility spike is likely!")
        
        st.divider()

        # --- BLOCK 2: REGIME SCORING ---
        score = 0
        reasons = []
        
        if abs(last_bar['ma_slope_per_bar']) < slope_thresh: score += 1; reasons.append("✅ Low Slope")
        if last_bar['ADX_14'] < adx_thresh: score += 1; reasons.append("✅ Low ADX")
        if last_bar['range_ratio'] < range_thresh: score += 1; reasons.append("✅ Tight Range")
        if half_life < hl_thresh: score += 1; reasons.append("✅ Low Half-Life")
        if btc_corr < 0.6: score += 1; reasons.append("✅ Low Correlation")

        c_s1, c_s2 = st.columns([1, 2])
        with c_s1:
            state_text = "FLAT / MEAN REVERSION" if score >= 3 else "TREND / IMPULSE"
            st.metric("CURRENT MARKET REGIME", state_text, f"Score: {score}/5")
        with c_s2:
            st.caption("Diagnostic Details:")
            st.text(" | ".join(reasons))

        # --- BLOCK 3: CHARTS ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA'], line=dict(color='orange'), name='Trend Baseline'), row=1, col=1)
        
        # Highlight historical consolidation zones
        df['change'] = df['is_flat_hist'].astype(int).diff()
        starts = df[df['change'] == 1].index
        ends = df[df['change'] == -1].index
        if len(starts) > len(ends): ends = ends.append(pd.Index([df.index[-1]]))
            
        for s, e in zip(starts, ends):
            fig.add_vrect(x0=s, x1=e, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['ADX_14'], line=dict(color='purple'), name='ADX (Trend Strength)'), row=2, col=1)
        fig.add_hline(y=adx_thresh, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark", 
                          title="Price Chart with Historical Flat Zones Highlighted (Green Areas)")
        st.plotly_chart(fig, use_container_width=True)
