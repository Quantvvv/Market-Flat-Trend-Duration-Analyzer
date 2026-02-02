import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(layout="wide", page_title="Trend/Flat Time Analyzer")

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data(ttl=300)
def get_binance_data(symbol, timeframe, limit=1000):
    """Получаем больше данных (1000 свечей) для статистики"""
    exchange = ccxt.binance()
    try:
        # Binance отдает максимум 1000 за раз
        bars = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Ошибка получения данных: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_market_correlation(base_symbol='BTC/USDT', alts=['ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT'], timeframe='4h'):
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

# --- МАТЕМАТИКА (HALF-LIFE) ---
def calculate_half_life(series):
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

# --- АНАЛИЗ ИНДИКАТОРОВ ---
def calculate_indicators(df, ma_len, ma_lookback, adx_len, atr_len, range_lookback):
    # 1. MA & Slope
    df['MA'] = ta.sma(df['close'], length=ma_len)
    df['ma_slope_pct'] = (df['MA'] - df['MA'].shift(ma_lookback)) / df['MA'].shift(ma_lookback) * 100
    df['ma_slope_per_bar'] = df['ma_slope_pct'] / ma_lookback
    
    # 2. ADX
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=adx_len)
    if adx_df is not None:
        df = pd.concat([df, adx_df], axis=1)
    else:
        df['ADX_14'] = 0

    # 3. Range Ratio
    df['HH'] = df['high'].rolling(window=range_lookback).max()
    df['LL'] = df['low'].rolling(window=range_lookback).min()
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=atr_len)
    df['range_ratio'] = (df['HH'] - df['LL']) / df['ATR']
    
    return df

# --- АНАЛИЗ ИСТОРИИ ФЛЭТОВ ---
def analyze_historical_flats(df, slope_th, adx_th, range_th):
    """
    Проходит по всей истории и ищет периоды, соответствующие критериям.
    Возвращает статистику длительности.
    """
    # Создаем маску: 1 если флэт, 0 если нет
    # Для истории используем упрощенную модель (без Half-Life и корреляции, так как они тяжелые/внешние)
    # Считаем флэтом, если соблюдаются ХОТЯ БЫ Slope и ADX (база)
    
    # Чтобы не было слишком строго, можно требовать 2 из 3 условий, но для чистоты возьмем основные
    cond_slope = df['ma_slope_per_bar'].abs() < slope_th
    cond_adx = df['ADX_14'] < adx_th
    cond_range = df['range_ratio'] < range_th
    
    # Основное условие флэта в прошлом: Наклон ок + (ADX ок ИЛИ Range ок)
    # Это позволяет находить и тихие флэты, и волатильные боковики
    df['is_flat_hist'] = cond_slope & (cond_adx | cond_range)
    
    # Группируем подряд идущие True
    df['group'] = (df['is_flat_hist'] != df['is_flat_hist'].shift()).cumsum()
    
    # Считаем длительность каждой группы
    flats = df[df['is_flat_hist'] == True].groupby('group').size()
    
    # Отсеиваем "шум" (флэты короче 5 свечей)
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
    
    # Проверяем текущий статус (последняя свеча)
    if df['is_flat_hist'].iloc[-1]:
        stats['is_currently_flat'] = True
        # Находим группу последней свечи
        last_group_id = df['group'].iloc[-1]
        if last_group_id in flats:
            stats['last_flat_len'] = flats[last_group_id]
            
    return stats, df

# --- ИНТЕРФЕЙС ---

st.title("⏳ Time-Based Trend/Flat Analyzer")
st.markdown("Поиск точки входа на основе **продолжительности** исторических флэтов.")

with st.sidebar:
    st.header("Настройки")
    symbol = st.text_input("Пара", "BTC/JPY") # Поставил JPY по дефолту для примера
    tf = st.selectbox("Таймфрейм", ["4h", "1d", "1h"])
    
    st.divider()
    st.caption("Критерии Флэта (Влияют на поиск в истории!)")
    
    ma_len = st.number_input("MA Period", 100, 300, 100)
    slope_thresh = st.number_input("Slope Threshold (%)", 0.01, 0.1, 0.05, step=0.01)
    adx_thresh = st.number_input("ADX Threshold", 15, 60, 45) # Чуть поднял дефолт для волатильных пар
    range_thresh = st.number_input("Range Ratio", 4.0, 15.0, 8.0)
    
    st.divider()
    hl_thresh = st.number_input("Half-Life Threshold", 10, 100, 30)

if st.button("Анализировать Историю и Тренд", type="primary"):
    with st.spinner('Сканируем 1000 свечей истории...'):
        # 1. Загрузка
        df = get_binance_data(symbol, tf, limit=1000)
        if df.empty or len(df) < ma_len:
            st.error("Нет данных или история слишком коротка")
            st.stop()
            
        # 2. Расчет индикаторов
        df = calculate_indicators(df, ma_len, 20, 14, 14, 50)
        
        # 3. Анализ истории флэтов
        hist_stats, df = analyze_historical_flats(df, slope_thresh, adx_thresh, range_thresh)
        
        # 4. Текущие метрики (для Score)
        last_bar = df.iloc[-1]
        half_life = calculate_half_life(df['close'].tail(200))
        btc_corr = get_market_correlation(timeframe=tf) # Тут упростил, не возвращаем таблицу корреляций для скорости

        # --- БЛОК 1: ВРЕМЕННОЙ АНАЛИЗ (САМОЕ ВАЖНОЕ) ---
        st.subheader("⏱️ Анализ Длительности (Time Decay)")
        
        if hist_stats:
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.metric("Всего флэтов найдено", f"{hist_stats['count']}")
                
            with c2:
                avg_bars = int(hist_stats['avg_len'])
                # Перевод в дни/часы
                if tf == '4h': time_str = f"~{int(avg_bars*4/24)} дн."
                elif tf == '1h': time_str = f"~{int(avg_bars)} ч."
                else: time_str = f"~{avg_bars} дн."
                st.metric("Средняя длина", f"{avg_bars} свеч", delta=time_str, delta_color="off")
                
            with c3:
                st.metric("Максимальная длина", f"{hist_stats['max_len']} свеч")
                
            with c4:
                curr_len = hist_stats['last_flat_len']
                if hist_stats['is_currently_flat']:
                    avg = hist_stats['avg_len']
                    progress = min(curr_len / avg, 1.5) # 1.0 = 100% средней длины
                    
                    state_color = "normal"
                    risk_msg = "🟢 Начало цикла"
                    
                    if curr_len > avg * 1.3:
                        state_color = "inverse" # Красный
                        risk_msg = "🔥 ВЫСОКИЙ РИСК ПРОБОЯ"
                    elif curr_len > avg * 0.8:
                        state_color = "off" # Серый/Желтый
                        risk_msg = "⚠️ Зрелый флэт"
                        
                    st.metric("ТЕКУЩИЙ ФЛЭТ", f"{curr_len} свеч", delta=risk_msg, delta_color=state_color)
                else:
                    st.metric("ТЕКУЩИЙ ФЛЭТ", "НЕТ (Тренд)", delta="Ждем условий", delta_color="off")
            
            # Прогресс бар вероятности окончания
            if hist_stats['is_currently_flat']:
                pct = min(int((curr_len / hist_stats['avg_len']) * 100), 100)
                st.write(f"Исчерпание потенциала флэта (относительно среднего): **{pct}%**")
                st.progress(pct)
                if curr_len > hist_stats['avg_len']:
                    st.warning(f"Внимание: Текущий боковик ({curr_len}) уже длиннее среднего исторического ({int(hist_stats['avg_len'])}). Вероятность импульса высока!")
        else:
            st.warning("По текущим критериям в истории не найдено ни одного флэта. Попробуйте смягчить настройки (повысить ADX, Slope).")

        st.divider()

        # --- БЛОК 2: ТЕКУЩИЙ СКОРИНГ (Как раньше) ---
        score = 0
        reasons = []
        
        # Slope
        slope_val = abs(last_bar['ma_slope_per_bar'])
        if slope_val < slope_thresh: score += 1; reasons.append(f"✅ MA Slope: {slope_val:.4f}%")
        else: reasons.append(f"❌ MA Slope: {slope_val:.4f}%")
        
        # ADX
        adx_val = last_bar['ADX_14']
        if adx_val < adx_thresh: score += 1; reasons.append(f"✅ ADX: {adx_val:.1f}")
        else: reasons.append(f"❌ ADX: {adx_val:.1f}")
        
        # Range
        range_val = last_bar['range_ratio']
        if range_val < range_thresh: score += 1; reasons.append(f"✅ Range: {range_val:.1f}")
        else: reasons.append(f"❌ Range: {range_val:.1f}")
        
        # Half-Life
        if half_life < hl_thresh: score += 1; reasons.append(f"✅ Half-Life: {half_life:.1f}")
        else: reasons.append(f"❌ Half-Life: {half_life:.1f}")
        
        # Corr (только для крипты)
        if btc_corr < 0.6: score += 1; reasons.append(f"✅ Corr: {btc_corr:.2f}")
        else: reasons.append(f"❌ Corr: {btc_corr:.2f}")

        c_s1, c_s2 = st.columns([1, 2])
        with c_s1:
            color = "normal" if score >= 3 else "inverse"
            state_text = "ФЛЭТ / ГРИД" if score >= 3 else "ТРЕНД / ОЖИДАНИЕ"
            st.metric("ТЕКУЩИЙ СТАТУС", state_text, f"Score: {score}/5", delta_color=color)
        with c_s2:
            st.caption("Детали:")
            st.text(" | ".join(reasons))

        # --- БЛОК 3: ГРАФИКИ ---
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # Свечи
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA'], line=dict(color='orange'), name=f'MA {ma_len}'), row=1, col=1)
        
        # Подсветка исторических флэтов (Серые зоны)
        # Ищем начало и конец каждой группы True
        df['change'] = df['is_flat_hist'].astype(int).diff()
        starts = df[df['change'] == 1].index
        ends = df[df['change'] == -1].index
        
        # Костыль для отрисовки, если флэт идет прямо сейчас (нет конца)
        if len(starts) > len(ends):
            ends = ends.append(pd.Index([df.index[-1]]))
            
        for s, e in zip(starts, ends):
            # Рисуем только если длительность была заметной (для чистоты графика)
            # Тут можно добавить условие длительности, но оставим все найденные
            fig.add_vrect(x0=s, x1=e, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=1, col=1)

        # Индикатор ADX
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX_14'], line=dict(color='purple'), name='ADX'), row=2, col=1)
        fig.add_hline(y=adx_thresh, line_dash="dash", line_color="red", row=2, col=1)
        
        # Индикатор Slope (визуализация флэтовости)
        # fig.add_trace(go.Scatter(x=df.index, y=df['ma_slope_per_bar'], line=dict(color='yellow'), name='Slope%'), row=2, col=1)
        
        fig.update_layout(height=700, xaxis_rangeslider_visible=False, template="plotly_dark", 
                          title="График с подсветкой исторических флэт-зон (Зеленый фон)")
        st.plotly_chart(fig, use_container_width=True)