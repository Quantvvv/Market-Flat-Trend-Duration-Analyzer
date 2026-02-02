# Market-Flat-Trend-Duration-Analyzer
A financial analytics dashboard that identifies market phases. The unique feature is Time-based analysis: the tool scans 1,000 bars of history to calculate the average duration of sideways movements, helping traders estimate the probability of an imminent breakout. Uses ADX, MA Slope, Correlation, and Half-Life metrics for robust regime detection.
⏳ Market Regime & Consolidation Analyzer

![alt text](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-the-badge&logo=Streamlit&logoColor=white)
![alt text](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![alt text](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

🎯 Core Methodology: The Flat Score
The engine does not rely on a single indicator. Instead, it implements a Voting System based on 5 independent metrics. Each metric contributes to a "Flat Score":
Score ≥ 3: Market is in a Consolidation phase (Ideal for Grid/Mean Reversion).
Score < 3: Market is Trending or highly volatile (High risk for sideways strategies).

🔍 Detailed Metric Breakdown
1. Moving Average Slope (Critical Weight ⭐️)
The "Low-Pass Filter" of the system. It measures the global direction of the market by calculating the angle of a long-term Moving Average (default: 100 SMA).
Logic: If the MA changes by less than a specific threshold (e.g., 0.03% per bar), the trend is considered "horizontal."
Significance: This is the primary trend-following defense.
2. ADX - Average Directional Index (Critical Weight ⭐️)
The industry standard for measuring trend strength regardless of direction.
Logic: e.g., ADX < 20 indicates a "dead" or sideways market. ADX > 25 confirms the presence of a strong trend.
3. Range Ratio (Secondary Filter 🔹)
Measures how tightly the price is compressed within a historical window relative to its volatility (ATR).
Formula: (High_N - Low_N) / ATR
Interpretation: E.g., a ratio < 6.0 suggests an extremely tight sideways channel. A ratio > 10.0 indicates a wide range or trend expansion.
4. Half-Life of Mean Reversion (Advanced Filter 🔹)
An econometric measure using the Ornstein-Uhlenbeck process to estimate the speed at which price returns to its mean.
Logic: A low Half-Life (e.g., < 25 bars) indicates a "spring-like" behavior where price deviations are quickly corrected. A high or infinite Half-Life suggests trending behavior where the price "drifts" away without returning.
5. Cross-Asset Correlation (Crypto-Specific 🔹)
Analyzes the correlation between the target asset and market leaders (e.g., BTC).
Logic:
If you set it to 0.3 (Very Strict): The program will only give a point for flat trading when the market is completely flat. This is super safe.
If you set it to 0.9 (Very Strict): The program will almost always give a point, even if the market is following Bitcoin.
If you set it to 1.0: You effectively turn off this filter.

Parameters need to be adjusted for each coin; visualization is available for this purpose. Adjust the settings until the green zones cover the flat areas. Optimal settings for the BTC/USDT pair:
---

## ⚙️ Recommended Presets

| Parameter	| Stable (4h) BTC 
| :--- | :--- |
| **MA Period** | 100 |
| **Slope Threshold**| 0.05% |
| **ADX Threshold** | 45 |
| **Range Ratio** | 8.0 |
| **Half-Life** | 30 |

---

## 🚀 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/Market-Flat-Trend-Duration-Analyzer.git
    cd Market-Flat-Trend-Duration-Analyzer
    ```

2.  **Install dependencies**:
    You can install all required libraries using the following command:
    ```bash
    pip install streamlit ccxt pandas pandas-ta numpy plotly
    ```

3.  **Launch the dashboard**:
    ```bash
    streamlit run TrendDetermination.py
    ```

---

## 🛠 Tech Stack

*   **Language**: Python 3.9+
*   **Data Source**: **Binance API** (connected via **CCXT** for professional exchange connectivity).
*   **Data Processing**: 
    *   **Pandas**: For time-series data manipulation and cleaning.
    *   **Pandas-TA**: For calculating technical indicators (ADX, ATR, SMA).
    *   **NumPy**: For advanced mathematical calculations (Half-Life, Regression).
*   **Visualization**: 
    *   **Plotly**: For high-performance, interactive financial charts.
    *   **Streamlit**: For building the web-based analytical dashboard.

---

Disclaimer: This tool is for analytical purposes only. Past performance does not guarantee future results. Trading involves significant risk.
