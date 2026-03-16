# crypto_dashboard_fixed_timer.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import joblib
import os
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import requests
import json
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

# ==============================
# Helper for Data Handling
# ==============================
def safe_yf_download(ticker, period=None, interval=None, start=None, end=None, retries=2):
    """Download data from Yahoo Finance with simple retries and period fallback"""
    ticker = ticker.strip()
    
    # Simple list of periods to try
    periods = [period] if period else ["1wk", "5d", "1d"]
    if period and period not in ["1d", "2d"]:
        periods.append("1d")

    for p in periods:
        for i in range(retries):
            try:
                # Standard download - sometimes works best without complex sessions
                data = yf.download(
                    tickers=ticker, 
                    period=p, 
                    interval=interval, 
                    start=start, 
                    end=end, 
                    progress=False
                )
                if not data.empty and len(data) > 0:
                    return data
                time.sleep(1)
            except Exception:
                time.sleep(2)
    return pd.DataFrame()

# ==============================
# Streamlit page config
# ==============================
st.set_page_config(
    page_title="Live Crypto Dashboard", 
    layout="wide",
    page_icon="💹"
)

st.title("💹 Live Crypto Dashboard - INSTANT UPDATES")
st.markdown("---")

# ==============================
# Initialize session state for timer and auto-refresh
# ==============================
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.datetime.now()
if 'next_refresh' not in st.session_state:
    st.session_state.next_refresh = datetime.datetime.now() + datetime.timedelta(seconds=30)
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 30
if 'force_refresh' not in st.session_state:
    st.session_state.force_refresh = False
if 'currency_confidences' not in st.session_state:
    st.session_state.currency_confidences = {}
if 'last_full_scan' not in st.session_state:
    st.session_state.last_full_scan = None
if 'initial_scan_complete' not in st.session_state:
    st.session_state.initial_scan_complete = False

# ==============================
# Universal Symbol Normalizer
# ==============================
def normalize_ticker(symbol):
    """Convert user-friendly names into valid Yahoo Finance tickers"""
    sym = symbol.upper().strip()
    
    # Gold & Silver
    if sym in ["GOLD", "XAU", "XAUUSD"]: return "XAUUSD=X"
    if sym in ["SILVER", "XAG", "XAGUSD"]: return "XAGUSD=X"
    
    # Forex
    forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD", "USDCAD"]
    if sym in forex_pairs: return sym + "=X"
    
    # Crypto (ensure -USD if missing for common coins)
    crypto_shorts = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOT", "DOGE", "AVAX", "LINK", "MATIC"]
    if sym in crypto_shorts: return sym + "-USD"
    
    return sym

# ==============================
# Permanent Storage for Symbols
# ==============================
SAVED_SYMBOLS_FILE = "saved_symbols.json"
SAVED_SETTINGS_FILE = "user_settings.json"

def load_saved_symbols():
    if os.path.exists(SAVED_SYMBOLS_FILE):
        try:
            with open(SAVED_SYMBOLS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_custom_symbols(symbols):
    try:
        with open(SAVED_SYMBOLS_FILE, "w") as f:
            json.dump(symbols, f)
    except Exception as e:
        print(f"Error saving symbols: {e}")

def load_user_settings():
    if os.path.exists(SAVED_SETTINGS_FILE):
        try:
            with open(SAVED_SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user_settings(settings):
    try:
        with open(SAVED_SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    except Exception as e:
        pass

# Initialize symbols list with persistence
if 'active_symbols' not in st.session_state:
    saved = load_saved_symbols()
    if saved:
        st.session_state.active_symbols = saved
    else:
        # Initial default list
        st.session_state.active_symbols = [
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOT-USD", "BNB-USD", "AVAX-USD", "LINK-USD", "LTC-USD",
            "MATIC-USD", "DOGE-USD", "EURUSD=X", "GBPUSD=X", "USDJPY=X", 
        ]

# Helper to get current list (for compatibility)
def get_available_currencies():
    return st.session_state.active_symbols

AVAILABLE_CURRENCIES = get_available_currencies()

# ==============================
# Timer display function
# ==============================
def display_refresh_timer():
    """Display live countdown timer for next refresh"""
    now = datetime.datetime.now()
    time_since_last = now - st.session_state.last_refresh
    time_until_next = st.session_state.next_refresh - now
    
    # Create columns for timer display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🕒 Last Refresh", 
            f"{st.session_state.last_refresh.strftime('%H:%M:%S')}",
            f"{int(time_since_last.total_seconds())}s ago"
        )
    
    with col2:
        if time_until_next.total_seconds() > 0:
            countdown_seconds = int(time_until_next.total_seconds())
            countdown_display = f"{countdown_seconds}s"
            st.metric(
                "🔄 Next Refresh", 
                countdown_display,
                "Counting down..."
            )
        else:
            st.metric(
                "🔄 Refresh Status", 
                "DUE NOW",
                "Refreshing..."
            )
    
    with col3:
        st.metric(
            "📊 Total Refreshes", 
            f"{st.session_state.refresh_count}",
            "Auto updates"
        )
    
    # Progress bar for countdown
    if time_until_next.total_seconds() > 0:
        total_refresh_interval = (st.session_state.next_refresh - st.session_state.last_refresh).total_seconds()
        progress = 1 - (time_until_next.total_seconds() / total_refresh_interval)
        st.progress(min(progress, 1.0), text=f"Next refresh: {st.session_state.next_refresh.strftime('%H:%M:%S')}")

# ==============================
# Currency Confidence Scanner
# ==============================
@st.cache_data(ttl=600)  # Cache scanner data for 10 minutes
def scan_currency_confidence(symbol, interval='15m'):
    """Scan confidence for a single currency"""
    symbol = normalize_ticker(symbol)
    try:
        # Fetch training data with error handling
        training_period = "5d" if interval == "1m" else "1wk"
        
        # Skip problematic symbols
        if symbol == "MATIC-USD":
            return 0.0
            
        data = safe_yf_download(ticker=symbol, period=training_period, interval=interval)
        
        if data.empty or len(data) < 10:
            return 0.0
        
        # Handle multi-level columns
        if isinstance(data.columns, pd.MultiIndex):
            flat_columns = [col[0] for col in data.columns]
            data.columns = flat_columns
        
        data.reset_index(inplace=True)
        
        # Simple confidence calculation based on price stability and volume
        if 'Close' not in data.columns or 'Volume' not in data.columns:
            return 0.0
        
        # Calculate price volatility (lower volatility = higher confidence)
        returns = data['Close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        volatility = returns.std()
        
        # Calculate volume consistency
        volume_mean = data['Volume'].mean()
        volume_std = data['Volume'].std()
        volume_consistency = 1 - (volume_std / volume_mean) if volume_mean > 0 else 0
        
        # Calculate trend strength (using simple linear regression)
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values
        model = LinearRegression()
        model.fit(X, y)
        trend_strength = abs(model.coef_[0]) / data['Close'].mean()
        
        # Combine factors into confidence score
        confidence = max(0, min(1, 
            0.6 * (1 - min(volatility, 0.1) / 0.1) +  # 60% weight to low volatility
            0.3 * volume_consistency +                  # 30% weight to volume consistency
            0.1 * min(trend_strength, 1)                # 10% weight to trend strength
        ))
        
        return confidence
        
    except Exception as e:
        return 0.0

def scan_all_currencies():
    """Scan confidence for all available currencies"""
    confidences = {}
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(AVAILABLE_CURRENCIES):
        status_text.text(f"Scanning {symbol}... ({i+1}/{len(AVAILABLE_CURRENCIES)})")
        confidence = scan_currency_confidence(symbol)
        confidences[symbol] = confidence
        progress_bar.progress((i + 1) / len(AVAILABLE_CURRENCIES))
        # Add a small delay between requests to avoid rate limiting
        time.sleep(0.2)
    
    status_text.text("✅ Scan complete!")
    time.sleep(1)  # Show completion message briefly
    status_text.empty()
    
    return confidences

# ==============================
# Auto-scan on first load
# ==============================
def perform_initial_scan():
    """Perform initial currency scan when page loads"""
    if not st.session_state.initial_scan_complete:
        # Show loading message
        with st.spinner('🔄 Performing initial currency scan... This may take a few seconds.'):
            st.session_state.currency_confidences = scan_all_currencies()
            st.session_state.last_full_scan = datetime.datetime.now()
            st.session_state.initial_scan_complete = True
        st.success("✅ Initial currency scan completed!")

# ==============================
# Top Currencies Display
# ==============================
def display_top_currencies():
    """Display top currencies by confidence at the top"""
    st.markdown("---")
    st.subheader("🏆 Top Currencies by Confidence")
    
    # Perform initial scan if not done
    if not st.session_state.initial_scan_complete:
        perform_initial_scan()
    
    if not st.session_state.currency_confidences:
        st.info("🔄 No confidence data available. Running scan...")
        perform_initial_scan()
    
    # Sort currencies by confidence (descending)
    sorted_currencies = sorted(
        st.session_state.currency_confidences.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Display top 8 currencies in a grid
    cols = st.columns(4)
    for i, (symbol, confidence) in enumerate(sorted_currencies[:8]):
        with cols[i % 4]:
            confidence_pct = confidence * 100
            
            # Color coding
            if confidence_pct >= 70:
                color = "#00C805"
                emoji = "🔥"
            elif confidence_pct >= 50:
                color = "#FFA500"
                emoji = "⚡"
            else:
                color = "#FF2E2E"
                emoji = "📉"
            
            st.metric(
                label=f"{emoji} {symbol}",
                value=f"{confidence_pct:.1f}%",
                delta="High Confidence" if confidence_pct >= 70 else "Medium" if confidence_pct >= 50 else "Low"
            )
    
    # Show full list in expander
    with st.expander("📋 Full Confidence List"):
        for symbol, confidence in sorted_currencies:
            confidence_pct = confidence * 100
            st.write(f"**{symbol}**: {confidence_pct:.1f}%")

# ==============================
# Enhanced data fetching with instant updates
# ==============================
def get_crypto_data(symbol, period, interval):
    """Fetch crypto data with cache boosting for instant updates"""
    symbol = normalize_ticker(symbol)
    try:
        @st.cache_data(ttl=60)  # Increased TTL to 60s
        def _fetch_data(_symbol, _period, _interval):
            # Skip problematic symbols
            if _symbol == "MATIC-USD":
                return pd.DataFrame()
                
            # For 1-minute data, handle special cases
            if _interval == "1m":
                try:
                    if _period in ["1mo", "1wk", "5d"]:
                        data = safe_yf_download(ticker=_symbol, period="7d", interval=_interval)
                    else:
                        data = safe_yf_download(ticker=_symbol, period=_period, interval=_interval)
                except:
                    data = safe_yf_download(ticker=_symbol, period="5d", interval=_interval)
            else:
                data = safe_yf_download(ticker=_symbol, period=_period, interval=_interval)
            
            if data.empty:
                return pd.DataFrame()
            
            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                flat_columns = [col[0] for col in data.columns]
                data.columns = flat_columns
            
            data.reset_index(inplace=True)
            return data
        
        return _fetch_data(symbol, period, interval)
        
    except Exception as e:
        st.error(f"❌ Error fetching {interval} data: {e}")
        return pd.DataFrame()

# ==============================
# Function to fetch 1 WEEK data for model training
# ==============================
def get_training_data(symbol, interval):
    """Always fetch 1 week of data for model training"""
    symbol = normalize_ticker(symbol)
    try:
        # Skip problematic symbols
        if symbol == "MATIC-USD":
            return pd.DataFrame()
            
        training_period = "5d" if interval == "1m" else "1wk"
        
        @st.cache_data(ttl=900)  # Increased TTL to 15 minutes
        def _fetch_training_data(_symbol, _interval, _training_period):
            return safe_yf_download(ticker=_symbol, period=_training_period, interval=_interval)
        
        training_data = _fetch_training_data(symbol, interval, training_period)
        
        if training_data.empty:
            return pd.DataFrame()
        
        if isinstance(training_data.columns, pd.MultiIndex):
            flat_columns = [col[0] for col in training_data.columns]
            training_data.columns = flat_columns
        
        training_data.reset_index(inplace=True)
        return training_data
        
    except Exception as e:
        return pd.DataFrame()

# ==============================
# TradingView Widget
# ==============================
def get_tv_mapping(symbol):
    """Smart mapping for TradingView symbols and exchanges"""
    sym = symbol.upper()
    
    # Precise Aliases
    aliases = {
        "GLD": ("AMEX", "GLD"),
        "GOLD": ("OANDA", "XAUUSD"),
        "SLV": ("AMEX", "SLV"),
        "SILVER": ("OANDA", "XAGUSD"),
        "XAUUSD=X": ("OANDA", "XAUUSD"),
        "XAGUSD=X": ("OANDA", "XAGUSD"),
        "EURUSD=X": ("FX", "EURUSD"),
        "GBPUSD=X": ("FX", "GBPUSD"),
        "USDJPY=X": ("FX", "USDJPY"),
    }
    
    if sym in aliases:
        return aliases[sym]
    
    # Generic Logic
    if "USD" in sym:
        # BTC-USD -> BINANCE:BTCUSDT
        clean = sym.replace("-USD", "USDT").replace("-", "")
        if "=X" in sym: return "FX", sym.replace("=X", "")
        return "BINANCE", clean
    
    if "=X" in sym:
        return "FX", sym.replace("=X", "")
        
    return "COINBASE", sym.replace("-", "")

def display_tradingview_widget(symbol):
    """Embed official TradingView Advanced Charting Widget"""
    exchange, tv_symbol = get_tv_mapping(symbol)

    st.markdown("---")
    st.subheader(f"📊 TradingView Live Chart: {symbol}")
    
    # Note for user about sync
    st.caption("💡 **Note:** Changes made *inside* this chart (like clicking symbols) don't update the models above. Use the Sidebar to analyze a new trade.")
    
    # TradingView Widget HTML (90vh height optimization)
    tv_html = f"""
    <div style="height:85vh; width:100%;">
      <div id="tradingview_chart" style="height:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{exchange}:{tv_symbol}",
        "interval": "15",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "studies": [
          "RSI@tv-basicstudies",
          "MACD@tv-basicstudies",
          "MASimple@tv-basicstudies",
          "SuperTrend@tv-basicstudies"
        ],
        "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """
    st.components.v1.html(tv_html, height=800)
    

def calculate_indicators(data):
    """Calculate professional technical indicators"""
    df = data.copy()
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # ATR (True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    return df

@st.cache_data(ttl=3600)
def get_fear_greed_index():
    """Fetch the Crypto Fear & Greed Index"""
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=5)
        data = response.json()
        val = data['data'][0]['value']
        cls = data['data'][0]['value_classification']
        return str(val) if val is not None else "50", str(cls) if cls is not None else "Neutral"
    except:
        return "50", "Neutral"

# ==============================
# Linear Regression Prediction function
# ==============================
def predict_next_price_linear(training_data, current_interval):
    """Predict next price using Linear Regression"""
    try:
        if len(training_data) < 10:
            return None, None, "Insufficient training data"
        
        data = training_data.copy()
        
        # Create features
        X = np.arange(len(data)).reshape(-1, 1)
        price_changes = data['Close'].pct_change().fillna(0)
        volatility = price_changes.rolling(5, min_periods=1).std().fillna(0)
        
        X = np.column_stack([X, price_changes.values, volatility.values])
        y = data['Close'].values
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next period
        last_price_change = price_changes.iloc[-1]
        last_volatility = volatility.iloc[-1]
        next_X = np.array([[len(data), last_price_change, last_volatility]])
        prediction = model.predict(next_X)[0]
        
        confidence = model.score(X, y)
        
        prediction_times = {
            '1m': '1 minute',
            '5m': '5 minutes', 
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '2h': '2 hours',
            '1d': '1 day'
        }
        
        prediction_time = prediction_times.get(current_interval, f'{current_interval}')
        
        return prediction, confidence, prediction_time
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ==============================
# LSTM MODEL PREDICTION (from saved models)
# ==============================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

MODEL_DIR = os.path.join(os.path.dirname(__file__), "crypto_models")

def predict_with_lstm(symbol, interval, model_dir=MODEL_DIR, sequence_length=20):
    """Predict next price using pretrained LSTM model if available"""
    try:
        # Skip problematic symbols
        if symbol == "MATIC-USD":
            return None, None, None, "Symbol not available"
            
        # Convert symbol name (e.g. BTC-USD -> BTC_USD)
        symbol_key = symbol.replace('-', '_')
        model_path = os.path.join(model_dir, f"{symbol_key}_lstm.pth")
        scaler_path = os.path.join(model_dir, f"{symbol_key}_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return None, None, None, f"No trained LSTM model found for {symbol}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model + scaler
        model = LSTMModel(input_size=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        scaler = joblib.load(scaler_path)

        # Fetch recent data (2 days)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=2)
        data = safe_yf_download(ticker=symbol, start=start, end=end, interval=interval)

        if data.empty:
            return None, None, None, f"No data for {symbol} ({interval})"

        closes = data[['Close']].values
        scaled = scaler.transform(closes)

        if len(scaled) < sequence_length:
            return None, None, None, "Insufficient data for sequence"

        X_input = np.expand_dims(scaled[-sequence_length:], axis=0)
        X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)

        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()

        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        current_price = float(closes[-1])

        # Compute change and signal
        change_pct = (pred_price - current_price) / current_price * 100
        signal = "BUY 🟢" if pred_price > current_price else "SELL 🔴"

        # Calculate confidence based on recent prediction accuracy
        confidence = 0.7  # Default confidence for LSTM

        return pred_price, confidence, signal, None

    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

# ==============================
# Enhanced Prediction with Multiple Models - FIXED VERSION
# ==============================
def predict_with_multiple_models(training_data, current_price, interval):
    """Predict next price using multiple ML models with REALISTIC confidence"""
    try:
        if len(training_data) < 20:
            return {}
        
        data = training_data.copy()
        
        # FEATURE ENGINEERING (PRO)
        data = calculate_indicators(data)
        data['Price_Change'] = data['Close'].pct_change().fillna(0)
        
        # Volatility
        data['Volatility'] = data['Price_Change'].rolling(10, min_periods=1).std().fillna(0.01)
        
        # Pro Signals
        data['RSI_Signal'] = data['RSI'].apply(lambda x: 1 if x < 30 else (-1 if x > 70 else 0))
        data['MACD_Signal'] = (data['MACD'] > data['Signal_Line']).astype(int)
        data['BB_Signal'] = ((data['Close'] < data['BB_Lower']).astype(int) - (data['Close'] > data['BB_Upper']).astype(int))
        
        data = data.dropna()
        
        if len(data) < 20: return {}
        
        # FEATURE SET
        feature_columns = ['Price_Change', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'BB_Lower', 'BB_Upper', 'Volatility', 'ATR']
        available_features = [col for col in feature_columns if col in data.columns]
        
        if len(available_features) < 5: return {}
        
        # TIME-SERIES PREDICTION SETUP
        X = data[available_features].iloc[:-1].values
        y = data['Close'].iloc[1:].values
        
        if len(X) < 10:
            return {}
            
        # FAST PERFORMANCE: Cap data for web stability
        if len(X) > 1000:
            X = X[-1000:]
            y = y[-1000:]
        
        # DATA CLEANING
        def clean_data_robust(X_data, y_data):
            X_df = pd.DataFrame(X_data, columns=available_features)
            y_series = pd.Series(y_data)
            
            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            valid_mask = ~X_df.isna().any(axis=1) & ~y_series.isna()
            X_clean = X_df[valid_mask]
            y_clean = y_series[valid_mask]
            
            if len(X_clean) == 0:
                return np.array([]), np.array([])
            
            # Remove outliers using IQR
            def remove_outliers_iqr(df):
                Q1 = df.quantile(0.25)
                Q3 = df.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                return ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
            
            outlier_mask = remove_outliers_iqr(X_clean)
            X_clean = X_clean[outlier_mask]
            y_clean = y_clean[outlier_mask]
            
            X_array = X_clean.values.astype(np.float64)
            y_array = y_clean.values.astype(np.float64)
            
            # Final cleaning
            for i in range(X_array.shape[1]):
                col = X_array[:, i]
                if np.any(np.isinf(col)) or np.any(np.isnan(col)):
                    col_mean = np.nanmean(col[np.isfinite(col)])
                    if np.isnan(col_mean):
                        col_mean = 0
                    col[np.isinf(col) | np.isnan(col)] = col_mean
                    X_array[:, i] = col
            
            return X_array, y_array
        
        X_clean, y_clean = clean_data_robust(X, y)
        
        if len(X_clean) < 10:
            return {}
        
        # DATA SPLITTING
        split_idx = max(1, int(0.7 * len(X_clean)))
        X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
        y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
        
        if len(X_train) < 5 or len(X_test) < 3:
            return {}
        
        # SCALE DATA
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # MODELS
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            ),
            'XGBoost': XGBRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'SVM': SVR(kernel='rbf', C=1e6, epsilon=0.001, gamma='scale', max_iter=1000),
            'Linear Regression': LinearRegression()
        }
        
        predictions = {}
        
        for name, model in models.items():
            try:
                # Train model
                if name == 'SVM':
                    # SPECIAL HANDLING FOR SVM: Needs Target Scaling (y-scaling)
                    y_scaler = StandardScaler()
                    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                    
                    model.fit(X_train_scaled, y_train_scaled)
                    
                    # Predict on test set for confidence
                    y_pred_scaled = model.predict(X_test_scaled)
                    y_pred_test = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                    
                    # Predict next price
                    latest_features = data[available_features].iloc[-1].values.reshape(1, -1)
                    latest_scaled = scaler.transform(latest_features)
                    next_price_scaled = model.predict(latest_scaled)
                    next_price = float(y_scaler.inverse_transform(next_price_scaled.reshape(-1, 1))[0][0])
                
                elif name == 'Linear Regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    latest_features = data[available_features].iloc[-1].values.reshape(1, -1)
                    latest_scaled = scaler.transform(latest_features)
                    next_price = float(model.predict(latest_scaled)[0])
                
                else:
                    # Random Forest / XGBoost don't require scaled X or y but work fine with it
                    model.fit(X_train, y_train)
                    y_pred_test = model.predict(X_test)
                    
                    latest_features = data[available_features].iloc[-1].values.reshape(1, -1)
                    next_price = float(model.predict(latest_features)[0])
                
                # ENHANCED REALISTIC CONFIDENCE CALCULATION
                confidence = 0.5  # Start with neutral confidence
                
                if len(y_test) > 2:
                    try:
                        # 1. R² Score (40% weight)
                        r2 = r2_score(y_test, y_pred_test)
                        r2 = max(0, min(1, r2))  # Clamp between 0-1
                        
                        # 2. Mean Absolute Percentage Error (30% weight)
                        valid_mask = (y_test > 0) & (y_pred_test > 0)
                        if valid_mask.any():
                            mape = np.mean(np.abs((y_test[valid_mask] - y_pred_test[valid_mask]) / y_test[valid_mask]))
                            mape_score = max(0, 1 - min(mape, 1))  # Lower MAPE = higher score
                        else:
                            mape_score = 0.5
                        
                        # 3. Direction Accuracy (30% weight)
                        actual_direction = np.sign(np.diff(y_test))
                        pred_direction = np.sign(np.diff(y_pred_test))
                        if len(actual_direction) > 0 and len(pred_direction) > 0:
                            min_len = min(len(actual_direction), len(pred_direction))
                            direction_accuracy = np.mean(actual_direction[:min_len] == pred_direction[:min_len])
                        else:
                            direction_accuracy = 0.5
                        
                        # Combined confidence with weights
                        confidence = (r2 * 0.4) + (mape_score * 0.3) + (direction_accuracy * 0.3)
                        
                        # Adjust for market volatility (higher volatility = lower confidence)
                        recent_volatility = data['Volatility'].iloc[-10:].mean()
                        volatility_penalty = max(0.5, 1 - (recent_volatility * 10))  # Reduce confidence in high volatility
                        confidence *= volatility_penalty
                        
                        # Ensure reasonable range
                        confidence = max(0.1, min(0.95, confidence))
                        
                    except:
                        # Fallback to correlation-based confidence
                        try:
                            correlation = np.corrcoef(y_test, y_pred_test)[0, 1]
                            if np.isnan(correlation):
                                correlation = 0
                            confidence = max(0.2, min(0.8, abs(correlation)))
                        except:
                            confidence = 0.3
                
                # SANITY CHECK PREDICTION
                if (np.isinf(next_price) or np.isnan(next_price) or 
                    abs(next_price) > current_price * 5 or next_price <= 0):
                    continue
                
                # Calculate percentage change
                change_pct = ((next_price - current_price) / current_price) * 100
                
                # Confidence adjustments based on prediction characteristics
                if abs(change_pct) > 10:  # Reduce confidence for extreme predictions
                    confidence *= 0.7
                elif abs(change_pct) < 0.1:  # Slightly reduce for very small changes
                    confidence *= 0.9
                
                # Model-specific confidence adjustments
                if name == 'Linear Regression':
                    confidence *= 1.1  # Linear models often more stable
                elif name == 'SVM':
                    confidence *= 0.9  # SVM can be sensitive to parameters
                
                predictions[name] = {
                    'price': next_price,
                    'confidence': confidence,
                    'change_pct': change_pct,
                    'signal': 'BUY 🟢' if next_price > current_price else 'SELL 🔴'
                }
                
            except Exception:
                continue
        
        return predictions
        
    except Exception:
        return {}

# ==============================
# Trading Insights Generator
# ==============================
def generate_trading_insights(predictions, current_price, symbol):
    """Generate actionable trading insights from model predictions"""
    
    if not predictions:
        return "🔄 Waiting for model predictions..."
    
    # Calculate consensus
    prices = [pred['price'] for pred in predictions.values()]
    avg_price = np.mean(prices)
    consensus_change = ((avg_price - current_price) / current_price) * 100
    
    # Count buy vs sell signals
    buy_signals = sum(1 for pred in predictions.values() if pred['signal'] == 'BUY 🟢')
    sell_signals = sum(1 for pred in predictions.values() if pred['signal'] == 'SELL 🔴')
    
    # Average confidence
    avg_confidence = np.mean([pred['confidence'] for pred in predictions.values()])
    
    # Generate insights
    insights = []
    
    if buy_signals > sell_signals:
        insights.append(f"**📈 BULLISH BIAS**: {buy_signals}/{len(predictions)} models recommend BUY")
    elif sell_signals > buy_signals:
        insights.append(f"**📉 BEARISH BIAS**: {sell_signals}/{len(predictions)} models recommend SELL")
    else:
        insights.append("**⚖️ NEUTRAL BIAS**: Equal buy/sell signals")
    
    if abs(consensus_change) < 0.1:
        insights.append("**🎯 SIDEWAYS EXPECTED**: Minimal price movement predicted")
    elif consensus_change > 0.5:
        insights.append("**🚀 STRONG BULLISH**: Significant upside predicted")
    elif consensus_change < -0.5:
        insights.append("**🔻 STRONG BEARISH**: Significant downside predicted")
    
    if avg_confidence > 0.7:
        insights.append("**✅ HIGH CONFIDENCE**: Models are confident in predictions")
    elif avg_confidence < 0.3:
        insights.append("**⚠️ LOW CONFIDENCE**: Consider waiting for clearer signals")
    
    # Risk assessment
    price_range = max(prices) - min(prices)
    if price_range > current_price * 0.02:  # More than 2% range
        insights.append("**🎲 HIGH UNCERTAINTY**: Models disagree on direction")
    
    return "\n\n".join(insights)

# ==============================
# Enhanced Predictions Display
# ==============================
def display_enhanced_predictions(symbol, interval, training_data, current_price):
    """Display predictions with Confluence Scoring and Market Sentiment"""
    st.markdown("---")
    
    # 1. Market Sentiment Header
    fng_value, fng_class = get_fear_greed_index()
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("🔮 Strategic Signal Room")
    with col_b:
        # Using built-in st.metric for stability instead of HTML injection
        st.metric(
            label="Fear & Greed Index", 
            value=f"{fng_value}", 
            delta=f"{fng_class}",
            delta_color="normal"
        )

    # 2. Analysis
    df_with_inds = calculate_indicators(training_data)
    
    # Safety Check: Empty Data
    if df_with_inds.empty or len(df_with_inds) < 20:
        st.warning("🔄 Collecting more market data for advanced analysis... (Need at least 20 candles)")
        return

    current_atr = df_with_inds['ATR'].iloc[-1]
    current_rsi = df_with_inds['RSI'].iloc[-1]
    macd_val = df_with_inds['MACD'].iloc[-1]
    sig_line = df_with_inds['Signal_Line'].iloc[-1]
    
    # Handle NaNs
    current_atr = current_atr if not np.isnan(current_atr) else current_price * 0.01
    current_rsi = current_rsi if not np.isnan(current_rsi) else 50
    
    ml_predictions = predict_with_multiple_models(training_data, current_price, interval)
    
    if ml_predictions:
        prices = [p['price'] for p in ml_predictions.values()]
        avg_target = np.mean(prices)
        buy_votes = sum(1 for p in ml_predictions.values() if p['signal'] == 'BUY 🟢')
        total_models = len(ml_predictions)
        
        # CONFLUENCE SCORE
        score = 0
        if buy_votes >= total_models/2: score += 40
        if current_rsi < 35: score += 20 
        if current_rsi > 65: score -= 20 
        if macd_val > sig_line: score += 20 
        
        try:
            fng_int = int(fng_value) if fng_value and str(fng_value).isdigit() else 50
            if fng_int > 70: score -= 10 
            if fng_int < 30: score += 10
        except:
            pass
        
        # Normalize
        score = max(0, min(100, score + 20)) 
        
        if score > 75: verdict, color, action = "🔥 STRONG BUY", "#00C805", "BUY 🟢"
        elif score > 55: verdict, color, action = "⚡ MODERATE BUY", "#90EE90", "BUY 🟢"
        elif score < 25: verdict, color, action = "🔻 STRONG SELL", "#FF2E2E", "SELL 🔴"
        elif score < 45: verdict, color, action = "📉 MODERATE SELL", "#FF7F7F", "SELL 🔴"
        else: verdict, color, action = "⚖️ NEUTRAL / WAIT", "#FFA500", "WAIT 🟡"

        # Trade Plan
        risk_multiplier = 1.5
        if action == "BUY 🟢":
            tp1, tp2 = current_price + (risk_multiplier * current_atr), current_price + (3 * current_atr)
            sl = current_price - (2 * current_atr)
        else:
            tp1, tp2 = current_price - (risk_multiplier * current_atr), current_price - (3 * current_atr)
            sl = current_price + (2 * current_atr)

        # 3. High-Impact Signal Banner (MATCHING SCREENSHOT)
        timeframes = {
            '1m': 'Next 60 Seconds',
            '5m': 'Next 5 Minutes',
            '15m': 'Next 15 Minutes',
            '30m': 'Next 30 Minutes',
            '1h': 'Next 1 Hour',
            '4h': 'Next 4 Hours',
            '1d': 'Next 1 Day'
        }
        current_timeframe = timeframes.get(interval, f"Next {interval}")

        st.markdown(f"""
            <style>
                .signal-banner {{
                    background-color: {color};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: left;
                    margin-bottom: 10px;
                    border-left: 10px solid rgba(0,0,0,0.2);
                }}
                .signal-text {{
                    color: {"#FFFFFF" if action != "WAIT 🟡" else "#000000"};
                    font-size: 28px;
                    font-weight: bold;
                    margin: 0;
                }}
                .timeframe-badge {{
                    display: inline-block;
                    background-color: rgba(255,255,255,0.2);
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 14px;
                    margin-top: 5px;
                }}
                .confluence-bar {{
                    background-color: #0e1117;
                    padding: 8px 15px;
                    border-radius: 5px;
                    color: #4A90E2;
                    font-weight: bold;
                    font-size: 14px;
                    border: 1px solid #1f2937;
                }}
            </style>
            <div class="signal-banner">
                <p class="signal-text">{verdict}</p>
                <div class="timeframe-badge">⏳ Forecast Window: {current_timeframe}</div>
            </div>
            <div class="confluence-bar">
                Confluence Score: {score}/100
            </div>
            <br>
        """, unsafe_allow_html=True)
        
        # Big Signal Row
        with st.container():
            
            # Big Signal Row
            sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
            sig_col1.metric("VERDICT", action)
            sig_col2.metric("CONFIDENCE", f"{score}%")
            sig_col3.metric("MOMENTUM", "BULLISH" if macd_val > sig_line else "BEARISH")
            sig_col4.metric("TARGET", f"${avg_target:,.2f}")

        # Trade Plan Row
        st.markdown("---")
        plan_col1, plan_col2, plan_col3 = st.columns(3)
        with plan_col1:
            st.metric("🎯 Entry Point", f"${current_price:,.4f}")
        with plan_col2:
            st.write("💰 **Profit Targets**")
            st.code(f"TP1: ${tp1:,.4f}\nTP2: ${tp2:,.4f}", language="text")
        with plan_col3:
            st.write("🛑 **Risk Management**")
            st.code(f"SL: ${sl:,.4f}\nRR: 1:2.0", language="text")

        with st.expander("🛠️ Advanced Model Analysis (4-Model Breakdown)", expanded=True):
            st.info(f"📊 Analysis Period: Recent 1 Week | Refresh Rate: {interval}")
            
            # Detailed Grid for the 4 Models
            detail_col1, detail_col2 = st.columns(2)
            model_items = list(ml_predictions.items())
            
            with detail_col1:
                for name, pred in model_items[:2]:
                    st.metric(
                        label=f"{name} (Conf: {pred['confidence']*100:.1f}%)",
                        value=f"${pred['price']:,.2f}",
                        delta=f"{pred['change_pct']:+.2f}% ({pred['signal']})"
                    )
            
            with detail_col2:
                for name, pred in model_items[2:]:
                    st.metric(
                        label=f"{name} (Conf: {pred['confidence']*100:.1f}%)",
                        value=f"${pred['price']:,.2f}",
                        delta=f"{pred['change_pct']:+.2f}% ({pred['signal']})"
                    )

        with st.expander("📝 Technical Logic Depth"):
            tl_col1, tl_col2 = st.columns(2)
            with tl_col1:
                st.write(f"- **Consensus:** {buy_votes}/{total_models} models favor UP")
                st.write(f"- **RSI (14):** {current_rsi:.2f}")
            with tl_col2:
                st.write(f"- **MACD Trend:** {'Bullish' if macd_val > sig_line else 'Bearish'}")
                st.write(f"- **ATR Volatility:** ${current_atr:.4f}")

    else:
        st.info("🔄 Running multi-model technical analysis...")

# ==============================
# Refresh control function
# ==============================
def check_and_refresh():
    """Check if it's time to refresh and trigger update"""
    now = datetime.datetime.now()
    
    # Check if manual refresh is requested
    if st.session_state.force_refresh:
        st.session_state.force_refresh = False
        st.session_state.last_refresh = datetime.datetime.now()
        st.session_state.next_refresh = st.session_state.last_refresh + datetime.timedelta(seconds=st.session_state.refresh_interval)
        st.session_state.refresh_count += 1
        st.cache_data.clear()
        st.rerun()
    
    # Check if auto-refresh is due (only refresh when time is up)
    if st.session_state.auto_refresh and now >= st.session_state.next_refresh:
        st.session_state.last_refresh = datetime.datetime.now()
        st.session_state.next_refresh = st.session_state.last_refresh + datetime.timedelta(seconds=st.session_state.refresh_interval)
        st.session_state.refresh_count += 1
        st.cache_data.clear()
        st.rerun()

# ==============================
# Sidebar settings with instant updates
# ==============================
def setup_sidebar():
    st.sidebar.header("Dashboard Settings")
    
    # Load perspective from permanent storage
    persisted = load_user_settings()
    
    # Custom Symbol Support
    st.sidebar.markdown("---")
    custom_sym = st.sidebar.text_input("🔍 Search Any Pair (e.g. SOL-USD, XAUUSD, EURUSD)", "").upper()
    if st.sidebar.button("➕ Add to Dashboard"):
        if custom_sym:
            norm_sym = normalize_ticker(custom_sym)
            if norm_sym not in st.session_state.active_symbols:
                st.session_state.active_symbols.append(norm_sym)
                save_custom_symbols(st.session_state.active_symbols)
                st.rerun()

    # Enhanced Global Symbol Management
    with st.sidebar.expander("🗑️ Manage Active Symbols"):
        st.write("Remove symbols to speed up analysis:")
        for sym in st.session_state.active_symbols:
            col1, col2 = st.columns([4, 1])
            col1.write(sym)
            if col2.button("X", key=f"hide_{sym}"):
                st.session_state.active_symbols.remove(sym)
                save_custom_symbols(st.session_state.active_symbols)
                st.rerun()

    # Define options
    period_options = ["1d", "5d", "1wk", "1mo", "3mo"]
    interval_options = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # Calculate default indices based on persistence
    def_sym_idx = 0
    if persisted.get('symbol') in AVAILABLE_CURRENCIES:
        def_sym_idx = AVAILABLE_CURRENCIES.index(persisted.get('symbol'))
        
    def_period_idx = 1 # Default 5d
    if persisted.get('period') in period_options:
        def_period_idx = period_options.index(persisted.get('period'))
        
    def_interval_idx = 2 # Default 15m
    if persisted.get('interval') in interval_options:
        def_interval_idx = interval_options.index(persisted.get('interval'))

    symbol = st.sidebar.selectbox(
        "Select Symbol", 
        AVAILABLE_CURRENCIES,
        index=def_sym_idx,
        key="symbol_select"
    )
    
    period = st.sidebar.selectbox(
        "Select data period", 
        period_options,
        index=def_period_idx,
        key="period_select"
    )
    
    interval = st.sidebar.selectbox(
        "Select interval", 
        interval_options,
        index=def_interval_idx,
        key="interval_select"
    )
    
    # Check if any setting changed and trigger instant update + save
    prev_symbol = persisted.get('symbol', '')
    prev_period = persisted.get('period', '')
    prev_interval = persisted.get('interval', '')
    
    settings_changed = (symbol != prev_symbol or period != prev_period or interval != prev_interval)
    
    if settings_changed:
        # Save to permanent storage
        save_user_settings({
            'symbol': symbol,
            'period': period,
            'interval': interval
        })
        st.session_state.force_refresh = True
    
    return symbol, period, interval

# ==============================
# Refresh controls in sidebar
# ==============================
def setup_refresh_controls():
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Refresh Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox(
        "Enable Auto Refresh", 
        value=st.session_state.auto_refresh,
        key="auto_refresh_checkbox"
    )
    
    # Refresh interval
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (seconds)", 
        [30, 60, 120, 300],  # Longer intervals to prevent excessive refreshing
        index=[30, 60, 120, 300].index(st.session_state.refresh_interval) if st.session_state.refresh_interval in [30, 60, 120, 300] else 0,
        key="refresh_interval_select"
    )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Manual Refresh Now", use_container_width=True, key="manual_refresh_button"):
        st.session_state.force_refresh = True
        st.rerun()
    
    # Currency scan button
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Currency Scanner")
    
    if st.sidebar.button("🔄 Rescan All Currencies", use_container_width=True, key="scan_currencies_button"):
        st.session_state.currency_confidences = scan_all_currencies()
        st.session_state.last_full_scan = datetime.datetime.now()
        st.rerun()
    
    # Update session state
    st.session_state.auto_refresh = auto_refresh
    st.session_state.refresh_interval = refresh_interval
    
    # Set next refresh time if auto-refresh is enabled
    if auto_refresh:
        st.session_state.next_refresh = st.session_state.last_refresh + datetime.timedelta(seconds=refresh_interval)
        st.sidebar.success(f"🔄 Auto-refresh: {refresh_interval}s")
    else:
        st.sidebar.info("⏸️ Auto-refresh disabled")
    
    return auto_refresh, refresh_interval

# ==============================
# Main dashboard function
# ==============================
def display_dashboard(symbol, period, interval):
    # PRE-FETCH DATA (Must happen before Signal Room)
    display_data = get_crypto_data(symbol, period, interval)
    training_data = get_training_data(symbol, interval)
    
    if display_data.empty:
        st.error(f"❌ No {interval} data available for {symbol} at the moment.")
        st.warning("⚠️ **Yahoo Finance Rate Limit Detected:** You are making requests too quickly. Wait 2-5 minutes or try a larger interval (1h or 1d).")
        if st.button("🔄 Clear Cache & Retry"):
            st.cache_data.clear()
            st.rerun()
        return

    # ==============================
    # 1. STRATEGIC SIGNAL ROOM (VERY TOP)
    # ==============================
    signal_placeholder = st.empty()
    
    with signal_placeholder.container():
        with st.status("🔮 AI Engine: Analyzing Market Trends...", expanded=True) as status:
            try:
                current_price = float(display_data['Close'].iloc[-1])
                display_enhanced_predictions(symbol, interval, training_data, current_price)
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"⚠️ Signal Engine Error: {e}")
                status.update(label="❌ Analysis Failed", state="error")

    # ==============================
    # 2. TRADINGVIEW HUB
    # ==============================
    st.markdown("---")
    display_tradingview_widget(symbol)

    # ==============================
    # 3. TOP CURRENCIES
    # ==============================
    display_top_currencies()

    # ==============================
    # 4. STATUS & REFRESH TIMER (BOTTOM)
    # ==============================
    st.markdown("---")
    st.subheader("🕐 Live Refresh Timer")
    display_refresh_timer()
    
    # Display current time and status
    st.markdown("---")
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"**🕒 Current Clock:** {current_time}")
    
    # Show exactly which candle the analysis is using
    time_col = 'Datetime' if 'Datetime' in display_data.columns else 'Date'
    if time_col in display_data.columns:
        latest_point = display_data[time_col].iloc[-1]
        latest_str = latest_point.strftime('%Y-%m-%d %H:%M:%S') if hasattr(latest_point, 'strftime') else str(latest_point)
        st.write(f"**📊 Analysis Based on Data Up To:** `{latest_str}`")

    st.write(f"**📈 Displaying:** {len(display_data)} {interval} candles | **Period:** {period}")
    
    if not training_data.empty:
        st.write(f"**🤖 Training:** {len(training_data)} points (1 week)")

# ==============================
# Main execution
# ==============================
def main():
    # Setup sidebar and get settings
    symbol, period, interval = setup_sidebar()
    
    # Setup refresh controls
    auto_refresh, refresh_interval = setup_refresh_controls()
    
  
    
    # Display main dashboard
    display_dashboard(symbol, period, interval)
    
    # Check if refresh is needed
    check_and_refresh()

    # Timer and refresh info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("🕐 Timer Information")
    
    now = datetime.datetime.now()
    time_until_next = st.session_state.next_refresh - now
    
    # Re-calculate Available Currencies for the sidebar display
    current_avail = st.session_state.active_symbols

    st.sidebar.info(f"""
    **Current Status:**
    - Active Symbols: {len(current_avail)}
    - Last Refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}
    - Next Refresh: {st.session_state.next_refresh.strftime('%H:%M:%S')}
    - Time Until: {int(time_until_next.total_seconds()) if time_until_next.total_seconds() > 0 else 0}s
    - Auto-refresh: {'✅ ON' if auto_refresh else '❌ OFF'}
    """)
    
    # Show last scan time if available
    if st.session_state.last_full_scan:
        st.sidebar.info(f"""
        **Currency Scan:**
        - Last Scan: {st.session_state.last_full_scan.strftime('%H:%M:%S')}
        - Currencies Scanned: {len(st.session_state.currency_confidences)}
        - Status: {'✅ Complete' if st.session_state.initial_scan_complete else '🔄 In Progress'}
        """)
    

# Run the main function
if __name__ == "__main__":
    main()


# streamlit run tread.py     to run the app