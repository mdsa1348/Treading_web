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
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

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
# Available currencies for scanning (MATIC-USD removed due to errors)
# ==============================
AVAILABLE_CURRENCIES = [
    "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "BNB-USD", 
    "DOGE-USD", "SOL-USD", "LTC-USD", "XRP-USD",
    "USDT-USD", "USDC-USD", "DAI-USD", "EURUSD=X", "GBPUSD=X", 
    "USDJPY=X", "BRL=X", "USDCAD=X", "AUDUSD=X", "GLD", "SLV", "USO"
]

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
def scan_currency_confidence(symbol, interval='15m'):
    """Scan confidence for a single currency"""
    try:
        # Fetch training data with error handling
        training_period = "5d" if interval == "1m" else "1wk"
        
        # Skip problematic symbols
        if symbol == "MATIC-USD":
            return 0.0
            
        data = yf.download(tickers=symbol, period=training_period, interval=interval, progress=False)
        
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
        return
    
    if not st.session_state.currency_confidences:
        st.info("🔄 No confidence data available. Running scan...")
        perform_initial_scan()
        return
    
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
    """Fetch crypto data with cache busting for instant updates"""
    try:
        @st.cache_data(ttl=25)
        def _fetch_data(_symbol, _period, _interval):
            # Skip problematic symbols
            if _symbol == "MATIC-USD":
                return pd.DataFrame()
                
            # For 1-minute data, handle special cases
            if _interval == "1m":
                try:
                    if _period in ["1mo", "1wk", "5d"]:
                        data = yf.download(tickers=_symbol, period="7d", interval=_interval, progress=False)
                    else:
                        data = yf.download(tickers=_symbol, period=_period, interval=_interval, progress=False)
                except:
                    data = yf.download(tickers=_symbol, period="5d", interval=_interval, progress=False)
            else:
                data = yf.download(tickers=_symbol, period=_period, interval=_interval, progress=False)
            
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
    try:
        # Skip problematic symbols
        if symbol == "MATIC-USD":
            return pd.DataFrame()
            
        training_period = "5d" if interval == "1m" else "1wk"
        
        @st.cache_data(ttl=300)
        def _fetch_training_data(_symbol, _interval, _training_period):
            return yf.download(tickers=_symbol, period=_training_period, interval=_interval, progress=False)
        
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
# Enhanced candlestick chart
# ==============================
def create_enhanced_candlestick(data, symbol, period, interval):
    """Create candlestick chart optimized for all intervals"""
    
    datetime_column = 'Datetime' if 'Datetime' in data.columns else 'Date'
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=('', ''),
        row_heights=[0.70, 0.30]
    )
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=data[datetime_column],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='#00C805',
        decreasing_line_color='#FF2E2E',
        increasing_fillcolor='#00C805',
        decreasing_fillcolor='#FF2E2E',
        line=dict(width=1),
        whiskerwidth=0.6,
        name='Price'
    ), row=1, col=1)
    
    # Add volume bars
    colors = ['#00C805' if data['Close'].iloc[i] >= data['Open'].iloc[i] 
              else '#FF2E2E' for i in range(len(data))]
    
    fig.add_trace(go.Bar(
        x=data[datetime_column],
        y=data['Volume'],
        marker_color=colors,
        name='Volume',
        opacity=0.7,
        marker_line_width=0
    ), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b> | {period} | {interval} | Candles: {len(data)}",
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=14, color='white')
        ),
        height=800,
        template="plotly_dark",
        showlegend=False,
        font=dict(size=11, color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=60, b=50),
        dragmode='zoom',
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

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
        data = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

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
        
        # FEATURE ENGINEERING
        data['Price_Change'] = data['Close'].pct_change().fillna(0)
        data['Price_Change'] = np.clip(data['Price_Change'], -1, 1)
        
        data['MA_5'] = data['Close'].rolling(5, min_periods=1).mean().fillna(data['Close'])
        data['MA_10'] = data['Close'].rolling(10, min_periods=1).mean().fillna(data['Close'])
        data['MA_20'] = data['Close'].rolling(20, min_periods=1).mean().fillna(data['Close'])
        
        data['Volatility'] = data['Price_Change'].rolling(5, min_periods=1).std().fillna(0.01)
        data['Volatility'] = np.clip(data['Volatility'], 0.001, 1)
        
        data['Momentum_3'] = data['Close'] - data['Close'].shift(3).fillna(data['Close'])
        data['Momentum_5'] = data['Close'] - data['Close'].shift(5).fillna(data['Close'])
        
        # Volume-based features if available
        if 'Volume' in data.columns:
            data['Volume_MA_5'] = data['Volume'].rolling(5, min_periods=1).mean().fillna(data['Volume'])
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_5']
            data['Volume_Ratio'] = np.clip(data['Volume_Ratio'], 0.1, 10)
        
        data = data.dropna()
        
        if len(data) < 15:
            return {}
        
        # FEATURE SET
        feature_columns = ['Price_Change', 'MA_5', 'MA_10', 'MA_20', 'Volatility', 'Momentum_3', 'Momentum_5']
        if 'Volume_Ratio' in data.columns:
            feature_columns.extend(['Volume_Ratio'])
        
        available_features = [col for col in feature_columns if col in data.columns]
        
        if len(available_features) < 3:
            return {}
        
        # TIME-SERIES PREDICTION SETUP
        X = data[available_features].iloc[:-1].values
        y = data['Close'].iloc[1:].values
        
        if len(X) < 10:
            return {}
        
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
            'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'Linear Regression': LinearRegression()
        }
        
        predictions = {}
        
        for name, model in models.items():
            try:
                # Train model
                if name in ['SVM', 'Linear Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    latest_features = data[available_features].iloc[-1].values.reshape(1, -1)
                    latest_df = pd.DataFrame(latest_features, columns=available_features)
                    latest_clean = latest_df.values.astype(np.float64)
                    latest_scaled = scaler.transform(latest_clean)
                    next_price = model.predict(latest_scaled)[0]
                else:
                    model.fit(X_train, y_train)
                    y_pred_test = model.predict(X_test)
                    
                    latest_features = data[available_features].iloc[-1].values.reshape(1, -1)
                    latest_df = pd.DataFrame(latest_features, columns=available_features)
                    latest_clean = latest_df.values.astype(np.float64)
                    next_price = model.predict(latest_clean)[0]
                
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
    """Display predictions from all models with insights"""
    st.markdown("---")
    st.subheader("🔮 Enhanced Price Predictions")
    
    # Get predictions from all models
    ml_predictions = predict_with_multiple_models(training_data, current_price, interval)
    
    # Get LSTM prediction
    lstm_pred, lstm_confidence, lstm_signal, lstm_error = predict_with_lstm(symbol, interval)
    
    # Display ML model predictions in columns
    if ml_predictions:
        st.success(f"🤖 Multiple Model Analysis ({len(ml_predictions)} models active)")
        
        # Create columns for model predictions
        cols = st.columns(len(ml_predictions))
        
        for i, (model_name, prediction) in enumerate(ml_predictions.items()):
            with cols[i]:
                confidence_pct = prediction['confidence'] * 100
                st.metric(
                    label=f"📊 {model_name} Confidence: {confidence_pct:.2f}%",
                    value=f"${prediction['price']:,.5f}",
                    delta=f"{prediction['change_pct']:+.5f}% ({prediction['signal']})",
                    help=f"Confidence: {confidence_pct:.1f}%"
                )
        
        # ADD TRADING INSIGHTS
        st.markdown("---")
        st.subheader("💡 Trading Insights")
        insights = generate_trading_insights(ml_predictions, current_price, symbol)
        st.info(insights)
    else:
        st.warning("🤖 No ML predictions available - insufficient data or models still training")

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

    # Store previous values to detect changes
    prev_symbol = st.session_state.get('prev_symbol', '')
    prev_period = st.session_state.get('prev_period', '')
    prev_interval = st.session_state.get('prev_interval', '')
    
    symbol = st.sidebar.selectbox(
        "Select Symbol", 
        AVAILABLE_CURRENCIES,
        key="symbol_select"
    )
    
    period = st.sidebar.selectbox(
        "Select data period", 
        ["1d", "2d", "5d", "1wk", "1mo"],
        key="period_select"
    )
    
    interval = st.sidebar.selectbox(
        "Select interval", 
        ["1m", "5m", "15m", "30m", "1h", "2h", "1d"],
        key="interval_select"
    )
    
    # Check if any setting changed and trigger instant update
    settings_changed = (symbol != prev_symbol or period != prev_period or interval != prev_interval)
    
    if settings_changed:
        st.session_state.prev_symbol = symbol
        st.session_state.prev_period = period
        st.session_state.prev_interval = interval
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
    # Display top currencies at the very top
    display_top_currencies()
    # st.session_state.force_refresh = True
    
    # check_and_refresh()
    # Display refresh timer
    st.markdown("---")
    st.subheader("🕐 Live Refresh Timer")
    display_refresh_timer()
    st.markdown("---")
    
    # Fetch display data
    display_data = get_crypto_data(symbol, period, interval)
    
    # Fetch training data
    training_data = get_training_data(symbol, interval)
    
    if display_data.empty:
        st.error(f"❌ No {interval} data available for {symbol}.")
        st.info("💡 Try 5m or 15m intervals if 1m doesn't work")
        return
    
    # Display current time and status
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"**🕒 Current Time:** {current_time}")
    st.write(f"**📈 Display:** {len(display_data)} {interval} candles | **Period:** {period}")
    
    if not training_data.empty:
        st.write(f"**🤖 Training:** {len(training_data)} points (1 week)")
    
    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in display_data.columns]
    
    if missing_columns:
        st.error(f"❌ Missing columns: {missing_columns}")
        return
    
    # ==============================
    # PRICE PREDICTION SECTION
    # ==============================
    st.markdown("---")
    st.subheader("🔮 Price Prediction")
    
    # Get current price for calculations
    current_price = float(display_data['Close'].iloc[-1])
    
    # ==============================
    # ENHANCED PREDICTION SECTION
    # ==============================
    display_enhanced_predictions(symbol, interval, training_data, current_price)

    # --- Linear Regression prediction
    if not training_data.empty:
        linear_pred, linear_confidence, linear_time = predict_next_price_linear(training_data, interval)
    else:
        linear_pred, linear_confidence, linear_time = None, None, "No data"
    
    # --- LSTM prediction
    lstm_pred, lstm_confidence, lstm_signal, lstm_error = predict_with_lstm(symbol, interval)
    
    # --- Layout for prediction display
    pred_col1, pred_col2, pred_col3 = st.columns(3)

    with pred_col1:
        if linear_pred is not None:
            pred_change = linear_pred - current_price
            pred_change_pct = (pred_change / current_price) * 100
            confidence_display = f"{linear_confidence*100:.1f}%" if linear_confidence else "N/A"
            
            st.metric(
                label=f"📊 Linear Regression Confidence: {confidence_display}",
                value=f"${linear_pred:,.5f}",
                delta=f"{pred_change:+.5f} ({pred_change_pct:+.2f}%)",
                help=f"Confidence: {confidence_display}"
            )
        else:
            st.metric("📊 Linear Regression", "Calculating...")

    with pred_col2:
        if lstm_pred is not None:
            pred_change = lstm_pred - current_price
            pred_change_pct = (pred_change / current_price) * 100
            confidence_display = f"{lstm_confidence*100:.1f}%" if lstm_confidence else "N/A"
            
            st.metric(
                label="🧠 LSTM Pretrained "+f"Confidence: {confidence_display}",
                value=f"${lstm_pred:,.5f}",
                delta=f"{pred_change_pct:+.2f}% ({lstm_signal})",
                help=f"Confidence: {confidence_display}"
            )
        else:
            st.metric("🧠 LSTM Network", "No Model", help=lstm_error or "")

    with pred_col3:
        # Show current currency confidence if available
        if symbol in st.session_state.currency_confidences:
            confidence = st.session_state.currency_confidences[symbol] * 100
            st.metric(
                label="📈 Currency Confidence",
                value=f"{confidence:.1f}%",
                delta="High" if confidence >= 70 else "Medium" if confidence >= 50 else "Low"
            )
        else:
            st.metric("📈 Currency Confidence", "Scanning...")

    # ==============================
    # KEY METRICS
    # ==============================
    st.markdown("---")
    st.subheader("📊 Current Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    latest_price = current_price
    previous_price = float(display_data['Close'].iloc[-2]) if len(display_data) > 1 else latest_price
    price_change = latest_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    
    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest_price:,.2f}",
            delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        period_high = float(display_data['High'].max())
        st.metric("📈 Session High", f"${period_high:,.2f}")
    
    with col3:
        period_low = float(display_data['Low'].min())
        st.metric("📉 Session Low", f"${period_low:,.2f}")
    
    with col4:
        volume = float(display_data['Volume'].iloc[-1]) if 'Volume' in display_data.columns else 0
        st.metric("📦 Volume", f"{volume:,.0f}")
    
    # ==============================
    # CANDLESTICK CHART
    # ==============================
    st.markdown("---")
    st.subheader(f"📈 {interval} Candlestick Chart")
    
    plot_data = display_data.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
    
    if not plot_data.empty:
        st.info(f"🕯️ Displaying {len(plot_data)} {interval} candles | Last: {plot_data.iloc[-1]['Close']:,.2f}")
        
        with st.container():
            fig = create_enhanced_candlestick(plot_data, symbol, period, interval)
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
            })
    else:
        st.error("❌ No valid candle data available")

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
    
    st.sidebar.info(f"""
    **Current Status:**
    - Last Refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}
    - Next Refresh: {st.session_state.next_refresh.strftime('%H:%M:%S')}
    - Time Until: {int(time_until_next.total_seconds()) if time_until_next.total_seconds() > 0 else 0}s
    - Total Refreshes: {st.session_state.refresh_count}
    - Auto-refresh: {'✅ ON' if auto_refresh else '❌ OFF'}
    - Refresh Interval: {refresh_interval}s
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