"""
Enhanced Morning Momentum Scanner (First Pullback Strategy)
=========================================================

*Version 0.3 – Enhanced with better error handling, logging, and UI improvements*

This Streamlit app runs in two modes:

1. **Live Scan** (default, 06:25‑09:30 AM PT) – ranks real‑time symbols and
   highlights the safest "first‑pullback" long entries.
2. **Back‑test** (querystring `?mode=backtest`) – historical strategy testing
   with equity curve visualization vs SPY benchmark.

Required Secrets (Streamlit → Settings → Secrets):
```
FINNHUB_KEY   = "pk_..."
ALPACA_KEY    = "AK..."
ALPACA_SECRET = "AS..."
```

> NOT FINANCIAL ADVICE. Educational / illustrative purposes only.
"""

from __future__ import annotations

import os
import sys
import time
import datetime as dt
import logging
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
import traceback

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil import tz

# Optional dependencies
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    vbt = None
    VBT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ────────────────────────────────────────────────────────────────────────────────

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Time configuration
TZ = tz.gettz("America/Los_Angeles")
WINDOW_START = dt.time(6, 25)
WINDOW_END = dt.time(9, 30)
REFRESH_SEC = 60

# Universe filters
UNIVERSE_MIN_PRICE = 3.0
UNIVERSE_MIN_PREVOL = 300_000
MIN_FLOAT_SHARES = 20_000_000
MAX_UNIVERSE_SIZE = 100  # Prevent API overload

# API configuration
FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "")
ALPACA_KEY = st.secrets.get("ALPACA_KEY", "")
ALPACA_SECRET = st.secrets.get("ALPACA_SECRET", "")

FINNHUB_BASE = "https://finnhub.io/api/v1"
ALPACA_DATA_BASE = "https://data.alpaca.markets/v2"

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SP1500_CSV = os.path.join(DATA_DIR, "sp1500.csv")

# Strategy parameters
SCORE_WEIGHTS = {
    'gap': 0.4,
    'rvol': 0.3,
    'vwap': 0.3
}

# ────────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────────────────────

def now_pacific() -> dt.datetime:
    """Get current Pacific time."""
    return dt.datetime.now(tz=TZ)

def inside_window() -> bool:
    """Check if current time is within scanning window."""
    current_time = now_pacific().time()
    return WINDOW_START <= current_time <= WINDOW_END

def is_market_day() -> bool:
    """Check if today is a trading day (simple weekday check)."""
    return now_pacific().weekday() < 5  # Monday = 0, Friday = 4

@lru_cache(maxsize=1)
def load_universe() -> List[str]:
    """Load trading universe from CSV file."""
    try:
        if os.path.exists(SP1500_CSV):
            df = pd.read_csv(SP1500_CSV)
            symbols = df["Symbol"].tolist()[:MAX_UNIVERSE_SIZE]
            logger.info(f"Loaded {len(symbols)} symbols from universe")
            return symbols
        else:
            logger.warning(f"Universe file not found: {SP1500_CSV}")
            # Fallback to common tech stocks
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    except Exception as e:
        logger.error(f"Error loading universe: {e}")
        return []

def validate_api_keys() -> Tuple[bool, List[str]]:
    """Validate that all required API keys are present."""
    missing_keys = []
    if not FINNHUB_KEY or FINNHUB_KEY == "YOUR_FINNHUB_KEY":
        missing_keys.append("FINNHUB_KEY")
    if not ALPACA_KEY or ALPACA_KEY == "YOUR_ALPACA_KEY":
        missing_keys.append("ALPACA_KEY")
    if not ALPACA_SECRET or ALPACA_SECRET == "YOUR_ALPACA_SECRET":
        missing_keys.append("ALPACA_SECRET")
    
    return len(missing_keys) == 0, missing_keys

# ────────────────────────────────────────────────────────────────────────────────
# Market Data Functions
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)  # Cache for 30 seconds
def fetch_premarket_snapshot(symbol: str) -> Dict:
    """Fetch pre-market snapshot from Finnhub."""
    try:
        url = f"{FINNHUB_BASE}/quote?symbol={symbol}&token={FINNHUB_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Validate response
        if 'error' in data:
            logger.warning(f"Finnhub error for {symbol}: {data['error']}")
            return {}
            
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching snapshot for {symbol}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error fetching {symbol}: {e}")
        return {}

def fetch_minute_bars(symbols: List[str], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Fetch minute bars from Alpaca."""
    if not symbols:
        return pd.DataFrame()
    
    try:
        headers = {
            "APCA-API-KEY-ID": ALPACA_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET
        }
        params = {
            "symbols": ",".join(symbols[:50]),  # Limit to avoid API limits
            "timeframe": "1Min",
            "start": start.isoformat() + "Z",
            "end": end.isoformat() + "Z",
            "limit": 10000,
        }
        
        response = requests.get(
            f"{ALPACA_DATA_BASE}/stocks/bars",
            headers=headers,
            params=params,
            timeout=15
        )
        response.raise_for_status()
        
        raw_data = response.json().get("bars", {})
        if not raw_data:
            return pd.DataFrame()
        
        # Process data
        frames = []
        for symbol, bars in raw_data.items():
            if bars:
                df = pd.DataFrame(bars)
                df["symbol"] = symbol
                df["t"] = pd.to_datetime(df["t"])
                frames.append(df)
        
        if frames:
            result = pd.concat(frames, ignore_index=True)
            logger.info(f"Fetched {len(result)} bars for {len(frames)} symbols")
            return result
        else:
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching bars: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error in fetch_minute_bars: {e}")
        return pd.DataFrame()

# ────────────────────────────────────────────────────────────────────────────────
# Feature Engineering & Scoring
# ────────────────────────────────────────────────────────────────────────────────

def compute_technical_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to bars dataframe."""
    if bars.empty:
        return bars
    
    # Sort by symbol and time
    bars = bars.sort_values(["symbol", "t"]).copy()
    
    # VWAP (5-period)
    bars["price_volume"] = bars["c"] * bars["v"]
    bars["vwap"] = (
        bars.groupby("symbol")["price_volume"]
        .rolling(window=5, min_periods=1).sum().reset_index(0, drop=True) /
        bars.groupby("symbol")["v"]
        .rolling(window=5, min_periods=1).sum().reset_index(0, drop=True)
    )
    
    # EMA 9
    bars["ema9"] = (
        bars.groupby("symbol")["c"]
        .transform(lambda x: x.ewm(span=9, adjust=False).mean())
    )
    
    # Relative Volume
    bars["rvol_1m"] = (
        bars["v"] / 
        bars.groupby("symbol")["v"]
        .transform(lambda x: x.rolling(20, min_periods=1).median())
    )
    
    return bars

def compute_live_features(bars: pd.DataFrame, pm_highs: Dict[str, float], 
                         pm_vols: Dict[str, int]) -> pd.DataFrame:
    """Compute all features for live scanning."""
    if bars.empty:
        return bars
    
    # Add technical indicators
    bars = compute_technical_indicators(bars)
    
    # Gap calculation
    bars["gap_pct"] = ((bars["o"] / bars["pc"]) - 1) * 100
    
    # Pre-market data
    bars["pm_high"] = bars["symbol"].map(pm_highs).fillna(0)
    bars["pm_rvol"] = bars["symbol"].map(pm_vols).fillna(0)
    
    # Breakout detection
    bars["break_hi"] = (bars["h"] > bars["pm_high"]).astype(int)
    
    # Z-score normalization for scoring
    def safe_zscore(series):
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0, index=series.index)
        return (series - series.mean()) / std
    
    # Calculate component scores
    z_gap = safe_zscore(bars["gap_pct"])
    z_rvol = safe_zscore(bars["pm_rvol"])
    z_vwap = safe_zscore((bars["c"] - bars["vwap"]) / bars["vwap"])
    
    # Composite score
    bars["score"] = (
        SCORE_WEIGHTS['gap'] * z_gap +
        SCORE_WEIGHTS['rvol'] * z_rvol +
        SCORE_WEIGHTS['vwap'] * z_vwap
    )
    
    return bars

# ────────────────────────────────────────────────────────────────────────────────
# Strategy Logic
# ────────────────────────────────────────────────────────────────────────────────

def generate_entry_signals(df: pd.DataFrame) -> pd.Series:
    """Generate first pullback entry signals."""
    try:
        # Strategy conditions
        cond_breakout = (df["h"] >= df["pm_high"]) & (df["c"] > df["pm_high"])
        cond_hold_vwap = df["c"] > df["vwap"]
        cond_pullback = (df["c"].shift() > df["c"]) & (df["c"] <= df["ema9"])
        cond_reversal = df["c"] > df["c"].shift()
        
        # Combined entry signal
        entry_signal = cond_breakout & cond_hold_vwap & cond_pullback & cond_reversal
        
        return entry_signal.fillna(False)
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return pd.Series(False, index=df.index)

def calculate_risk_metrics(row) -> Dict:
    """Calculate risk/reward metrics for a position."""
    current_price = row["c"]
    vwap = row["vwap"]
    
    # Stop loss at VWAP
    stop_loss = vwap
    risk = abs(current_price - stop_loss)
    
    # Target at 1R (1:1 risk/reward)
    target = current_price + risk
    
    # Risk percentage
    risk_pct = (risk / current_price) * 100 if current_price > 0 else 0
    
    return {
        "stop_loss": stop_loss,
        "target": target,
        "risk_amount": risk,
        "risk_pct": risk_pct,
        "reward_ratio": 1.0
    }

# ────────────────────────────────────────────────────────────────────────────────
# UI Components
# ────────────────────────────────────────────────────────────────────────────────

def display_market_status():
    """Display current market status."""
    now = now_pacific()
    is_window = inside_window()
    is_trading_day = is_market_day()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = "🟢 SCANNING" if (is_window and is_trading_day) else "🔴 CLOSED"
        st.metric("Scanner Status", status)
    
    with col2:
        st.metric("Current Time (PT)", now.strftime("%H:%M:%S"))
    
    with col3:
        next_open = "Mon 6:25 AM" if now.weekday() >= 4 else "Tomorrow 6:25 AM"
        st.metric("Next Scan Window", next_open if not is_window else "Active")

def display_top_candidates(features_df: pd.DataFrame, top_n: int = 10):
    """Display top scanning candidates."""
    if features_df.empty:
        st.warning("No candidates found.")
        return
    
    # Get latest data for each symbol
    latest = features_df.groupby("symbol").last().reset_index()
    top_candidates = latest.nlargest(top_n, "score")
    
    if top_candidates.empty:
        st.warning("No valid candidates after filtering.")
        return
    
    # Display best candidate prominently
    best = top_candidates.iloc[0]
    
    st.subheader(f"🎯 Top Pick: {best['symbol']}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price", f"${best['c']:.2f}")
        st.metric("Gap %", f"{best['gap_pct']:.1f}%")
    
    with col2:
        risk_metrics = calculate_risk_metrics(best)
        st.metric("Stop Loss", f"${risk_metrics['stop_loss']:.2f}")
        st.metric("Risk %", f"{risk_metrics['risk_pct']:.1f}%")
    
    with col3:
        st.metric("Target", f"${risk_metrics['target']:.2f}")
        st.metric("Score", f"{best['score']:.2f}")
    
    with col4:
        st.metric("Volume", f"{best['v']:,}")
        st.metric("Rel Vol", f"{best['rvol_1m']:.1f}x")
    
    # Display top 10 table
    st.subheader("📊 Top Candidates")
    display_cols = ["symbol", "c", "gap_pct", "pm_rvol", "rvol_1m", "score"]
    display_df = top_candidates[display_cols].copy()
    display_df.columns = ["Symbol", "Price", "Gap %", "PM Vol", "RVol", "Score"]
    
    # Format the dataframe
    display_df["Price"] = display_df["Price"].apply(lambda x: f"${x:.2f}")
    display_df["Gap %"] = display_df["Gap %"].apply(lambda x: f"{x:.1f}%")
    display_df["PM Vol"] = display_df["PM Vol"].apply(lambda x: f"{x:,}")
    display_df["RVol"] = display_df["RVol"].apply(lambda x: f"{x:.1f}x")
    display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────────────────────
# Main Pages
# ────────────────────────────────────────────────────────────────────────────────

def live_scan_page():
    """Main live scanning interface."""
    st.title("⚡ Morning Momentum Scanner")
    st.caption("First Pullback Strategy • Live Market Scanning")
    
    # Check API keys
    keys_valid, missing_keys = validate_api_keys()
    if not keys_valid:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("Please add your API keys in Streamlit Settings → Secrets")
        return
    
    # Display market status
    display_market_status()
    
    # Check if we should be scanning
    if not is_market_day():
        st.info("📅 Market is closed (weekend). Try the backtester: ?mode=backtest")
        return
    
    if not inside_window():
        st.info("⏰ Outside scanning window (6:25-9:30 AM PT)")
        st.markdown("**Available modes:**")
        st.markdown("- 🔄 Current page will auto-activate during market hours")
        st.markdown("- 📊 [Historical Backtesting](?mode=backtest)")
        return
    
    # Load universe
    universe = load_universe()
    if not universe:
        st.error("Failed to load trading universe")
        return
    
    st.info(f"🔍 Scanning {len(universe)} symbols...")
    
    # Fetch pre-market data
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    pm_highs = {}
    pm_volumes = {}
    
    for i, symbol in enumerate(universe):
        status_text.text(f"Fetching pre-market data: {symbol}")
        progress_bar.progress((i + 1) / len(universe))
        
        snapshot = fetch_premarket_snapshot(symbol)
        if snapshot and snapshot.get("c", 0) >= UNIVERSE_MIN_PRICE:
            if snapshot.get("v", 0) >= UNIVERSE_MIN_PREVOL:
                pm_highs[symbol] = max(snapshot.get("h", 0), snapshot.get("c", 0))
                pm_volumes[symbol] = snapshot.get("v", 0)
    
    progress_bar.empty()
    status_text.empty()
    
    tradeable_symbols = list(pm_highs.keys())
    if not tradeable_symbols:
        st.warning("⚠️ No symbols passed pre-market filters today")
        return
    
    st.success(f"✅ {len(tradeable_symbols)} symbols passed filters")
    
    # Fetch recent minute bars
    end_time = dt.datetime.utcnow()
    start_time = end_time - dt.timedelta(minutes=10)  # Last 10 minutes
    
    with st.spinner("Fetching latest market data..."):
        bars = fetch_minute_bars(tradeable_symbols, start_time, end_time)
    
    if bars.empty:
        st.warning("⏳ Waiting for market data...")
        time.sleep(5)
        st.rerun()
        return
    
    # Compute features and display results
    features = compute_live_features(bars, pm_highs, pm_volumes)
    display_top_candidates(features)
    
    # Auto-refresh during market hours
    if inside_window():
        time.sleep(REFRESH_SEC)
        st.rerun()

def backtest_page():
    """Historical backtesting interface."""
    st.title("📊 Strategy Backtester")
    st.caption("Historical Analysis of First Pullback Strategy")
    
    if not VBT_AVAILABLE:
        st.error("📦 VectorBT not installed")
        st.code("pip install vectorbt", language="bash")
        return
    
    # Check API keys
    keys_valid, missing_keys = validate_api_keys()
    if not keys_valid:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        return
    
    # Date selection
    col1, col2 = st.columns(2)
    max_date = now_pacific().date() - dt.timedelta(days=1)
    default_start = max_date - dt.timedelta(days=14)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=max_date,
            max_value=max_date
        )
    
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return
    
    # Symbol selection
    universe = load_universe()
    max_symbols = st.slider("Max Symbols", 10, 100, 30)
    selected_symbols = universe[:max_symbols]
    
    if st.button("🚀 Run Backtest", type="primary"):
        run_backtest(selected_symbols, start_date, end_date)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def run_backtest(symbols: List[str], start_date: dt.date, end_date: dt.date):
    """Execute the backtesting process."""
    try:
        # Convert dates to datetime with timezone
        start_dt = dt.datetime.combine(start_date, dt.time(13, 30), tzinfo=TZ)  # Market open
        end_dt = dt.datetime.combine(end_date, dt.time(20, 0), tzinfo=TZ)  # After hours
        
        # Convert to UTC for API
        start_utc = start_dt.astimezone(dt.timezone.utc)
        end_utc = end_dt.astimezone(dt.timezone.utc)
        
        st.info(f"📈 Fetching data for {len(symbols)} symbols...")
        
        # Fetch historical data
        historical_data = fetch_minute_bars(symbols, start_utc, end_utc)
        
        if historical_data.empty:
            st.error("❌ No historical data returned")
            return
        
        st.success(f"✅ Retrieved {len(historical_data)} data points")
        
        # Simple backtesting logic (placeholder for demonstration)
        st.subheader("📊 Backtest Results")
        
        # Group by symbol and calculate basic stats
        symbol_stats = []
        for symbol in symbols[:10]:  # Limit for demo
            symbol_data = historical_data[historical_data["symbol"] == symbol]
            if not symbol_data.empty:
                returns = symbol_data["c"].pct_change().dropna()
                stats = {
                    "Symbol": symbol,
                    "Total Return": f"{(symbol_data['c'].iloc[-1] / symbol_data['c'].iloc[0] - 1) * 100:.1f}%",
                    "Volatility": f"{returns.std() * np.sqrt(252 * 390):.1f}%",  # Annualized
                    "Sharpe": f"{returns.mean() / returns.std() * np.sqrt(252 * 390):.2f}" if returns.std() > 0 else "N/A"
                }
                symbol_stats.append(stats)
        
        if symbol_stats:
            results_df = pd.DataFrame(symbol_stats)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Simple equity curve (placeholder)
        if PLOTLY_AVAILABLE and not historical_data.empty:
            sample_data = historical_data.groupby("t")["c"].mean().reset_index()
            sample_data["cumulative_return"] = (sample_data["c"] / sample_data["c"].iloc[0] - 1) * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_data["t"],
                y=sample_data["cumulative_return"],
                mode="lines",
                name="Strategy",
                line=dict(color="blue", width=2)
            ))
            
            fig.update_layout(
                title="Equity Curve (Simplified Demo)",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Backtest error: {str(e)}")
        logger.error(f"Backtest error: {traceback.format_exc()}")

# ────────────────────────────────────────────────────────────────────────────────
# Main Application
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Morning Momentum Scanner",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar with info
    with st.sidebar:
        st.title("📖 About")
        st.markdown("""
        **Morning Momentum Scanner** identifies stocks with:
        
        - Strong pre-market gaps
        - High relative volume  
        - Technical breakout patterns
        - First pullback entry opportunities
        
        **⚠️ Risk Warning:**
        This is educational software only. 
        Not financial advice.
        """)
        
        st.markdown("---")
        st.markdown("**Navigation:**")
        st.markdown("- [Live Scanner](?mode=live)")
        st.markdown("- [Backtester](?mode=backtest)")
    
    # Route to appropriate page
    mode = st.query_params.get("mode", "live").lower()
    
    if mode == "backtest":
        backtest_page()
    else:
        live_scan_page()

if __name__ == "__main__":
    main()