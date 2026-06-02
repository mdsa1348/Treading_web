#!/usr/bin/env python3
"""
SuperTrend Trading Bot with Telegram Integration
- Paper trading + Real money trading (switchable)
- Telegram alerts for every trade open/close
- Commands: /status, /pnl, /trades, /start, /stop
"""

import os
import sys
import json
import time
import threading
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    print("Install: pip install python-binance")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Install: pip install numpy")
    sys.exit(1)


# ──────────────────────────────────────────────
# CONFIGURATION — Edit these before running
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "8700744509:AAHQRb0eThsSuc3wc-cDZTKYFaAm9-I6-jQ"
TELEGRAM_CHAT_ID   = "1893889763"

BINANCE_API_KEY    = "9RQo4GWQNRgmh3oGb9vjj5VrLUZSSKNu7baSmu5ymaUeCyNWFp7QYpsDQdr3zETd"
BINANCE_API_SECRET = "FTDJZlg0OOOnXlLE0b398Xvn39pxUA3u09YDBeA8O5fH9KxUN8yyn11b2UzoIvZ0"

TRADE_MODE         = "paper"
SYMBOLS            = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "PROMUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"]
INITIAL_BALANCE    = 1000.0
LEVERAGE           = 5
RISK_PER_TRADE_PCT = 0.01  # Risk 1% of balance per trade (conservative)
ATR_PERIOD         = 14
ATR_MULTIPLIER_TP  = 2.0   # Tighter TP = more frequent wins
ATR_MULTIPLIER_SL  = 2.5   # Wider SL = survive noise, don't get shaken out
PARTIAL_TP_PCT     = 0.5   # Close 50% at first TP
PARTIAL_TP_TRIGGER = 0.03  # 3% profit for first partial TP
BREAK_EVEN_TRIGGER = 0.015 # 1.5% profit to move SL to break-even
INTERVAL_SECONDS   = 60    # Scan every 60s (less noise, more patience)
MAX_CONCURRENT     = 3     # Max 3 trades open at once
MAX_SAME_DIRECTION = 2     # Max 2 trades in same direction
COOLDOWN_MINUTES   = 10    # Wait 10 min after a loss before trading same symbol
LOSE_STREAK_LIMIT  = 3     # After 3 consecutive losses, pause trading for 30 min

# TQI Settings (from SATS - Self-Aware Trend System)
TQI_MIN_THRESHOLD  = 0.30  # Reject signals below this trend quality
TQI_ER_WEIGHT      = 0.35  # Efficiency ratio weight
TQI_VOL_WEIGHT     = 0.20  # Volatility regime weight
TQI_STRUCT_WEIGHT  = 0.25  # Price structure weight
TQI_MOM_WEIGHT     = 0.20  # Momentum persistence weight

# Dynamic TP Settings (from SATS)
DYN_TP_MIN_SCALE   = 0.6   # Min TP scale in choppy markets
DYN_TP_MAX_SCALE   = 1.6   # Max TP scale in trending markets

# Order Flow Settings (from ORB+VWAP+RSI indicator)
ORDER_FLOW_SMOOTH  = 8     # Delta smoothing period
# ──────────────────────────────────────────────


class PaperTradeMode(Enum):
    LIVE_PRICES = 1
    TESTNET     = 2


@dataclass
class Trade:
    trade_id:          int
    symbol:            str
    mode:              str          # "paper" or "real"
    entry_time:        datetime
    entry_price:       float
    quantity:          float
    side:              str          # LONG / SHORT
    stop_loss:         float
    take_profit:       float
    signal_confidence: float
    status:            str          # ACTIVE / CLOSED_TP / CLOSED_SL / CLOSED_TIMEOUT / TREND_FLIP
    remaining_quantity: float           = 0.0
    is_partial_tp_done: bool           = False
    is_break_even_done: bool           = False
    exit_time:         Optional[datetime] = None
    exit_price:        Optional[float]    = None
    profit_loss:       Optional[float]    = None
    profit_loss_pct:   Optional[float]    = None
    exit_reason:       Optional[str]      = None

    def to_dict(self):
        return {
            "trade_id":          self.trade_id,
            "symbol":            self.symbol,
            "mode":              self.mode,
            "entry_time":        self.entry_time.isoformat(),
            "entry_price":       self.entry_price,
            "quantity":          self.quantity,
            "side":              self.side,
            "stop_loss":         self.stop_loss,
            "take_profit":       self.take_profit,
            "signal_confidence": self.signal_confidence,
            "status":            self.status,
            "exit_time":         self.exit_time.isoformat() if self.exit_time else None,
            "exit_price":        self.exit_price,
            "profit_loss":       self.profit_loss,
            "profit_loss_pct":   self.profit_loss_pct,
            "exit_reason":       self.exit_reason,
            "duration_minutes":  (self.exit_time - self.entry_time).total_seconds() / 60
                                  if self.exit_time else None,
        }


# ──────────────────────────────────────────────
# TELEGRAM MESSENGER
# ──────────────────────────────────────────────
class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.base    = f"https://api.telegram.org/bot{token}"
        self.offset  = 0
        self._validate()

    def _validate(self):
        try:
            r = requests.get(f"{self.base}/getMe", timeout=5)
            if r.ok:
                name = r.json()["result"]["username"]
                print(f"✅ Telegram connected: @{name}")
            else:
                print(f"⚠️  Telegram token invalid: {r.text}")
        except Exception as e:
            print(f"⚠️  Telegram connection failed: {e}")

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram chat."""
        try:
            r = requests.post(
                f"{self.base}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode},
                timeout=10
            )
            return r.ok
        except Exception as e:
            print(f"⚠️  Telegram send failed: {e}")
            return False

    def get_updates(self) -> List[dict]:
        """Poll for new messages."""
        try:
            r = requests.get(
                f"{self.base}/getUpdates",
                params={"offset": self.offset, "timeout": 5},
                timeout=10
            )
            if r.ok:
                updates = r.json().get("result", [])
                if updates:
                    self.offset = updates[-1]["update_id"] + 1
                return updates
        except Exception:
            pass
        return []


# ──────────────────────────────────────────────
# MAIN BOT
# ──────────────────────────────────────────────
class SuperTrendBot:
    def __init__(
        self,
        trade_mode: str       = "paper",
        api_key:    str       = "",
        api_secret: str       = "",
        symbols:    List[str] = None,
        initial_balance: float = 1000.0,
        leverage:   int       = 5,
        tg_token:   str       = "",
        tg_chat_id: str       = "",
    ):
        self.trade_mode      = trade_mode.lower()   # "paper" or "real"
        self.symbols         = symbols or ["DOGEUSDT"]
        self.initial_balance = initial_balance
        self.balance         = initial_balance
        self.leverage        = leverage
        self.risk_pct        = RISK_PER_TRADE_PCT
        self.TAKER_FEE       = 0.0004
        self.min_confidence  = 0.75   # Only take high-quality signals

        # SuperTrend params
        self.st_period     = 10
        self.st_multiplier = 3.0

        # Losing streak tracker
        self.consecutive_losses = 0
        self.last_loss_time     = None
        self.is_paused          = False
        self.pause_until        = None

        # Binance client
        self.client = None
        try:
            self.client = Client(api_key, api_secret)
            print("✅ Binance connected")
        except Exception as e:
            print(f"⚠️  Binance: {e}")

        # Telegram
        self.tg = TelegramBot(tg_token, tg_chat_id) if tg_token and tg_chat_id else None

        # Market data buffers
        self.market_data = {
            sym: {
                "1m": deque(maxlen=300),
                "5m": deque(maxlen=300),
                "2h": deque(maxlen=300),
            }
            for sym in self.symbols
        }

        # Trade state
        self.active_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        self.cooldowns: Dict[str, datetime] = {}

        self.stats = {
            "total_trades":   0,
            "winning_trades": 0,
            "losing_trades":  0,
            "total_pnl":      0.0,
            "total_fees":     0.0,
            "gross_profit":   0.0,
            "gross_loss":     0.0,
            "max_drawdown":   0.0,
            "peak_balance":   initial_balance,
        }

        self.memory_file    = "bot_memory.json"
        self.dashboard_file = "dashboard_data.json"
        self._load_memory()

        self.is_running  = False
        self.start_time  = datetime.now()
        self._cmd_thread = None

        mode_label = "💵 REAL MONEY" if self.trade_mode == "real" else "📄 PAPER"
        print(f"\n{'='*65}")
        print(f"  SuperTrend Bot Pro  |  {mode_label}")
        print(f"{'='*65}")
        print(f"  Symbols   : {', '.join(self.symbols)}")
        print(f"  Balance   : ${self.balance:,.2f}")
        print(f"  Risk/Trade: {self.risk_pct*100:.1f}%  |  Leverage: {leverage}x")
        print(f"  Fee       : {self.TAKER_FEE*100:.3f}%")
        print(f"{'='*65}\n")

    # ── PERSISTENCE ──────────────────────────────────────────────────────────

    def _restore_trade(self, t: dict) -> Trade:
        trade = Trade(
            trade_id          = t.get("trade_id", 0),
            symbol            = t.get("symbol", "UNKNOWN"),
            mode              = t.get("mode", "paper"),
            entry_time        = datetime.fromisoformat(t["entry_time"]) if t.get("entry_time") else datetime.now(),
            entry_price       = t.get("entry_price", 0),
            quantity          = t.get("quantity", 0),
            side              = t.get("side", "LONG"),
            stop_loss         = t.get("stop_loss", 0),
            take_profit       = t.get("take_profit", 0),
            signal_confidence = t.get("signal_confidence", 0),
            status            = t.get("status", "CLOSED"),
            remaining_quantity = t.get("remaining_quantity", t.get("quantity", 0)),
            is_partial_tp_done = t.get("is_partial_tp_done", False),
            is_break_even_done = t.get("is_break_even_done", False),
        )
        if t.get("exit_time"):
            trade.exit_time = datetime.fromisoformat(t["exit_time"])
        trade.exit_price      = t.get("exit_price")
        trade.profit_loss     = t.get("profit_loss")
        trade.profit_loss_pct = t.get("profit_loss_pct")
        trade.exit_reason     = t.get("exit_reason")
        trade._pos_val        = t.get("_pos_val",   self.position_value)
        trade._entry_fee      = t.get("_entry_fee", self.position_value * self.TAKER_FEE)
        return trade

    def _load_memory(self):
        if not os.path.exists(self.memory_file):
            return
        try:
            with open(self.memory_file) as f:
                data = json.load(f)
            self.balance       = data.get("balance",       self.initial_balance)
            self.stats         = data.get("stats",         self.stats)
            self.trade_counter = data.get("trade_counter", 0)
            for t in data.get("active_trades", []):
                self.active_trades.append(self._restore_trade(t))
            for t in data.get("closed_trades", []):
                self.closed_trades.append(self._restore_trade(t))
            print(f"🧠 Memory loaded: ${self.balance:,.2f} | "
                  f"{len(self.active_trades)} active | {len(self.closed_trades)} closed")
        except Exception as e:
            print(f"⚠️  Load memory: {e}")

    def _save_memory(self):
        try:
            state = {
                "balance":       self.balance,
                "stats":         self.stats,
                "trade_counter": self.trade_counter,
                "last_update":   datetime.now().isoformat(),
                "active_trades": [t.to_dict() for t in self.active_trades],
                "closed_trades": [t.to_dict() for t in self.closed_trades[-100:]],
            }
            with open(self.memory_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️  Save memory: {e}")

    def _update_dashboard(self):
        try:
            active_data = []
            for t in self.active_trades:
                # Get current price from latest data
                curr_price = t.entry_price
                if t.symbol in self.market_data and self.market_data[t.symbol]["1m"]:
                    curr_price = self.market_data[t.symbol]["1m"][-1]["c"]
                
                # Calculate live PnL
                raw_pnl = (curr_price - t.entry_price) * t.quantity if t.side == "LONG" else (t.entry_price - curr_price) * t.quantity
                pnl_pct = (curr_price - t.entry_price) / t.entry_price * 100 if t.side == "LONG" else (t.entry_price - curr_price) / t.entry_price * 100
                
                d = t.to_dict()
                d["current_price"] = curr_price
                d["live_pnl"] = round(raw_pnl, 4)
                d["live_pnl_pct"] = round(pnl_pct, 2)
                active_data.append(d)

            # Calculate win rate for UI
            total = self.stats["total_trades"]
            win_rate = (self.stats["winning_trades"] / total * 100) if total > 0 else 0.0

            data = {
                "symbols":      self.symbols,
                "mode":         self.trade_mode,
                "balance":      round(self.balance, 4),
                "performance":  {**self.stats, "win_rate": win_rate},
                "activeTrades": active_data,
                "recentTrades": [t.to_dict() for t in self.closed_trades[-10:]],
                "lastUpdate":   datetime.now().strftime("%H:%M:%S"),
                "threshold":    round(self.min_confidence * 100, 1)
            }
            with open(self.dashboard_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Dashboard update failed: {e}")

    # ── MARKET DATA ──────────────────────────────────────────────────────────

    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> List[dict]:
        try:
            if not self.client:
                return []
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            return [{"o": float(k[1]), "h": float(k[2]), "l": float(k[3]), "c": float(k[4]), "v": float(k[5])} for k in klines]
        except Exception:
            return []

    def update_market_data(self, symbol: str):
        for tf, limit in [("1m", 100), ("5m", 100), ("2h", 250)]:
            candles = self._fetch_klines(symbol, tf, limit)
            if candles:
                # Only update if we have a full new candle (for confirmation logic)
                self.market_data[symbol][tf].clear()
                for c in candles:
                    self.market_data[symbol][tf].append(c)

    # ── INDICATORS ───────────────────────────────────────────────────────────

    @staticmethod
    def _ema(data: List[float], period: int) -> float:
        if len(data) < period:
            return 0
        alpha = 2 / (period + 1)
        ema = np.mean(data[:period])  # Initialize with SMA
        for v in data[period:]:
            ema = (v - ema) * alpha + ema
        return ema

    def calculate_atr(self, symbol: str, interval: str = "5m", period: int = 14) -> float:
        candles = list(self.market_data[symbol][interval])
        if len(candles) < period + 1:
            return 0
        tr_values = []
        for i in range(1, len(candles)):
            h, l, pc = candles[i]["h"], candles[i]["l"], candles[i - 1]["c"]
            tr_values.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(np.mean(tr_values[-period:]))

    def check_candle_patterns(self, symbol: str, interval: str = "5m") -> str:
        """Returns 'ENGULFING_LONG', 'ENGULFING_SHORT', 'PIN_BAR_LONG', 'PIN_BAR_SHORT' or 'NONE'."""
        candles = list(self.market_data[symbol][interval])
        if len(candles) < 3:
            return "NONE"
        
        c1, c2 = candles[-2], candles[-1] # Previous and current (closed)
        
        # Engulfing
        if c2["c"] > c2["h"] - (c2["h"]-c2["l"])*0.3 and c1["c"] < c1["h"] and c2["c"] > c1["h"] and c2["l"] < c1["l"]:
            return "ENGULFING_LONG"
        if c2["c"] < c2["l"] + (c2["h"]-c2["l"])*0.3 and c1["c"] > c1["l"] and c2["c"] < c1["l"] and c2["h"] > c1["h"]:
            return "ENGULFING_SHORT"
        
        # Pin Bar
        body = abs(c2["c"] - (c2["h"] + c2["l"])/2)
        range_ = c2["h"] - c2["l"]
        if range_ > 0:
            if (c2["h"] - max(c2["c"], c2["h"]-range_*0.3)) > range_ * 0.6 and c2["c"] > c2["l"] + range_*0.4:
                return "PIN_BAR_LONG"
            if (min(c2["c"], c2["l"]+range_*0.3) - c2["l"]) > range_ * 0.6 and c2["c"] < c2["h"] - range_*0.4:
                return "PIN_BAR_SHORT"
                
        return "NONE"

    def calculate_supertrend(self, symbol: str, interval: str = "5m") -> Tuple[str, float]:
        candles = list(self.market_data[symbol][interval])
        if len(candles) < self.st_period + 5:
            return "NEUTRAL", 0

        tr_values = []
        for i in range(1, len(candles)):
            h, l, pc = candles[i]["h"], candles[i]["l"], candles[i - 1]["c"]
            tr_values.append(max(h - l, abs(h - pc), abs(l - pc)))

        trend, final_upper, final_lower = "BUY", 0.0, 0.0

        for i in range(self.st_period, len(candles)):
            h, l, c = candles[i]["h"], candles[i]["l"], candles[i]["c"]
            prev_c  = candles[i - 1]["c"]
            hl2     = (h + l) / 2
            atr     = np.mean(tr_values[i - self.st_period : i])
            basic_u = hl2 + self.st_multiplier * atr
            basic_l = hl2 - self.st_multiplier * atr

            if basic_u < final_upper or prev_c > final_upper:
                final_upper = basic_u
            if basic_l > final_lower or prev_c < final_lower:
                final_lower = basic_l

            if trend == "BUY":
                if c < final_lower:
                    trend = "SELL"
            else:
                if c > final_upper:
                    trend = "BUY"

        return trend, (final_lower if trend == "BUY" else final_upper)

    def calculate_rsi(self, symbol: str, period: int = 14) -> float:
        candles = list(self.market_data[symbol]["1m"])
        if len(candles) < period + 1:
            return 50
        prices  = [c["c"] for c in candles]
        deltas  = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains   = [d if d > 0 else 0 for d in deltas]
        losses  = [-d if d < 0 else 0 for d in deltas]
        avg_g   = np.mean(gains[-period:])
        avg_l   = np.mean(losses[-period:])
        if avg_l == 0:
            return 100 if avg_g > 0 else 50
        return 100 - 100 / (1 + avg_g / avg_l)

    def calculate_adx(self, symbol: str, period: int = 14) -> float:
        candles = list(self.market_data[symbol]["5m"])
        if len(candles) < period + 5:
            return 0
        up_m, dn_m, tr = [], [], []
        for i in range(1, len(candles)):
            up = candles[i]["h"] - candles[i - 1]["h"]
            dn = candles[i - 1]["l"] - candles[i]["l"]
            up_m.append(max(up, 0) if up > dn else 0)
            dn_m.append(max(dn, 0) if dn > up else 0)
            tr.append(max(
                candles[i]["h"] - candles[i]["l"],
                abs(candles[i]["h"] - candles[i - 1]["c"]),
                abs(candles[i]["l"] - candles[i - 1]["c"]),
            ))
        sum_tr = sum(tr[-period:]) or 1
        plus_di  = 100 * sum(up_m[-period:]) / sum_tr
        minus_di = 100 * sum(dn_m[-period:]) / sum_tr
        return 100 * abs(plus_di - minus_di) / ((plus_di + minus_di) or 1)

    def get_volume_bias(self, symbol: str, interval: str = "5m") -> float:
        candles = list(self.market_data[symbol][interval])
        if len(candles) < 21:
            return 1.0
        recent_vol = candles[-1]["v"]
        avg_vol = np.mean([c["v"] for c in candles[-21:-1]])
        return recent_vol / avg_vol if avg_vol > 0 else 1.0

    # ── ADVANCED INDICATORS (from PineScript strategies) ─────────────────────

    def calculate_tqi(self, candles: List[dict], er_period: int = 20,
                      struct_period: int = 20, mom_period: int = 10) -> float:
        """
        Trend Quality Index (inspired by SATS - Self-Aware Trend System)
        Returns 0.0 (choppy/noise) to 1.0 (strong clean trend)
        Combines: Efficiency Ratio, Volatility Regime, Structure, Momentum
        """
        if len(candles) < max(er_period, struct_period, mom_period) + 5:
            return 0.5

        closes = [c["c"] for c in candles]

        # Factor 1: Efficiency Ratio (directional movement vs total movement)
        net_change = abs(closes[-1] - closes[-er_period])
        total_move = sum(abs(closes[i] - closes[i-1]) for i in range(-er_period, 0))
        er = net_change / total_move if total_move > 0 else 0.0
        tqi_er = min(er, 1.0)

        # Factor 2: Volatility Regime (current ATR vs baseline)
        atr_vals = []
        for i in range(1, len(candles)):
            tr = max(candles[i]["h"] - candles[i]["l"],
                     abs(candles[i]["h"] - candles[i-1]["c"]),
                     abs(candles[i]["l"] - candles[i-1]["c"]))
            atr_vals.append(tr)
        if len(atr_vals) >= 100:
            current_atr = float(np.mean(atr_vals[-14:]))
            baseline_atr = float(np.mean(atr_vals[-100:]))
            vol_ratio = current_atr / baseline_atr if baseline_atr > 0 else 1.0
            # Map: low vol (0.5) → 0, normal (1.0) → 0.5, high vol (2.0) → 1.0
            tqi_vol = max(0.0, min(1.0, (vol_ratio - 0.5) / 1.5))
        else:
            tqi_vol = 0.5

        # Factor 3: Structure (price position within recent range)
        recent_hi = max(c["h"] for c in candles[-struct_period:])
        recent_lo = min(c["l"] for c in candles[-struct_period:])
        rng = recent_hi - recent_lo
        if rng > 0:
            pos = (closes[-1] - recent_lo) / rng
            tqi_struct = abs(pos - 0.5) * 2.0  # 0 at middle, 1 at extremes
        else:
            tqi_struct = 0.5

        # Factor 4: Momentum Persistence (how many bars agree with direction)
        window_change = closes[-1] - closes[-mom_period]
        aligned = 0
        for i in range(1, min(mom_period, len(closes))):
            bar_change = closes[-i] - closes[-i-1]
            if (window_change > 0 and bar_change > 0) or \
               (window_change < 0 and bar_change < 0):
                aligned += 1
        tqi_mom = aligned / mom_period if mom_period > 0 else 0.5

        # Weighted blend
        w_sum = TQI_ER_WEIGHT + TQI_VOL_WEIGHT + TQI_STRUCT_WEIGHT + TQI_MOM_WEIGHT
        tqi = (tqi_er * TQI_ER_WEIGHT + tqi_vol * TQI_VOL_WEIGHT +
               tqi_struct * TQI_STRUCT_WEIGHT + tqi_mom * TQI_MOM_WEIGHT) / w_sum
        return max(0.0, min(1.0, tqi))

    def calculate_order_flow(self, candles: List[dict], smooth: int = 8) -> Tuple[float, bool]:
        """
        Order Flow Proxy (inspired by ORB+VWAP+RSI indicator)
        Uses volume * sign(close - open) as delta proxy.
        Returns: (smoothed_delta, is_cvd_rising)
        """
        if len(candles) < smooth + 5:
            return 0.0, False

        # Raw delta per bar: volume * direction
        deltas = []
        for c in candles:
            open_price = c.get("o", c["c"])  # Fallback if open not available
            direction = 1.0 if c["c"] > open_price else (-1.0 if c["c"] < open_price else 0.0)
            deltas.append(c["v"] * direction)

        # Smoothed delta (EMA)
        k = 2.0 / (smooth + 1)
        ema = deltas[0]
        for d in deltas[1:]:
            ema = d * k + ema * (1 - k)
        smoothed_delta = ema

        # CVD (Cumulative Volume Delta) - last 50 bars
        cvd_window = deltas[-50:] if len(deltas) >= 50 else deltas
        cvd = sum(cvd_window)
        cvd_prev = sum(cvd_window[:-1]) if len(cvd_window) > 1 else 0
        cvd_rising = cvd > cvd_prev

        return smoothed_delta, cvd_rising

    def calculate_dynamic_tp_scale(self, tqi: float, candles: List[dict]) -> float:
        """
        Dynamic TP Scaling (inspired by SATS)
        High TQI + trending → wider targets (scale > 1.0)
        Low TQI + choppy → tighter targets (scale < 1.0)
        """
        # Volatility component
        atr_vals = []
        for i in range(1, len(candles)):
            tr = max(candles[i]["h"] - candles[i]["l"],
                     abs(candles[i]["h"] - candles[i-1]["c"]),
                     abs(candles[i]["l"] - candles[i-1]["c"]))
            atr_vals.append(tr)
        if len(atr_vals) >= 100:
            vol_ratio = float(np.mean(atr_vals[-14:])) / float(np.mean(atr_vals[-100:]))
        else:
            vol_ratio = 1.0

        # TQI component (0-1) and vol component (mapped 0.5-2.0 → 0-1)
        vol_comp = max(0.0, min(1.0, (vol_ratio - 0.5) / 1.5))
        raw_scale = tqi * 0.6 + vol_comp * 0.4  # Weighted blend
        # Map to [DYN_TP_MIN_SCALE, DYN_TP_MAX_SCALE]
        scale = DYN_TP_MIN_SCALE + raw_scale * (DYN_TP_MAX_SCALE - DYN_TP_MIN_SCALE)
        return scale

    # ── SIGNAL ───────────────────────────────────────────────────────────────

    def calculate_rsi_on_data(self, candles: List[dict], period: int = 14) -> float:
        """RSI calculated on closed candle data."""
        if len(candles) < period + 1:
            return 50
        prices = [c["c"] for c in candles]
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains  = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_g  = np.mean(gains[-period:])
        avg_l  = np.mean(losses[-period:])
        if avg_l == 0:
            return 100 if avg_g > 0 else 50
        return 100 - 100 / (1 + avg_g / avg_l)

    def calculate_ema_crossover(self, candles: List[dict], fast: int = 9, slow: int = 21) -> str:
        """Returns 'BULL_CROSS', 'BEAR_CROSS', or 'NONE'."""
        closes = [c["c"] for c in candles]
        if len(closes) < slow + 2:
            return "NONE"
        ema_fast_now  = self._ema(closes, fast)
        ema_slow_now  = self._ema(closes, slow)
        ema_fast_prev = self._ema(closes[:-1], fast)
        ema_slow_prev = self._ema(closes[:-1], slow)
        if ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now:
            return "BULL_CROSS"
        if ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now:
            return "BEAR_CROSS"
        # Also check if fast is already above/below slow (trend continuation)
        if ema_fast_now > ema_slow_now:
            return "BULL_TREND"
        if ema_fast_now < ema_slow_now:
            return "BEAR_TREND"
        return "NONE"

    def check_two_candle_confirmation(self, candles: List[dict], direction: str) -> bool:
        """Check if the last 2 closed candles agree with the direction."""
        if len(candles) < 3:
            return False
        c1, c2 = candles[-2], candles[-1]
        if direction == "BUY":
            return c1["c"] > c1["l"] + (c1["h"] - c1["l"]) * 0.4 and \
                   c2["c"] > c2["l"] + (c2["h"] - c2["l"]) * 0.4 and \
                   c2["c"] >= c1["c"]
        else:  # SELL
            return c1["c"] < c1["h"] - (c1["h"] - c1["l"]) * 0.4 and \
                   c2["c"] < c2["h"] - (c2["h"] - c2["l"]) * 0.4 and \
                   c2["c"] <= c1["c"]

    def generate_signal(self, symbol: str) -> Tuple[str, float]:
        # Use candles[:-1] to ensure we only look at CLOSED candles
        candles_5m = list(self.market_data[symbol]["5m"])[:-1]
        candles_2h = list(self.market_data[symbol]["2h"])[:-1]
        candles_1m = list(self.market_data[symbol]["1m"])[:-1]

        if len(candles_5m) < 30 or len(candles_2h) < 30:
            return "NEUTRAL", 0.0

        # ── LAYER 0: TQI Gate (from SATS) — reject choppy markets early ──
        tqi = self.calculate_tqi(candles_5m)
        if tqi < TQI_MIN_THRESHOLD:
            return "NEUTRAL", 0.0  # Market too choppy, skip

        # ── LAYER 0.5: Order Flow confirmation ──
        of_delta, cvd_rising = self.calculate_order_flow(candles_5m)

        # ── LAYER 1: SuperTrend direction (primary) ──
        trend_5m, _ = self.calculate_supertrend_on_data(candles_5m)
        trend_2h, _ = self.calculate_supertrend_on_data(candles_2h)

        # ── LAYER 2: EMA 200 macro bias (2h) ──
        close_2h = [c["c"] for c in candles_2h]
        ema200_2h = self._ema(close_2h, 200)
        price_2h = close_2h[-1] if close_2h else 0
        ema_bias = "BULL" if price_2h > ema200_2h else "BEAR"

        # ── LAYER 3: RSI filter (avoid overbought/oversold entries) ──
        rsi_5m = self.calculate_rsi_on_data(candles_5m)
        rsi_ok_long  = 35 < rsi_5m < 65   # Don't buy when already overbought
        rsi_ok_short = 35 < rsi_5m < 65   # Don't sell when already oversold

        # ── LAYER 4: EMA 9/21 crossover on 5m (momentum) ──
        ema_cross = self.calculate_ema_crossover(candles_5m)

        # ── LAYER 5: Two-candle confirmation ──
        two_candle_bull = self.check_two_candle_confirmation(candles_5m, "BUY")
        two_candle_bear = self.check_two_candle_confirmation(candles_5m, "SELL")

        # ── LAYER 6: Volume confirmation ──
        vol_ratio = self.get_volume_bias_on_data(candles_5m)
        vol_conf = vol_ratio > 1.1

        # ── LAYER 7: ADX trend strength ──
        adx = self.calculate_adx_on_data(candles_5m)

        # ── LAYER 8: Candlestick patterns ──
        pattern = self.check_candle_patterns_on_data(candles_5m)

        # ── BUILD SIGNAL with strict multi-layer confirmation ──
        confidence = 0.0
        signal = "NEUTRAL"

        # === LONG SIGNAL ===
        if trend_5m == "BUY" and ema_bias == "BULL" and rsi_ok_long:
            confidence = 0.45  # Base: SuperTrend + EMA200 + RSI OK

            # Must-have: 2h trend alignment
            if trend_2h == "BUY":
                confidence += 0.20
            else:
                return "NEUTRAL", 0.0  # REJECT: don't fight the higher timeframe

            # EMA crossover momentum
            if ema_cross in ("BULL_CROSS", "BULL_TREND"):
                confidence += 0.15
            else:
                confidence -= 0.10  # Penalty if momentum is against us

            # Two-candle confirmation
            if two_candle_bull:
                confidence += 0.10

            # Volume boost
            if vol_conf:
                confidence += 0.05

            # ADX trend strength
            if adx > 25:
                confidence += 0.05

            # Order Flow confirmation (from ORB+VWAP+RSI indicator)
            if of_delta > 0 and cvd_rising:
                confidence += 0.10  # Strong buying pressure
            elif of_delta < 0:
                confidence -= 0.10  # Selling pressure contradicts long

            # TQI quality boost (from SATS)
            confidence *= (0.8 + 0.2 * tqi)  # TQI modulates final confidence

            # Candlestick pattern bonus
            if "LONG" in pattern:
                confidence += 0.05

            signal = "STRONG_BUY" if confidence >= 0.85 else "BUY"

        # === SHORT SIGNAL ===
        elif trend_5m == "SELL" and ema_bias == "BEAR" and rsi_ok_short:
            confidence = 0.45

            if trend_2h == "SELL":
                confidence += 0.20
            else:
                return "NEUTRAL", 0.0  # REJECT

            if ema_cross in ("BEAR_CROSS", "BEAR_TREND"):
                confidence += 0.15
            else:
                confidence -= 0.10

            if two_candle_bear:
                confidence += 0.10

            if vol_conf:
                confidence += 0.05

            if adx > 25:
                confidence += 0.05

            # Order Flow confirmation (from ORB+VWAP+RSI indicator)
            if of_delta < 0 and not cvd_rising:
                confidence += 0.10  # Strong selling pressure
            elif of_delta > 0:
                confidence -= 0.10  # Buying pressure contradicts short

            # TQI quality boost (from SATS)
            confidence *= (0.8 + 0.2 * tqi)

            if "SHORT" in pattern:
                confidence += 0.05

            signal = "STRONG_SELL" if confidence >= 0.85 else "SELL"

        return signal, max(min(confidence, 1.0), 0.0)

    # Helper methods for closed data
    def calculate_supertrend_on_data(self, candles: List[dict]) -> Tuple[str, float]:
        if len(candles) < self.st_period + 5: return "NEUTRAL", 0
        tr_values = []
        for i in range(1, len(candles)):
            h, l, pc = candles[i]["h"], candles[i]["l"], candles[i - 1]["c"]
            tr_values.append(max(h - l, abs(h - pc), abs(l - pc)))
        trend, final_upper, final_lower = "BUY", 0.0, 0.0
        for i in range(self.st_period, len(candles)):
            h, l, c = candles[i]["h"], candles[i]["l"], candles[i]["c"]
            prev_c  = candles[i - 1]["c"]
            hl2, atr = (h + l) / 2, np.mean(tr_values[i - self.st_period : i])
            basic_u, basic_l = hl2 + self.st_multiplier * atr, hl2 - self.st_multiplier * atr
            if basic_u < final_upper or prev_c > final_upper: final_upper = basic_u
            if basic_l > final_lower or prev_c < final_lower: final_lower = basic_l
            if trend == "BUY":
                if c < final_lower: trend = "SELL"
            else:
                if c > final_upper: trend = "BUY"
        return trend, (final_lower if trend == "BUY" else final_upper)

    def get_volume_bias_on_data(self, candles: List[dict]) -> float:
        if len(candles) < 21: return 1.0
        recent_vol, avg_vol = candles[-1]["v"], np.mean([c["v"] for c in candles[-21:-1]])
        return recent_vol / avg_vol if avg_vol > 0 else 1.0

    def check_candle_patterns_on_data(self, candles: List[dict]) -> str:
        if len(candles) < 3: return "NONE"
        c1, c2 = candles[-2], candles[-1]
        if c2["c"] > c2["h"] - (c2["h"]-c2["l"])*0.3 and c1["c"] < c1["h"] and c2["c"] > c1["h"] and c2["l"] < c1["l"]: return "ENGULFING_LONG"
        if c2["c"] < c2["l"] + (c2["h"]-c2["l"])*0.3 and c1["c"] > c1["l"] and c2["c"] < c1["l"] and c2["h"] > c1["h"]: return "ENGULFING_SHORT"
        body, range_ = abs(c2["c"] - (c2["h"] + c2["l"])/2), c2["h"] - c2["l"]
        if range_ > 0:
            if (c2["h"] - max(c2["c"], c2["h"]-range_*0.3)) > range_ * 0.6 and c2["c"] > c2["l"] + range_*0.4: return "PIN_BAR_LONG"
            if (min(c2["c"], c2["l"]+range_*0.3) - c2["l"]) > range_ * 0.6 and c2["c"] < c2["h"] - range_*0.4: return "PIN_BAR_SHORT"
        return "NONE"

    def calculate_adx_on_data(self, candles: List[dict], period: int = 14) -> float:
        if len(candles) < period + 5: return 0
        up_m, dn_m, tr = [], [], []
        for i in range(1, len(candles)):
            up, dn = candles[i]["h"] - candles[i - 1]["h"], candles[i - 1]["l"] - candles[i]["l"]
            up_m.append(max(up, 0) if up > dn else 0)
            dn_m.append(max(dn, 0) if dn > up else 0)
            tr.append(max(candles[i]["h"] - candles[i]["l"], abs(candles[i]["h"] - candles[i - 1]["c"]), abs(candles[i]["l"] - candles[i - 1]["c"])))
        sum_tr = sum(tr[-period:]) or 1
        plus_di, minus_di = 100 * sum(up_m[-period:]) / sum_tr, 100 * sum(dn_m[-period:]) / sum_tr
        return 100 * abs(plus_di - minus_di) / ((plus_di + minus_di) or 1)

    # ── ORDER PLACEMENT ───────────────────────────────────────────────────────

    def place_order(self, symbol: str, signal: str, confidence: float, price: float) -> bool:
        if confidence < self.min_confidence or signal == "NEUTRAL":
            return False
        if len(self.active_trades) >= MAX_CONCURRENT:
            return False
        if any(t.symbol == symbol for t in self.active_trades):
            return False

        # Losing streak protection: pause after consecutive losses
        if self.is_paused and self.pause_until and datetime.now() < self.pause_until:
            print(f"   ⏸️ Bot paused until {self.pause_until.strftime('%H:%M:%S')} (lose streak protection)")
            return False
        elif self.is_paused:
            self.is_paused = False
            self.consecutive_losses = 0
            print(f"   ▶️ Pause ended, resuming trading")

        # Cooldown check per symbol
        if symbol in self.cooldowns and datetime.now() < self.cooldowns[symbol]:
            return False

        # Reduce risk after losses
        current_risk = self.risk_pct
        if self.consecutive_losses >= 2:
            current_risk = self.risk_pct * 0.5  # Half risk after 2 consecutive losses
            print(f"   ⚠️ Reduced risk to {current_risk*100:.1f}% (lose streak: {self.consecutive_losses})")

        atr = self.calculate_atr(symbol, "5m")
        if atr == 0: atr = price * 0.02

        # Dynamic TP Scaling (from SATS - Self-Aware Trend System)
        candles_5m = list(self.market_data[symbol]["5m"])[:-1]
        tqi = self.calculate_tqi(candles_5m) if len(candles_5m) > 30 else 0.5
        tp_scale = self.calculate_dynamic_tp_scale(tqi, candles_5m) if len(candles_5m) > 30 else 1.0
        dynamic_tp_mult = ATR_MULTIPLIER_TP * tp_scale

        if "BUY" in signal:
            side        = "LONG"
            stop_loss   = price - (atr * ATR_MULTIPLIER_SL)
            take_profit = price + (atr * dynamic_tp_mult)
        else:
            side        = "SHORT"
            stop_loss   = price + (atr * ATR_MULTIPLIER_SL)
            take_profit = price - (atr * dynamic_tp_mult)

        # Log TQI and TP scaling
        print(f"   📐 TQI: {tqi:.2f} | TP Scale: {tp_scale:.2f}x → TP Mult: {dynamic_tp_mult:.2f}")

        # Dynamic Sizing: Risk % of balance
        risk_amount = self.balance * current_risk
        price_risk = abs(price - stop_loss)
        if price_risk == 0: price_risk = price * 0.01

        raw_qty = (risk_amount / price_risk)
        quantity = min(raw_qty, (self.balance * self.leverage) / price * 0.5)  # Cap at 50% of max buying power

        pos_val   = quantity * price
        entry_fee = pos_val * self.TAKER_FEE
        self.balance -= entry_fee
        self.stats["total_fees"] += entry_fee

        self.trade_counter += 1
        trade             = Trade(
            trade_id          = self.trade_counter,
            symbol            = symbol,
            mode              = self.trade_mode,
            entry_time        = datetime.now(),
            entry_price       = price,
            quantity          = quantity,
            remaining_quantity = quantity,
            side              = side,
            stop_loss         = stop_loss,
            take_profit       = take_profit,
            signal_confidence = confidence,
            status            = "ACTIVE",
        )
        trade._pos_val   = pos_val
        trade._entry_fee = entry_fee
        self.active_trades.append(trade)

        # ── REAL MONEY: place actual Binance Futures order ─────────────────
        if self.trade_mode == "real" and self.client:
            try:
                order_side = "BUY" if side == "LONG" else "SELL"
                self.client.futures_create_order(
                    symbol     = symbol,
                    side       = order_side,
                    type       = "MARKET",
                    quantity   = round(quantity, 4),
                    reduceOnly = False,
                )
                print(f"✅ REAL ORDER placed on Binance: {order_side} {symbol}")
            except BinanceAPIException as e:
                print(f"❌ Real order failed: {e}")
                self.active_trades.remove(trade)
                self.balance += entry_fee
                return False

        dir_icon = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
        mode_tag = "💵 REAL" if self.trade_mode == "real" else "📄 PAPER"
        print(f"\n✅ [{symbol}] {dir_icon} ({mode_tag})")
        print(f"   Entry: ${price:,.4f} | SL: ${stop_loss:,.4f} | TP: ${take_profit:,.4f}")
        print(f"   Entry Fee: -${entry_fee:.4f} | Balance: ${self.balance:,.4f}")

        # ── TELEGRAM ALERT ─────────────────────────────────────────────────
        if self.tg:
            emoji = "🟢" if side == "LONG" else "🔴"
            msg = (
                f"{emoji} <b>NEW TRADE — {mode_tag}</b>\n"
                f"📌 <b>{symbol}</b>  |  {side}\n"
                f"💰 Entry : <code>${price:,.4f}</code>\n"
                f"🛑 SL    : <code>${stop_loss:,.4f}</code>\n"
                f"🎯 TP    : <code>${take_profit:,.4f}</code>\n"
                f"📊 Pos   : ${pos_val:.0f}  ({self.leverage}x leverage)\n"
                f"🔑 Conf  : {confidence:.0%}\n"
                f"🕐 Time  : {datetime.now().strftime('%H:%M:%S')}"
            )
            self.tg.send(msg)

        return True

    # ── POSITION MANAGEMENT ───────────────────────────────────────────────────

    def update_positions(self, symbol: str, price: float):
        for trade in [t for t in self.active_trades if t.symbol == symbol]:
            # Current PnL %
            pnl_pct = (price - trade.entry_price) / trade.entry_price if trade.side == "LONG" else (trade.entry_price - price) / trade.entry_price
            
            # 1. Break-even SL
            if not trade.is_break_even_done and pnl_pct >= BREAK_EVEN_TRIGGER:
                trade.stop_loss = trade.entry_price
                trade.is_break_even_done = True
                print(f"   🛡️ [{symbol}] Break-even SL activated")
                if self.tg: self.tg.send(f"🛡️ <b>{symbol}</b>: SL moved to Break-even")

            # 2. Partial Take Profit
            if not trade.is_partial_tp_done and pnl_pct >= PARTIAL_TP_TRIGGER:
                close_qty = trade.quantity * PARTIAL_TP_PCT
                self._partial_close(trade, price, close_qty, "PARTIAL_TP")
                trade.is_partial_tp_done = True

            # 3. Full Exit Conditions
            if trade.side == "LONG":
                if price <= trade.stop_loss:
                    self._close_trade(trade, price, "STOP_LOSS")
                elif price >= trade.take_profit:
                    self._close_trade(trade, price, "TAKE_PROFIT")
            else:
                if price >= trade.stop_loss:
                    self._close_trade(trade, price, "STOP_LOSS")
                elif price <= trade.take_profit:
                    self._close_trade(trade, price, "TAKE_PROFIT")

            # 4h timeout (increased from 2h)
            if trade.status == "ACTIVE" and (datetime.now() - trade.entry_time).total_seconds() > 14400:
                self._close_trade(trade, price, "TIMEOUT")

    def _partial_close(self, trade: Trade, price: float, qty: float, reason: str):
        gross = (price - trade.entry_price) * qty if trade.side == "LONG" else (trade.entry_price - price) * qty
        fee = (qty * price) * self.TAKER_FEE
        net = gross - fee
        self.balance += net
        self.stats["total_fees"] += fee
        trade.remaining_quantity -= qty
        
        print(f"   💰 [{trade.symbol}] Partial TP: +${net:.4f}")
        if self.tg:
            self.tg.send(f"💰 <b>{trade.symbol}</b>: Partial TP Closed 50%\nNet P&L: ${net:.4f}")
        
        if self.trade_mode == "real" and self.client:
            try:
                self.client.futures_create_order(
                    symbol=trade.symbol, side="SELL" if trade.side=="LONG" else "BUY",
                    type="MARKET", quantity=round(qty, 4), reduceOnly=True
                )
            except Exception as e: print(f"❌ Partial close failed: {e}")

    def _close_trade(self, trade: Trade, exit_price: float, reason: str):
        if trade not in self.active_trades:
            return

        trade.exit_time  = datetime.now()
        trade.exit_price = exit_price
        trade.exit_reason = reason

        qty       = trade.remaining_quantity
        entry_fee = getattr(trade, "_entry_fee", 0.0)
        gross     = ((exit_price - trade.entry_price) * qty
                 if trade.side == "LONG"
                 else (trade.entry_price - exit_price) * qty)
        exit_fee  = (qty * exit_price) * self.TAKER_FEE
        net_pnl   = gross - exit_fee

        trade.profit_loss     = net_pnl
        trade.profit_loss_pct = (net_pnl / (qty * trade.entry_price)) * 100
        trade.status = {"TAKE_PROFIT": "CLOSED_TP", "STOP_LOSS": "CLOSED_SL",
                        "TREND_FLIP": "CLOSED_TF"}.get(reason, "CLOSED_TIMEOUT")

        # ── REAL MONEY: close Binance position ────────────────────────────
        if trade.mode == "real" and self.client:
            try:
                close_side = "SELL" if trade.side == "LONG" else "BUY"
                self.client.futures_create_order(
                    symbol     = trade.symbol,
                    side       = close_side,
                    type       = "MARKET",
                    quantity   = round(qty, 4),
                    reduceOnly = True,
                )
            except BinanceAPIException as e:
                print(f"❌ Real close failed: {e}")

        self.balance += net_pnl

        # Update stats
        self.stats["total_trades"]  += 1
        self.stats["total_pnl"]     += net_pnl
        self.stats["total_fees"]    += exit_fee
        if net_pnl > 0:
            self.stats["winning_trades"] += 1
            self.stats["gross_profit"]   += net_pnl
            self.consecutive_losses = 0  # Reset streak on win
        else:
            self.stats["losing_trades"]  += 1
            self.stats["gross_loss"]     += abs(net_pnl)
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
            # Set cooldown for this symbol
            self.cooldowns[trade.symbol] = datetime.now() + timedelta(minutes=COOLDOWN_MINUTES)
            # Pause bot after too many consecutive losses
            if self.consecutive_losses >= LOSE_STREAK_LIMIT:
                self.is_paused = True
                self.pause_until = datetime.now() + timedelta(minutes=30)
                print(f"\n   ⏸️ LOSE STREAK ({self.consecutive_losses}) — pausing until {self.pause_until.strftime('%H:%M:%S')}")
                if self.tg:
                    self.tg.send(f"⏸️ <b>Bot Paused</b>\n{self.consecutive_losses} consecutive losses.\nResuming at {self.pause_until.strftime('%H:%M:%S')}")
        if self.balance > self.stats["peak_balance"]:
            self.stats["peak_balance"] = self.balance
        dd = (self.stats["peak_balance"] - self.balance) / self.stats["peak_balance"]
        if dd > self.stats["max_drawdown"]:
            self.stats["max_drawdown"] = dd

        self.active_trades.remove(trade)
        if trade not in self.closed_trades:
            self.closed_trades.append(trade)

        icon = "✓" if net_pnl > 0 else "✗"
        print(f"\n{icon} [{trade.symbol}] #{trade.trade_id} CLOSED ({reason})")
        print(f"   Net P&L: ${net_pnl:+.4f} | Balance: ${self.balance:,.4f}")

        # ── TELEGRAM CLOSE ALERT ──────────────────────────────────────────
        if self.tg:
            pnl_emoji = "✅" if net_pnl > 0 else "❌"
            duration  = (trade.exit_time - trade.entry_time).total_seconds() / 60
            mode_tag  = "💵 REAL" if trade.mode == "real" else "📄 PAPER"
            msg = (
                f"{pnl_emoji} <b>TRADE CLOSED — {mode_tag}</b>\n"
                f"📌 <b>{trade.symbol}</b>  |  {trade.side}\n"
                f"🔖 Reason : {reason}\n"
                f"💰 Entry  : <code>${trade.entry_price:,.4f}</code>\n"
                f"🏁 Exit   : <code>${exit_price:,.4f}</code>\n"
                f"📈 Net P&L: <code>${net_pnl:+.4f}</code>  ({trade.profit_loss_pct:+.2f}%)\n"
                f"⏱ Duration: {duration:.0f} min\n"
                f"💼 Balance: <code>${self.balance:,.4f}</code>"
            )
            self.tg.send(msg)

        self._save_memory()
        self._update_dashboard()

    # ── TRADING CYCLE ─────────────────────────────────────────────────────────

    def run_cycle(self):
        signals = []
        for symbol in self.symbols:
            active = next((t for t in self.active_trades if t.symbol == symbol), None)
            self.update_market_data(symbol)
            
            p1m = list(self.market_data[symbol]["1m"])
            if not p1m: continue
            price = p1m[-1]["c"]
            
            self.update_positions(symbol, price)
            
            signal, confidence = self.generate_signal(symbol)

            # --- ADDED LOGGING ---
            active_info = f" | {active.side} SL:{active.stop_loss:,.2f}" if active else ""
            print(f"   🔍 {symbol}: {signal} ({confidence:.1%}) | ${price:,.4f}{active_info}")
            # ---------------------

            # Trend flip check for active trades
            if active:
                is_flip = (active.side == "LONG" and "SELL" in signal) or \
                          (active.side == "SHORT" and "BUY" in signal)
                if is_flip and confidence >= 0.7:
                    print(f"   🔄 TREND FLIP on {symbol} — closing {active.side}")
                    self._close_trade(active, price, "TREND_FLIP")
                    active = None

            if not active and signal != "NEUTRAL" and confidence >= self.min_confidence:
                active_longs = [t for t in self.active_trades if t.side == "LONG"]
                active_shorts = [t for t in self.active_trades if t.side == "SHORT"]

                # Direction limit filter
                if ("BUY" in signal and len(active_longs) >= MAX_SAME_DIRECTION) or \
                   ("SELL" in signal and len(active_shorts) >= MAX_SAME_DIRECTION):
                    continue

                signals.append((symbol, signal, confidence, price))

        # Sort signals by confidence and pick the BEST only
        signals.sort(key=lambda x: x[2], reverse=True)

        # Print active trades summary with LIVE P&L
        if self.active_trades:
            print(f"\n   {'─'*40}")
            print(f"   💰 ACTIVE POSITIONS:")
            for t in self.active_trades:
                curr_p = self.market_data[t.symbol]["1m"][-1]["c"]
                pnl = (curr_p - t.entry_price) * t.quantity if t.side == "LONG" else (t.entry_price - curr_p) * t.quantity
                pnl_pct = (pnl / (t.quantity * t.entry_price)) * 100
                color = "🟢" if pnl >= 0 else "🔴"
                print(f"   {color} {t.symbol} {t.side}: ${pnl:+.2f} ({pnl_pct:+.2f}%) | SL:{t.stop_loss:,.2f}")
            print(f"   {'─'*40}\n")

        for symbol, signal, confidence, price in signals:
            if len(self.active_trades) < MAX_CONCURRENT:
                self.place_order(symbol, signal, confidence, price)

        self._save_memory()
        self._update_dashboard()
        self._print_status()

    # ── TELEGRAM COMMAND LISTENER ─────────────────────────────────────────────

    def _listen_commands(self):
        """Background thread: listen for Telegram commands."""
        if not self.tg:
            return
        print("📡 Listening for Telegram commands...")
        while self.is_running:
            updates = self.tg.get_updates()
            for update in updates:
                msg  = update.get("message", {})
                text = msg.get("text", "").strip().lower()
                if not text:
                    continue

                if text in ("/start", "start"):
                    self.tg.send("✅ Bot is <b>running</b>.\nUse /status, /pnl, /trades to check up.")

                elif text in ("/stop", "stop"):
                    self.tg.send("🛑 Stop command received. Finishing current cycle...")
                    self.is_running = False

                elif text in ("/status", "status"):
                    self.tg.send(self._build_status_msg())

                elif text in ("/pnl", "pnl"):
                    self.tg.send(self._build_pnl_msg())

                elif text in ("/trades", "trades"):
                    self.tg.send(self._build_trades_msg())

                elif text in ("/analysis", "analysis"):
                    self.tg.send(self._build_analysis_msg())

                elif text in ("/help", "help"):
                    self.tg.send(
                        "📋 <b>Available Commands</b>\n\n"
                        "/status   — Balance & open positions\n"
                        "/analysis — Market trend analysis\n"
                        "/pnl      — Profit/loss summary\n"
                        "/trades   — Last 5 closed trades\n"
                        "/stop     — Stop the bot\n"
                        "/start    — Confirm bot is running"
                    )

            time.sleep(2)

    def _build_status_msg(self) -> str:
        mode_tag = "💵 REAL" if self.trade_mode == "real" else "📄 PAPER"
        lines = [
            f"📊 <b>BOT STATUS — {mode_tag}</b>",
            f"💼 Balance : <code>${self.balance:,.4f}</code>",
            f"📈 Total P&L: <code>${self.stats['total_pnl']:+.4f}</code>  "
            f"({self.stats['total_pnl']/self.initial_balance*100:+.2f}%)",
            f"🏆 Trades  : {self.stats['winning_trades']}W / {self.stats['losing_trades']}L",
            f"🕐 Updated : {datetime.now().strftime('%H:%M:%S')}",
            "",
        ]
        if self.active_trades:
            lines.append(f"🔓 <b>Open Positions ({len(self.active_trades)})</b>")
            for t in self.active_trades:
                prices = list(self.market_data.get(t.symbol, {}).get("1m", []))
                curr   = prices[-1]["c"] if prices else t.entry_price
                pos_val = getattr(t, "_pos_val", self.position_value)
                pnl    = (curr - t.entry_price) * t.quantity if t.side == "LONG" else (t.entry_price - curr) * t.quantity
                lock   = " 🔒" if (t.side == "LONG" and t.stop_loss > t.entry_price) or \
                                   (t.side == "SHORT" and t.stop_loss < t.entry_price) else ""
                lines.append(
                    f"  • {t.symbol} {t.side}{lock}: <code>${pnl:+.4f}</code> ({pnl/pos_val*100:+.2f}%)"
                )
        else:
            lines.append("⏳ No open positions")
        return "\n".join(lines)

    def _build_pnl_msg(self) -> str:
        s = self.stats
        win_rate = (s["winning_trades"] / s["total_trades"] * 100) if s["total_trades"] > 0 else 0
        pf       = (s["gross_profit"] / s["gross_loss"]) if s["gross_loss"] > 0 else 0
        return (
            f"📈 <b>P&L REPORT</b>\n\n"
            f"💰 Starting  : <code>${self.initial_balance:,.2f}</code>\n"
            f"💼 Current   : <code>${self.balance:,.4f}</code>\n"
            f"📊 Net P&L   : <code>${s['total_pnl']:+.4f}</code>  "
            f"({s['total_pnl']/self.initial_balance*100:+.2f}%)\n"
            f"💸 Total Fees: <code>${s['total_fees']:.4f}</code>\n"
            f"🏆 Win Rate  : {win_rate:.1f}%  ({s['winning_trades']}W / {s['losing_trades']}L)\n"
            f"📉 Gross Loss: <code>${s['gross_loss']:.4f}</code>\n"
            f"📈 Gross Profit: <code>${s['gross_profit']:.4f}</code>\n"
            f"⚖️ Profit Factor: {pf:.2f}\n"
            f"📉 Max Drawdown: {s['max_drawdown']*100:.2f}%"
        )

    def _build_analysis_msg(self) -> str:
        lines = ["🔍 <b>MARKET ANALYSIS</b>\n"]
        for symbol in self.symbols[:5]: # Top 5
            self.update_market_data(symbol)
            candles_2h = [c["c"] for c in self.market_data[symbol]["2h"]]
            ema200 = self._ema(candles_2h, 200)
            curr = candles_2h[-1] if candles_2h else 0
            bias = "🐂 BULLISH" if curr > ema200 else "🐻 BEARISH"
            vol = self.get_volume_bias(symbol)
            sig, conf = self.generate_signal(symbol)
            
            lines.append(
                f"<b>{symbol}</b>: {bias}\n"
                f"  • Price: <code>${curr:,.2f}</code>\n"
                f"  • EMA 200: <code>${ema200:,.2f}</code>\n"
                f"  • Vol. Ratio: {vol:.2f}x\n"
                f"  • Signal: {sig} ({conf:.0%})\n"
            )
        return "\n".join(lines)

    def _build_trades_msg(self) -> str:
        recent = self.closed_trades[-5:]
        if not recent:
            return "📋 No closed trades yet."
        lines = ["📋 <b>LAST 5 CLOSED TRADES</b>\n"]
        for t in reversed(recent):
            pnl_emoji = "✅" if (t.profit_loss or 0) > 0 else "❌"
            dur = (t.exit_time - t.entry_time).total_seconds() / 60 if t.exit_time else 0
            lines.append(
                f"{pnl_emoji} #{t.trade_id} <b>{t.symbol}</b> {t.side}\n"
                f"   P&L: <code>${t.profit_loss:+.4f}</code> ({t.profit_loss_pct:+.2f}%) | "
                f"{t.exit_reason} | {dur:.0f}m"
            )
        return "\n".join(lines)

    # ── STATUS PRINT ──────────────────────────────────────────────────────────

    def _print_status(self):
        win_rate = (self.stats["winning_trades"] / self.stats["total_trades"] * 100
                    if self.stats["total_trades"] > 0 else 0)
        print(f"\n📊 {datetime.now().strftime('%H:%M:%S')} | "
              f"Balance: ${self.balance:,.4f} | "
              f"P&L: ${self.stats['total_pnl']:+.4f} | "
              f"W/L: {self.stats['winning_trades']}/{self.stats['losing_trades']} ({win_rate:.1f}%)")

    # ── START / STOP ──────────────────────────────────────────────────────────

    def start(self, interval: int = INTERVAL_SECONDS, max_cycles: Optional[int] = None):
        print(f"\n🚀 Bot started — scanning every {interval}s")
        print("   Press Ctrl+C to stop\n")

        self.is_running = True

        # Start Telegram command listener in background
        self._cmd_thread = threading.Thread(target=self._listen_commands, daemon=True)
        self._cmd_thread.start()

        if self.tg:
            mode_tag = "💵 REAL MONEY" if self.trade_mode == "real" else "📄 PAPER"
            self.tg.send(
                f"🚀 <b>Bot Started — {mode_tag}</b>\n"
                f"📌 Symbols: {', '.join(self.symbols)}\n"
                f"💼 Balance: ${self.balance:,.2f}\n"
                f"⚙️ Leverage: {self.leverage}x | Risk: {self.risk_pct*100:.1f}% per trade\n"
                f"📡 Send /help for commands"
            )

        cycle = 0
        try:
            while self.is_running:
                if max_cycles and cycle >= max_cycles:
                    print("\n⏹️  Max cycles reached.")
                    break
                cycle += 1
                print(f"\n{'='*65}")
                print(f"CYCLE #{cycle} | {datetime.now().isoformat()}")
                print(f"{'='*65}")
                self.run_cycle()
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
        finally:
            self.is_running = False
            self._save_memory()
            self._update_dashboard()
            if self.tg:
                self.tg.send(
                    f"🛑 <b>Bot Stopped</b>\n"
                    f"💼 Final Balance: <code>${self.balance:,.4f}</code>\n"
                    f"📊 Net P&L: <code>${self.stats['total_pnl']:+.4f}</code>\n"
                    f"🏆 Trades: {self.stats['winning_trades']}W / {self.stats['losing_trades']}L"
                )
            self._print_final_report()

    def _print_final_report(self):
        runtime  = datetime.now() - self.start_time
        win_rate = (self.stats["winning_trades"] / self.stats["total_trades"] * 100
                    if self.stats["total_trades"] > 0 else 0)
        pf = (self.stats["gross_profit"] / self.stats["gross_loss"]
              if self.stats["gross_loss"] > 0 else 0)
        print(f"\n{'='*65}")
        print("📊 FINAL SESSION REPORT")
        print(f"{'='*65}")
        print(f"  Runtime    : {runtime}")
        print(f"  Mode       : {self.trade_mode.upper()}")
        print(f"  Start Bal  : ${self.initial_balance:,.2f}")
        print(f"  End Bal    : ${self.balance:,.4f}")
        print(f"  Net P&L    : ${self.stats['total_pnl']:+.4f} ({self.stats['total_pnl']/self.initial_balance*100:+.2f}%)")
        print(f"  Trades     : {self.stats['total_trades']}  ({win_rate:.1f}% win rate)")
        print(f"  Gross Profit: ${self.stats['gross_profit']:.4f}")
        print(f"  Gross Loss  : ${self.stats['gross_loss']:.4f}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Max Drawdown: {self.stats['max_drawdown']*100:.2f}%")
        print(f"  Total Fees  : ${self.stats['total_fees']:.4f}")
        print(f"{'='*65}\n")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="SuperTrend Bot with Telegram")
    parser.add_argument("--mode",     choices=["paper", "real"], default=TRADE_MODE)
    parser.add_argument("--balance",  type=float, default=INITIAL_BALANCE)
    parser.add_argument("--symbols",  default=",".join(SYMBOLS))
    parser.add_argument("--leverage", type=int,   default=LEVERAGE)
    parser.add_argument("--interval", type=int,   default=INTERVAL_SECONDS)
    parser.add_argument("--cycles",   type=int,   default=None)
    args = parser.parse_args()

    bot = SuperTrendBot(
        trade_mode      = args.mode,
        api_key         = BINANCE_API_KEY,
        api_secret      = BINANCE_API_SECRET,
        symbols         = args.symbols.split(","),
        initial_balance = args.balance,
        leverage        = args.leverage,
        tg_token        = TELEGRAM_BOT_TOKEN,
        tg_chat_id      = TELEGRAM_CHAT_ID,
    )
    bot.start(interval=args.interval, max_cycles=args.cycles)


if __name__ == "__main__":
    main()
