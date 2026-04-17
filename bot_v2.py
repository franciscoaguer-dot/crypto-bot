"""
CryptoBot v3 — Tiered Confirmation Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sistema de trading serio con entradas por tiers y contexto dinámico.

ARQUITECTURA:
─────────────
1. CONTEXTO (modificador dinámico, no filtro binario)
   BULL     → BTC4h > EMA21 AND F&G > 25
   NEUTRAL  → lateral / sin dirección clara
   RISK_OFF → BTC4h < EMA21 OR crash OR F&G < 20

2. SCORE 0–3 (3 familias independientes)
   Tendencia → Supertrend + EMA50 + slope
   Momentum  → MACD histogram con umbral mínimo
   Volumen   → VWAP + conviction + body filter

3. TIERS DE ENTRADA
   Tier A (fuerte):   3/3 → entra siempre si contexto != RISK_OFF
   Tier B (reducido): 2/3 → solo en BULL + confirmación 15m
   1/3 → NO TRADE (ruido)

4. CONFIRMACIÓN 15m (solo Tier B)
   Vela a favor + micro momentum alineado + no sobreextendido

5. SIZING INTELIGENTE
   Tier A + major + BULL   → 2.5%
   Tier A + BULL/NEUTRAL   → 2.0%
   Tier A + sideways       → 1.5%
   Tier B                  → 1.0% (siempre reducido)
   Sideways                → todo × 0.75

6. SHORTS
   Solo majors + contexto RISK_OFF/NEUTRAL + Tier A

HERENCIA DE v2 (intacto):
   ✅ 3 familias
   ✅ Cooldown 3 ciclos
   ✅ Circuit breaker diario
   ✅ Trailing stop dinámico
   ✅ Partial TP
   ✅ Tablas Postgres separadas (v3_kv, v3_trades)
"""

import os, re, time, json, logging, requests, threading, collections
from datetime import datetime, timezone, timedelta
from enum import Enum
import ccxt
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string

# ─────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────
class Context(Enum):
    BULL     = "bull"
    NEUTRAL  = "neutral"
    RISK_OFF = "risk_off"

class Tier(Enum):
    A    = "A"      # 3/3
    B    = "B"      # 2/3 + confirmación
    NONE = "none"   # 1/3 o menos

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
PAPER_TRADING     = os.environ.get("PAPER_TRADING", "true").lower() == "true"
CAPITAL_TOTAL_USD = float(os.environ.get("CAPITAL_USD", 1000))
BINANCE_API_KEY   = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET= os.environ.get("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")

MAJORS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"}

# Sizing por tier y contexto
SIZE = {
    (Tier.A, "major", Context.BULL):    0.025,
    (Tier.A, "major", Context.NEUTRAL): 0.020,
    (Tier.A, "alt",   Context.BULL):    0.020,
    (Tier.A, "alt",   Context.NEUTRAL): 0.015,
    (Tier.B, "major", Context.BULL):    0.010,
    (Tier.B, "alt",   Context.BULL):    0.010,
}
SIZE_DEFAULT      = 0.015
SIZE_SIDEWAYS_MULT= 0.75
MAX_SIZE_PCT      = 0.025   # nunca más de 2.5%
MAX_EXPOSURE      = 0.20    # máx 20% capital total
MAX_ALTS_OPEN     = 3

# Risk management
STOP_LOSS_PCT     = 0.025
TRAILING_PCT      = 0.012
TP_PCT            = 0.030
TP_PARTIAL_PCT    = 0.018
PARTIAL_SIZE      = 0.50
ATR_MULT          = 1.5

# Señales
MACD_THRESHOLD    = 0.5     # abs(hist) > mean10 * this
VOL_MULT          = 1.3
BODY_ATR_MULT     = 1.2
VWAP_PERIODS      = 24

# Contexto
FG_BULL_MIN       = 25
FG_RISK_OFF_MAX   = 20
FG_GREED_MAX      = 75      # euforia → reducir size

# Liquidez
MIN_VOL_24H       = 20_000_000

# Timeframes
TF_SETUP    = "1h"
TF_CONTEXT  = "4h"
TF_CONFIRM  = "15m"

COOLDOWN_CANDLES  = 3
LOOP_SEC          = 60

ARG_TZ = timezone(timedelta(hours=-3))

EXCLUDE_SYMBOLS = {
    "USDT","USDC","BUSD","DAI","TUSD","FDUSD","USDP","USD1","RLUSD","EUR",
    "WBTC","WETH","STETH","BETH","BTC","ETH","SOL","BNB","LDUSDT","XAUT","PAXG"
}

# ─────────────────────────────────────────
# PERSISTENCIA
# ─────────────────────────────────────────
DATABASE_URL = os.environ.get("DATABASE_URL", "")
_pg = None
_USE_PG = False

if DATABASE_URL:
    try:
        import psycopg2
        _pg = psycopg2.connect(DATABASE_URL, sslmode="require")
        _pg.autocommit = True
        with _pg.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS v3_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS v3_trades (
                    id SERIAL PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        _USE_PG = True
    except Exception as e:
        pass

DATA_DIR       = "/data" if os.path.isdir("/data") else "."
POSITIONS_FILE = f"{DATA_DIR}/v3_positions.json"
STATE_FILE     = f"{DATA_DIR}/v3_state.json"
TRADE_LOG_FILE = f"{DATA_DIR}/v3_trades.json"

def _pg_get(key, default=None):
    try:
        with _pg.cursor() as cur:
            cur.execute("SELECT value FROM v3_kv WHERE key=%s", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else default
    except: return default

def _pg_set(key, value):
    try:
        with _pg.cursor() as cur:
            cur.execute("""
                INSERT INTO v3_kv (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()
            """, (key, json.dumps(value, default=str)))
    except: pass

def load_state():
    d = {"capital": CAPITAL_TOTAL_USD, "day_start_capital": CAPITAL_TOTAL_USD,
         "day_start_date": datetime.now(ARG_TZ).strftime("%Y-%m-%d"),
         "daily_circuit": False, "tier_a_count": 0, "tier_b_count": 0}
    if _USE_PG: return _pg_get("v3_state", d)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except: pass
    return d

def save_state(s):
    if _USE_PG: _pg_set("v3_state", s)
    else:
        with open(STATE_FILE, "w") as f: json.dump(s, f, indent=2, default=str)

def load_positions():
    if _USE_PG: return _pg_get("v3_positions", {})
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_positions(p):
    if _USE_PG: _pg_set("v3_positions", p)
    else:
        with open(POSITIONS_FILE, "w") as f: json.dump(p, f, indent=2, default=str)

def save_trade(r):
    if _USE_PG:
        try:
            with _pg.cursor() as cur:
                cur.execute("INSERT INTO v3_trades (data) VALUES (%s)",
                            (json.dumps(r, default=str),))
            return
        except: pass
    data = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: data = json.load(f)
        except: pass
    data.append(r)
    with open(TRADE_LOG_FILE, "w") as f: json.dump(data, f, indent=2, default=str)

def load_trades():
    if _USE_PG:
        try:
            with _pg.cursor() as cur:
                cur.execute("SELECT data FROM v3_trades ORDER BY created_at ASC")
                return [row[0] for row in cur.fetchall()]
        except: pass
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: return json.load(f)
        except: pass
    return []

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("v3")

_log_buffer = collections.deque(maxlen=40)
class LogBuf(logging.Handler):
    def emit(self, r):
        msg = re.sub(r'\x1b\[[0-9;]*m', '', self.format(r))
        _log_buffer.append({"ts": r.created, "msg": msg[-220:]})
_lbh = LogBuf()
_lbh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_lbh)

def log_decision(symbol, score, families, tier, context, action, reason, size=None):
    """Log estructurado de cada decisión para auditoría completa."""
    t = families.get("trend", 0)
    m = families.get("momentum", 0)
    v = families.get("volume", 0)
    tier_str = f"Tier {tier.value}" if tier != Tier.NONE else "NO TRADE"
    size_str = f" | size=${size:.2f}" if size else ""
    log.info(
        f"  [{symbol}] T:{t:+d} M:{m:+d} V:{v:+d} → {score}/3 | "
        f"CTX:{context.value.upper()} | {tier_str}{size_str} | {reason}"
    )

# ─────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10)
    except: pass

# ─────────────────────────────────────────
# EXCHANGES
# ─────────────────────────────────────────
def get_exchange():
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET,
        "enableRateLimit": True, "options": {"defaultType": "spot"},
        "urls": {"api": {"public": "https://testnet.binance.vision/api"}},
    })

def get_public_exchange():
    return ccxt.binance({"enableRateLimit": True})

# ─────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────
def get_ohlcv(exchange, symbol, timeframe="1h", limit=100):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema50_slope"] = df["ema50"].diff(3) / df["ema50"].shift(3)

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["macd_hist_mean10"] = df["macd_hist"].abs().rolling(10).mean()

    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    df["vwap"]     = (df["close"] * df["volume"]).rolling(VWAP_PERIODS).sum() / \
                      df["volume"].rolling(VWAP_PERIODS).sum()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["body"]     = (df["close"] - df["open"]).abs()

    # Supertrend
    mult = 3.0
    hl2  = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * df["atr"]
    lower = hl2 - mult * df["atr"]
    st    = pd.Series(index=df.index, dtype=float)
    st_dir = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper.iloc[i-1]:
            st_dir.iloc[i] = 1
            st.iloc[i] = lower.iloc[i]
        elif df["close"].iloc[i] < lower.iloc[i-1]:
            st_dir.iloc[i] = -1
            st.iloc[i] = upper.iloc[i]
        else:
            st_dir.iloc[i] = st_dir.iloc[i-1]
            st.iloc[i] = lower.iloc[i] if st_dir.iloc[i] == 1 else upper.iloc[i]
    df["supertrend_dir"] = st_dir
    return df

# ─────────────────────────────────────────
# LAS 3 FAMILIAS
# ─────────────────────────────────────────
def family_trend(df) -> int:
    last = df.iloc[-1]
    st_dir = int(last.get("supertrend_dir", 0))
    close  = float(last["close"])
    ema50  = float(last["ema50"])
    slope  = float(last.get("ema50_slope", 0))
    if st_dir == 1 and close > ema50 and slope > 0:   return +1
    if st_dir == -1 and close < ema50 and slope < 0:  return -1
    return 0

def family_momentum(df) -> int:
    last = df.iloc[-1]; prev = df.iloc[-2]
    hist      = float(last["macd_hist"])
    hist_prev = float(prev["macd_hist"])
    mean10    = float(last.get("macd_hist_mean10", 1e-9)) or 1e-9
    threshold = mean10 * MACD_THRESHOLD
    if hist > 0 and hist > hist_prev and abs(hist) > threshold: return +1
    if hist < 0 and hist < hist_prev and abs(hist) > threshold: return -1
    return 0

def family_volume(df, direction: int) -> int:
    last = df.iloc[-1]
    close  = float(last["close"])
    vwap   = float(last.get("vwap", close))
    vol    = float(last["volume"])
    vol_ma = float(last.get("vol_ma20", vol)) or 1
    body   = float(last["body"])
    atr    = float(last.get("atr", close * 0.01)) or 1
    if pd.isna(vwap) or pd.isna(vol_ma): return 0
    if vol < vol_ma * VOL_MULT: return 0
    if body >= atr * BODY_ATR_MULT: return 0
    if direction >= 0 and close > vwap: return +1
    if direction <= 0 and close < vwap: return -1
    return 0

# ─────────────────────────────────────────
# CONFIRMACIÓN 15m (Tier B)
# ─────────────────────────────────────────
def confirm_15m(exchange, symbol, direction: int) -> bool:
    """
    Confirmación en timeframe 15m para Tier B.
    Requiere:
    - Vela a favor (close > open para long, < para short)
    - Micro momentum alineado (MACD hist en dirección)
    - No sobreextendido (precio no lejos de VWAP)
    """
    try:
        df = calculate_indicators(get_ohlcv(exchange, symbol, TF_CONFIRM, limit=30))
        last = df.iloc[-1]
        close = float(last["close"])
        open_ = float(last["open"])
        hist  = float(last["macd_hist"])
        vwap  = float(last.get("vwap", close))

        candle_ok  = (close > open_) if direction > 0 else (close < open_)
        momentum_ok= (hist > 0) if direction > 0 else (hist < 0)
        vwap_dist  = abs(close - vwap) / vwap if vwap else 0
        not_extended = vwap_dist < 0.03  # no más de 3% lejos del VWAP

        result = candle_ok and momentum_ok and not_extended
        log.info(f"  15m confirm {symbol}: candle={'✅' if candle_ok else '❌'} "
                 f"momentum={'✅' if momentum_ok else '❌'} "
                 f"extended={'❌' if not not_extended else '✅'} → {'PASS' if result else 'FAIL'}")
        return result
    except Exception as e:
        log.warning(f"  15m confirm error {symbol}: {e}")
        return False

# ─────────────────────────────────────────
# CONTEXTO DINÁMICO
# ─────────────────────────────────────────
_btc4h_cache = {"bull": None, "ts": 0}

def get_market_context(exchange, regime: str, fg_value: int) -> Context:
    """
    Determina el contexto de mercado como modificador dinámico.

    BULL:     BTC4h > EMA21 AND F&G > 25
    RISK_OFF: BTC4h < EMA21 OR crash OR F&G < 20
    NEUTRAL:  todo lo demás
    """
    global _btc4h_cache
    now = time.time()

    if now - _btc4h_cache["ts"] > 900:
        try:
            df4h = calculate_indicators(get_ohlcv(exchange, "BTC/USDT", TF_CONTEXT, limit=30))
            last = df4h.iloc[-1]
            btc_bull = float(last["close"]) > float(last["ema21"])
            _btc4h_cache = {"bull": btc_bull, "ts": now}
            log.info(f"  🌍 BTC 4h: {'↑ BULL' if btc_bull else '↓ BEAR'} "
                     f"({float(last['close']):.0f} vs EMA21 {float(last['ema21']):.0f})")
        except Exception as e:
            log.warning(f"  BTC 4h context error: {e}")

    btc_bull = _btc4h_cache.get("bull", True)
    is_crash = regime == "crash"

    # RISK_OFF: cualquier condición negativa
    if not btc_bull or is_crash or fg_value < FG_RISK_OFF_MAX:
        ctx = Context.RISK_OFF
    # BULL: todas las condiciones positivas
    elif btc_bull and fg_value > FG_BULL_MIN and not is_crash:
        ctx = Context.BULL
    # NEUTRAL: todo lo demás
    else:
        ctx = Context.NEUTRAL

    log.info(f"  📊 Contexto: {ctx.value.upper()} "
             f"(BTC4h={'↑' if btc_bull else '↓'} | F&G={fg_value} | régimen={regime})")
    return ctx

# ─────────────────────────────────────────
# ENGINE DE ENTRADA — TIERS
# ─────────────────────────────────────────
def evaluate_entry(symbol, score_long, score_short, families_long, families_short,
                   context, fg_value, exchange, is_major) -> tuple:
    """
    Motor de decisión de entrada por tiers.

    Retorna (action, tier, reason) o (None, Tier.NONE, reason)

    Reglas:
    - Tier A Long:  score=3 AND contexto != RISK_OFF
    - Tier B Long:  score=2 AND contexto=BULL AND confirmación 15m
    - Tier A Short: score=3 AND is_major AND contexto=RISK_OFF/NEUTRAL
    - Tier B Short: NO (shorts siempre Tier A)
    - F&G > 75 (euforia): reducir size, no bloquear entrada
    """
    # ── LONGS ────────────────────────────────────────────────────────────────
    if context != Context.RISK_OFF:
        if score_long == 3:
            return "BUY", Tier.A, "score=3/3 Tier A"

        if score_long == 2 and context == Context.BULL:
            # Tier B: necesita confirmación 15m
            if confirm_15m(exchange, symbol, +1):
                return "BUY", Tier.B, "score=2/3 Tier B + 15m confirmado"
            else:
                return None, Tier.NONE, "score=2/3 Tier B pero 15m no confirma"

        if score_long == 2 and context == Context.NEUTRAL:
            return None, Tier.NONE, "score=2/3 en NEUTRAL → threshold insuficiente"

    # ── SHORTS (solo majors) ──────────────────────────────────────────────────
    if is_major and context in [Context.RISK_OFF, Context.NEUTRAL]:
        if score_short == 3:
            return "SELL", Tier.A, "score=3/3 short Tier A"

    return None, Tier.NONE, f"score LONG={score_long} SHORT={score_short} insuficiente"

# ─────────────────────────────────────────
# SIZING INTELIGENTE
# ─────────────────────────────────────────
def get_size(capital: float, symbol: str, tier: Tier,
             context: Context, regime: str, fg_value: int) -> float:
    """
    Sizing dinámico por tier, contexto y tipo de activo.
    """
    is_major = symbol in MAJORS
    asset_type = "major" if is_major else "alt"
    sideways = regime == "sideways"

    key = (tier, asset_type, context)
    pct = SIZE.get(key, SIZE_DEFAULT)

    # Reducción en sideways
    if sideways:
        pct *= SIZE_SIDEWAYS_MULT

    # Reducción en euforia (F&G > 75)
    if fg_value > FG_GREED_MAX:
        pct *= 0.75
        log.info(f"  ⚠️  Euforia (F&G={fg_value}) → size reducido 25%")

    pct = min(pct, MAX_SIZE_PCT)
    usd_size = round(capital * pct, 2)
    max_usd  = round(capital * MAX_EXPOSURE, 2)
    return min(usd_size, max_usd)

# ─────────────────────────────────────────
# COOLDOWN
# ─────────────────────────────────────────
_cooldown: dict = {}

def is_in_cooldown(symbol: str) -> bool:
    r = _cooldown.get(symbol, 0)
    if r > 0:
        log.info(f"  ⏳ Cooldown {symbol}: {r} ciclos restantes")
        return True
    return False

def set_cooldown(symbol: str):
    _cooldown[symbol] = COOLDOWN_CANDLES

def tick_cooldowns():
    for s in list(_cooldown.keys()):
        _cooldown[s] -= 1
        if _cooldown[s] <= 0: del _cooldown[s]

# ─────────────────────────────────────────
# GESTIÓN DE POSICIONES
# ─────────────────────────────────────────
def open_position(symbol, entry_price, usd_size, action, atr_value, tier, context):
    positions = load_positions()
    is_short  = action == "SELL"
    trail_pct = max((atr_value * ATR_MULT / entry_price) if atr_value else TRAILING_PCT,
                     TRAILING_PCT)
    trail_pct = min(trail_pct, 0.03)

    if is_short:
        trail_stop  = round(entry_price * (1 + trail_pct), 8)
        take_profit = round(entry_price * (1 - TP_PCT), 8)
        partial_tp  = round(entry_price * (1 - TP_PARTIAL_PCT), 8)
    else:
        trail_stop  = round(entry_price * (1 - trail_pct), 8)
        take_profit = round(entry_price * (1 + TP_PCT), 8)
        partial_tp  = round(entry_price * (1 + TP_PARTIAL_PCT), 8)

    positions[symbol] = {
        "symbol": symbol, "action": action,
        "entry_price": entry_price, "current_price": entry_price,
        "high_price": entry_price, "trail_stop": trail_stop,
        "take_profit": take_profit, "partial_tp": partial_tp,
        "partial_closed": False, "usd_size": usd_size,
        "trail_pct": trail_pct, "tier": tier.value,
        "context_entry": context.value,
        "opened_at": datetime.now().isoformat(),
    }
    save_positions(positions)
    log.info(f"  ✅ ABIERTA {symbol} {action} Tier {tier.value} @ {entry_price} "
             f"| size=${usd_size} | trail={trail_pct*100:.2f}%")

def update_trailing_stops(exchange, state):
    positions = load_positions()
    if not positions: return
    for symbol, pos in list(positions.items()):
        try:
            price = float(exchange.fetch_ticker(symbol)["last"])
            pos["current_price"] = price
            is_short  = pos["action"] == "SELL"
            trail_pct = pos.get("trail_pct", TRAILING_PCT)

            if is_short:
                if price < pos["high_price"]:
                    pos["high_price"] = price
                    pos["trail_stop"] = round(price * (1 + trail_pct), 8)
            else:
                if price > pos["high_price"]:
                    pos["high_price"] = price
                    pos["trail_stop"] = round(price * (1 - trail_pct), 8)
                unreal = (price - pos["entry_price"]) / pos["entry_price"]
                if unreal > 0.015:
                    new_tp = price * (1 + 0.01)
                    if new_tp > pos["take_profit"]:
                        pos["take_profit"] = round(new_tp, 8)
                        log.info(f"  🎯 TP subido {symbol}: {pos['take_profit']:.6f}")

            log.info(f"  📈 Trail {symbol}: {pos['trail_stop']:.6f}")

            # Partial TP
            if not pos["partial_closed"]:
                hit = (not is_short and price >= pos["partial_tp"]) or \
                      (is_short     and price <= pos["partial_tp"])
                if hit:
                    pos["partial_closed"] = True
                    pos["usd_size"] = round(pos["usd_size"] * (1 - PARTIAL_SIZE), 2)
                    log.info(f"  ✂️  Partial TP {symbol} @ {price:.6f}")

            # Salida completa
            exit_reason = None
            if not is_short:
                if price <= pos["trail_stop"]:    exit_reason = "trail"
                elif price >= pos["take_profit"]: exit_reason = "take_profit"
                elif price <= pos["entry_price"] * (1 - STOP_LOSS_PCT): exit_reason = "stop_loss"
            else:
                if price >= pos["trail_stop"]:    exit_reason = "trail"
                elif price <= pos["take_profit"]: exit_reason = "take_profit"
                elif price >= pos["entry_price"] * (1 + STOP_LOSS_PCT): exit_reason = "stop_loss"

            if exit_reason:
                pnl = ((price - pos["entry_price"]) / pos["entry_price"] * 100
                       if not is_short else
                       (pos["entry_price"] - price) / pos["entry_price"] * 100)
                pnl     = round(pnl, 3)
                usd_pnl = round(pnl * pos["usd_size"] / 100, 2)
                state["capital"] = round(state.get("capital", CAPITAL_TOTAL_USD) + usd_pnl, 2)
                save_state(state)
                emoji = "🟢" if pnl > 0 else "🔴"
                log.info(f"  {emoji} CERRADA {symbol} Tier {pos.get('tier','?')} "
                         f"@ {price:.6f} | PnL: {pnl:+.2f}% ({usd_pnl:+.2f}) | {exit_reason}")
                send_telegram(
                    f"{emoji} <b>v3 {symbol}</b> [Tier {pos.get('tier','?')}] {exit_reason.upper()}\n"
                    f"Entrada: {pos['entry_price']} → Salida: {price:.6f}\n"
                    f"PnL: {pnl:+.2f}% | ${usd_pnl:+.2f}\n"
                    f"Capital: ${state['capital']:.2f} | Ctx: {pos.get('context_entry','?')}"
                )
                save_trade({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol, "action": pos["action"],
                    "entry": pos["entry_price"], "exit": price,
                    "pnl_pct": pnl, "usd_pnl": usd_pnl,
                    "reason": exit_reason, "usd_size": pos["usd_size"],
                    "tier": pos.get("tier"), "context_entry": pos.get("context_entry"),
                    "partial": pos["partial_closed"],
                })
                set_cooldown(symbol)
                del positions[symbol]
                save_positions(positions)
        except Exception as e:
            log.error(f"  Trail error {symbol}: {e}")

# ─────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────
def check_daily_circuit(state) -> bool:
    today = datetime.now(ARG_TZ).strftime("%Y-%m-%d")
    if state.get("day_start_date") != today:
        state["day_start_capital"] = state.get("capital", CAPITAL_TOTAL_USD)
        state["day_start_date"]    = today
        state["daily_circuit"]     = False
    day_start = state.get("day_start_capital", CAPITAL_TOTAL_USD)
    current   = state.get("capital", CAPITAL_TOTAL_USD)
    daily_loss = (day_start - current) / day_start if day_start > 0 else 0
    if daily_loss > 0.03 and not state.get("daily_circuit"):
        state["daily_circuit"] = True
        save_state(state)
        send_telegram(f"🛑 <b>v3 Circuit Breaker</b>\nPérdida diaria: {daily_loss*100:.1f}%")
        log.info(f"🛑 v3 Circuit breaker — {daily_loss*100:.1f}%")
    return not state.get("daily_circuit", False)

# ─────────────────────────────────────────
# ANÁLISIS PRINCIPAL
# ─────────────────────────────────────────
def analyze_symbol(symbol, exchange, regime, fg_value, state, context) -> bool:
    positions = load_positions()
    if symbol in positions:
        log.info(f"  {symbol}: posición ya abierta — skip")
        return False
    if is_in_cooldown(symbol):
        return False

    is_major = symbol in MAJORS

    # En RISK_OFF solo majors y solo shorts
    if context == Context.RISK_OFF and not is_major:
        return False

    try:
        df = calculate_indicators(get_ohlcv(exchange, symbol, TF_SETUP, limit=100))
    except Exception as e:
        log.warning(f"  OHLCV error {symbol}: {e}")
        return False

    # Calcular familias
    t_long = family_trend(df)
    m_long = family_momentum(df)
    v_long = family_volume(df, +1)
    t_short = family_trend(df)
    m_short = family_momentum(df)
    v_short = family_volume(df, -1)

    score_long  = sum(1 for s in [t_long,  m_long,  v_long]  if s == +1)
    score_short = sum(1 for s in [t_short, m_short, v_short] if s == -1)

    fam_long  = {"trend": t_long,  "momentum": m_long,  "volume": v_long}
    fam_short = {"trend": t_short, "momentum": m_short, "volume": v_short}

    # Motor de entrada
    action, tier, reason = evaluate_entry(
        symbol, score_long, score_short, fam_long, fam_short,
        context, fg_value, exchange, is_major
    )

    # Log estructurado de la decisión
    if action:
        fam = fam_long if action == "BUY" else fam_short
        score = score_long if action == "BUY" else score_short
    else:
        fam = fam_long
        score = score_long

    log_decision(symbol, score, fam, tier, context, action, reason)

    if not action:
        return False

    # Control de exposición
    open_pos  = load_positions()
    allocated = sum(p.get("usd_size", 0) for p in open_pos.values())
    capital   = state.get("capital", CAPITAL_TOTAL_USD)
    if allocated >= capital * MAX_EXPOSURE:
        log.info(f"  ⏭️  Exposición máxima (${allocated:.0f}/${capital*MAX_EXPOSURE:.0f})")
        return False

    # Control correlación altcoins
    open_alts = [s for s in open_pos if s not in MAJORS]
    if not is_major and len(open_alts) >= MAX_ALTS_OPEN:
        log.info(f"  ⏭️  Correlación: {len(open_alts)} altcoins abiertas")
        return False

    # Sizing
    usd_size = get_size(capital, symbol, tier, context, regime, fg_value)
    if usd_size < 5:
        log.info(f"  ⏭️  Size demasiado pequeño (${usd_size})")
        return False

    last  = df.iloc[-1]
    price = float(last["close"])
    atr   = float(last["atr"]) if not pd.isna(last.get("atr", float("nan"))) else None

    tier_emoji = "🔥" if tier == Tier.A else "⚡"
    log.info(f"  {tier_emoji} SEÑAL {action} {symbol} @ {price:.6f} "
             f"| Tier {tier.value} | size=${usd_size} | ctx={context.value}")

    if PAPER_TRADING:
        open_position(symbol, price, usd_size, action, atr, tier, context)
        send_telegram(
            f"📝 <b>v3 PAPER {action} {symbol}</b> [Tier {tier.value}]\n"
            f"Score: {score}/3 | Contexto: {context.value.upper()}\n"
            f"Precio: {price:.6f} | Size: ${usd_size}\n"
            f"F&G: {fg_value} | Régimen: {regime}\n"
            f"Motivo: {reason}"
        )
        # Actualizar contador de tiers en state
        key = f"tier_{tier.value.lower()}_count"
        state[key] = state.get(key, 0) + 1
        save_state(state)
        return True
    return False

# ─────────────────────────────────────────
# SCANNER
# ─────────────────────────────────────────
_altcoins  = []
_last_scan = 0

def scan_altcoins(exchange) -> list:
    global _altcoins, _last_scan
    now = time.time()
    if now - _last_scan < 600 and _altcoins: return _altcoins
    try:
        tickers = exchange.fetch_tickers()
        pairs   = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT"): continue
            base = sym.replace("/USDT", "")
            if base in EXCLUDE_SYMBOLS: continue
            if not base.isascii() or len(base) > 10: continue
            if sym in MAJORS: continue
            if (t.get("quoteVolume") or 0) < MIN_VOL_24H: continue
            pairs.append(sym)
        pairs.sort()
        _altcoins  = pairs[:15]
        _last_scan = now
        log.info(f"🔍 v3 Altcoins: {_altcoins}")
    except Exception as e:
        log.error(f"Scanner error: {e}")
    return _altcoins

# ─────────────────────────────────────────
# RÉGIMEN Y F&G
# ─────────────────────────────────────────
def detect_regime(exchange) -> str:
    try:
        df = calculate_indicators(get_ohlcv(exchange, "BTC/USDT", "1d", limit=7))
        pct = (df.iloc[-1]["close"] - df.iloc[-7]["close"]) / df.iloc[-7]["close"]
        atr_pct = df.iloc[-1]["atr"] / df.iloc[-1]["close"]
        if pct < -0.08 and atr_pct > 0.04: return "crash"
        if pct < -0.03: return "bear"
        if pct > 0.03:  return "bull"
        return "sideways"
    except: return "sideways"

def get_fear_greed() -> tuple:
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except: return 50, "Neutral"

# ─────────────────────────────────────────
# DASHBOARD v3
# ─────────────────────────────────────────
DASHBOARD_V3 = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CryptoBot v3</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#04080f;--s1:#0b1220;--s2:#101828;--border:#162030;--green:#0dffb0;--red:#ff3366;--yellow:#ffd700;--blue:#38bdf8;--orange:#fb923c;--purple:#c084fc;--text:#a8bfd4;--text2:#6b8299;--white:#e2f0ff;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;min-height:100vh;padding:20px 24px}
h1{font-family:var(--sans);font-size:18px;color:var(--white)}
h1 span.v{color:var(--purple)}
.sub{font-size:10px;color:var(--text2);margin-bottom:20px;margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:20px}
.kpi{background:var(--s1);border:1px solid var(--border);border-radius:6px;padding:12px;position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--k,var(--green))}
.kl{font-size:9px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
.kv{font-family:var(--sans);font-weight:700;font-size:20px;color:var(--white)}
.kv.g{color:var(--green)}.kv.r{color:var(--red)}.kv.y{color:var(--yellow)}.kv.b{color:var(--blue)}.kv.p{color:var(--purple)}.kv.o{color:var(--orange)}
.ks{font-size:10px;color:var(--text2);margin-top:2px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.panel{background:var(--s1);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-bottom:12px}
.ph{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:var(--s2);border-bottom:1px solid var(--border)}
.pt{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--text2);font-weight:600}
/* Tiers */
.tiers{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px 14px}
.tier{border-radius:5px;padding:12px;border:1px solid var(--border)}
.tier.a{border-color:rgba(192,132,252,.4);background:rgba(192,132,252,.05)}
.tier.b{border-color:rgba(251,146,60,.3);background:rgba(251,146,60,.04)}
.tier-title{font-size:10px;color:var(--text2);margin-bottom:6px;letter-spacing:.08em;text-transform:uppercase}
.tier-count{font-family:var(--sans);font-weight:700;font-size:28px}
.tier-count.a{color:var(--purple)}.tier-count.b{color:var(--orange)}
.tier-sub{font-size:10px;color:var(--text2);margin-top:2px}
/* Context */
.ctx-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:100px;font-size:11px;font-weight:600;border:1px solid}
.ctx-bull{color:var(--green);border-color:rgba(13,255,176,.4);background:rgba(13,255,176,.06)}
.ctx-neutral{color:var(--yellow);border-color:rgba(255,215,0,.3);background:rgba(255,215,0,.04)}
.ctx-risk_off{color:var(--red);border-color:rgba(255,51,102,.3);background:rgba(255,51,102,.05)}
/* Positions */
.pos-list{display:flex;flex-direction:column;gap:6px;padding:10px}
.pos{border-radius:5px;padding:10px 12px;border:1px solid var(--border);background:var(--s2)}
.pos.long{border-left:3px solid var(--green)}.pos.short{border-left:3px solid var(--red)}
.pos.tier-a{border-right:2px solid var(--purple)}.pos.tier-b{border-right:2px solid var(--orange)}
.pt2{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px}
.psym{font-family:var(--sans);font-weight:700;font-size:13px;color:var(--white)}
.ppnl{font-family:var(--sans);font-weight:700;font-size:13px}
.ppnl.pos{color:var(--green)}.ppnl.neg{color:var(--red)}
.pmeta{display:grid;grid-template-columns:1fr 1fr;gap:2px;font-size:10px}
.pl{color:var(--text2)}.pv{color:var(--text);text-align:right}
/* Families */
.fam-row{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;padding:10px 14px}
.fam{border-radius:4px;padding:8px;border:1px solid var(--border);text-align:center}
.fam.ok{border-color:rgba(13,255,176,.3);background:rgba(13,255,176,.04)}
.fam.fail{border-color:rgba(255,51,102,.2)}
.fn{font-size:9px;color:var(--text2);letter-spacing:.07em;text-transform:uppercase;margin-bottom:3px}
/* Table */
.tbl{width:100%;border-collapse:collapse;font-size:11px}
.tbl th{padding:6px 10px;font-size:8px;letter-spacing:.1em;text-transform:uppercase;color:var(--text2);background:var(--s2);text-align:left}
.tbl td{padding:6px 10px;border-bottom:1px solid rgba(22,32,48,.6)}
/* Log */
#live-log{padding:8px 14px;font-size:10px;max-height:200px;overflow-y:auto;display:flex;flex-direction:column;gap:2px}
.dot{width:5px;height:5px;border-radius:50%;background:currentColor;animation:blink 2s infinite;display:inline-block}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.pill{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:100px;font-size:10px;border:1px solid currentColor}
.pill.g{color:var(--green)}.pill.p{color:var(--purple)}.pill.b{color:var(--blue)}
footer{text-align:center;font-size:9px;color:var(--text2);margin-top:16px;padding-top:12px;border-top:1px solid var(--border)}
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
  <h1>Crypto<span class="v">Bot</span> <small style="font-size:12px;color:var(--text2)">v3 Tiered Confirmation</small></h1>
  <div style="display:flex;gap:6px;align-items:center">
    <span id="ctx-badge" class="ctx-badge ctx-neutral">NEUTRAL</span>
    <span class="pill g"><span class="dot"></span> LIVE</span>
    <span class="pill p" id="mode-pill">PAPER</span>
    <span style="font-size:10px;color:var(--text2);padding:2px 8px;background:var(--s1);border:1px solid var(--border);border-radius:3px">↻ <span id="cd">15</span>s</span>
  </div>
</div>
<div class="sub">Tier A (3/3) · Tier B (2/3+15m) · Contexto dinámico · Sin RL · Pesos fijos</div>

<div class="kpis">
  <div class="kpi" style="--k:var(--green)"><div class="kl">Capital</div><div class="kv g" id="k-cap">—</div><div class="ks" id="k-cap-d">—</div></div>
  <div class="kpi" style="--k:var(--yellow)"><div class="kl">P&L Total</div><div class="kv y" id="k-pnl">—</div><div class="ks">trades cerrados</div></div>
  <div class="kpi" style="--k:var(--blue)"><div class="kl">Win Rate</div><div class="kv b" id="k-wr">—</div><div class="ks" id="k-wr-d">—</div></div>
  <div class="kpi" style="--k:var(--purple)"><div class="kl">Tier A / B</div><div class="kv p" id="k-tiers">—</div><div class="ks" id="k-tiers-d">—</div></div>
  <div class="kpi" style="--k:var(--red)"><div class="kl">F&G</div><div class="kv" id="k-fg">—</div><div class="ks" id="k-fg-l">—</div></div>
  <div class="kpi" style="--k:var(--blue)"><div class="kl">Régimen</div><div class="kv b" id="k-reg">—</div><div class="ks" id="k-reg-s">—</div></div>
</div>

<div class="grid2">
  <div>
    <div class="panel">
      <div class="ph"><span class="pt">Entradas por tier</span><span style="font-size:10px;color:var(--text2)">histórico</span></div>
      <div class="tiers">
        <div class="tier a"><div class="tier-title">Tier A · 3/3</div><div class="tier-count a" id="ta-count">0</div><div class="tier-sub" id="ta-wr">sin datos</div></div>
        <div class="tier b"><div class="tier-title">Tier B · 2/3+15m</div><div class="tier-count b" id="tb-count">0</div><div class="tier-sub" id="tb-wr">sin datos</div></div>
      </div>
    </div>
    <div class="panel">
      <div class="ph"><span class="pt">Performance por familia</span></div>
      <div class="fam-row" id="fam-row"></div>
    </div>
    <div class="panel">
      <div class="ph"><span class="pt">Actividad</span></div>
      <div id="live-log"></div>
    </div>
  </div>
  <div>
    <div class="panel">
      <div class="ph"><span class="pt">Posiciones abiertas</span><span id="pos-count" style="font-size:10px;color:var(--text2)">ninguna</span></div>
      <div id="pos-list" class="pos-list"></div>
    </div>
    <div class="panel">
      <div class="ph"><span class="pt">Scanner activo</span></div>
      <div id="scanner" style="padding:8px 14px;display:flex;flex-wrap:wrap;gap:4px"></div>
    </div>
  </div>
</div>

<div class="panel">
  <div class="ph"><span class="pt">Historial v3</span></div>
  <table class="tbl">
    <thead><tr><th>Par</th><th>Tier</th><th>Acción</th><th>Entrada</th><th>Salida</th><th>P&L</th><th>Motivo</th><th>Ctx</th><th>Size</th><th>Hora</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<footer>CryptoBot v3 · Tier A(3/3) + Tier B(2/3+15m) · Contexto dinámico · Circuit breaker · Sin RL</footer>

<script>
let cd=15; const ICAP=1000;
function fmt(n,d=2){return(+n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d})}

async function load(){
  try{
    const d=await fetch('/v3/api').then(r=>r.json());
    const st=d.state||{};
    const trades=d.trades||[];
    const closed=trades.filter(t=>t.pnl_pct!==undefined);
    const wins=closed.filter(t=>t.pnl_pct>0);
    const cap=st.capital||ICAP;

    // KPIs
    document.getElementById('k-cap').textContent='$'+fmt(cap);
    document.getElementById('k-cap').className='kv '+(cap>=ICAP?'g':'r');
    document.getElementById('k-cap-d').textContent=(cap-ICAP>=0?'+':'')+fmt(cap-ICAP)+' desde inicio';

    const tpnl=closed.reduce((s,t)=>s+(t.pnl_pct||0)*(t.usd_size||20)/100,0);
    document.getElementById('k-pnl').textContent=(tpnl>=0?'+':'')+'$'+fmt(tpnl);
    document.getElementById('k-pnl').className='kv '+(tpnl>=0?'g':'r');

    const wr=closed.length?Math.round(wins.length/closed.length*100):null;
    document.getElementById('k-wr').textContent=wr!==null?wr+'%':'—';
    document.getElementById('k-wr-d').textContent=closed.length?wins.length+'/'+closed.length+' trades':'sin datos';

    const ta=closed.filter(t=>t.tier==='A');
    const tb=closed.filter(t=>t.tier==='B');
    document.getElementById('k-tiers').textContent=(st.tier_a_count||0)+'/'+(st.tier_b_count||0);
    document.getElementById('k-tiers-d').textContent='entradas A / B';

    if(d.fear_greed){const fv=+d.fear_greed.value;document.getElementById('k-fg').textContent=fv;document.getElementById('k-fg').className='kv '+(fv<35?'r':fv>65?'g':'y');document.getElementById('k-fg-l').textContent=d.fear_greed.label;}
    const rm={bull:'BULL 📈',bear:'BEAR 📉',sideways:'SIDE ↔',crash:'CRASH 💥'};
    const reg=d.regime||'?';
    document.getElementById('k-reg').textContent=rm[reg]||reg.toUpperCase();
    document.getElementById('k-reg-s').textContent=reg;

    // Context badge
    const ctx=d.context||'neutral';
    const cbadge=document.getElementById('ctx-badge');
    cbadge.className='ctx-badge ctx-'+ctx;
    cbadge.textContent={bull:'🟢 BULL',neutral:'🟡 NEUTRAL',risk_off:'🔴 RISK-OFF'}[ctx]||ctx.toUpperCase();

    // Tier stats
    const taWr=ta.length?Math.round(ta.filter(t=>t.pnl_pct>0).length/ta.length*100):null;
    const tbWr=tb.length?Math.round(tb.filter(t=>t.pnl_pct>0).length/tb.length*100):null;
    document.getElementById('ta-count').textContent=ta.length;
    document.getElementById('tb-count').textContent=tb.length;
    document.getElementById('ta-wr').textContent=taWr!==null?taWr+'% WR':'sin datos';
    document.getElementById('tb-wr').textContent=tbWr!==null?tbWr+'% WR':'sin datos';

    // Familia stats
    const fams=[{k:'trend',n:'Tendencia',i:'📈'},{k:'momentum',n:'Momentum',i:'⚡'},{k:'volume',n:'Volumen',i:'📊'}];
    const fs=d.family_stats||{};
    document.getElementById('fam-row').innerHTML=fams.map(f=>{
      const s=fs[f.k]||{wins:0,total:0};
      const wr2=s.total?Math.round(s.wins/s.total*100):null;
      const cls=wr2===null?'':wr2>=55?'ok':'fail';
      const color=wr2===null?'var(--text2)':wr2>=55?'var(--green)':'var(--red)';
      return '<div class="fam '+cls+'"><div class="fn">'+f.i+' '+f.n+'</div><div style="color:'+color+';font-size:12px;font-weight:700">'+(wr2!==null?wr2+'% ('+s.total+')'  :'—')+'</div></div>';
    }).join('');

    // Posiciones
    const pos=d.positions||[];
    document.getElementById('pos-count').textContent=pos.length||'ninguna';
    document.getElementById('pos-list').innerHTML=pos.length?pos.map(p=>{
      const isS=p.action==='SELL';
      const pnl=isS?((p.entry_price-p.current_price)/p.entry_price*100):((p.current_price-p.entry_price)/p.entry_price*100);
      const tierCls='tier-'+(p.tier||'a').toLowerCase();
      const oa=p.opened_at?new Date(p.opened_at):null;
      const el=oa?Math.round((Date.now()-oa)/60000):null;
      return '<div class="pos '+(isS?'short':'long')+' '+tierCls+'"><div class="pt2"><span class="psym">'+p.symbol.replace('/USDT','')+'<small style="font-size:9px;color:var(--text2);margin-left:4px">Tier '+( p.tier||'?')+'</small></span><span class="ppnl '+(pnl>=0?'pos':'neg')+'">'+(pnl>=0?'+':'')+pnl.toFixed(2)+'%</span></div><div class="pmeta"><span class="pl">Entrada</span><span class="pv">'+p.entry_price+'</span><span class="pl">Trail</span><span class="pv" style="color:var(--orange)">'+p.trail_stop+'</span><span class="pl">TP</span><span class="pv" style="color:var(--green)">'+p.take_profit+'</span><span class="pl">Size</span><span class="pv">$'+p.usd_size+'</span></div>'+(el!==null?'<div style="font-size:9px;color:var(--text2);margin-top:4px">Hace '+(el<60?el+'m':(el/60).toFixed(1)+'h')+'</div>':'')+'</div>';
    }).join(''):'<div style="padding:20px;text-align:center;color:var(--text2);font-size:11px">Sin posiciones abiertas</div>';

    // Scanner
    document.getElementById('scanner').innerHTML=(d.scanner||[]).map(s=>'<span style="padding:2px 7px;border-radius:3px;font-size:10px;border:1px solid var(--border);color:var(--text2)">'+s.replace('/USDT','')+'</span>').join('');

    // Log
    try{
      const logs=await fetch('/api/log').then(r=>r.json());
      const el=document.getElementById('live-log');
      el.innerHTML=logs.slice(-20).map(l=>{
        const m=l.msg||'';
        const c=m.includes('ABIERTA')||m.includes('✅')?'var(--green)':m.includes('CERRADA')||m.includes('🔴')?'var(--red)':m.includes('Tier A')||m.includes('🔥')?'var(--purple)':m.includes('Tier B')||m.includes('⚡')?'var(--orange)':m.includes('15m')?'var(--yellow)':m.includes('⏭️')?'var(--text2)':'var(--text)';
        return '<div style="color:'+c+';line-height:1.4">'+m+'</div>';
      }).join('');
      el.scrollTop=el.scrollHeight;
    }catch(e){}

    // Tabla
    const tb2=document.getElementById('tbody');
    if(!closed.length){tb2.innerHTML='<tr><td colspan="10" style="text-align:center;padding:30px;color:var(--text2)">Sin trades aún</td></tr>';return;}
    tb2.innerHTML=[...closed].reverse().map(t=>{
      const ts=new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const pc=t.pnl_pct>=0?'var(--green)':'var(--red)';
      const tc=t.tier==='A'?'var(--purple)':'var(--orange)';
      return '<tr><td style="color:var(--white);font-weight:600">'+t.symbol+'</td><td style="color:'+tc+'">'+( t.tier||'?')+'</td><td><span style="padding:1px 5px;border-radius:2px;font-size:10px;background:'+(t.action==='BUY'?'rgba(13,255,176,.1)':'rgba(255,51,102,.1)')+';color:'+(t.action==='BUY'?'var(--green)':'var(--red)')+'">'+t.action+'</span></td><td>'+t.entry+'</td><td>'+t.exit+'</td><td style="color:'+pc+';font-weight:700">'+(t.pnl_pct>=0?'+':'')+t.pnl_pct+'%</td><td style="color:var(--text2)">'+t.reason+'</td><td style="color:var(--text2)">'+( t.context_entry||'?')+'</td><td style="color:var(--orange)">$'+t.usd_size+'</td><td style="color:var(--text2);font-size:10px">'+ts+'</td></tr>';
    }).join('');
  }catch(e){console.error(e);}
}
function tick(){cd--;document.getElementById('cd').textContent=cd;if(cd<=0){cd=15;load();}}
load();setInterval(tick,1000);
</script>
</body>
</html>"""

# ─────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────
flask_v3 = Flask("v3")

_v3_regime  = "unknown"
_v3_fg      = {"value": 50, "label": "Neutral"}
_v3_scanner = []
_v3_ctx     = "neutral"
_v3_fam_stats = {
    "trend":    {"wins": 0, "total": 0},
    "momentum": {"wins": 0, "total": 0},
    "volume":   {"wins": 0, "total": 0},
}

@flask_v3.route("/")
def v3_index():
    return render_template_string(DASHBOARD_V3)

@flask_v3.route("/v3/api")
@flask_v3.route("/api/trades")
def v3_api():
    return jsonify({
        "trades": load_trades(), "positions": list(load_positions().values()),
        "state": load_state(), "fear_greed": _v3_fg, "regime": _v3_regime,
        "scanner": _v3_scanner, "context": _v3_ctx, "family_stats": _v3_fam_stats,
    })

@flask_v3.route("/api/log")
def v3_log():
    return jsonify(list(_log_buffer))

@flask_v3.route("/health")
def v3_health():
    return jsonify({"status": "ok", "version": "v3", "paper": PAPER_TRADING})

def run_dashboard():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"🌐 v3 Dashboard en puerto {port}")
    flask_v3.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run_bot():
    global _v3_regime, _v3_fg, _v3_scanner, _v3_ctx

    log.info("=" * 55)
    log.info("🚀 RUNNING BOT V3 — TIERED CONFIRMATION EDITION")
    log.info("=" * 55)
    log.info(f"Mode: {'📝 PAPER' if PAPER_TRADING else '💰 REAL'} | Capital: ${CAPITAL_TOTAL_USD}")
    log.info("Tier A: 3/3 → entrada fuerte")
    log.info("Tier B: 2/3 + confirmación 15m → entrada reducida (solo en BULL)")
    log.info("Contexto: BULL / NEUTRAL / RISK_OFF (modificador dinámico)")

    send_telegram(
        f"🚀 <b>CryptoBot v3 — Tiered Confirmation</b>\n"
        f"Mode: {'📝 PAPER' if PAPER_TRADING else '💰 REAL'}\n"
        f"Tier A: 3/3 (entrada fuerte)\n"
        f"Tier B: 2/3+15m solo en contexto BULL\n"
        f"Contexto dinámico: BULL / NEUTRAL / RISK_OFF\n"
        f"Sin RL · Pesos fijos · Circuit breaker activo"
    )

    pub = get_public_exchange()
    state = load_state()
    last_regime = 0

    while True:
        now = time.time()
        log.info(f"\n{'='*55}")
        log.info(f"⏰ v3 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        state = load_state()

        # Régimen cada 15 min
        if now - last_regime > 900:
            _v3_regime  = detect_regime(pub)
            last_regime = now
            log.info(f"🧭 Régimen: {_v3_regime.upper()}")

        # Fear & Greed
        fg_val, fg_label = get_fear_greed()
        _v3_fg = {"value": fg_val, "label": fg_label}
        log.info(f"😱 F&G: {fg_val} — {fg_label}")

        # Circuit breaker
        if not check_daily_circuit(state):
            log.info("🛑 v3 Circuit breaker — skip entradas")
            update_trailing_stops(pub, state)
            tick_cooldowns()
            log.info(f"💤 {LOOP_SEC}s...")
            time.sleep(LOOP_SEC)
            continue

        # Trailing stops
        update_trailing_stops(pub, state)
        tick_cooldowns()

        # Contexto dinámico
        ctx = get_market_context(pub, _v3_regime, fg_val)
        _v3_ctx = ctx.value

        open_positions = load_positions()
        log.info(f"📂 v3 Posiciones: {list(open_positions.keys()) or 'ninguna'} "
                 f"| Capital: ${state.get('capital', CAPITAL_TOTAL_USD):.2f} "
                 f"| Ctx: {ctx.value.upper()}")

        # Majors
        log.info(f"\n--- MAJORS [{TF_SETUP}] ---")
        for symbol in list(MAJORS):
            try:
                log.info(f"\n📊 {symbol}...")
                analyze_symbol(symbol, pub, _v3_regime, fg_val, state, ctx)
            except Exception as e:
                log.error(f"Error {symbol}: {e}")

        # Altcoins (solo si no es RISK_OFF)
        if ctx != Context.RISK_OFF:
            alts = scan_altcoins(pub)
            _v3_scanner = alts
            log.info(f"\n--- ALTCOINS [{TF_SETUP}] ---")
            for symbol in alts:
                try:
                    log.info(f"\n📊 {symbol}...")
                    analyze_symbol(symbol, pub, _v3_regime, fg_val, state, ctx)
                except Exception as e:
                    log.error(f"Error {symbol}: {e}")
        else:
            log.info("⏭️  Altcoins: contexto RISK_OFF — solo majors y shorts")

        log.info(f"\n💤 {LOOP_SEC}s...")
        time.sleep(LOOP_SEC)

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_dashboard, daemon=True).start()
    run_bot()
