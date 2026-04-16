"""
CryptoBot v3 — Tiered Entries Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Arquitectura limpia de 3 familias + filtro de contexto.

FILOSOFÍA:
- Contexto = permiso de mercado (filtro, no score)
- Score 0-3 por familias independientes
- Sin dynamic weights, sin RL, sin news scoring
- Sin OB/Funding como familias (solo filtros de calidad)
- Shorts solo en majors

FAMILIAS:
1. Tendencia  → Supertrend + EMA50 slope
2. Momentum   → MACD histogram con umbral mínimo
3. Volumen    → VWAP + conviction + body filter

CONTEXTO (filtro):
  Long:  BTC4h > EMA21 AND régimen != crash AND F&G > 15
  Short: BTC4h < EMA21 OR régimen = crash

SIZING:
  3/3 + bull/bear → 2%
  3/3 + sideways  → 1.5%
  3/3 + major + bull → 2.5% (máximo)

REGLAS ESPECIALES:
  Altcoin + sideways → exigir 3/3 estricto + size * 0.75
  Shorts → solo majors (BTC/ETH/SOL/BNB)
  Cooldown → no reentrar en mismo activo por 3 velas
"""

import os, re, time, json, logging, requests, threading, collections
from datetime import datetime, timezone, timedelta
import ccxt
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
PAPER_TRADING        = os.environ.get("PAPER_TRADING", "true").lower() == "true"
CAPITAL_TOTAL_USD    = float(os.environ.get("CAPITAL_USD", 1000))
BINANCE_API_KEY      = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET   = os.environ.get("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN       = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")

# Majors vs Altcoins
MAJORS = {"BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"}

# Sizing
SIZE_BASE        = 0.020   # 2%
SIZE_SIDEWAYS    = 0.015   # 1.5%
SIZE_MAJOR_BULL  = 0.025   # 2.5% max
SIZE_ALT_SIDEWAYS_MULT = 0.75

# Gestión de riesgo
STOP_LOSS_PCT        = 0.025   # 2.5%
TRAILING_STOP_PCT    = 0.012   # 1.2%
TAKE_PROFIT_PCT      = 0.030   # 3%
TAKE_PROFIT_PARTIAL  = 0.018   # 1.8% parcial
PARTIAL_EXIT_PCT     = 0.50
ATR_MULT             = 1.5
MAX_CAPITAL_EXPOSURE = 0.20    # máx 20% capital total
MAX_ALTS_OPEN        = 3
COOLDOWN_CANDLES     = 3       # no reentrar por 3 velas
LOOP_INTERVAL_SEC    = 60

# Contexto
MIN_FG_LONG          = 15
MACD_HIST_THRESHOLD  = 0.5    # abs(hist) > mean * this
VOL_MULT             = 1.3
BODY_ATR_MULT        = 1.2
VWAP_PERIODS         = 24

# Liquidez mínima
MIN_VOLUME_24H       = 20_000_000  # $20M
MIN_VOLUME_24H_TIERB = 75_000_000  # majors o alts muy líquidas para 2/3
VOL_MULT_SOFT        = 1.15
BODY_ATR_MULT_SOFT   = 1.5
CONFIRM_EMA_PERIOD   = 21

# Timeframes
TF_SETUP    = "1h"    # tendencia, momentum, volumen
TF_CONTEXT  = "4h"    # BTC macro
TF_CONFIRM  = "15m"   # confirmación opcional

EXCLUDE_SYMBOLS = {
    "USDT","USDC","BUSD","DAI","TUSD","FDUSD","USDP","USD1","RLUSD","EUR",
    "WBTC","WETH","STETH","BETH","BTC","ETH","SOL","BNB","LDUSDT","XAUT","PAXG"
}

ARG_TZ = timezone(timedelta(hours=-3))

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
                CREATE TABLE IF NOT EXISTS v2_kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS v2_trades (
                    id SERIAL PRIMARY KEY,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        _USE_PG = True
    except Exception as e:
        pass

DATA_DIR       = "/data" if os.path.isdir("/data") else "."
POSITIONS_FILE = f"{DATA_DIR}/v2_positions.json"
STATE_FILE     = f"{DATA_DIR}/v2_state.json"
TRADE_LOG_FILE = f"{DATA_DIR}/v2_trades.json"

def _pg_get(key, default=None):
    try:
        with _pg.cursor() as cur:
            cur.execute("SELECT value FROM v2_kv WHERE key=%s", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else default
    except:
        return default

def _pg_set(key, value):
    try:
        with _pg.cursor() as cur:
            cur.execute("""
                INSERT INTO v2_kv (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()
            """, (key, json.dumps(value, default=str)))
    except:
        pass

def load_state():
    default = {"capital": CAPITAL_TOTAL_USD, "day_start_capital": CAPITAL_TOTAL_USD,
                "day_start_date": datetime.now(ARG_TZ).strftime("%Y-%m-%d"),
                "daily_circuit": False}
    if _USE_PG: return _pg_get("v2_state", default)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except: pass
    return default

def save_state(state):
    if _USE_PG: _pg_set("v2_state", state)
    else:
        with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2, default=str)

def load_positions():
    if _USE_PG: return _pg_get("v2_positions", {})
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_positions(positions):
    if _USE_PG: _pg_set("v2_positions", positions)
    else:
        with open(POSITIONS_FILE, "w") as f: json.dump(positions, f, indent=2, default=str)

def save_trade(record):
    if _USE_PG:
        try:
            with _pg.cursor() as cur:
                cur.execute("INSERT INTO v2_trades (data) VALUES (%s)",
                            (json.dumps(record, default=str),))
            return
        except: pass
    data = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: data = json.load(f)
        except: pass
    data.append(record)
    with open(TRADE_LOG_FILE, "w") as f: json.dump(data, f, indent=2, default=str)

def load_trades():
    if _USE_PG:
        try:
            with _pg.cursor() as cur:
                cur.execute("SELECT data FROM v2_trades ORDER BY created_at ASC")
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
log = logging.getLogger("v2")

_log_buffer = collections.deque(maxlen=30)
class LogBufHandler(logging.Handler):
    def emit(self, record):
        msg = re.sub(r'\x1b\[[0-9;]*m', '', self.format(record))
        _log_buffer.append({"ts": record.created, "msg": msg[-200:]})
_lbh = LogBufHandler()
_lbh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
log.addHandler(_lbh)

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
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
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

    # EMAs
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema50_slope"] = df["ema50"].diff(3) / df["ema50"].shift(3)  # pendiente %

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]      = ema12 - ema26
    df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    df["macd_hist_mean10"] = df["macd_hist"].abs().rolling(10).mean()

    # ATR
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # VWAP (últimas VWAP_PERIODS velas)
    df["vwap"] = (df["close"] * df["volume"]).rolling(VWAP_PERIODS).sum() / \
                  df["volume"].rolling(VWAP_PERIODS).sum()

    # Volume MA
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # Body size
    df["body"] = (df["close"] - df["open"]).abs()

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
    df["supertrend"]     = st
    df["supertrend_dir"] = st_dir

    return df

# ─────────────────────────────────────────
# LAS 3 FAMILIAS
# ─────────────────────────────────────────

def family_trend(df) -> int:
    """
    Tendencia: Supertrend + EMA50 + slope positivo
    Long:  supertrend bullish AND close > ema50 AND slope > 0
    Short: supertrend bearish AND close < ema50 AND slope < 0
    """
    last = df.iloc[-1]
    st_dir = int(last.get("supertrend_dir", 0))
    close  = float(last["close"])
    ema50  = float(last["ema50"])
    slope  = float(last.get("ema50_slope", 0))

    if st_dir == 1 and close > ema50 and slope > 0:
        log.info("  📈 Tendencia: BULLISH (ST bullish + close > EMA50 + slope+)")
        return +1
    if st_dir == -1 and close < ema50 and slope < 0:
        log.info("  📉 Tendencia: BEARISH (ST bearish + close < EMA50 + slope-)")
        return -1
    return 0

def family_momentum(df) -> int:
    """
    Momentum: MACD histogram con umbral mínimo de amplitud
    Long:  hist > 0 AND creciente AND abs(hist) > mean_10 * 0.5
    Short: hist < 0 AND decreciente AND abs(hist) > mean_10 * 0.5
    """
    last = df.iloc[-1]; prev = df.iloc[-2]
    hist      = float(last["macd_hist"])
    hist_prev = float(prev["macd_hist"])
    mean10    = float(last.get("macd_hist_mean10", 1e-9))
    if mean10 == 0: mean10 = 1e-9
    threshold = mean10 * MACD_HIST_THRESHOLD

    if hist > 0 and hist > hist_prev and abs(hist) > threshold:
        log.info(f"  ⚡ Momentum: BULLISH (hist={hist:.6f} > threshold={threshold:.6f})")
        return +1
    if hist < 0 and hist < hist_prev and abs(hist) > threshold:
        log.info(f"  ⚡ Momentum: BEARISH (hist={hist:.6f})")
        return -1
    return 0


def family_volume(df, direction: int, soft: bool = False) -> int:
    """
    Volumen: VWAP + conviction + body filter
    Tier A (strict):
      vol > vol_ma20 * 1.3 AND body < ATR * 1.2
    Tier B (soft):
      vol > vol_ma20 * 1.15 AND body < ATR * 1.5
    """
    last = df.iloc[-1]
    close  = float(last["close"])
    vwap   = float(last.get("vwap", close))
    vol    = float(last["volume"])
    vol_ma = float(last.get("vol_ma20", vol))
    body   = float(last["body"])
    atr    = float(last.get("atr", close * 0.01))

    if pd.isna(vwap) or pd.isna(vol_ma) or vol_ma == 0:
        return 0

    vol_mult = VOL_MULT_SOFT if soft else VOL_MULT
    body_mult = BODY_ATR_MULT_SOFT if soft else BODY_ATR_MULT
    vol_ok  = vol > vol_ma * vol_mult
    body_ok = body < atr * body_mult

    if not vol_ok:
        return 0
    if not body_ok:
        log.info(f"  ⏭️  Volumen: body {body:.4f} > ATR*{body_mult} {atr*body_mult:.4f} — skip")
        return 0

    tier = "soft" if soft else "strict"
    if direction >= 0 and close > vwap:
        log.info(f"  📊 Volumen: BULLISH/{tier} (close {close:.4f} > VWAP {vwap:.4f}, vol {vol/vol_ma:.2f}x)")
        return +1
    if direction <= 0 and close < vwap:
        log.info(f"  📊 Volumen: BEARISH/{tier} (close {close:.4f} < VWAP {vwap:.4f}, vol {vol/vol_ma:.2f}x)")
        return -1
    return 0

def confirmation_signal(exchange, symbol: str, direction: int) -> bool:
    """
    Confirmación rápida en 15m para habilitar entradas Tier B (2/3).
    Long:
      close > EMA21, close > VWAP, macd_hist > 0 y creciendo
    Short:
      close < EMA21, close < VWAP, macd_hist < 0 y decreciendo
    """
    try:
        df = calculate_indicators(get_ohlcv(exchange, symbol, TF_CONFIRM, limit=80))
        last = df.iloc[-1]
        prev = df.iloc[-2]

        close  = float(last["close"])
        ema21  = float(last["ema21"])
        vwap   = float(last.get("vwap", close))
        hist   = float(last["macd_hist"])
        hprev  = float(prev["macd_hist"])

        if direction > 0:
            ok = close > ema21 and close > vwap and hist > 0 and hist > hprev
        else:
            ok = close < ema21 and close < vwap and hist < 0 and hist < hprev

        if ok:
            log.info(f"  ✅ Confirmación {TF_CONFIRM}: {'LONG' if direction > 0 else 'SHORT'}")
        else:
            log.info(f"  ⏭️  Confirmación {TF_CONFIRM}: falló para {'LONG' if direction > 0 else 'SHORT'}")
        return ok
    except Exception as e:
        log.warning(f"  Confirm {symbol} error: {e}")
        return False

def classify_context(ctx: dict, regime: str) -> str:
    """
    bull     = longs habilitados y BTC4h alcista, sin sideways
    neutral  = longs habilitados pero contexto no expansivo
    risk_off = longs bloqueados
    """
    if not ctx.get("long_ok", False):
        return "risk_off"
    if ctx.get("btc4h_bull", False) and regime == "bull":
        return "bull"
    return "neutral"

def ticker_quality(exchange, symbol: str) -> dict:
    try:
        t = exchange.fetch_ticker(symbol)
        qv = float(t.get("quoteVolume") or 0)
        bid = float(t.get("bid") or 0)
        ask = float(t.get("ask") or 0)
        spread = ((ask - bid) / bid) if bid > 0 and ask > 0 else 0
        return {"quote_volume": qv, "spread": spread}
    except Exception:
        return {"quote_volume": 0, "spread": 999}

# ─────────────────────────────────────────
# FILTRO DE CONTEXTO
# ─────────────────────────────────────────
_btc4h_cache = {"long": None, "short": None, "ts": 0}


def context_filter(exchange, regime: str, fg_value: int) -> dict:
    """
    Retorna dict con:
      long_ok:  bool
      short_ok: bool
      sideways: bool
      btc4h_bull: bool
      market_state: bull | neutral | risk_off
    """
    global _btc4h_cache
    now = time.time()
    sideways = regime == "sideways"

    if now - _btc4h_cache["ts"] > 900:
        try:
            df4h = calculate_indicators(get_ohlcv(exchange, "BTC/USDT", "4h", limit=30))
            last = df4h.iloc[-1]
            btc_close = float(last["close"])
            btc_ema21 = float(last["ema21"])
            _btc4h_cache = {
                "long":  btc_close > btc_ema21,
                "short": btc_close < btc_ema21,
                "ts": now
            }
            trend = "↑ BULL" if _btc4h_cache["long"] else "↓ BEAR"
            log.info(f"  🌍 BTC 4h: {trend} ({btc_close:.0f} vs EMA21 {btc_ema21:.0f})")
        except Exception as e:
            log.warning(f"  BTC 4h context error: {e}")

    long_ok = (
        _btc4h_cache.get("long", True) and
        regime != "crash" and
        fg_value > MIN_FG_LONG
    )
    short_ok = (
        _btc4h_cache.get("short", False) or
        regime == "crash"
    )

    ctx = {
        "long_ok": long_ok,
        "short_ok": short_ok,
        "sideways": sideways,
        "btc4h_bull": _btc4h_cache.get("long", True),
    }
    ctx["market_state"] = classify_context(ctx, regime)

    if not long_ok:
        reasons = []
        if not _btc4h_cache.get("long", True): reasons.append("BTC4h bajista")
        if regime == "crash": reasons.append("crash")
        if fg_value <= MIN_FG_LONG: reasons.append(f"F&G={fg_value}≤{MIN_FG_LONG}")
        log.info(f"  🚫 Contexto: LONG bloqueado ({', '.join(reasons)})")
    else:
        log.info(f"  🧭 Contexto v3: {ctx['market_state'].upper()}")

    return ctx

# ─────────────────────────────────────────
# COOLDOWN TRACKER
# ─────────────────────────────────────────
_cooldown: dict[str, int] = {}   # symbol → ciclos restantes

def is_in_cooldown(symbol: str) -> bool:
    remaining = _cooldown.get(symbol, 0)
    if remaining > 0:
        log.info(f"  ⏳ Cooldown {symbol}: {remaining} ciclos restantes")
        return True
    return False

def set_cooldown(symbol: str):
    _cooldown[symbol] = COOLDOWN_CANDLES

def tick_cooldowns():
    for sym in list(_cooldown.keys()):
        _cooldown[sym] -= 1
        if _cooldown[sym] <= 0:
            del _cooldown[sym]

# ─────────────────────────────────────────
# SIZING
# ─────────────────────────────────────────

def get_position_size(capital: float, symbol: str, regime: str, score: int, tier: str = "A", context_state: str = "neutral") -> float:
    """
    Tier A:
      3/3 + bull major -> 2.5%
      3/3 resto        -> 2.0%
      sideways         -> 1.5%
    Tier B:
      2/3 + confirm    -> 1.0%
      sideways         -> 0.75%
    """
    is_major = symbol in MAJORS

    if tier == "B":
        pct = 0.010
        if regime == "sideways":
            pct = 0.0075
    else:
        if regime == "sideways":
            pct = SIZE_SIDEWAYS
        elif is_major and context_state == "bull":
            pct = SIZE_MAJOR_BULL
        else:
            pct = SIZE_BASE

    if regime == "sideways" and not is_major:
        pct *= SIZE_ALT_SIDEWAYS_MULT

    usd_size = round(capital * pct, 2)
    max_size = round(capital * MAX_CAPITAL_EXPOSURE, 2)
    return min(usd_size, max_size)

# ─────────────────────────────────────────
# MERCADO / RÉGIMEN
# ─────────────────────────────────────────
_regime     = "unknown"
_fear_greed = {"value": 50, "label": "Neutral"}

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

def get_fear_greed() -> tuple[int, str]:
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except: return 50, "Neutral"

# ─────────────────────────────────────────
# GESTIÓN DE POSICIONES
# ─────────────────────────────────────────
def open_position(symbol, entry_price, usd_size, action, atr_value):
    positions = load_positions()
    is_short  = action == "SELL"

    trail_pct = max((atr_value * ATR_MULT / entry_price) if atr_value else TRAILING_STOP_PCT,
                     TRAILING_STOP_PCT)
    trail_pct = min(trail_pct, 0.03)  # cap 3%
    tp_pct    = TAKE_PROFIT_PCT

    if is_short:
        trail_stop  = round(entry_price * (1 + trail_pct), 6)
        take_profit = round(entry_price * (1 - tp_pct), 6)
        partial_tp  = round(entry_price * (1 - TAKE_PROFIT_PARTIAL), 6)
    else:
        trail_stop  = round(entry_price * (1 - trail_pct), 6)
        take_profit = round(entry_price * (1 + tp_pct), 6)
        partial_tp  = round(entry_price * (1 + TAKE_PROFIT_PARTIAL), 6)

    positions[symbol] = {
        "symbol": symbol, "action": action,
        "entry_price": entry_price, "current_price": entry_price,
        "high_price": entry_price, "trail_stop": trail_stop,
        "take_profit": take_profit, "partial_tp": partial_tp,
        "partial_closed": False, "usd_size": usd_size,
        "trail_pct": trail_pct, "opened_at": datetime.now().isoformat(),
    }
    save_positions(positions)
    log.info(f"  ✅ POSICIÓN ABIERTA {symbol} {action} @ {entry_price} | size=${usd_size} | trail={trail_pct*100:.2f}%")

def update_trailing_stops(exchange, state):
    positions = load_positions()
    if not positions: return
    for symbol, pos in list(positions.items()):
        try:
            price = float(exchange.fetch_ticker(symbol)["last"])
            pos["current_price"] = price
            is_short = pos["action"] == "SELL"
            trail_pct = pos.get("trail_pct", TRAILING_STOP_PCT)

            if is_short:
                if price < pos["high_price"]:
                    pos["high_price"] = price
                    pos["trail_stop"] = round(price * (1 + trail_pct), 6)
            else:
                if price > pos["high_price"]:
                    pos["high_price"] = price
                    pos["trail_stop"] = round(price * (1 - trail_pct), 6)
                # Subir TP si va bien
                unreal = (price - pos["entry_price"]) / pos["entry_price"]
                if unreal > 0.015:
                    new_tp = price * (1 + 0.01)
                    if new_tp > pos["take_profit"]:
                        pos["take_profit"] = round(new_tp, 6)
                        log.info(f"  🎯 TP subido {symbol}: {pos['take_profit']} (+{unreal*100:.1f}%)")

            log.info(f"  📈 Trail {symbol}: {pos['trail_stop']}")

            # ── Partial TP ──────────────────────────────────────────────
            if not pos["partial_closed"]:
                hit_partial = (
                    (not is_short and price >= pos["partial_tp"]) or
                    (is_short     and price <= pos["partial_tp"])
                )
                if hit_partial:
                    pos["partial_closed"] = True
                    pos["usd_size"] = round(pos["usd_size"] * (1 - PARTIAL_EXIT_PCT), 2)
                    log.info(f"  ✂️  Partial TP {symbol} @ {price}")

            # ── Salida completa ─────────────────────────────────────────
            exit_reason = None
            if not is_short:
                if price <= pos["trail_stop"]:   exit_reason = "trail"
                elif price >= pos["take_profit"]: exit_reason = "take_profit"
                elif price <= pos["entry_price"] * (1 - 0.025): exit_reason = "stop_loss"
            else:
                if price >= pos["trail_stop"]:   exit_reason = "trail"
                elif price <= pos["take_profit"]: exit_reason = "take_profit"
                elif price >= pos["entry_price"] * (1 + 0.025): exit_reason = "stop_loss"

            if exit_reason:
                pnl = ((price - pos["entry_price"]) / pos["entry_price"] * 100
                       if not is_short else
                       (pos["entry_price"] - price) / pos["entry_price"] * 100)
                pnl = round(pnl, 3)
                usd_pnl = round(pnl * pos["usd_size"] / 100, 2)
                state["capital"] = round(state.get("capital", CAPITAL_TOTAL_USD) + usd_pnl, 2)
                save_state(state)

                emoji = "🟢" if pnl > 0 else "🔴"
                log.info(f"  {emoji} CERRADO {symbol} @ {price} | PnL: {pnl:+.2f}% ({'+' if usd_pnl>=0 else ''}{usd_pnl}) | {exit_reason}")
                send_telegram(
                    f"{emoji} <b>v3 {symbol}</b> {exit_reason.upper()}\n"
                    f"Entrada: {pos['entry_price']} → Salida: {price}\n"
                    f"PnL: {pnl:+.2f}% | ${usd_pnl:+.2f}\n"
                    f"Capital: ${state['capital']:.2f}"
                )
                save_trade({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol, "action": pos["action"],
                    "entry": pos["entry_price"], "exit": price,
                    "pnl_pct": pnl, "usd_pnl": usd_pnl,
                    "reason": exit_reason, "usd_size": pos["usd_size"],
                    "partial": pos["partial_closed"],
                })
                set_cooldown(symbol)
                del positions[symbol]
                save_positions(positions)
        except Exception as e:
            log.error(f"  Trail error {symbol}: {e}")

# ─────────────────────────────────────────
# MOTOR PRINCIPAL DE ANÁLISIS
# ─────────────────────────────────────────

def analyze_symbol(symbol, exchange, regime, fg_value, state, ctx) -> bool:
    """
    v3:
      Tier A = 3/3
      Tier B = 2/3 + confirmación 15m + liquidez suficiente
    """
    positions = load_positions()
    if symbol in positions:
        log.info(f"  {symbol}: posición ya abierta — skip")
        return False
    if is_in_cooldown(symbol):
        return False

    is_major = symbol in MAJORS
    context_state = ctx.get("market_state", "neutral")

    if not ctx["long_ok"] and not ctx["short_ok"]:
        return False
    if not is_major and not ctx["long_ok"]:
        return False

    quality = ticker_quality(exchange, symbol)
    if quality["quote_volume"] < MIN_VOLUME_24H:
        log.info(f"  ⏭️  Liquidez insuficiente ({quality['quote_volume']:.0f}) — skip")
        return False
    if quality["spread"] > 0.004:
        log.info(f"  ⏭️  Spread alto ({quality['spread']*100:.2f}%) — skip")
        return False

    try:
        df = calculate_indicators(get_ohlcv(exchange, symbol, TF_SETUP, limit=100))
    except Exception as e:
        log.warning(f"  OHLCV error {symbol}: {e}")
        return False

    # Familias strict (Tier A)
    t_sig = family_trend(df)
    m_sig = family_momentum(df)
    v_sig_long = family_volume(df, direction=+1, soft=False)
    v_sig_short = family_volume(df, direction=-1, soft=False)

    score_long = sum(1 for s in [t_sig, m_sig, v_sig_long] if s == +1) if ctx["long_ok"] else 0
    score_short = sum(1 for s in [t_sig, m_sig, v_sig_short] if s == -1) if (ctx["short_ok"] and is_major) else 0

    # Tier B: volumen soft + confirmación
    v_sig_long_soft = family_volume(df, direction=+1, soft=True)
    v_sig_short_soft = family_volume(df, direction=-1, soft=True)
    score_long_soft = sum(1 for s in [t_sig, m_sig, v_sig_long_soft] if s == +1) if ctx["long_ok"] else 0
    score_short_soft = sum(1 for s in [t_sig, m_sig, v_sig_short_soft] if s == -1) if (ctx["short_ok"] and is_major) else 0

    log.info(f"  [{TF_SETUP}] TierA LONG={score_long}/3 SHORT={score_short}/3 | TierB LONG={score_long_soft}/3 SHORT={score_short_soft}/3")

    action = None
    score = 0
    tier = None

    # Tier A first
    if score_long == 3 and ctx["long_ok"]:
        action, score, tier = "BUY", 3, "A"
    elif score_short == 3 and ctx["short_ok"] and is_major:
        action, score, tier = "SELL", 3, "A"

    # Tier B fallback: 2/3 + confirmación
    if not action:
        if ctx["long_ok"] and score_long_soft >= 2:
            allow_tierb = is_major or (context_state == "bull" and quality["quote_volume"] >= MIN_VOLUME_24H_TIERB)
            if allow_tierb and confirmation_signal(exchange, symbol, +1):
                action, score, tier = "BUY", 2, "B"

        if (not action) and ctx["short_ok"] and is_major and score_short_soft >= 2:
            if confirmation_signal(exchange, symbol, -1):
                action, score, tier = "SELL", 2, "B"

    # Altcoins en sideways: bloquear Tier B
    if action and (not is_major) and regime == "sideways" and tier == "B":
        log.info("  ⏭️  Sideways + altcoin: Tier B bloqueado")
        action = None

    if not action:
        log.info("  ⏭️  Score insuficiente / confirmación fallida — skip")
        return False

    open_pos = load_positions()
    allocated = sum(p.get("usd_size", 0) for p in open_pos.values())
    capital = state.get("capital", CAPITAL_TOTAL_USD)
    if allocated >= capital * MAX_CAPITAL_EXPOSURE:
        log.info(f"  ⏭️  Exposición máxima alcanzada (${allocated:.0f} / ${capital*MAX_CAPITAL_EXPOSURE:.0f})")
        return False

    open_alts = [s for s in open_pos if s not in MAJORS]
    if symbol not in MAJORS and len(open_alts) >= MAX_ALTS_OPEN:
        log.info(f"  ⏭️  Correlación: {len(open_alts)} altcoins abiertas — skip")
        return False

    usd_size = get_position_size(capital, symbol, regime, score, tier=tier, context_state=context_state)
    if usd_size < 5:
        log.info(f"  ⏭️  Size demasiado pequeño (${usd_size})")
        return False

    last = df.iloc[-1]
    price = float(last["close"])
    atr = float(last["atr"]) if not pd.isna(last["atr"]) else None

    log.info(f"  🟢 SEÑAL {action} {symbol} @ {price:.6f} | size=${usd_size} | score={score}/3 | tier={tier} | ctx={context_state}")

    if PAPER_TRADING:
        open_position(symbol, price, usd_size, action, atr)
        send_telegram(
            f"📝 <b>v3 PAPER {action} {symbol}</b>\n"
            f"Tier: {tier} | Score: {score}/3 | Régimen: {regime.upper()}\n"
            f"Precio: {price:.6f} | Size: ${usd_size}\n"
            f"F&G: {fg_value} | Contexto: {context_state}"
        )
        return True
    return False

# ─────────────────────────────────────────
# SCANNER DE ALTCOINS
# ─────────────────────────────────────────
_altcoins = []
_last_scan = 0

def scan_altcoins(exchange) -> list:
    global _altcoins, _last_scan
    now = time.time()
    if now - _last_scan < 600 and _altcoins:
        return _altcoins
    try:
        tickers = exchange.fetch_tickers()
        pairs   = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT"): continue
            base = sym.replace("/USDT", "")
            if base in EXCLUDE_SYMBOLS: continue
            if not base.isascii() or len(base) > 10: continue
            if sym in MAJORS: continue
            vol = t.get("quoteVolume") or 0
            if vol < MIN_VOLUME_24H: continue
            pairs.append(sym)
        pairs.sort()
        _altcoins = pairs[:15]
        _last_scan = now
        log.info(f"🔍 v3 Altcoins: {_altcoins}")
    except Exception as e:
        log.error(f"Scanner error: {e}")
    return _altcoins

# ─────────────────────────────────────────
# CIRCUIT BREAKER DIARIO
# ─────────────────────────────────────────
def check_daily_circuit(state) -> bool:
    """Retorna True si el bot puede operar (no activado)."""
    today = datetime.now(ARG_TZ).strftime("%Y-%m-%d")
    if state.get("day_start_date") != today:
        state["day_start_capital"] = state.get("capital", CAPITAL_TOTAL_USD)
        state["day_start_date"]    = today
        state["daily_circuit"]     = False

    day_start = state.get("day_start_capital", state.get("capital", CAPITAL_TOTAL_USD))
    current   = state.get("capital", CAPITAL_TOTAL_USD)
    daily_loss = (day_start - current) / day_start if day_start > 0 else 0

    if daily_loss > 0.03 and not state.get("daily_circuit"):
        state["daily_circuit"] = True
        save_state(state)
        send_telegram(f"🛑 <b>v3 Circuit Breaker</b>\nPérdida diaria: {daily_loss*100:.1f}%\nSin entradas hasta mañana")
        log.info(f"🛑 v3 Circuit breaker — pérdida diaria {daily_loss*100:.1f}%")

    return not state.get("daily_circuit", False)

# ─────────────────────────────────────────
# DASHBOARD v2
# ─────────────────────────────────────────
DASHBOARD_V2 = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CryptoBot v3</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#04080f;--s1:#0b1220;--s2:#101828;--border:#162030;--green:#0dffb0;--red:#ff3366;--yellow:#ffd700;--blue:#38bdf8;--orange:#fb923c;--text:#a8bfd4;--text2:#6b8299;--white:#e2f0ff;--mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;min-height:100vh;padding:20px 24px}
h1{font-family:var(--sans);font-size:18px;color:var(--white);margin-bottom:4px}
h1 span{color:var(--green)}
.sub{font-size:10px;color:var(--text2);margin-bottom:20px}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:20px}
.kpi{background:var(--s1);border:1px solid var(--border);border-radius:6px;padding:12px;position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--k,var(--green))}
.kl{font-size:9px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:4px}
.kv{font-family:var(--sans);font-weight:700;font-size:20px;color:var(--white)}
.kv.g{color:var(--green)}.kv.r{color:var(--red)}.kv.y{color:var(--yellow)}.kv.b{color:var(--blue)}.kv.o{color:var(--orange)}
.ks{font-size:10px;color:var(--text2);margin-top:2px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.panel{background:var(--s1);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-bottom:12px}
.ph{display:flex;justify-content:space-between;align-items:center;padding:8px 14px;background:var(--s2);border-bottom:1px solid var(--border)}
.pt{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--text2);font-weight:600}
.pb{padding:10px 14px}
/* Families */
.fam-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:10px 14px}
.fam{border-radius:5px;padding:10px;border:1px solid var(--border);text-align:center}
.fam.ok{border-color:rgba(13,255,176,.3);background:rgba(13,255,176,.05)}
.fam.fail{border-color:rgba(255,51,102,.2);background:rgba(255,51,102,.03)}
.fam.neutral{background:var(--s2)}
.fn{font-size:9px;color:var(--text2);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}
.fi{font-size:22px}
.fv{font-size:10px;margin-top:2px}
/* Positions */
.pos-list{display:flex;flex-direction:column;gap:6px;padding:10px}
.pos{border-radius:5px;padding:10px 12px;border:1px solid var(--border);background:var(--s2)}
.pos.long{border-left:2px solid var(--green)}.pos.short{border-left:2px solid var(--red)}
.pt2{display:flex;justify-content:space-between;margin-bottom:5px}
.psym{font-family:var(--sans);font-weight:700;font-size:13px;color:var(--white)}
.ppnl{font-family:var(--sans);font-weight:700;font-size:13px}
.ppnl.pos{color:var(--green)}.ppnl.neg{color:var(--red)}
.pmeta{display:grid;grid-template-columns:1fr 1fr;gap:2px;font-size:10px}
.pl{color:var(--text2)}.pv{color:var(--text);text-align:right}
/* Table */
.tbl{width:100%;border-collapse:collapse;font-size:11px}
.tbl th{padding:6px 10px;font-size:8px;letter-spacing:.1em;text-transform:uppercase;color:var(--text2);font-weight:500;text-align:left;background:var(--s2)}
.tbl td{padding:6px 10px;border-bottom:1px solid rgba(22,32,48,.6)}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:rgba(255,255,255,.01)}
/* Log */
#live-log{padding:8px 14px;font-size:10px;max-height:180px;overflow-y:auto;display:flex;flex-direction:column;gap:2px}
.dot{width:5px;height:5px;border-radius:50%;background:currentColor;animation:blink 2s infinite;display:inline-block}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.pill{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:100px;font-size:10px;border:1px solid currentColor}
.pill.g{color:var(--green)}.pill.b{color:var(--blue)}.pill.p{color:#c084fc}
footer{text-align:center;font-size:9px;color:var(--text2);margin-top:16px;padding-top:12px;border-top:1px solid var(--border)}
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
  <h1>Crypto<span>Bot</span> <small style="font-size:12px;color:var(--text2)">v3 Tiered Entries</small></h1>
  <div style="display:flex;gap:6px">
    <span class="pill g"><span class="dot"></span> LIVE</span>
    <span class="pill b" id="mode-pill">PAPER</span>
    <span style="font-size:10px;color:var(--text2);padding:2px 8px;background:var(--s1);border:1px solid var(--border);border-radius:3px">↻ <span id="cd">15</span>s</span>
  </div>
</div>
<div class="sub">3 familias · Tier A/B · Confirmación 15m · Contexto como filtro</div>

<div class="kpis">
  <div class="kpi" style="--k:var(--green)"><div class="kl">Capital</div><div class="kv g" id="k-cap">—</div><div class="ks" id="k-cap-d">—</div></div>
  <div class="kpi" style="--k:var(--yellow)"><div class="kl">P&L Total</div><div class="kv y" id="k-pnl">—</div><div class="ks">trades cerrados</div></div>
  <div class="kpi" style="--k:var(--blue)"><div class="kl">Win Rate</div><div class="kv b" id="k-wr">—</div><div class="ks" id="k-wr-d">—</div></div>
  <div class="kpi" style="--k:var(--orange)"><div class="kl">Trades</div><div class="kv o" id="k-trades">—</div><div class="ks" id="k-trades-d">—</div></div>
  <div class="kpi" style="--k:var(--red)"><div class="kl">F&G</div><div class="kv" id="k-fg">—</div><div class="ks" id="k-fg-l">—</div></div>
  <div class="kpi" style="--k:var(--blue)"><div class="kl">Régimen</div><div class="kv b" id="k-reg">—</div><div class="ks" id="k-reg-s">—</div></div>
</div>

<div class="grid2">
  <div>
    <div class="panel">
      <div class="ph"><span class="pt">Contexto de mercado</span><span id="ctx-status" style="font-size:10px"></span></div>
      <div style="padding:10px 14px;display:grid;grid-template-columns:1fr 1fr;gap:6px" id="ctx-grid"></div>
    </div>
    <div class="panel">
      <div class="ph"><span class="pt">Performance por familia</span></div>
      <div class="fam-grid" id="fam-grid"></div>
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
      <div class="ph"><span class="pt">Scanner</span></div>
      <div id="scanner-row" style="padding:8px 14px;display:flex;flex-wrap:wrap;gap:4px"></div>
    </div>
  </div>
</div>

<div class="panel">
  <div class="ph"><span class="pt">Historial de trades v2</span></div>
  <div style="overflow:hidden">
    <table class="tbl">
      <thead><tr><th>Par</th><th>Acción</th><th>Entrada</th><th>Salida</th><th>P&L</th><th>Motivo</th><th>Size</th><th>Hora</th></tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<footer>CryptoBot v3 · 3 Familias · Contexto filtro · Tier A/B · Confirmación 15m</footer>

<script>
let cd=15;
const ICAP=1000;
function fmt(n,d=2){return(+n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d})}

async function load(){
  try{
    const d=await fetch('/v2/api').then(r=>r.json());
    const st=d.state||{};
    const trades=d.trades||[];
    const closed=trades.filter(t=>t.pnl_pct!==undefined);
    const wins=closed.filter(t=>t.pnl_pct>0);
    const cap=st.capital||ICAP;

    document.getElementById('k-cap').textContent='$'+fmt(cap);
    document.getElementById('k-cap').className='kv '+(cap>=ICAP?'g':'r');
    document.getElementById('k-cap-d').textContent=(cap-ICAP>=0?'+':'')+fmt(cap-ICAP)+' desde inicio';

    const tpnl=closed.reduce((s,t)=>s+(t.pnl_pct||0)*(t.usd_size||20)/100,0);
    document.getElementById('k-pnl').textContent=(tpnl>=0?'+':'')+'$'+fmt(tpnl);
    document.getElementById('k-pnl').className='kv '+(tpnl>=0?'g':'r');

    const wr=closed.length?Math.round(wins.length/closed.length*100):null;
    document.getElementById('k-wr').textContent=wr!==null?wr+'%':'—';
    document.getElementById('k-wr-d').textContent=closed.length?wins.length+'/'+closed.length:'sin datos';

    document.getElementById('k-trades').textContent=trades.length;
    document.getElementById('k-trades-d').textContent=closed.length+' cerrados';

    if(d.fear_greed){const fg=d.fear_greed,fv=+fg.value;document.getElementById('k-fg').textContent=fv;document.getElementById('k-fg').className='kv '+(fv<35?'r':fv>65?'g':'y');document.getElementById('k-fg-l').textContent=fg.label;}
    const rm={bull:'BULL 📈',bear:'BEAR 📉',sideways:'SIDE ↔',crash:'CRASH 💥'};
    const reg=d.regime||'?';
    document.getElementById('k-reg').textContent=rm[reg]||reg.toUpperCase();
    document.getElementById('k-reg-s').textContent=reg;

    // Contexto
    const ctx=d.context||{};
    const ctxDefs=[
      {k:'long_ok',  label:'Long permitido', ok:ctx.long_ok},
      {k:'short_ok', label:'Short permitido', ok:ctx.short_ok},
      {k:'btc4h',    label:'BTC 4h alcista',  ok:ctx.btc4h_bull},
      {k:'circuit',  label:'Circuit breaker', ok:!st.daily_circuit, inv:true},
    ];
    document.getElementById('ctx-grid').innerHTML=ctxDefs.map(c=>{
      const ok=c.ok; const color=ok?'var(--green)':'var(--red)';
      return '<div style="background:var(--s2);border:1px solid var(--border);border-radius:4px;padding:6px 8px"><div style="font-size:9px;color:var(--text2)">'+c.label+'</div><div style="font-size:12px;color:'+color+';font-weight:700;margin-top:2px">'+(ok?'✅ OK':'🔴 OFF')+'</div></div>';
    }).join('');
    document.getElementById('ctx-status').textContent=ctx.long_ok?'🟢 Longs habilitados':'🔴 Longs bloqueados';

    // Familia performance
    const fams=[
      {k:'tendencia', n:'Tendencia', icon:'📈'},
      {k:'momentum',  n:'Momentum',  icon:'⚡'},
      {k:'volumen',   n:'Volumen',   icon:'📊'},
    ];
    const stats=d.family_stats||{};
    document.getElementById('fam-grid').innerHTML=fams.map(f=>{
      const s=stats[f.k]||{wins:0,total:0};
      const wr=s.total?Math.round(s.wins/s.total*100):null;
      const cls=wr===null?'neutral':wr>=55?'ok':'fail';
      const color=wr===null?'var(--text2)':wr>=55?'var(--green)':'var(--red)';
      return '<div class="fam '+cls+'"><div class="fn">'+f.n+'</div><div class="fi">'+f.icon+'</div><div class="fv" style="color:'+color+'">'+(wr!==null?wr+'% WR ('+s.total+')'  :'sin datos')+'</div></div>';
    }).join('');

    // Posiciones
    const pos=d.positions||[];
    document.getElementById('pos-count').textContent=pos.length||'ninguna';
    document.getElementById('pos-list').innerHTML=pos.length?pos.map(p=>{
      const isS=p.action==='SELL';
      const pnl=isS?((p.entry_price-p.current_price)/p.entry_price*100):((p.current_price-p.entry_price)/p.entry_price*100);
      const isp=pnl>=0;
      const oa=p.opened_at?new Date(p.opened_at):null;
      const el=oa?Math.round((Date.now()-oa)/60000):null;
      return '<div class="pos '+(isS?'short':'long')+'"><div class="pt2"><span class="psym">'+p.symbol.replace('/USDT','')+'</span><span class="ppnl '+(isp?'pos':'neg')+'">'+(isp?'+':'')+pnl.toFixed(2)+'%</span></div><div class="pmeta"><span class="pl">Entrada</span><span class="pv">'+p.entry_price+'</span><span class="pl">Trail</span><span class="pv" style="color:var(--orange)">'+p.trail_stop+'</span><span class="pl">TP</span><span class="pv" style="color:var(--green)">'+p.take_profit+'</span><span class="pl">Size</span><span class="pv">$'+p.usd_size+'</span></div>'+(el!==null?'<div style="font-size:9px;color:var(--text2);margin-top:4px">Hace '+(el<60?el+'m':(el/60).toFixed(1)+'h')+'</div>':'')+'</div>';
    }).join(''):'<div style="padding:20px;text-align:center;color:var(--text2);font-size:11px">Sin posiciones abiertas</div>';

    // Scanner
    const sc=d.scanner||[];
    document.getElementById('scanner-row').innerHTML=sc.map(s=>'<span style="padding:2px 7px;border-radius:3px;font-size:10px;border:1px solid var(--border);color:var(--text2)">'+s.replace('/USDT','')+'</span>').join('');

    // Live log
    try{
      const logs=await fetch('/v2/log').then(r=>r.json());
      const el=document.getElementById('live-log');
      el.innerHTML=logs.slice(-15).map(l=>{
        const m=l.msg||'';
        const c=m.includes('ERROR')?'var(--red)':m.includes('CERRADO')||m.includes('✅')?'var(--green)':m.includes('⏭️')?'var(--text2)':m.includes('⏳')||m.includes('trail')?'var(--yellow)':'var(--text)';
        return '<div style="color:'+c+';line-height:1.4">'+m+'</div>';
      }).join('');
      el.scrollTop=el.scrollHeight;
    }catch(e){}

    // Tabla
    const tb=document.getElementById('tbody');
    if(!closed.length){tb.innerHTML='<tr><td colspan="8" style="text-align:center;padding:30px;color:var(--text2)">Sin trades aún</td></tr>';return;}
    tb.innerHTML=[...closed].reverse().map(t=>{
      const ts=new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const pc=t.pnl_pct>=0?'var(--green)':'var(--red)';
      return '<tr><td style="color:var(--white);font-weight:600">'+t.symbol+'</td><td><span style="padding:1px 5px;border-radius:2px;font-size:10px;background:'+(t.action==='BUY'?'rgba(13,255,176,.1)':'rgba(255,51,102,.1)')+';color:'+(t.action==='BUY'?'var(--green)':'var(--red)')+'">'+t.action+'</span></td><td>'+t.entry+'</td><td>'+t.exit+'</td><td style="color:'+pc+';font-weight:700">'+(t.pnl_pct>=0?'+':'')+t.pnl_pct+'%</td><td style="color:var(--text2)">'+t.reason+'</td><td style="color:var(--orange)">$'+t.usd_size+'</td><td style="color:var(--text2);font-size:10px">'+ts+'</td></tr>';
    }).join('');
  }catch(e){console.error(e);}
}
function tick(){cd--;document.getElementById('cd').textContent=cd;if(cd<=0){cd=15;load();}}
load();setInterval(tick,1000);
</script>
</body>
</html>"""

# ─────────────────────────────────────────
# FLASK API
# ─────────────────────────────────────────
flask_v3   = Flask("v2")
_v2_regime = "unknown"
_v2_fg     = {"value": 50, "label": "Neutral"}
_v2_scanner = []
_v2_ctx    = {"long_ok": True, "short_ok": False, "btc4h_bull": True, "market_state": "neutral"}
_v2_fam_stats = {"tendencia": {"wins":0,"total":0}, "momentum": {"wins":0,"total":0}, "volumen": {"wins":0,"total":0}}

@flask_v2.route("/")
def v2_index():
    return render_template_string(DASHBOARD_V2)

@flask_v2.route("/v2/api")
@flask_v2.route("/api/trades")   # compatibilidad con monitor externo
def v2_api():
    trades    = load_trades()
    positions = load_positions()
    state     = load_state()
    return jsonify({
        "trades": trades, "positions": list(positions.values()),
        "state": state, "fear_greed": _v2_fg, "regime": _v2_regime,
        "scanner": _v2_scanner, "context": _v2_ctx,
        "family_stats": _v2_fam_stats,
    })

@flask_v2.route("/api/log")
def v2_log_compat():
    return jsonify(list(_log_buffer))



@flask_v2.route("/health")
def v2_health():
    return jsonify({"status": "ok", "version": "v3", "paper": PAPER_TRADING})

def run_dashboard():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"🌐 v3 Dashboard en puerto {port}")
    flask_v2.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run_bot():
    global _v2_regime, _v2_fg, _v2_scanner, _v2_ctx

    log.info("🤖 CryptoBot v3 — Tiered Entries Edition")
    log.info(f"Mode: {'PAPER' if PAPER_TRADING else 'REAL'} | Capital: ${CAPITAL_TOTAL_USD}")
    send_telegram(
        f"🤖 <b>CryptoBot v3 — Tiered Entries</b>\n"
        f"Mode: {'📝 PAPER' if PAPER_TRADING else '💰 REAL'}\n"
        f"3 familias: Tendencia + Momentum + Volumen\n"
        f"Tier A/B | Confirmación 15m | Contexto como filtro\n"
        f"Shorts solo en majors | Cooldown {COOLDOWN_CANDLES} ciclos"
    )

    exchange     = get_exchange()
    pub_exchange = get_public_exchange()
    state        = load_state()
    last_regime  = 0

    while True:
        now = time.time()
        log.info(f"\n{'='*50}\n⏰ v3 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        state = load_state()

        # Régimen cada 15 min
        if now - last_regime > 900:
            _v2_regime = detect_regime(pub_exchange)
            last_regime = now
            log.info(f"🧭 Régimen: {_v2_regime.upper()}")

        # Fear & Greed
        _v2_fg_val, _v2_fg_label = get_fear_greed()
        _v2_fg = {"value": _v2_fg_val, "label": _v2_fg_label}
        log.info(f"😱 F&G: {_v2_fg_val} — {_v2_fg_label}")

        # Circuit breaker
        if not check_daily_circuit(state):
            log.info("🛑 v3 Circuit breaker activo — skip entradas")
            update_trailing_stops(pub_exchange, state)
            tick_cooldowns()
            log.info(f"💤 {LOOP_INTERVAL_SEC}s...")
            time.sleep(LOOP_INTERVAL_SEC)
            continue

        # Trailing stops
        update_trailing_stops(pub_exchange, state)
        tick_cooldowns()

        # Contexto
        ctx = context_filter(pub_exchange, _v2_regime, _v2_fg_val)
        _v2_ctx = {**ctx, "btc4h_bull": _btc4h_cache.get("long", True)}

        open_positions = load_positions()
        log.info(f"📂 v3 Posiciones: {list(open_positions.keys()) or 'ninguna'} | Capital: ${state.get('capital',CAPITAL_TOTAL_USD):.2f}")

        # Analizar majors
        log.info(f"\n--- MAJORS [{TF_SETUP}] ---")
        for symbol in list(MAJORS):
            try:
                log.info(f"\n📊 {symbol}...")
                analyze_symbol(symbol, pub_exchange, _v2_regime, _v2_fg_val, state, ctx)
                open_positions = load_positions()
            except Exception as e:
                log.error(f"Error {symbol}: {e}")

        # Altcoins (solo si contexto long OK)
        if ctx["long_ok"]:
            alts = scan_altcoins(pub_exchange)
            _v2_scanner = alts
            log.info(f"\n--- ALTCOINS [{TF_SETUP}] ---")
            for symbol in alts:
                try:
                    log.info(f"\n📊 {symbol}...")
                    analyze_symbol(symbol, pub_exchange, _v2_regime, _v2_fg_val, state, ctx)
                    open_positions = load_positions()
                except Exception as e:
                    log.error(f"Error {symbol}: {e}")
        else:
            log.info("⏭️  Altcoins: contexto long bloqueado — skip")

        log.info(f"\n💤 {LOOP_INTERVAL_SEC}s...")
        time.sleep(LOOP_INTERVAL_SEC)

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_dashboard, daemon=True).start()
    run_bot()
