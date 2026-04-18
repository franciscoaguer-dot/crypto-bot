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
MAX_ALTS_OPEN     = 5

# Risk management
STOP_LOSS_PCT     = 0.025
BINANCE_FEE_RT    = 0.001   # 0.1% entrada + 0.1% salida = 0.2% round-trip (spot)
                             # Con BNB: 0.075% × 2 = 0.15% — conservador usar 0.2%
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
MIN_VOL_24H       = 10_000_000  # bajado de 20M → más pares califican

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

def _pg_reconnect():
    """Reconecta a Postgres si la conexión se perdió (deploy, timeout, etc.)"""
    global _pg, _USE_PG
    if not DATABASE_URL: return False
    try:
        import psycopg2
        _pg = psycopg2.connect(DATABASE_URL, sslmode="require")
        _pg.autocommit = True
        _USE_PG = True
        log.info("✅ Postgres reconectado")
        return True
    except Exception as e:
        log.warning(f"PG reconnect failed: {e}")
        _USE_PG = False
        return False

def _pg_get(key, default=None):
    for attempt in range(2):
        try:
            with _pg.cursor() as cur:
                cur.execute("SELECT value FROM v3_kv WHERE key=%s", (key,))
                row = cur.fetchone()
                return json.loads(row[0]) if row else default
        except Exception as e:
            if attempt == 0:
                log.warning(f"PG get error, reconectando: {e}")
                _pg_reconnect()
            else:
                log.error(f"PG get falló: {e}")
    return default

def _pg_set(key, value):
    for attempt in range(2):
        try:
            with _pg.cursor() as cur:
                cur.execute("""
                    INSERT INTO v3_kv (key, value, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()
                """, (key, json.dumps(value, default=str)))
            return
        except Exception as e:
            if attempt == 0:
                log.warning(f"PG set error, reconectando: {e}")
                _pg_reconnect()
            else:
                log.error(f"PG set falló: {e}")

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
        for attempt in range(2):
            try:
                with _pg.cursor() as cur:
                    cur.execute("INSERT INTO v3_trades (data) VALUES (%s)",
                                (json.dumps(r, default=str),))
                return
            except Exception as e:
                if attempt == 0:
                    log.warning(f"PG save_trade error, reconectando: {e}")
                    _pg_reconnect()
                else:
                    log.error(f"PG save_trade falló: {e}")
    data = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: data = json.load(f)
        except: pass
    data.append(r)
    with open(TRADE_LOG_FILE, "w") as f: json.dump(data, f, indent=2, default=str)

def load_trades():
    if _USE_PG:
        for attempt in range(2):
            try:
                with _pg.cursor() as cur:
                    cur.execute("SELECT data FROM v3_trades ORDER BY created_at ASC")
                    return [row[0] for row in cur.fetchall()]
            except Exception as e:
                if attempt == 0:
                    log.warning(f"PG load_trades error, reconectando: {e}")
                    _pg_reconnect()
                else:
                    log.error(f"PG load_trades falló: {e}")
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
    """
    Momentum: MACD histogram con dos niveles de sensibilidad.

    +1 (fuerte):   hist > 0 AND creciente AND > threshold
    +1 (suave):    hist negativo pero creciendo claramente (> threshold)
                   → mercado aún bajista pero momentum girando alcista
    -1 (fuerte):   hist < 0 AND decreciente AND > threshold
    -1 (suave):    hist positivo pero cayendo claramente
     0:            sin señal clara
    """
    last = df.iloc[-1]; prev = df.iloc[-2]
    hist      = float(last["macd_hist"])
    hist_prev = float(prev["macd_hist"])
    mean10    = float(last.get("macd_hist_mean10", 1e-9)) or 1e-9
    threshold = mean10 * MACD_THRESHOLD
    # Threshold suave: 30% del threshold normal para capturar giros tempranos
    threshold_soft = mean10 * MACD_THRESHOLD * 0.3

    # ── LONG momentum ───────────────────────────────────────────────────────
    # Fuerte: hist claramente positivo y creciendo
    if hist > 0 and hist > hist_prev and abs(hist) > threshold:
        return +1
    # Suave: hist negativo pero acelerando hacia arriba con convicción
    # (histograma bajista que gira → señal temprana de reversión)
    if hist < 0 and hist > hist_prev and (hist_prev - hist) > threshold_soft:
        return +1

    # ── SHORT momentum ──────────────────────────────────────────────────────
    # Fuerte: hist claramente negativo y cayendo
    if hist < 0 and hist < hist_prev and abs(hist) > threshold:
        return -1
    # Suave: hist positivo pero cayendo con convicción
    if hist > 0 and hist < hist_prev and (hist - hist_prev) > threshold_soft:
        return -1

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

    # RISK_OFF: BTC bajista O crash — F&G ya no lo baja a RISK_OFF
    if not btc_bull or is_crash:
        ctx = Context.RISK_OFF
    # BULL: BTC alcista + régimen no crash
    # F&G bajo (< FG_BULL_MIN) → sigue siendo BULL pero con subtype CAUTIOUS
    # El sizing se ajusta en get_size(), no acá
    elif btc_bull and regime in ("bull", "sideways", "bear"):
        ctx = Context.BULL
    else:
        ctx = Context.NEUTRAL

    # Subtype para logging y sizing
    cautious = fg_value < FG_BULL_MIN and ctx == Context.BULL
    subtype  = " CAUTIOUS" if cautious else ""
    log.info(f"  📊 Contexto: {ctx.value.upper()}{subtype} "
             f"(BTC4h={'↑' if btc_bull else '↓'} | F&G={fg_value} | régimen={regime})")

    # Guardar subtype en cache para que get_size lo use
    _btc4h_cache["cautious"] = cautious
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
            # Tier B: en BULL siempre se intenta (incluye BULL_CAUTIOUS)
            # Confirmación 15m requerida — si falla, logueamos y saltamos
            if confirm_15m(exchange, symbol, +1):
                cautious_note = " [F&G bajo → size reducido]" if _btc4h_cache.get("cautious") else ""
                return "BUY", Tier.B, f"score=2/3 Tier B + 15m confirmado{cautious_note}"
            else:
                return None, Tier.NONE, "score=2/3 Tier B pero 15m no confirma"

        if score_long == 2 and context == Context.NEUTRAL:
            return None, Tier.NONE, "score=2/3 en NEUTRAL → no entra (solo BULL)"

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

    # Reducción cuando F&G bajo (BULL_CAUTIOUS) — mercado con miedo
    if _btc4h_cache.get("cautious"):
        pct *= 0.75
        log.info(f"  ⚠️  BULL_CAUTIOUS (F&G bajo) → size reducido 25%")

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
                # Descontar comisión Binance: 0.1% entrada + 0.1% salida = 0.2% round-trip
                pnl_gross = pnl
                pnl       = round(pnl - BINANCE_FEE_RT * 100, 3)
                usd_pnl   = round(pnl * pos["usd_size"] / 100, 2)
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
                    "pnl_pct": pnl, "pnl_gross_pct": pnl_gross, "usd_pnl": usd_pnl,
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
    # Tier A (3/3): ignora límite — setup fuerte, vale la pena
    # Tier B (2/3): límite aumentado a 5 altcoins abiertas
    open_alts = [s for s in open_pos if s not in MAJORS]
    alt_limit = 999 if tier == Tier.A else 5
    if not is_major and len(open_alts) >= alt_limit:
        log.info(f"  ⏭️  Correlación: {len(open_alts)} altcoins abiertas (límite Tier B={alt_limit})")
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
        _altcoins  = pairs[:25]
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
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root{
  --bg:#060a12;--s1:#0c1423;--s2:#111c2e;--s3:#162138;
  --border:#1e2d45;--border2:#243550;
  --green:#00e5a0;--green2:rgba(0,229,160,.12);
  --red:#ff4060;--red2:rgba(255,64,96,.1);
  --yellow:#fbbf24;--blue:#38bdf8;--purple:#a78bfa;
  --orange:#f97316;--cyan:#22d3ee;
  --text:#8ba4c0;--text2:#4d6a85;--white:#deeeff;
  --mono:'JetBrains Mono',monospace;--sans:'Space Grotesk',sans-serif;
  --radius:8px;--radius-sm:5px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--sans);font-size:13px;min-height:100vh;padding:18px 22px;
  background-image:radial-gradient(ellipse 80% 50% at 50% -20%,rgba(0,229,160,.06),transparent),
                   radial-gradient(ellipse 60% 40% at 80% 80%,rgba(56,189,248,.04),transparent)}
/* ── HEADER ──────────────────────────────────────────── */
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.logo{font-size:20px;font-weight:700;color:var(--white);letter-spacing:-.5px}
.logo .v3{color:var(--green);font-size:12px;font-weight:500;margin-left:6px;padding:2px 7px;
  border:1px solid rgba(0,229,160,.3);border-radius:100px;vertical-align:middle}
.badges{display:flex;gap:6px;align-items:center}
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:100px;
  font-size:10px;font-weight:600;border:1px solid;font-family:var(--mono)}
.badge-live{color:var(--green);border-color:rgba(0,229,160,.3);background:rgba(0,229,160,.07)}
.badge-paper{color:var(--blue);border-color:rgba(56,189,248,.3);background:rgba(56,189,248,.06)}
.dot{width:5px;height:5px;border-radius:50%;background:currentColor;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
.ctx-wrap{display:flex;align-items:center;gap:8px}
.ctx-badge{padding:4px 12px;border-radius:100px;font-size:11px;font-weight:600;border:1px solid;font-family:var(--mono);transition:all .3s}
.ctx-bull{color:var(--green);border-color:rgba(0,229,160,.4);background:rgba(0,229,160,.08)}
.ctx-bull-cautious{color:var(--yellow);border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.07)}
.ctx-neutral{color:var(--blue);border-color:rgba(56,189,248,.3);background:rgba(56,189,248,.06)}
.ctx-risk_off{color:var(--red);border-color:rgba(255,64,96,.3);background:rgba(255,64,96,.07)}
.timer{font-size:10px;color:var(--text2);font-family:var(--mono);padding:3px 8px;
  background:var(--s1);border:1px solid var(--border);border-radius:var(--radius-sm)}
.sub{font-size:10px;color:var(--text2);margin-bottom:18px;display:flex;gap:12px;flex-wrap:wrap}
.sub span{display:flex;align-items:center;gap:4px}
.sub span::before{content:'·';color:var(--border2)}
.sub span:first-child::before{display:none}
/* ── KPIS ────────────────────────────────────────────── */
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:16px}
@media(max-width:900px){.kpis{grid-template-columns:repeat(3,1fr)}}
@media(max-width:500px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--s1);border:1px solid var(--border);border-radius:var(--radius);padding:13px 14px;position:relative;overflow:hidden;transition:border-color .2s}
.kpi:hover{border-color:var(--border2)}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--ka,var(--green));opacity:.7}
.kpi-label{font-size:9px;color:var(--text2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;font-family:var(--mono)}
.kpi-val{font-size:22px;font-weight:700;color:var(--white);line-height:1;margin-bottom:3px}
.kpi-sub{font-size:10px;color:var(--text2);font-family:var(--mono)}
.kpi-val.g{color:var(--green)}.kpi-val.r{color:var(--red)}.kpi-val.y{color:var(--yellow)}
.kpi-val.b{color:var(--blue)}.kpi-val.p{color:var(--purple)}.kpi-val.o{color:var(--orange)}
/* ── GRID ────────────────────────────────────────────── */
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
/* ── PANELS ──────────────────────────────────────────── */
.panel{background:var(--s1);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:10px}
.ph{display:flex;justify-content:space-between;align-items:center;padding:9px 14px;
  background:var(--s2);border-bottom:1px solid var(--border)}
.ph-title{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--text2);font-weight:600;font-family:var(--mono)}
.ph-sub{font-size:10px;color:var(--text2);font-family:var(--mono)}
/* ── CAPITAL CHART ───────────────────────────────────── */
.chart-wrap{padding:12px 14px;height:130px;position:relative}
.chart-empty{display:flex;align-items:center;justify-content:center;height:100%;
  color:var(--text2);font-size:11px;font-family:var(--mono)}
/* ── METRICS GRID ────────────────────────────────────── */
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;padding:10px 14px}
.metric{background:var(--s2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:9px 11px}
.metric-label{font-size:9px;color:var(--text2);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;font-family:var(--mono)}
.metric-val{font-size:16px;font-weight:700;color:var(--white);font-family:var(--mono)}
.metric-val.g{color:var(--green)}.metric-val.r{color:var(--red)}.metric-val.y{color:var(--yellow)}
/* ── TIERS ───────────────────────────────────────────── */
.tiers{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px 14px}
.tier-card{border-radius:var(--radius-sm);padding:11px;border:1px solid var(--border);position:relative;overflow:hidden}
.tier-card.ta{border-color:rgba(167,139,250,.35);background:rgba(167,139,250,.05)}
.tier-card.tb{border-color:rgba(249,115,22,.3);background:rgba(249,115,22,.04)}
.tier-name{font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;font-family:var(--mono)}
.tier-num{font-size:30px;font-weight:700;line-height:1}
.tier-num.ta{color:var(--purple)}.tier-num.tb{color:var(--orange)}
.tier-detail{font-size:10px;color:var(--text2);margin-top:4px;font-family:var(--mono)}
/* ── FAMILIES ────────────────────────────────────────── */
.families{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;padding:10px 14px}
.fam{border-radius:var(--radius-sm);padding:10px;border:1px solid var(--border);text-align:center;transition:all .2s}
.fam.ok{border-color:rgba(0,229,160,.3);background:rgba(0,229,160,.05)}
.fam.warn{border-color:rgba(251,191,36,.2);background:rgba(251,191,36,.03)}
.fam.fail{border-color:rgba(255,64,96,.2);background:rgba(255,64,96,.03)}
.fam-icon{font-size:18px;margin-bottom:4px}
.fam-name{font-size:9px;color:var(--text2);text-transform:uppercase;letter-spacing:.08em;font-family:var(--mono);margin-bottom:4px}
.fam-wr{font-size:13px;font-weight:600;font-family:var(--mono)}
/* ── POSITIONS ───────────────────────────────────────── */
.pos-list{display:flex;flex-direction:column;gap:6px;padding:10px}
.pos-card{border-radius:var(--radius-sm);padding:10px 12px;border:1px solid var(--border);background:var(--s2);transition:border-color .2s}
.pos-card:hover{border-color:var(--border2)}
.pos-card.long{border-left:3px solid var(--green)}
.pos-card.short{border-left:3px solid var(--red)}
.pos-card.tier-a{border-right:2px solid var(--purple)}
.pos-card.tier-b{border-right:2px solid var(--orange)}
.pos-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px}
.pos-sym{font-weight:700;font-size:14px;color:var(--white)}
.pos-tier-badge{font-size:8px;color:var(--text2);font-family:var(--mono);padding:1px 5px;
  border:1px solid var(--border);border-radius:3px;margin-left:5px}
.pos-pnl{font-weight:700;font-size:14px;font-family:var(--mono)}
.pos-pnl.pos{color:var(--green)}.pos-pnl.neg{color:var(--red)}
.pos-grid{display:grid;grid-template-columns:1fr 1fr;gap:3px;font-size:10px;font-family:var(--mono)}
.pos-l{color:var(--text2)}.pos-v{color:var(--text);text-align:right}
.pos-bar{height:2px;background:var(--border);border-radius:1px;margin-top:7px;overflow:hidden}
.pos-bar-fill{height:100%;border-radius:1px;transition:width .5s}
.pos-time{font-size:9px;color:var(--text2);margin-top:5px;font-family:var(--mono)}
/* ── SCANNER ─────────────────────────────────────────── */
.scanner-wrap{padding:8px 14px;display:flex;flex-wrap:wrap;gap:4px}
.scan-tag{padding:2px 8px;border-radius:3px;font-size:10px;border:1px solid var(--border);
  color:var(--text2);font-family:var(--mono);transition:all .15s}
.scan-tag:hover{border-color:var(--border2);color:var(--text)}
/* ── LOG ─────────────────────────────────────────────── */
#live-log{padding:8px 14px;font-size:10px;max-height:200px;overflow-y:auto;
  display:flex;flex-direction:column;gap:2px;font-family:var(--mono)}
#live-log::-webkit-scrollbar{width:3px}
#live-log::-webkit-scrollbar-track{background:transparent}
#live-log::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
.log-line{line-height:1.5;padding:1px 0}
/* ── TABLE ───────────────────────────────────────────── */
.tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono)}
.tbl th{padding:7px 10px;font-size:8px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--text2);background:var(--s2);text-align:left;font-weight:500}
.tbl td{padding:7px 10px;border-bottom:1px solid rgba(22,32,48,.5)}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:rgba(255,255,255,.015)}
.action-buy{padding:1px 6px;border-radius:2px;font-size:10px;background:rgba(0,229,160,.1);color:var(--green)}
.action-sell{padding:1px 6px;border-radius:2px;font-size:10px;background:rgba(255,64,96,.1);color:var(--red)}
/* ── DRAWDOWN BAR ────────────────────────────────────── */
.dd-bar{height:4px;background:var(--border);border-radius:2px;margin:6px 14px 10px;overflow:hidden}
.dd-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--green),var(--yellow),var(--red));transition:width .5s}
/* ── FOOTER ──────────────────────────────────────────── */
footer{text-align:center;font-size:9px;color:var(--text2);margin-top:14px;padding-top:10px;
  border-top:1px solid var(--border);font-family:var(--mono);letter-spacing:.05em}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div>
    <div class="logo">CryptoBot <span class="v3">v3</span></div>
  </div>
  <div class="badges ctx-wrap">
    <span id="ctx-badge" class="ctx-badge ctx-neutral">— CTX</span>
    <span class="badge badge-live"><span class="dot"></span> LIVE</span>
    <span class="badge badge-paper" id="mode-badge">PAPER</span>
    <span class="timer">↻ <span id="cd">15</span>s</span>
  </div>
</div>
<div class="sub">
  <span>Tier A · 3/3</span>
  <span>Tier B · 2/3 + 15m</span>
  <span>Contexto dinámico</span>
  <span>Sin RL · Pesos fijos</span>
  <span id="sub-regime">régimen —</span>
</div>

<!-- KPIs -->
<div class="kpis">
  <div class="kpi" style="--ka:var(--green)">
    <div class="kpi-label">Capital</div>
    <div class="kpi-val g" id="k-cap">—</div>
    <div class="kpi-sub" id="k-cap-d">—</div>
  </div>
  <div class="kpi" style="--ka:var(--yellow)">
    <div class="kpi-label">P&L Total</div>
    <div class="kpi-val" id="k-pnl">—</div>
    <div class="kpi-sub" id="k-pnl-d">trades cerrados</div>
  </div>
  <div class="kpi" style="--ka:var(--blue)">
    <div class="kpi-label">Win Rate</div>
    <div class="kpi-val b" id="k-wr">—</div>
    <div class="kpi-sub" id="k-wr-d">sin datos</div>
  </div>
  <div class="kpi" style="--ka:var(--cyan)">
    <div class="kpi-label">Expectancy</div>
    <div class="kpi-val" id="k-exp">—</div>
    <div class="kpi-sub">por trade</div>
  </div>
  <div class="kpi" style="--ka:var(--red)">
    <div class="kpi-label">Max DD</div>
    <div class="kpi-val r" id="k-dd">—</div>
    <div class="kpi-sub" id="k-dd-d">desde pico</div>
  </div>
  <div class="kpi" style="--ka:var(--purple)">
    <div class="kpi-label">Profit Factor</div>
    <div class="kpi-val p" id="k-pf">—</div>
    <div class="kpi-sub">wins/losses</div>
  </div>
</div>

<!-- GRID PRINCIPAL -->
<div class="grid">
  <!-- COLUMNA IZQUIERDA -->
  <div>
    <!-- CURVA DE CAPITAL -->
    <div class="panel">
      <div class="ph">
        <span class="ph-title">Curva de capital</span>
        <span class="ph-sub" id="cap-range">—</span>
      </div>
      <div class="chart-wrap">
        <canvas id="cap-chart"></canvas>
        <div class="chart-empty" id="chart-empty" style="display:none">Sin trades aún</div>
      </div>
    </div>

    <!-- MÉTRICAS AVANZADAS -->
    <div class="panel">
      <div class="ph">
        <span class="ph-title">Métricas</span>
        <span class="ph-sub" id="fg-badge">F&G —</span>
      </div>
      <div class="metrics">
        <div class="metric">
          <div class="metric-label">Tier A WR</div>
          <div class="metric-val" id="m-tawr">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Tier B WR</div>
          <div class="metric-val" id="m-tbwr">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Trades total</div>
          <div class="metric-val" id="m-trades">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Avg win</div>
          <div class="metric-val g" id="m-avgwin">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Avg loss</div>
          <div class="metric-val r" id="m-avgloss">—</div>
        </div>
        <div class="metric">
          <div class="metric-label">Payoff ratio</div>
          <div class="metric-val y" id="m-payoff">—</div>
        </div>
      </div>
      <!-- Drawdown bar -->
      <div style="padding:0 14px 4px;font-size:9px;color:var(--text2);font-family:var(--mono)">MAX DRAWDOWN ACTUAL</div>
      <div class="dd-bar"><div class="dd-fill" id="dd-fill" style="width:0%"></div></div>
    </div>

    <!-- TIERS + FAMILIAS -->
    <div class="panel">
      <div class="ph"><span class="ph-title">Entradas por tier</span></div>
      <div class="tiers">
        <div class="tier-card ta">
          <div class="tier-name">Tier A · 3/3</div>
          <div class="tier-num ta" id="ta-n">0</div>
          <div class="tier-detail" id="ta-wr">sin datos</div>
        </div>
        <div class="tier-card tb">
          <div class="tier-name">Tier B · 2/3+15m</div>
          <div class="tier-num tb" id="tb-n">0</div>
          <div class="tier-detail" id="tb-wr">sin datos</div>
        </div>
      </div>
    </div>

    <!-- FAMILIAS -->
    <div class="panel">
      <div class="ph"><span class="ph-title">Performance por familia</span></div>
      <div class="families" id="fam-row"></div>
    </div>

    <!-- LOG -->
    <div class="panel">
      <div class="ph"><span class="ph-title">Actividad en vivo</span><span class="ph-sub" id="log-time">—</span></div>
      <div id="live-log"></div>
    </div>
  </div>

  <!-- COLUMNA DERECHA -->
  <div>
    <!-- POSICIONES -->
    <div class="panel">
      <div class="ph">
        <span class="ph-title">Posiciones abiertas</span>
        <span class="ph-sub" id="pos-count">ninguna</span>
      </div>
      <div id="pos-list" class="pos-list">
        <div style="padding:20px;text-align:center;color:var(--text2);font-size:11px;font-family:var(--mono)">
          Sin posiciones abiertas
        </div>
      </div>
    </div>

    <!-- SCANNER -->
    <div class="panel">
      <div class="ph"><span class="ph-title">Scanner activo</span><span class="ph-sub" id="scan-count">—</span></div>
      <div class="scanner-wrap" id="scanner"></div>
    </div>
  </div>
</div>

<!-- HISTORIAL -->
<div class="panel">
  <div class="ph">
    <span class="ph-title">Historial de trades</span>
    <span class="ph-sub" id="hist-count">—</span>
  </div>
  <div style="overflow:auto">
    <table class="tbl">
      <thead>
        <tr>
          <th>Par</th><th>Tier</th><th>Acción</th>
          <th>Entrada</th><th>Salida</th><th>P&L%</th><th>P&L $</th>
          <th>Motivo</th><th>Ctx</th><th>Hora</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<footer>CryptoBot v3 · Tiered Confirmation · Contexto dinámico BULL/NEUTRAL/RISK-OFF · Circuit breaker · Sin RL</footer>

<script>
const ICAP = 1000;
let cd = 15;
let capChart = null;

function fmt(n, d=2){ return (+n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}); }
function fmtPct(n){ return (n>=0?'+':'')+fmt(n)+'%'; }
function fmtUSD(n){ return (n>=0?'+':'-')+'$'+fmt(Math.abs(n)); }

function computeMetrics(closed){
  if(!closed.length) return {};
  const wins   = closed.filter(t=>t.pnl_pct>0);
  const losses = closed.filter(t=>t.pnl_pct<=0);
  const wr     = wins.length/closed.length*100;
  const avgWin = wins.length   ? wins.reduce((s,t)=>s+t.pnl_pct,0)/wins.length   : 0;
  const avgLoss= losses.length ? Math.abs(losses.reduce((s,t)=>s+t.pnl_pct,0)/losses.length) : 0;
  const payoff = avgLoss > 0 ? avgWin/avgLoss : 0;
  const exp    = (wr/100)*avgWin - (1-wr/100)*avgLoss;
  const grossW = wins.reduce((s,t)=>s+(t.pnl_pct*(t.usd_size||20)/100),0);
  const grossL = Math.abs(losses.reduce((s,t)=>s+(t.pnl_pct*(t.usd_size||20)/100),0));
  const pf     = grossL > 0 ? grossW/grossL : grossW > 0 ? 99 : 0;
  // max drawdown from equity curve
  let peak = ICAP, maxDD = 0;
  let cap  = ICAP;
  closed.forEach(t=>{
    cap += (t.pnl_pct||0)*(t.usd_size||20)/100;
    if(cap > peak) peak = cap;
    const dd = (peak - cap)/peak*100;
    if(dd > maxDD) maxDD = dd;
  });
  return {wr, avgWin, avgLoss, payoff, exp, pf, maxDD};
}

function buildCapCurve(closed){
  const pts = [{x:0, y:ICAP, label:'Inicio'}];
  let cap = ICAP;
  closed.forEach((t,i)=>{
    cap += (t.pnl_pct||0)*(t.usd_size||20)/100;
    pts.push({x:i+1, y:+cap.toFixed(2), label:t.symbol});
  });
  return pts;
}

function renderChart(pts){
  const canvas = document.getElementById('cap-chart');
  const empty  = document.getElementById('chart-empty');
  if(pts.length < 2){
    canvas.style.display='none'; empty.style.display='flex'; return;
  }
  canvas.style.display='block'; empty.style.display='none';
  const labels = pts.map(p=>p.label);
  const data   = pts.map(p=>p.y);
  const last   = data[data.length-1];
  const color  = last >= ICAP ? '#00e5a0' : '#ff4060';
  if(capChart){ capChart.destroy(); }
  capChart = new Chart(canvas, {
    type:'line',
    data:{
      labels,
      datasets:[{
        data, borderColor:color, borderWidth:2,
        backgroundColor: last>=ICAP ? 'rgba(0,229,160,.08)' : 'rgba(255,64,96,.06)',
        fill:true, tension:.35, pointRadius:0, pointHoverRadius:4,
        pointHoverBackgroundColor:color,
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:400},
      plugins:{legend:{display:false},tooltip:{
        callbacks:{label:ctx=>'$'+fmt(ctx.parsed.y)},
        backgroundColor:'rgba(11,20,35,.95)',borderColor:'rgba(30,45,70,.8)',borderWidth:1,
        titleFont:{family:'JetBrains Mono',size:10},bodyFont:{family:'JetBrains Mono',size:11},
      }},
      scales:{
        x:{display:false},
        y:{
          grid:{color:'rgba(30,45,70,.5)',drawBorder:false},
          ticks:{color:'#4d6a85',font:{family:'JetBrains Mono',size:10},
            callback:v=>'$'+v.toLocaleString()},
          border:{display:false}
        }
      }
    }
  });
}

async function load(){
  try{
    const d = await fetch('/v3/api').then(r=>r.json());
    const st = d.state||{};
    const trades = d.trades||[];
    const closed = trades.filter(t=>t.pnl_pct!==undefined);
    const cap    = st.capital||ICAP;
    const m      = computeMetrics(closed);

    // ── Capital KPI ──
    document.getElementById('k-cap').textContent = '$'+fmt(cap);
    document.getElementById('k-cap').className = 'kpi-val '+(cap>=ICAP?'g':'r');
    document.getElementById('k-cap-d').textContent = (cap-ICAP>=0?'+':'')+fmt(cap-ICAP)+' desde inicio';

    // ── P&L ──
    const tpnl = closed.reduce((s,t)=>s+(t.pnl_pct||0)*(t.usd_size||20)/100,0);
    document.getElementById('k-pnl').textContent = fmtUSD(tpnl);
    document.getElementById('k-pnl').className = 'kpi-val '+(tpnl>=0?'g':'r');

    // ── Win Rate ──
    const wr = m.wr!=null ? m.wr : null;
    document.getElementById('k-wr').textContent  = wr!=null ? Math.round(wr)+'%' : '—';
    document.getElementById('k-wr').className    = 'kpi-val '+(wr>=50?'g':wr<50?'r':'b');
    document.getElementById('k-wr-d').textContent= closed.length
      ? closed.filter(t=>t.pnl_pct>0).length+'/'+closed.length+' trades' : 'sin datos';

    // ── Expectancy ──
    const expEl = document.getElementById('k-exp');
    if(m.exp!=null){ expEl.textContent=(m.exp>=0?'+':'')+fmt(m.exp)+'%'; expEl.className='kpi-val '+(m.exp>=0?'g':'r'); }

    // ── Max DD ──
    if(m.maxDD!=null){
      document.getElementById('k-dd').textContent   = '-'+fmt(m.maxDD)+'%';
      document.getElementById('k-dd-d').textContent = 'máximo histórico';
      document.getElementById('dd-fill').style.width = Math.min(m.maxDD*5,100)+'%';
    }

    // ── Profit Factor ──
    const pfEl = document.getElementById('k-pf');
    if(m.pf!=null){ pfEl.textContent=fmt(m.pf,2)+'x'; pfEl.className='kpi-val '+(m.pf>=1.5?'g':m.pf>=1?'y':'r'); }

    // ── Métricas avanzadas ──
    const ta = closed.filter(t=>t.tier==='A');
    const tb = closed.filter(t=>t.tier==='B');
    const taWr = ta.length ? Math.round(ta.filter(t=>t.pnl_pct>0).length/ta.length*100) : null;
    const tbWr = tb.length ? Math.round(tb.filter(t=>t.pnl_pct>0).length/tb.length*100) : null;
    document.getElementById('m-tawr').textContent   = taWr!=null ? taWr+'%' : '—';
    document.getElementById('m-tawr').className     = 'metric-val '+(taWr>=50?'g':taWr<50?'r':'');
    document.getElementById('m-tbwr').textContent   = tbWr!=null ? tbWr+'%' : '—';
    document.getElementById('m-tbwr').className     = 'metric-val '+(tbWr>=50?'g':tbWr<50?'r':'');
    document.getElementById('m-trades').textContent = closed.length||'—';
    document.getElementById('m-avgwin').textContent = m.avgWin ? '+'+fmt(m.avgWin)+'%' : '—';
    document.getElementById('m-avgloss').textContent= m.avgLoss ? '-'+fmt(m.avgLoss)+'%' : '—';
    document.getElementById('m-payoff').textContent = m.payoff ? fmt(m.payoff,2)+'x' : '—';

    // ── F&G ──
    if(d.fear_greed){
      const fv = +d.fear_greed.value;
      const fc = fv<25?'var(--red)':fv<45?'var(--orange)':fv>65?'var(--green)':'var(--yellow)';
      document.getElementById('fg-badge').textContent = 'F&G '+fv+' — '+d.fear_greed.label;
      document.getElementById('fg-badge').style.color = fc;
    }

    // ── Régimen ──
    const rm={bull:'BULL 📈',bear:'BEAR 📉',sideways:'SIDE ↔',crash:'CRASH 💥'};
    const reg = d.regime||'?';
    document.getElementById('sub-regime').textContent = 'régimen '+(rm[reg]||reg);

    // ── Contexto badge ──
    const ctx = (d.context||'neutral').toLowerCase();
    const cb  = document.getElementById('ctx-badge');
    const ctxMap = {
      'bull':'🟢 BULL', 'bull cautious':'🟡 BULL CAUTIOUS',
      'neutral':'🔵 NEUTRAL', 'risk_off':'🔴 RISK-OFF'
    };
    const ctxClass = ctx.includes('cautious')?'bull-cautious':ctx.replace(' ','_').replace('off','off');
    cb.className = 'ctx-badge ctx-'+(ctx.startsWith('bull')?ctx.includes('cautious')?'bull-cautious':'bull':ctx);
    cb.textContent = ctxMap[ctx]||ctx.toUpperCase();

    // ── Curva de capital ──
    const pts = buildCapCurve(closed);
    renderChart(pts);
    if(pts.length > 1){
      const lo = Math.min(...pts.map(p=>p.y)), hi = Math.max(...pts.map(p=>p.y));
      document.getElementById('cap-range').textContent = '$'+fmt(lo)+' — $'+fmt(hi);
    }

    // ── Tiers ──
    document.getElementById('ta-n').textContent  = ta.length;
    document.getElementById('tb-n').textContent  = tb.length;
    document.getElementById('ta-wr').textContent = taWr!=null ? taWr+'% WR · '+ta.length+' trades':'sin datos';
    document.getElementById('tb-wr').textContent = tbWr!=null ? tbWr+'% WR · '+tb.length+' trades':'sin datos';

    // ── Familias ──
    const fams=[{k:'trend',n:'Tendencia',i:'📈'},{k:'momentum',n:'Momentum',i:'⚡'},{k:'volume',n:'Volumen',i:'📊'}];
    const fs = d.family_stats||{};
    document.getElementById('fam-row').innerHTML = fams.map(f=>{
      const s  = fs[f.k]||{wins:0,total:0};
      const wr2= s.total ? Math.round(s.wins/s.total*100) : null;
      const cls= wr2==null?'':wr2>=60?'ok':wr2>=45?'warn':'fail';
      const col= wr2==null?'var(--text2)':wr2>=60?'var(--green)':wr2>=45?'var(--yellow)':'var(--red)';
      return `<div class="fam ${cls}">
        <div class="fam-icon">${f.i}</div>
        <div class="fam-name">${f.n}</div>
        <div class="fam-wr" style="color:${col}">${wr2!=null?wr2+'%':'—'}</div>
        ${s.total?`<div style="font-size:9px;color:var(--text2);margin-top:2px;font-family:var(--mono)">${s.total} trades</div>`:''}
      </div>`;
    }).join('');

    // ── Posiciones ──
    const pos = d.positions||[];
    document.getElementById('pos-count').textContent = pos.length ? pos.length+' abierta'+(pos.length>1?'s':'') : 'ninguna';
    document.getElementById('pos-list').innerHTML = pos.length ? pos.map(p=>{
      const isS = p.action==='SELL';
      const entry= +p.entry_price, curr= +p.current_price, tp= +p.take_profit, sl= +(p.trail_stop);
      const pnlPct = isS ? (entry-curr)/entry*100 : (curr-entry)/entry*100;
      const isPos  = pnlPct >= 0;
      const tierCls= 'tier-'+(p.tier||'a').toLowerCase();
      const oa     = p.opened_at ? new Date(p.opened_at) : null;
      const elapsed= oa ? Math.round((Date.now()-oa)/60000) : null;
      const tpPct  = isS ? (entry-tp)/entry*100 : (tp-entry)/entry*100;
      const slPct  = isS ? (sl-entry)/entry*100  : (entry-sl)/entry*100;
      const progress= isS
        ? Math.max(0,Math.min(100,(entry-curr)/(entry-tp)*100))
        : Math.max(0,Math.min(100,(curr-entry)/(tp-entry)*100));
      return `<div class="pos-card ${isS?'short':'long'} ${tierCls}">
        <div class="pos-top">
          <div><span class="pos-sym">${p.symbol.replace('/USDT','')}</span><span class="pos-tier-badge">Tier ${p.tier||'?'}</span></div>
          <span class="pos-pnl ${isPos?'pos':'neg'}">${fmtPct(pnlPct)}</span>
        </div>
        <div class="pos-grid">
          <span class="pos-l">Entrada</span><span class="pos-v">${entry}</span>
          <span class="pos-l">Precio</span><span class="pos-v" style="color:${isPos?'var(--green)':'var(--red)'}">${curr}</span>
          <span class="pos-l">Trail stop</span><span class="pos-v" style="color:var(--orange)">${sl}</span>
          <span class="pos-l">Take profit</span><span class="pos-v" style="color:var(--green)">${tp} <small style="color:var(--text2)">(+${fmt(tpPct)}%)</small></span>
          <span class="pos-l">Size</span><span class="pos-v">$${p.usd_size}</span>
          <span class="pos-l">Parcial</span><span class="pos-v">${p.partial_closed?'✅ cerrado':'pendiente'}</span>
        </div>
        <div class="pos-bar"><div class="pos-bar-fill" style="width:${progress}%;background:${isPos?'var(--green)':'var(--red)'}"></div></div>
        ${elapsed!=null?`<div class="pos-time">abierta hace ${elapsed<60?elapsed+'m':(elapsed/60).toFixed(1)+'h'} · ctx: ${p.context_entry||'?'}</div>`:''}
      </div>`;
    }).join('') : '<div style="padding:20px;text-align:center;color:var(--text2);font-size:11px;font-family:var(--mono)">Sin posiciones abiertas</div>';

    // ── Scanner ──
    const sc = d.scanner||[];
    document.getElementById('scan-count').textContent = sc.length ? sc.length+' pares' : '—';
    document.getElementById('scanner').innerHTML = sc.map(s=>
      `<span class="scan-tag">${s.replace('/USDT','')}</span>`).join('');

    // ── Log ──
    try{
      const logs = await fetch('/api/log').then(r=>r.json());
      const logEl = document.getElementById('live-log');
      if(logs.length){
        document.getElementById('log-time').textContent = new Date(logs[logs.length-1].ts*1000).toLocaleTimeString('es-AR');
      }
      logEl.innerHTML = logs.slice(-20).map(l=>{
        const m2 = l.msg||'';
        let c='var(--text)';
        if(m2.includes('ABIERTA')||m2.includes('🔥')||m2.includes('✅')) c='var(--green)';
        else if(m2.includes('CERRADA')||m2.includes('🔴')) c='var(--red)';
        else if(m2.includes('Tier A')||m2.includes('score=3')) c='var(--purple)';
        else if(m2.includes('Tier B')||m2.includes('15m')) c='var(--orange)';
        else if(m2.includes('CAUTIOUS')) c='var(--yellow)';
        else if(m2.includes('⏭️')||m2.includes('skip')) c='var(--text2)';
        return `<div class="log-line" style="color:${c}">${m2}</div>`;
      }).join('');
      logEl.scrollTop = logEl.scrollHeight;
    }catch(e){}

    // ── Tabla historial ──
    const tb2 = document.getElementById('tbody');
    document.getElementById('hist-count').textContent = closed.length ? closed.length+' trades' : 'sin trades';
    if(!closed.length){
      tb2.innerHTML='<tr><td colspan="10" style="text-align:center;padding:30px;color:var(--text2)">Sin trades aún — esperando primera entrada</td></tr>';
      return;
    }
    tb2.innerHTML = [...closed].reverse().map(t=>{
      const ts = new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const pc = t.pnl_pct>=0?'var(--green)':'var(--red)';
      const tc = t.tier==='A'?'var(--purple)':'var(--orange)';
      const usdPnl = (t.pnl_pct||0)*(t.usd_size||20)/100;
      return `<tr>
        <td style="color:var(--white);font-weight:600">${t.symbol}</td>
        <td style="color:${tc};font-weight:600">${t.tier||'?'}</td>
        <td><span class="${t.action==='BUY'?'action-buy':'action-sell'}">${t.action}</span></td>
        <td>${t.entry}</td><td>${t.exit}</td>
        <td style="color:${pc};font-weight:700">${fmtPct(t.pnl_pct||0)}</td>
        <td style="color:${pc}">${fmtUSD(usdPnl)}</td>
        <td style="color:var(--text2);font-size:10px">${t.reason||'—'}</td>
        <td style="color:var(--text2)">${t.context_entry||'—'}</td>
        <td style="color:var(--text2);font-size:10px">${ts}</td>
      </tr>`;
    }).join('');
  }catch(e){ console.error('Dashboard error:', e); }
}

function tick(){
  cd--;
  document.getElementById('cd').textContent = cd;
  if(cd <= 0){ cd = 15; load(); }
}

load();
setInterval(tick, 1000);
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
