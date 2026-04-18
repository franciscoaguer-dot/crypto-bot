"""
CryptoBot v10 — Aggressive Self-Learning Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
APRENDIZAJE AUTOMÁTICO:
- Signal Memory: guarda qué señales llevaron a cada resultado
- Dynamic Weights: ponderación por win rate histórico (no +1 fijo)
- Reinforcement Learning simple: ajusta MIN_SIGNALS según racha
- Reporte semanal con Claude: genera config optimizada

MEJORAS WIN RATE:
- Entry confirmation delay: espera 1 vela de confirmación
- Salida parcial: cierra 50% al primer TP, deja correr el resto
- Stop loss absoluto 3%
- Profit trailing dinámico

INFRAESTRUCTURA:
- Rate limiter Claude + retry con backoff
- Market regime detector
- Sideways filter (score ≥3)
- Correlación máx 3 altcoins
- Compounding + anti-drawdown
- Futuros 2x para score ≥4
- 15 altcoins dinámicas
"""

import os, re, time, json, logging, requests, threading
from datetime import datetime, timezone, timedelta
import ccxt
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

_claude_semaphore = threading.Semaphore(1)
_claude_last_call = 0

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN     = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
CAPITAL_TOTAL_USD  = float(os.environ.get("CAPITAL_USD", "100"))
PAPER_TRADING      = os.environ.get("PAPER_TRADING", "true").lower() == "true"

POSITION_SIZE_MAP  = {2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.06}
TRAILING_STOP_PCT  = 0.010
TAKE_PROFIT_PCT    = 0.025
TAKE_PROFIT_PARTIAL = 0.015  # TP parcial al 1.5% — cierra 50%
FUTURES_TRAILING   = 0.008
FUTURES_TP         = 0.020
FUTURES_MIN_SCORE  = 4
FUTURES_LEVERAGE   = 2
MIN_SIGNALS        = 2
MIN_SIGNALS_SIDEWAYS = 3
CONFIDENCE_MIN     = 0.35  # bajado de 0.50 — Claude vetaba todo con RSI overbought en 3m
LOOP_INTERVAL_SEC  = 60
MAX_CAPITAL_EXPOSURE = 0.25  # 25% — con $500 permite posiciones más grandes
MAX_WEEKLY_LOSS_PCT  = 0.10
DRAWDOWN_REDUCE_PCT  = 0.50
STOP_LOSS_PCT        = 0.03
BINANCE_FEE_RT       = 0.001   # 0.1% entrada + 0.1% salida = 0.2% round-trip (spot)
PROFIT_TRAIL_TRIGGER = 0.015
PROFIT_TRAIL_STEP    = 0.010
MAX_CORRELATION_ALTS = 3
CLAUDE_RATE_LIMIT_SEC = 2
CLAUDE_MAX_RETRIES   = 3
FUNDING_BULLISH_THRESHOLD = -0.0001
FUNDING_BEARISH_THRESHOLD =  0.0015
OB_IMBALANCE_THRESHOLD    =  0.60
BB_PERIOD = 20
BB_STD    = 2.0
REGIME_TREND_THRESHOLD = 0.02
REGIME_CRASH_THRESHOLD = -0.05

# v10 — Aggressive Self-Learning
SHORT_ENABLED          = True    # habilitar shorts en futuros
SHORT_MIN_SCORE        = -2.5    # score ponderado mínimo para short
SHORT_MAX_ALTS         = 2       # máx altcoins short simultáneas
ATR_PERIOD             = 14      # período ATR para trailing dinámico
ATR_MULTIPLIER_SPOT    = 1.5     # trailing = ATR * multiplier (spot)
ATR_MULTIPLIER_FUT     = 1.0     # trailing = ATR * multiplier (futuros)
SIDEWAYS_MIN_FLOAT     = 2.0     # bajado de 2.5 a 2.0 — más entradas
WEIGHTS_UPDATE_INTERVAL = 3600   # recalcular pesos cada 1h
MIN_SAMPLES_FOR_WEIGHT  = 5      # mínimo trades para confiar en un peso
RL_STREAK_THRESHOLD     = 3      # 3 pérdidas seguidas → subir MIN_SIGNALS
ENTRY_CONFIRM_CANDLES   = 1      # esperar N velas de confirmación antes de entrar
PARTIAL_EXIT_PCT        = 0.50   # cerrar 50% al TP parcial

# ── Persistencia PostgreSQL ───────────────────────────────────────────────────
# Si DATABASE_URL existe (Railway Postgres) → datos sobreviven redeploys.
# Si no → fallback a archivos JSON locales (se pierden en redeploy).

DATABASE_URL = os.environ.get("DATABASE_URL", "")

if DATABASE_URL:
    try:
        import psycopg2, psycopg2.extras
        _pg = psycopg2.connect(DATABASE_URL, sslmode="require")
        _pg.autocommit = True
        # Crear tablas si no existen
        with _pg.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id         SERIAL PRIMARY KEY,
                    data       JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
        import logging as _l; _l.getLogger("bot").info("💾 Persistencia: PostgreSQL ✅")
        _USE_PG = True
    except Exception as _pg_err:
        import logging as _l; _l.getLogger("bot").warning(f"⚠️  PostgreSQL no disponible ({_pg_err}) — usando archivos JSON")
        _pg = None; _USE_PG = False
else:
    import logging as _l; _l.getLogger("bot").warning("⚠️  DATABASE_URL no encontrada — usando archivos JSON (datos se pierden en redeploy)")
    _pg = None; _USE_PG = False

def _pg_get(key: str, default=None):
    """Lee un valor JSON de PostgreSQL."""
    try:
        with _pg.cursor() as cur:
            cur.execute("SELECT value FROM kv_store WHERE key=%s", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else default
    except Exception as e:
        log.warning(f"PG get error ({key}): {e}")
        return default

def _pg_set(key: str, value) -> bool:
    """Escribe un valor JSON en PostgreSQL."""
    try:
        with _pg.cursor() as cur:
            cur.execute("""
                INSERT INTO kv_store (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE
                SET value=EXCLUDED.value, updated_at=NOW()
            """, (key, json.dumps(value, default=str)))
        return True
    except Exception as e:
        log.warning(f"PG set error ({key}): {e}")
        return False

# Rutas de fallback JSON
DATA_DIR             = "/data" if os.path.isdir("/data") else "."
SIGNAL_MEMORY_FILE   = f"{DATA_DIR}/signal_memory.json"
DYNAMIC_WEIGHTS_FILE = f"{DATA_DIR}/dynamic_weights.json"
TRADE_LOG_FILE       = f"{DATA_DIR}/trade_log.json"
POSITIONS_FILE       = f"{DATA_DIR}/positions.json"
STATE_FILE           = f"{DATA_DIR}/bot_state.json"

BASE_WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
EXCLUDE_SYMBOLS = {
    "USDT","USDC","BUSD","DAI","TUSD","FDUSD","USDP","USD1","RLUSD","EUR",
    "WBTC","WETH","STETH","BETH","BTC","ETH","SOL","BNB","LDUSDT","XAUT","PAXG"
}
ARG_TZ = timezone(timedelta(hours=-3))

# ─────────────────────────────────────────
# SIGNAL MEMORY — el cerebro del aprendizaje
# ─────────────────────────────────────────
def load_signal_memory():
    if _USE_PG:
        return _pg_get("signal_memory", {})
    if os.path.exists(SIGNAL_MEMORY_FILE):
        try:
            with open(SIGNAL_MEMORY_FILE) as f: return json.load(f)
        except: pass
    return {}


def apply_fee(pnl_pct: float, mode: str = "SPOT") -> float:
    """
    Descuenta comisión Binance del PnL bruto.
    SPOT:    0.1% × 2 = 0.2% round-trip
    FUTURES: 0.04% × 2 = 0.08% (taker rate futuros)
    Con BNB sería 25% menos, pero usamos el valor conservador sin BNB.
    """
    fee = 0.0004 if "FUT" in mode.upper() else BINANCE_FEE_RT
    return round(pnl_pct - fee * 100, 3)

def save_signal_to_memory(signals_dict, pnl_pct, regime):
    memory = load_signal_memory()
    sig_names = ["tech","macd","bb","ob","rsi_div","vol","funding","news","tf4h"]
    for sig in sig_names:
        val = signals_dict.get(sig, 0)
        if val == 0: continue
        key = f"{sig}_{regime}"
        if key not in memory:
            memory[key] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}
        memory[key]["count"] += 1
        memory[key]["total_pnl"] = round(memory[key]["total_pnl"] + pnl_pct, 4)
        if pnl_pct > 0: memory[key]["wins"] += 1
        else:           memory[key]["losses"] += 1
    if _USE_PG:
        _pg_set("signal_memory", memory)
    else:
        with open(SIGNAL_MEMORY_FILE, "w") as f: json.dump(memory, f, indent=2)

# ─────────────────────────────────────────
# DYNAMIC WEIGHTS — ponderación por historial
# ─────────────────────────────────────────
_weights_cache = {}
_weights_last_update = 0

DEFAULT_WEIGHTS = {
    # ── SCORE PRINCIPAL: 3 grupos, máx ~6 pts ──────────────────────
    # Trend group
    "tech":         1.0,  # EMA cross
    "supertrend":   1.3,  # Supertrend
    # Momentum group
    "macd":         1.0,  # MACD histogram
    "rsi_div":      1.0,  # RSI divergence
    # Confirmation group
    "vwap":         1.2,  # VWAP
    "vol":          0.8,  # Volume
    "trend_struct": 0.9,  # HH/LL structure
    # ── FILTROS CONTEXTUALES: no suman al score ─────────────────────
    # Estas señales se calculan pero se usan como filtros/ajuste de size
    "tf4h":        0.0,   # 4h confirmation → filtro separado
    "bb":          0.0,   # Bollinger → filtro pump/dump
    "ob":          0.0,   # Order book → ajuste de size
    "news":        0.0,   # News → filtro contextual
    "funding":     0.0,   # Funding → ajuste de size
    "stoch_rsi":   0.0,   # Stoch RSI → filtro sobreextensión
    "williams_r":  0.0,   # Williams → filtro sobrecompra
    "cci":         0.0,   # CCI → filtro secundario
    "squeeze":     0.0,   # Squeeze → confirmación opcional
    "support_res": 0.0,   # S/R → filtro niveles
    "candle":      0.0,   # Candlestick → filtro confirmación
}

def recalculate_weights(regime):
    """
    Recalcula el peso de cada señal según su win rate histórico.
    Señales con >65% win rate → peso hasta 1.5
    Señales con <40% win rate → peso hasta 0.5
    Sin datos suficientes → peso default 1.0
    """
    global _weights_cache, _weights_last_update
    now = time.time()
    if now - _weights_last_update < WEIGHTS_UPDATE_INTERVAL and _weights_cache:
        return _weights_cache

    memory = load_signal_memory()
    weights = DEFAULT_WEIGHTS.copy()
    updated = []

    for sig in DEFAULT_WEIGHTS.keys():
        key = f"{sig}_{regime}"
        if key not in memory: continue
        data = memory[key]
        if data["count"] < MIN_SAMPLES_FOR_WEIGHT: continue

        win_rate = data["wins"] / data["count"]
        avg_pnl  = data["total_pnl"] / data["count"]

        # Mapear win rate a peso: 40% → 0.5, 50% → 1.0, 65% → 1.5
        if win_rate >= 0.65:
            weight = 1.5
        elif win_rate >= 0.55:
            weight = 1.2
        elif win_rate >= 0.45:
            weight = 1.0
        elif win_rate >= 0.35:
            weight = 0.7
        else:
            weight = 0.5

        weights[sig] = round(weight, 2)
        updated.append(f"{sig}:{weight:.1f}(WR={win_rate*100:.0f}%,n={data['count']})")

    _weights_cache = weights
    _weights_last_update = now

    if _USE_PG:
        _pg_set("dynamic_weights", {"weights": weights, "updated": datetime.now().isoformat(), "regime": regime})
    else:
        with open(DYNAMIC_WEIGHTS_FILE, "w") as f:
            json.dump({"weights": weights, "updated": datetime.now().isoformat(), "regime": regime}, f, indent=2)

    if updated:
        log.info(f"  🧠 Pesos actualizados: {' | '.join(updated)}")
    return weights

def weighted_score(signals_dict, regime):
    """
    Calcula el score total usando pesos dinámicos en vez de +1 fijo.
    Retorna score ponderado (puede ser float) y score entero para comparaciones.
    """
    weights = recalculate_weights(regime)
    score_float = 0.0
    for sig, val in signals_dict.items():
        if sig == "rsi_div_val" or val == 0: continue
        w = weights.get(sig, 1.0)
        score_float += val * w

    score_int = round(score_float)
    return score_float, score_int

# ─────────────────────────────────────────
# REINFORCEMENT LEARNING — ajuste por racha
# ─────────────────────────────────────────
def rl_adjust_min_signals(state):
    """
    Si hay ≥3 pérdidas seguidas → subir MIN_SIGNALS_SIDEWAYS a 4.
    Si hay ≥3 ganancias seguidas → bajar MIN_SIGNALS_SIDEWAYS a 2.
    """
    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: trades = json.load(f)
        except: pass

    closed = [t for t in trades if t.get("pnl_pct") is not None]
    if len(closed) < RL_STREAK_THRESHOLD: return MIN_SIGNALS_SIDEWAYS

    recent = closed[-RL_STREAK_THRESHOLD:]
    all_losses = all(t.get("pnl_pct", 0) < 0 for t in recent)
    all_wins   = all(t.get("pnl_pct", 0) > 0 for t in recent)

    if all_losses:
        new_min = min(5, state.get("rl_min_signals", MIN_SIGNALS_SIDEWAYS) + 1)
        if new_min != state.get("rl_min_signals", MIN_SIGNALS_SIDEWAYS):
            state["rl_min_signals"] = new_min
            log.info(f"  🔴 RL: {RL_STREAK_THRESHOLD} pérdidas seguidas → min_signals={new_min}")
            send_telegram(f"🤖 <b>RL ajuste</b>\n{RL_STREAK_THRESHOLD} pérdidas seguidas\nMin signals: {new_min}")
        return new_min
    elif all_wins:
        new_min = max(2, state.get("rl_min_signals", MIN_SIGNALS_SIDEWAYS) - 1)
        if new_min != state.get("rl_min_signals", MIN_SIGNALS_SIDEWAYS):
            state["rl_min_signals"] = new_min
            log.info(f"  🟢 RL: {RL_STREAK_THRESHOLD} ganancias seguidas → min_signals={new_min}")
        return new_min

    return state.get("rl_min_signals", MIN_SIGNALS_SIDEWAYS)

# ─────────────────────────────────────────
# ENTRY CONFIRMATION DELAY
# ─────────────────────────────────────────
_pending_entries = {}  # symbol → {signals, timestamp, df_snapshot}

def check_entry_confirmation(symbol, signals_dict, df, timeframe):
    """
    Espera confirmación antes de entrar, pero con tiempos razonables:
    - 3m  → espera 3 min  (1 vela completa)
    - 1h  → espera 5 min  (no 1h completa — demasiado conservador)
    - 15m → espera 5 min
    Si la señal sigue activa tras la espera, ejecuta. Si no, cancela.
    """
    now = time.time()
    # Tiempos de espera razonables — no bloquear 1h para entrar en 1h
    confirm_wait = {"1m": 60, "3m": 180, "5m": 180, "15m": 300, "1h": 300}
    wait_time = confirm_wait.get(timeframe, 180)

    if symbol not in _pending_entries:
        _pending_entries[symbol] = {"time": now, "signals": signals_dict.copy()}
        log.info(f"  ⏳ Entrada pendiente {symbol} — esperando {wait_time}s de confirmación")
        return False  # no entrar todavía

    pending = _pending_entries[symbol]
    elapsed = now - pending["time"]

    if elapsed < wait_time:
        log.info(f"  ⏳ {symbol}: {elapsed:.0f}s / {wait_time}s — confirmando...")
        return False

    # Ya pasó el tiempo — verificar que la señal general sigue siendo válida
    # Comparamos dirección del score, no señal técnica exacta (más flexible)
    prev_direction = 1 if sum(v for k,v in pending["signals"].items() if k != "rsi_div_val") > 0 else -1
    cur_macd  = macd_signal(df)
    cur_t_sig = technical_signal(df)
    # Confirmar si al menos 1 señal principal persiste en la misma dirección
    still_valid = (
        (cur_macd == prev_direction) or
        (cur_t_sig == prev_direction) or
        (pending["signals"].get("ob", 0) == prev_direction and pending["signals"].get("funding", 0) == prev_direction)
    )
    if still_valid:
        log.info(f"  ✅ Confirmado {symbol} — dirección persiste tras {elapsed:.0f}s")
        del _pending_entries[symbol]
        return True
    else:
        log.info(f"  ❌ {symbol}: señal no confirmada — cancelando")
        del _pending_entries[symbol]
        return False

# ─────────────────────────────────────────
# STATE
# ─────────────────────────────────────────
def load_state():
    default = {
        "capital": CAPITAL_TOTAL_USD,
        "week_start_capital": CAPITAL_TOTAL_USD,
        "week_start_date": datetime.now(ARG_TZ).strftime("%Y-%m-%d"),
        "drawdown_mode": False,
        "last_backtest": None,
        "rl_min_signals": MIN_SIGNALS_SIDEWAYS,
    }
    if _USE_PG:
        return _pg_get("bot_state", default)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except: pass
    return default

def save_state(state):
    if _USE_PG:
        _pg_set("bot_state", state)
    else:
        with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2, default=str)

def get_effective_capital(state):
    cap = state.get("capital", CAPITAL_TOTAL_USD)
    if state.get("drawdown_mode"):
        cap = cap * DRAWDOWN_REDUCE_PCT
        log.info(f"  ⚠️ DRAWDOWN MODE: capital reducido a ${cap:.2f}")
    return cap

def update_compounding(state, pnl_usd):
    state["capital"] = round(state["capital"] + pnl_usd, 2)
    save_state(state)
    log.info(f"  💰 Capital: ${state['capital']:.2f} ({'+' if pnl_usd>=0 else ''}{pnl_usd:.2f})")

def check_drawdown(state):
    week_start = state.get("week_start_capital", CAPITAL_TOTAL_USD)
    current    = state.get("capital", CAPITAL_TOTAL_USD)
    loss_pct   = (week_start - current) / week_start if week_start > 0 else 0
    now_arg    = datetime.now(ARG_TZ)

    # ── Reset semanal (lunes) ──
    if now_arg.weekday() == 0:
        week_date = now_arg.strftime("%Y-%m-%d")
        if state.get("week_start_date") != week_date:
            state["week_start_capital"] = current
            state["week_start_date"]    = week_date
            state["drawdown_mode"]      = False
            state["daily_drawdown_mode"] = False
            state["day_start_capital"]  = current
            state["day_start_date"]     = now_arg.strftime("%Y-%m-%d")
            state["rl_min_signals"]     = MIN_SIGNALS_SIDEWAYS
            log.info(f"📅 Reset semanal — capital base: ${current:.2f}")

    # ── Reset diario ──
    today = now_arg.strftime("%Y-%m-%d")
    if state.get("day_start_date") != today:
        state["day_start_capital"] = current
        state["day_start_date"]    = today
        state["daily_drawdown_mode"] = False
        log.info(f"📅 Reset diario — capital base: ${current:.2f}")

    # ── Drawdown diario > 3% → pausar hasta mañana ──
    day_start = state.get("day_start_capital", current)
    daily_loss = (day_start - current) / day_start if day_start > 0 else 0
    if daily_loss > 0.03 and not state.get("daily_drawdown_mode"):
        state["daily_drawdown_mode"] = True
        send_telegram(f"🛑 <b>DAILY CIRCUIT BREAKER</b>\nPérdida diaria: {daily_loss*100:.1f}%\nSin nuevas entradas hasta mañana")
        log.info(f"🛑 Circuit breaker diario — pérdida {daily_loss*100:.1f}%")
    elif daily_loss <= 0.01 and state.get("daily_drawdown_mode"):
        state["daily_drawdown_mode"] = False

    # ── Drawdown semanal ──
    if loss_pct > MAX_WEEKLY_LOSS_PCT and not state.get("drawdown_mode"):
        state["drawdown_mode"] = True
        send_telegram(f"⚠️ <b>DRAWDOWN MODE</b>\nPérdida semanal: {loss_pct*100:.1f}%\nSizing reducido 50%")
    elif loss_pct <= MAX_WEEKLY_LOSS_PCT * 0.5 and state.get("drawdown_mode"):
        state["drawdown_mode"] = False
    save_state(state)


def trading_hours_filter() -> bool:
    """No operar entre 00:00 y 06:00 UTC — volumen bajo, spreads amplios."""
    hour_utc = datetime.utcnow().hour
    if 0 <= hour_utc < 6:
        log.info(f"  ⏭️  Hora UTC {hour_utc:02d}:xx — fuera de horario (00-06 UTC)")
        return False
    return True


def losing_positions_filter(open_positions: dict) -> bool:
    """No abrir nuevas posiciones si ya hay 2 o más en pérdida simultánea."""
    losing = [s for s, p in open_positions.items()
              if p.get("current_price") and p.get("entry_price") and
              ((p["action"] == "BUY"  and p["current_price"] < p["entry_price"]) or
               (p["action"] == "SELL" and p["current_price"] > p["entry_price"]))]
    if len(losing) >= 2:
        log.info(f"  ⏭️  {len(losing)} posiciones en pérdida ({', '.join(losing)}) — no abrir más")
        return False
    return True

# ─────────────────────────────────────────
# FLASK DASHBOARD
# ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CryptoBot v10</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
:root{
  --bg:#050911;--s1:#090f1c;--s2:#0d1525;--s3:#111b30;
  --border:#19273d;--border2:#1f3050;
  --g:#00f5a0;--g2:rgba(0,245,160,.1);--g3:rgba(0,245,160,.04);
  --r:#ff3d5a;--r2:rgba(255,61,90,.1);
  --y:#f5c518;--b:#3db8f5;--p:#a78bfa;--o:#f97316;--c:#22d3ee;
  --t:#7a9bb8;--t2:#3d5a75;--w:#d8eeff;
  --mono:'DM Mono',monospace;--sans:'Syne',sans-serif;
  --rad:8px;--rad-sm:5px;
}
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--t);font-family:var(--mono);font-size:12px;min-height:100vh;
  background-image:
    radial-gradient(ellipse 100% 60% at 10% 0%,rgba(0,245,160,.04) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 80%,rgba(61,184,245,.03) 0%,transparent 50%)}

/* ── LAYOUT ── */
.wrap{max-width:1500px;margin:0 auto;padding:18px 20px}
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}
.logo{font-family:var(--sans);font-size:22px;font-weight:800;color:var(--w);letter-spacing:-1px}
.logo sup{font-size:11px;color:var(--g);font-weight:600;letter-spacing:0;margin-left:3px;
  border:1px solid rgba(0,245,160,.3);border-radius:100px;padding:1px 6px;vertical-align:top;margin-top:4px}
.badges{display:flex;gap:6px;align-items:center}
.badge{font-size:10px;padding:3px 9px;border-radius:100px;border:1px solid;font-family:var(--mono);font-weight:500}
.badge-live{color:var(--g);border-color:rgba(0,245,160,.3);background:rgba(0,245,160,.06)}
.badge-paper{color:var(--b);border-color:rgba(61,184,245,.3);background:rgba(61,184,245,.05)}
.dot{width:5px;height:5px;border-radius:50%;background:currentColor;display:inline-block;animation:blink 2s infinite;margin-right:3px}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.timer{font-size:10px;color:var(--t2);padding:3px 8px;background:var(--s1);border:1px solid var(--border);border-radius:var(--rad-sm)}
.sub{font-size:10px;color:var(--t2);margin-bottom:16px;display:flex;gap:14px;flex-wrap:wrap}
.sub-item::before{content:'·';margin-right:6px;color:var(--border2)}
.sub-item:first-child::before{display:none}

/* ── KPIs ── */
.kpis{display:grid;grid-template-columns:repeat(8,1fr);gap:6px;margin-bottom:14px}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(4,1fr)}}
@media(max-width:600px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--s1);border:1px solid var(--border);border-radius:var(--rad);padding:11px 13px;position:relative;overflow:hidden;transition:border-color .2s}
.kpi:hover{border-color:var(--border2)}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--ka,var(--g));opacity:.6}
.kpi-lbl{font-size:9px;color:var(--t2);letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px}
.kpi-val{font-family:var(--sans);font-weight:700;font-size:20px;color:var(--w);line-height:1.1}
.kpi-val.g{color:var(--g)}.kpi-val.r{color:var(--r)}.kpi-val.y{color:var(--y)}
.kpi-val.b{color:var(--b)}.kpi-val.p{color:var(--p)}.kpi-val.o{color:var(--o)}
.kpi-sub{font-size:10px;color:var(--t2);margin-top:2px}

/* ── PANELS ── */
.panel{background:var(--s1);border:1px solid var(--border);border-radius:var(--rad);overflow:hidden;margin-bottom:10px}
.ph{display:flex;justify-content:space-between;align-items:center;padding:8px 13px;background:var(--s2);border-bottom:1px solid var(--border)}
.ph-t{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--t2);font-weight:600}
.ph-s{font-size:10px;color:var(--t2)}

/* ── GRID ── */
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.grid2-3{display:grid;grid-template-columns:2fr 1fr;gap:10px}
@media(max-width:900px){.grid3,.grid2,.grid2-3{grid-template-columns:1fr}}

/* ── CAPITAL CHART ── */
.chart-wrap{padding:10px 13px;height:120px;position:relative}

/* ── TRADES WINS/LOSS ── */
.trades-split{display:grid;grid-template-columns:1fr 1fr;gap:0}
.trades-side{padding:10px 13px}
.trades-side.wins{border-right:1px solid var(--border)}
.side-title{font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--t2);margin-bottom:8px;display:flex;align-items:center;gap:5px}
.side-dot{width:6px;height:6px;border-radius:50%}
.trade-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(25,39,61,.5);font-size:11px}
.trade-row:last-child{border-bottom:none}
.trade-sym{color:var(--w);font-weight:600;font-family:var(--sans);font-size:11px}
.trade-meta{color:var(--t2);font-size:10px}
.trade-pnl{font-weight:600;font-size:12px;text-align:right}
.trade-pnl.win{color:var(--g)}.trade-pnl.loss{color:var(--r)}
.trade-usd{font-size:10px;text-align:right}
.trade-usd.win{color:rgba(0,245,160,.6)}.trade-usd.loss{color:rgba(255,61,90,.5)}

/* ── DYNAMIC WEIGHTS ── */
.weights-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px;padding:10px 13px}
.w-item{background:var(--s2);border:1px solid var(--border);border-radius:var(--rad-sm);padding:9px 10px;transition:border-color .2s}
.w-item:hover{border-color:var(--border2)}
.w-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.w-name{font-size:10px;color:var(--t2);text-transform:uppercase;letter-spacing:.07em}
.w-val{font-family:var(--sans);font-weight:700;font-size:14px}
.w-track{height:4px;background:var(--border);border-radius:2px;overflow:hidden;position:relative}
.w-fill{height:100%;border-radius:2px;transition:width .5s ease}
.w-baseline{position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border2)}

/* ── RL PANEL ── */
.rl-wrap{padding:12px 13px}
.rl-score-display{display:flex;align-items:center;gap:16px;margin-bottom:14px}
.rl-num{font-family:var(--sans);font-weight:800;font-size:48px;line-height:1;color:var(--w)}
.rl-info{flex:1}
.rl-label{font-size:10px;color:var(--t2);text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px}
.rl-desc{font-size:11px;color:var(--t)}
.rl-track{height:6px;background:var(--border);border-radius:3px;overflow:hidden;margin-bottom:6px}
.rl-fill{height:100%;border-radius:3px;transition:width .5s}
.rl-scale{display:flex;justify-content:space-between;font-size:9px;color:var(--t2)}
.rl-history{display:flex;gap:3px;margin-top:10px}
.rl-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}
.streak-info{display:flex;gap:10px;margin-top:10px}
.streak-badge{flex:1;background:var(--s2);border:1px solid var(--border);border-radius:var(--rad-sm);padding:7px 9px;text-align:center}
.streak-num{font-family:var(--sans);font-weight:700;font-size:18px}
.streak-lbl{font-size:9px;color:var(--t2);text-transform:uppercase;margin-top:2px}

/* ── HEATMAP SEÑALES ── */
.heatmap{display:grid;grid-template-columns:repeat(auto-fill,minmax(72px,1fr));gap:4px;padding:10px 13px}
.sig-cell{background:var(--s2);border:1px solid var(--border);border-radius:var(--rad-sm);padding:7px 8px;text-align:center;transition:all .15s}
.sig-cell:hover{border-color:var(--border2)}
.sig-cell.hot{border-color:rgba(0,245,160,.25);background:rgba(0,245,160,.04)}
.sig-cell.cold{border-color:rgba(255,61,90,.15);background:rgba(255,61,90,.03)}
.sig-name{font-size:9px;color:var(--t2);margin-bottom:3px;text-transform:uppercase;letter-spacing:.06em}
.sig-n{font-family:var(--sans);font-weight:700;font-size:15px;color:var(--w)}
.sig-wr{font-size:10px;margin-top:2px}

/* ── POSITIONS ── */
.pos-list{display:flex;flex-direction:column;gap:6px;padding:8px}
.pos-card{border-radius:var(--rad-sm);padding:10px 11px;border:1px solid var(--border);background:var(--s2);transition:border-color .2s}
.pos-card:hover{border-color:var(--border2)}
.pos-card.long{border-left:3px solid var(--g)}.pos-card.short{border-left:3px solid var(--r)}
.pos-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.pos-sym{font-family:var(--sans);font-weight:700;font-size:13px;color:var(--w)}
.pos-dir{font-size:8px;padding:1px 5px;border-radius:2px;margin-left:4px;font-weight:600}
.pos-dir.long{background:rgba(0,245,160,.12);color:var(--g)}.pos-dir.short{background:rgba(255,61,90,.12);color:var(--r)}
.pos-pnl{font-family:var(--sans);font-weight:700;font-size:14px}
.pos-pnl.pos{color:var(--g)}.pos-pnl.neg{color:var(--r)}
.pos-meta{display:grid;grid-template-columns:1fr 1fr;gap:3px;font-size:10px}
.lbl{color:var(--t2)}.val{color:var(--t);text-align:right}
.pos-bar{height:3px;background:var(--border);border-radius:2px;margin-top:7px;overflow:hidden}
.pos-bar-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--g),var(--b));transition:width .5s}
.pos-time{font-size:9px;color:var(--t2);margin-top:4px}

/* ── FILTERS ── */
.filter-row{padding:8px 13px;display:flex;flex-direction:column;gap:5px}
.f-item{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid rgba(25,39,61,.5)}
.f-item:last-child{border-bottom:none}
.f-label{font-size:10px;color:var(--t2)}.f-val{font-size:10px}

/* ── SCANNER ── */
.scanner-row{padding:8px 13px;display:flex;flex-wrap:wrap;gap:4px}
.chip{padding:2px 8px;border-radius:3px;font-size:10px;border:1px solid var(--border);color:var(--t2);transition:all .15s}
.chip:hover{border-color:var(--border2);color:var(--t)}
.chip.base{border-color:rgba(0,245,160,.2);color:rgba(0,245,160,.7)}

/* ── LOG ── */
#live-log{padding:8px 13px;font-size:10px;max-height:190px;overflow-y:auto;display:flex;flex-direction:column;gap:2px;line-height:1.5}
#live-log::-webkit-scrollbar{width:3px}
#live-log::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

/* ── TABLE ── */
.tbl-wrap{overflow-x:auto}
.tbl{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono)}
.tbl th{padding:7px 10px;font-size:8px;letter-spacing:.1em;text-transform:uppercase;color:var(--t2);background:var(--s2);text-align:left;white-space:nowrap}
.tbl td{padding:7px 10px;border-bottom:1px solid rgba(25,39,61,.4);vertical-align:middle;white-space:nowrap}
.tbl tr:hover td{background:rgba(255,255,255,.015)}
.tbl tr.win-row td{border-left:2px solid rgba(0,245,160,.3)}
.tbl tr.loss-row td{border-left:2px solid rgba(255,61,90,.2)}
.bdg{padding:1px 6px;border-radius:2px;font-size:10px;font-weight:600}
.bdg-buy{background:rgba(0,245,160,.1);color:var(--g)}.bdg-sell{background:rgba(255,61,90,.1);color:var(--r)}
.mini{font-size:8px;padding:0 3px;border-radius:2px;margin-left:2px}
.mini-p{background:rgba(61,184,245,.15);color:var(--b)}
.mini-f{background:rgba(167,139,250,.15);color:var(--p)}
.mini-h{background:rgba(249,115,22,.15);color:var(--o)}
.bar-wrap{display:flex;align-items:center;gap:4px;min-width:70px}
.bar-bg{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
.bar-fg{height:100%;border-radius:2px;background:var(--b)}
.pair{color:var(--w);font-weight:600}
.ts{color:var(--t2);font-size:10px}
.empty-row{text-align:center;padding:30px;color:var(--t2)}
footer{text-align:center;font-size:9px;color:var(--t2);margin-top:14px;padding-top:10px;border-top:1px solid var(--border);letter-spacing:.04em}
</style>
</head>
<body>
<div class="wrap">

<!-- HEADER -->
<div class="hdr">
  <div>
    <div class="logo">CryptoBot<sup>v10</sup></div>
  </div>
  <div class="badges">
    <span class="badge badge-live"><span class="dot"></span> LIVE</span>
    <span id="mode-pill" class="badge badge-paper" id="mode-txt">PAPER</span>
    <span class="timer">↻ <span id="cd">15</span>s</span>
  </div>
</div>
<div class="sub">
  <span class="sub-item">Aggressive Self-Learning</span>
  <span class="sub-item">Dynamic Weights</span>
  <span class="sub-item">RL Min Score</span>
  <span class="sub-item">ATR Trailing</span>
  <span class="sub-item" id="sub-regime">régimen —</span>
</div>

<!-- KPIs -->
<div class="kpis">
  <div class="kpi" style="--ka:var(--g)">
    <div class="kpi-lbl">Capital</div>
    <div class="kpi-val g" id="k-cap">—</div>
    <div class="kpi-sub" id="k-cap-d">—</div>
  </div>
  <div class="kpi" style="--ka:var(--y)">
    <div class="kpi-lbl">P&L Total</div>
    <div class="kpi-val" id="k-pnl">—</div>
    <div class="kpi-sub">en trades cerrados</div>
  </div>
  <div class="kpi" style="--ka:var(--b)">
    <div class="kpi-lbl">Win Rate</div>
    <div class="kpi-val b" id="k-wr">—</div>
    <div class="kpi-sub" id="k-wr-d">sin datos</div>
  </div>
  <div class="kpi" style="--ka:var(--c)">
    <div class="kpi-lbl">Expectancy</div>
    <div class="kpi-val" id="k-exp">—</div>
    <div class="kpi-sub">por trade</div>
  </div>
  <div class="kpi" style="--ka:var(--p)">
    <div class="kpi-lbl">Trades</div>
    <div class="kpi-val p" id="k-trades">—</div>
    <div class="kpi-sub" id="k-trades-d">—</div>
  </div>
  <div class="kpi" style="--ka:var(--r)">
    <div class="kpi-lbl">Fear & Greed</div>
    <div class="kpi-val" id="k-fg">—</div>
    <div class="kpi-sub" id="k-fg-l">—</div>
  </div>
  <div class="kpi" style="--ka:var(--o)">
    <div class="kpi-lbl">Posiciones</div>
    <div class="kpi-val o" id="k-pos">0</div>
    <div class="kpi-sub" id="k-cap-u">$0 usado</div>
  </div>
  <div class="kpi" style="--ka:var(--p)">
    <div class="kpi-lbl">RL Min Score</div>
    <div class="kpi-val p" id="k-rl">—</div>
    <div class="kpi-sub">ajuste automático</div>
  </div>
</div>

<!-- ROW 1: Capital chart + Wins/Losses -->
<div class="grid2-3">
  <div class="panel">
    <div class="ph"><span class="ph-t">Curva de capital</span><span class="ph-s" id="chart-stats">—</span></div>
    <div class="chart-wrap"><canvas id="pnl-chart"></canvas></div>
  </div>
  <div class="panel">
    <div class="ph"><span class="ph-t">Performance</span></div>
    <div style="padding:10px 13px;display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div style="background:var(--s2);border:1px solid rgba(0,245,160,.15);border-radius:var(--rad-sm);padding:9px 10px">
        <div style="font-size:9px;color:var(--t2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Mejor trade</div>
        <div class="kpi-val g" style="font-size:16px" id="m-best">—</div>
      </div>
      <div style="background:var(--s2);border:1px solid rgba(255,61,90,.12);border-radius:var(--rad-sm);padding:9px 10px">
        <div style="font-size:9px;color:var(--t2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Peor trade</div>
        <div class="kpi-val r" style="font-size:16px" id="m-worst">—</div>
      </div>
      <div style="background:var(--s2);border:1px solid var(--border);border-radius:var(--rad-sm);padding:9px 10px">
        <div style="font-size:9px;color:var(--t2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Avg Win</div>
        <div class="kpi-val g" style="font-size:16px" id="m-avgw">—</div>
      </div>
      <div style="background:var(--s2);border:1px solid var(--border);border-radius:var(--rad-sm);padding:9px 10px">
        <div style="font-size:9px;color:var(--t2);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Avg Loss</div>
        <div class="kpi-val r" style="font-size:16px" id="m-avgl">—</div>
      </div>
    </div>
  </div>
</div>

<!-- ROW 2: Wins/Losses lado a lado -->
<div class="panel">
  <div class="ph"><span class="ph-t">Trades recientes — Ganados vs Perdidos</span><span class="ph-s" id="wl-count">—</span></div>
  <div class="trades-split">
    <div class="trades-side wins">
      <div class="side-title"><span class="side-dot" style="background:var(--g)"></span>Ganados</div>
      <div id="wins-list"></div>
    </div>
    <div class="trades-side">
      <div class="side-title"><span class="side-dot" style="background:var(--r)"></span>Perdidos</div>
      <div id="losses-list"></div>
    </div>
  </div>
</div>

<!-- ROW 3: Dynamic Weights + RL -->
<div class="grid2">
  <div class="panel">
    <div class="ph">
      <span class="ph-t">Dynamic Weights</span>
      <span class="ph-s">ajuste por win rate · base = 1.0x</span>
    </div>
    <div class="weights-grid" id="weights-row"></div>
  </div>
  <div class="panel">
    <div class="ph"><span class="ph-t">Reinforcement Learning</span><span class="ph-s">ajuste por racha</span></div>
    <div class="rl-wrap">
      <div class="rl-score-display">
        <div class="rl-num" id="rl-big">3</div>
        <div class="rl-info">
          <div class="rl-label">Min Score actual</div>
          <div class="rl-desc" id="rl-desc">—</div>
        </div>
      </div>
      <div class="rl-track">
        <div class="rl-fill" id="rl-fill" style="width:40%;background:var(--b)"></div>
      </div>
      <div class="rl-scale"><span>1 (agresivo)</span><span>3 (normal)</span><span>5 (conservador)</span></div>
      <div class="streak-info">
        <div class="streak-badge">
          <div class="streak-num g" id="rl-wins" style="color:var(--g)">—</div>
          <div class="streak-lbl">racha wins</div>
        </div>
        <div class="streak-badge">
          <div class="streak-num r" id="rl-losses" style="color:var(--r)">—</div>
          <div class="streak-lbl">racha losses</div>
        </div>
        <div class="streak-badge">
          <div class="streak-num" id="rl-total" style="color:var(--t)">—</div>
          <div class="streak-lbl">trades totales</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ROW 4: Señales + Posiciones -->
<div class="grid2">
  <div>
    <div class="panel">
      <div class="ph"><span class="ph-t">Performance por señal</span><span class="ph-s">últimos 20 trades</span></div>
      <div class="heatmap" id="sig-heatmap"></div>
    </div>
    <div class="panel">
      <div class="ph"><span class="ph-t">Filtros activos</span><span class="ph-s" id="filter-time">—</span></div>
      <div class="filter-row" id="filter-status"></div>
    </div>
  </div>
  <div>
    <div class="panel">
      <div class="ph"><span class="ph-t">Posiciones abiertas</span><span class="ph-s" id="pos-count">ninguna</span></div>
      <div id="pos-list" class="pos-list">
        <div style="padding:20px;text-align:center;color:var(--t2)">Sin posiciones abiertas</div>
      </div>
    </div>
    <div class="panel">
      <div class="ph"><span class="ph-t">Scanner activo</span></div>
      <div class="scanner-row" id="scanner-row"></div>
    </div>
  </div>
</div>

<!-- LOG -->
<div class="panel">
  <div class="ph"><span class="ph-t">Actividad en vivo</span></div>
  <div id="live-log"></div>
</div>

<!-- TABLE -->
<div class="panel">
  <div class="ph"><span class="ph-t">Historial completo</span><span class="ph-s" id="tbl-count">—</span></div>
  <div class="tbl-wrap">
    <table class="tbl">
      <thead><tr>
        <th>Par</th><th>P&L %</th><th>P&L $</th><th>TF</th>
        <th>Precio</th><th>Size</th><th>Tipo</th>
        <th>Score</th><th>Conf</th><th>Señales</th><th>Hora</th>
      </tr></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
</div>

<footer>CryptoBot v10 · Aggressive Self-Learning · Dynamic Weights · RL · ATR Trailing · Shorts · Entry Confirmation</footer>
</div>

<script>
const ICAP = 1000;
let cd = 15;
let capChart = null;

function fmt(n, d=2){ return (+n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}); }
function fmtP(n){ return (n>=0?'+':'')+fmt(n)+'%'; }
function fmtUSD(n){ return (n>=0?'+':'-')+'$'+fmt(Math.abs(n)); }

function renderChart(canvas, pts) {
  if(pts.length < 2){ canvas.style.display='none'; return; }
  canvas.style.display='block';
  const last = pts[pts.length-1];
  const color = last >= ICAP ? '0,245,160' : '255,61,90';
  const hex = last >= ICAP ? '#00f5a0' : '#ff3d5a';
  if(capChart) capChart.destroy();
  capChart = new Chart(canvas, {
    type:'line',
    data:{
      labels: pts.map((_,i)=>i),
      datasets:[{
        data: pts, borderColor: hex, borderWidth: 1.5,
        backgroundColor: `rgba(${color},.07)`,
        fill:true, tension:.35, pointRadius:0, pointHoverRadius:3,
        pointHoverBackgroundColor: hex
      }]
    },
    options:{
      responsive:true, maintainAspectRatio:false, animation:{duration:300},
      plugins:{legend:{display:false}, tooltip:{
        callbacks:{label: ctx => '$'+fmt(ctx.parsed.y)},
        backgroundColor:'rgba(9,15,28,.95)', borderColor:'rgba(31,48,80,.8)', borderWidth:1,
        titleFont:{family:'DM Mono',size:10}, bodyFont:{family:'DM Mono',size:11}
      }},
      scales:{
        x:{display:false},
        y:{grid:{color:'rgba(25,39,61,.5)',drawBorder:false},
           ticks:{color:'#3d5a75',font:{family:'DM Mono',size:10},callback:v=>'$'+v.toLocaleString()},
           border:{display:false}}
      }
    }
  });
}

async function load(){
  try{
    const d = await fetch('/api/trades').then(r=>r.json());
    const st = d.state||{};
    const trades = (d.trades||[]).filter(t=>t.action!=='HOLD');
    const closed = trades.filter(t=>t.pnl_pct!==undefined && !t.partial_exit);
    const wins   = closed.filter(t=>t.pnl_pct>0);
    const losses = closed.filter(t=>t.pnl_pct<=0);
    const cap    = st.capital||ICAP;
    const diff   = cap-ICAP;

    // ── KPIs ──
    document.getElementById('k-cap').textContent='$'+fmt(cap);
    document.getElementById('k-cap').className='kpi-val '+(cap>=ICAP?'g':'r');
    document.getElementById('k-cap-d').textContent=(diff>=0?'+':'')+fmt(diff)+' desde inicio';

    const tpnl = closed.reduce((s,t)=>s+(t.pnl_pct||0)*(t.usd_size||20)/100,0);
    document.getElementById('k-pnl').textContent = fmtUSD(tpnl);
    document.getElementById('k-pnl').className = 'kpi-val '+(tpnl>=0?'g':'r');

    const wr = closed.length ? Math.round(wins.length/closed.length*100) : null;
    document.getElementById('k-wr').textContent  = wr!==null ? wr+'%' : '—';
    document.getElementById('k-wr').className    = 'kpi-val '+(wr===null?'b':wr>=50?'b':'r');
    document.getElementById('k-wr-d').textContent= closed.length ? wins.length+'/'+closed.length+' trades':'sin datos';

    // Expectancy
    if(closed.length){
      const avgW = wins.length  ? wins.reduce((s,t)=>s+t.pnl_pct,0)/wins.length   : 0;
      const avgL = losses.length? Math.abs(losses.reduce((s,t)=>s+t.pnl_pct,0)/losses.length) : 0;
      const exp  = (wins.length/closed.length)*avgW - (losses.length/closed.length)*avgL;
      document.getElementById('k-exp').textContent = fmtP(exp);
      document.getElementById('k-exp').className   = 'kpi-val '+(exp>=0?'g':'r');
    }

    document.getElementById('k-trades').textContent   = trades.length;
    document.getElementById('k-trades-d').textContent = closed.length+' cerrados';

    if(d.fear_greed){
      const fv = +d.fear_greed.value;
      document.getElementById('k-fg').textContent = fv;
      document.getElementById('k-fg').className   = 'kpi-val '+(fv<35?'r':fv>65?'g':'y');
      document.getElementById('k-fg-l').textContent = d.fear_greed.label;
    }

    const pos = d.positions||[];
    document.getElementById('k-pos').textContent = pos.length;
    const alloc = pos.reduce((s,p)=>s+(p.usd_size||0),0);
    document.getElementById('k-cap-u').textContent = '$'+fmt(alloc)+' usado';

    const rlMin = st.rl_min_signals||3;
    document.getElementById('k-rl').textContent = rlMin;

    const rm = {bull:'BULL 📈',bear:'BEAR 📉',sideways:'SIDE ↔',crash:'CRASH 💥'};
    const reg = d.regime||'?';
    document.getElementById('sub-regime').textContent = 'régimen '+(rm[reg]||reg);

    const hasReal = trades.some(t=>!t.paper);
    document.getElementById('mode-pill').textContent  = hasReal?'REAL':'PAPER';
    document.getElementById('mode-pill').className    = 'badge '+(hasReal?'badge-live':'badge-paper');

    // ── CHART ──
    const curve=[ICAP]; let run=ICAP;
    [...closed].sort((a,b)=>new Date(a.timestamp)-new Date(b.timestamp))
      .forEach(t=>{run+=(t.pnl_pct||0)*(t.usd_size||20)/100; curve.push(run);});
    renderChart(document.getElementById('pnl-chart'), curve);
    if(curve.length>1){
      document.getElementById('chart-stats').textContent=
        'min $'+fmt(Math.min(...curve))+' · max $'+fmt(Math.max(...curve))+' · '+closed.length+' pts';
    }

    // ── PERFORMANCE STATS ──
    if(closed.length){
      const pnls = closed.map(t=>t.pnl_pct||0);
      const wP   = wins.map(t=>t.pnl_pct);
      const lP   = losses.map(t=>t.pnl_pct);
      document.getElementById('m-best').textContent  = fmtP(Math.max(...pnls));
      document.getElementById('m-worst').textContent = fmtP(Math.min(...pnls));
      document.getElementById('m-avgw').textContent  = wP.length ? fmtP(wP.reduce((a,b)=>a+b,0)/wP.length) : '—';
      document.getElementById('m-avgl').textContent  = lP.length ? fmtP(lP.reduce((a,b)=>a+b,0)/lP.length) : '—';
    }

    // ── WINS/LOSSES SIDE BY SIDE ──
    const recent = [...closed].sort((a,b)=>new Date(b.timestamp)-new Date(a.timestamp)).slice(0,20);
    const recentW = recent.filter(t=>t.pnl_pct>0).slice(0,8);
    const recentL = recent.filter(t=>t.pnl_pct<=0).slice(0,8);
    document.getElementById('wl-count').textContent =
      wins.length+' ganados · '+losses.length+' perdidos de '+closed.length+' totales';

    function tradeRow(t){
      const usdPnl = (t.pnl_pct||0)*(t.usd_size||20)/100;
      const win    = t.pnl_pct>0;
      const ts     = new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      return `<div class="trade-row">
        <div>
          <div class="trade-sym">${t.symbol.replace('/USDT','')}</div>
          <div class="trade-meta">${ts} · $${t.usd_size||'—'}</div>
        </div>
        <div>
          <div class="trade-pnl ${win?'win':'loss'}">${fmtP(t.pnl_pct||0)}</div>
          <div class="trade-usd ${win?'win':'loss'}">${fmtUSD(usdPnl)}</div>
        </div>
      </div>`;
    }
    document.getElementById('wins-list').innerHTML   = recentW.length ? recentW.map(tradeRow).join('') : '<div style="color:var(--t2);font-size:10px;padding:8px 0">Sin ganados aún</div>';
    document.getElementById('losses-list').innerHTML = recentL.length ? recentL.map(tradeRow).join('') : '<div style="color:var(--t2);font-size:10px;padding:8px 0">Sin pérdidas aún</div>';

    // ── DYNAMIC WEIGHTS ──
    const wnames = {
      tech:'EMA',macd:'MACD',bb:'BB',ob:'OB',rsi_div:'RSI▲',vol:'VOL',
      funding:'FR',news:'NEWS',tf4h:'CONF',vwap:'VWAP',supertrend:'STRND',
      stoch_rsi:'STOCH',williams_r:'WILLY',cci:'CCI',squeeze:'SQZ',
      support_res:'S/R',candle:'CNDLE',trend_struct:'TREND'
    };
    const wts = d.weights||{};
    document.getElementById('weights-row').innerHTML = Object.entries(wnames).map(([k,n])=>{
      const w = +(wts[k]||1.0);
      const isHigh = w >= 1.3, isLow = w <= 0.7, isNorm = !isHigh && !isLow;
      const color  = isHigh?'var(--g)':isLow?'var(--r)':'var(--t)';
      const bcolor = isHigh?'var(--g)':isLow?'var(--r)':'var(--b)';
      // Bar: 0x=0%, 1x=50%, 2x=100%
      const pct    = Math.min(100, Math.max(0, w/2*100));
      return `<div class="w-item">
        <div class="w-top">
          <span class="w-name">${n}</span>
          <span class="w-val" style="color:${color}">${w.toFixed(2)}x</span>
        </div>
        <div class="w-track">
          <div class="w-baseline"></div>
          <div class="w-fill" style="width:${pct}%;background:${bcolor}"></div>
        </div>
      </div>`;
    }).join('');

    // ── RL PANEL ──
    document.getElementById('rl-big').textContent = rlMin;
    const rlPct = Math.min(100, Math.max(0, (rlMin-1)/4*100));
    const rlColor = rlMin<=2?'var(--g)':rlMin>=4?'var(--r)':'var(--b)';
    document.getElementById('rl-fill').style.width = rlPct+'%';
    document.getElementById('rl-fill').style.background = rlColor;
    document.getElementById('rl-big').style.color = rlColor;
    const rlDescs = {1:'Agresivo — racha ganadora fuerte',2:'Relajado — racha ganadora',
                     3:'Normal — sin racha',4:'Conservador — racha perdedora',5:'Muy conservador — racha mala'};
    document.getElementById('rl-desc').textContent = rlDescs[rlMin]||'Estado desconocido';

    // Racha
    let winStreak=0, lossStreak=0;
    for(let i=closed.length-1;i>=0;i--){
      if(closed[i].pnl_pct>0&&lossStreak===0) winStreak++;
      else if(closed[i].pnl_pct<=0&&winStreak===0) lossStreak++;
      else break;
    }
    document.getElementById('rl-wins').textContent   = winStreak;
    document.getElementById('rl-losses').textContent = lossStreak;
    document.getElementById('rl-total').textContent  = closed.length;

    // ── HEATMAP SEÑALES ──
    const rec20 = closed.slice(-20);
    const sdefs = [
      {k:'tech',n:'EMA'},{k:'macd',n:'MACD'},{k:'bb',n:'BB'},{k:'ob',n:'OB'},
      {k:'vol',n:'VOL'},{k:'funding',n:'FR'},{k:'news',n:'NEWS'},{k:'rsi_div',n:'RSI▲'},
      {k:'tf4h',n:'CONF'},{k:'vwap',n:'VWAP'},{k:'supertrend',n:'STRND'},
      {k:'stoch_rsi',n:'STOCH'},{k:'williams_r',n:'WILLY'},{k:'cci',n:'CCI'},
      {k:'squeeze',n:'SQZ'},{k:'support_res',n:'S/R'},{k:'candle',n:'CNDLE'},
      {k:'trend_struct',n:'TREND'}
    ];
    document.getElementById('sig-heatmap').innerHTML = sdefs.map(s=>{
      const hits = rec20.filter(t=>t[s.k+'_signal']===1||t[s.k+'_signal']===-1||t[s.k]);
      const ws   = hits.filter(t=>(t.pnl_pct||0)>0);
      const sr   = hits.length ? Math.round(ws.length/hits.length*100) : null;
      const cls  = hits.length>=5?'hot':hits.length<=1?'cold':'';
      const wc   = sr===null?'var(--t2)':sr>=60?'var(--g)':sr<40?'var(--r)':'var(--y)';
      return `<div class="sig-cell ${cls}">
        <div class="sig-name">${s.n}</div>
        <div class="sig-n" style="color:${hits.length>=5?'var(--g)':'var(--w)'}">${hits.length}</div>
        <div class="sig-wr" style="color:${wc}">${sr!==null?sr+'%':'—'}</div>
      </div>`;
    }).join('');

    // ── POSITIONS ──
    document.getElementById('pos-count').textContent = pos.length||'ninguna';
    document.getElementById('pos-list').innerHTML = pos.length ? pos.map(p=>{
      const isS = p.action==='SELL';
      const pp  = isS?((p.entry_price-p.current_price)/p.entry_price*100):((p.current_price-p.entry_price)/p.entry_price*100);
      const isp = pp>=0;
      const oa  = p.opened_at ? new Date(p.opened_at) : null;
      const el  = oa ? Math.round((Date.now()-oa)/60000) : null;
      const dt  = isS?((p.trail_stop-p.current_price)/p.current_price*100):((p.current_price-p.trail_stop)/p.current_price*100);
      const bw  = Math.min(100,Math.max(0,(1-dt/5)*100));
      return `<div class="pos-card ${isS?'short':'long'}">
        <div class="pos-top">
          <span><span class="pos-sym">${p.symbol.replace('/USDT','')}</span><span class="pos-dir ${isS?'short':'long'}">${isS?'SHORT':'LONG'}</span></span>
          <span class="pos-pnl ${isp?'pos':'neg'}">${isp?'+':''}${pp.toFixed(2)}%</span>
        </div>
        <div class="pos-meta">
          <span class="lbl">Entrada</span><span class="val">${fmt(p.entry_price,4)}</span>
          <span class="lbl">Actual</span><span class="val">${fmt(p.current_price,4)}</span>
          <span class="lbl">Trail</span><span class="val" style="color:var(--o)">${fmt(p.trail_stop,4)}</span>
          <span class="lbl">TP</span><span class="val" style="color:var(--g)">${fmt(p.take_profit,4)}</span>
          <span class="lbl">Size</span><span class="val">$${p.usd_size}</span>
          <span class="lbl">Modo</span><span class="val" style="color:var(--p)">${p.mode||'SPOT'}</span>
        </div>
        ${p.partial_closed?'<div style="font-size:9px;color:var(--o);margin-top:4px">½ partial exit ejecutado</div>':''}
        <div class="pos-bar"><div class="pos-bar-fill" style="width:${bw}%"></div></div>
        ${el!==null?`<div class="pos-time">Hace ${el<60?el+'m':(el/60).toFixed(1)+'h'}</div>`:''}
      </div>`;
    }).join('') : '<div style="padding:20px;text-align:center;color:var(--t2);font-size:11px">Sin posiciones abiertas</div>';

    // ── SCANNER ──
    const sc = d.scanner||[];
    document.getElementById('scanner-row').innerHTML = sc.map(s=>{
      const b = ['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT'].includes(s.symbol);
      const v = s.volume?`<span style="color:var(--t2);font-size:9px"> $${Math.round(s.volume/1e6)}M</span>`:'';
      return `<span class="chip ${b?'base':'alt'}">${s.symbol.replace('/USDT','')}${v}</span>`;
    }).join('');

    // ── FILTERS ──
    const fl = d.filters||{};
    const now_utc = new Date().getUTCHours();
    const trading_hours = !(now_utc>=0 && now_utc<6);
    const filterDefs = [
      {label:'BTC 4h Macro',   ok:fl.btc_macro===true,  warn:fl.btc_macro===false,  okTxt:'Alcista ↑ OK',     warnTxt:'Bajista ↓ bloqueado'},
      {label:'Circuit Breaker',ok:!fl.daily_circuit,    warn:fl.daily_circuit,      okTxt:'OK — sin límite',  warnTxt:'🛑 ACTIVO — sin entradas'},
      {label:'Drawdown Semanal',ok:!fl.weekly_drawdown, warn:fl.weekly_drawdown,    okTxt:'OK — dentro límite',warnTxt:'⚠️ sizing reducido'},
      {label:'Horario (UTC)',   ok:trading_hours,        warn:!trading_hours,        okTxt:'Mercado activo',   warnTxt:'00-06 UTC — bloqueado'},
    ];
    document.getElementById('filter-status').innerHTML = filterDefs.map(f=>{
      const c = f.warn?'var(--r)':f.ok?'var(--g)':'var(--t2)';
      const dot = f.warn?'🔴':f.ok?'🟢':'⚪';
      return `<div class="f-item"><span class="f-label">${f.label}</span><span class="f-val" style="color:${c}">${dot} ${f.warn?f.warnTxt:f.ok?f.okTxt:'—'}</span></div>`;
    }).join('');
    document.getElementById('filter-time').textContent = 'UTC '+String(now_utc).padStart(2,'0')+':xx';

    // ── LOG ──
    try{
      const logs = await fetch('/api/log').then(r=>r.json());
      const logEl = document.getElementById('live-log');
      logEl.innerHTML = logs.slice(-20).map(l=>{
        const m = l.msg||'';
        const c = m.includes('ERROR')||m.includes('❌')?'var(--r)':
                  m.includes('✅')||m.includes('🟢')?'var(--g)':
                  m.includes('⏭️')||m.includes('skip')?'var(--t2)':
                  m.includes('⏳')||m.includes('confirmando')?'var(--y)':
                  m.includes('🔴 TRAIL')||m.includes('STOP')?'var(--o)':'var(--t)';
        return `<div style="color:${c}">${m}</div>`;
      }).join('');
      logEl.scrollTop = logEl.scrollHeight;
    }catch(e){}

    // ── TABLE ──
    const tb = document.getElementById('tbody');
    document.getElementById('tbl-count').textContent = trades.length ? trades.length+' trades' : '—';
    if(!trades.length){
      tb.innerHTML = '<tr><td colspan="11"><div class="empty-row">🤖 Sin trades aún</div></td></tr>';
      return;
    }
    tb.innerHTML = [...trades].reverse().map(t=>{
      const ts      = new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const win     = (t.pnl_pct||0) > 0;
      const hasResult = t.pnl_pct !== undefined;
      const usdPnl  = hasResult ? (t.pnl_pct||0)*(t.usd_size||20)/100 : null;
      const pnlPct  = hasResult ? `<span style="color:${win?'var(--g)':'var(--r)'};font-weight:700">${fmtP(t.pnl_pct||0)}</span>` : '<span style="color:var(--t2)">open</span>';
      const pnlUsd  = usdPnl!==null ? `<span style="color:${win?'var(--g)':'var(--r)'}">${fmtUSD(usdPnl)}</span>` : '—';
      const conf    = Math.round((t.confidence||0)*100);
      const sc_v    = (+(t.score_weighted||0)).toFixed(1);
      const sc_c    = +sc_v>0?'var(--g)':+sc_v<0?'var(--r)':'var(--t2)';
      const sigs = [];
      if(t.tech_signal===1)sigs.push('<span style="color:var(--g)">EMA+</span>');else if(t.tech_signal===-1)sigs.push('<span style="color:var(--r)">EMA-</span>');
      if(t.macd_signal===1)sigs.push('<span style="color:var(--g)">MACD+</span>');else if(t.macd_signal===-1)sigs.push('<span style="color:var(--r)">MACD-</span>');
      if(t.rsi_div)sigs.push('<span style="color:var(--y)">RSI▲</span>');
      if(t.ob_signal===1)sigs.push('<span style="color:var(--b)">OB+</span>');
      if(t.vol_signal===1)sigs.push('<span style="color:var(--p)">VOL+</span>');
      const rowClass = hasResult ? (win?'win-row':'loss-row') : '';
      return `<tr class="${rowClass}">
        <td class="pair">${t.symbol}</td>
        <td>${pnlPct}</td>
        <td>${pnlUsd}</td>
        <td style="color:${t.timeframe==='3m'?'var(--p)':'var(--t2)'}">${t.timeframe}</td>
        <td>${t.price?(+t.price).toLocaleString('en-US',{maximumFractionDigits:4}):'—'}</td>
        <td style="color:var(--o)">$${t.usd_size||'—'}</td>
        <td><span class="bdg bdg-${(t.action||'buy').toLowerCase()}">${t.action}</span>${t.paper?'<span class="mini mini-p">P</span>':''}${t.mode&&t.mode.includes('FUT')?'<span class="mini mini-f">F</span>':''}${t.partial_exit?'<span class="mini mini-h">½</span>':''}</td>
        <td style="color:${sc_c}">${sc_v}</td>
        <td><div class="bar-wrap"><div class="bar-bg"><div class="bar-fg" style="width:${conf}%"></div></div><span style="font-size:10px">${conf}%</span></div></td>
        <td style="font-size:10px">${sigs.join(' ')}</td>
        <td class="ts">${ts}</td>
      </tr>`;
    }).join('');
  }catch(e){ console.error(e); }
}

function tick(){ cd--; document.getElementById('cd').textContent=cd; if(cd<=0){cd=15;load();} }
load();
setInterval(tick, 1000);
window.addEventListener('resize', ()=>{ if(capChart) load(); });
</script>
</body>
</html>"""
flask_app   = Flask(__name__)
_fear_greed = {"value": 50, "label": "Neutral"}
_scanner    = []
_regime     = "unknown"

@flask_app.route("/")
def index(): return render_template_string(DASHBOARD_HTML)

# ── Live log buffer ──────────────────────────────────────────────────────────
import collections
_log_buffer = collections.deque(maxlen=30)

class LogBufferHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        # strip ANSI codes
        import re
        msg = re.sub(r'\[[0-9;]*m', '', msg)
        _log_buffer.append({"ts": record.created, "msg": msg[-200:]})

_log_buf_handler = LogBufferHandler()
_log_buf_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
logging.getLogger("bot").addHandler(_log_buf_handler)

@flask_app.route("/api/log")
def api_log():
    return jsonify(list(_log_buffer))

@flask_app.route("/api/trades")
def api_trades():
    trades = []
    if _USE_PG:
        try:
            with _pg.cursor() as cur:
                cur.execute("SELECT data FROM trades ORDER BY created_at ASC")
                trades = [row[0] for row in cur.fetchall()]
        except Exception as e:
            log.warning(f"PG api_trades error: {e}")
    elif os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: trades = json.load(f)
        except: pass
    positions = load_positions()
    state     = load_state()
    weights   = _weights_cache if _weights_cache else DEFAULT_WEIGHTS
    btc_macro = _btc_macro_cache.get("value")
    return jsonify({
        "trades": trades, "count": len(trades),
        "fear_greed": _fear_greed,
        "positions": list(positions.values()),
        "scanner": _scanner,
        "regime": _regime,
        "state": state,
        "weights": weights,
        "filters": {
            "btc_macro": btc_macro,          # True=alcista, False=bajista, None=desconocido
            "daily_circuit": state.get("daily_drawdown_mode", False),
            "weekly_drawdown": state.get("drawdown_mode", False),
        },
    })

@flask_app.route("/health")
def health(): return jsonify({"status": "ok", "paper": PAPER_TRADING})

def run_dashboard():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"🌐 Dashboard en puerto {port}")
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

# ─────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
        log.info("📱 Telegram enviado")
    except Exception as e: log.warning(f"Telegram error: {e}")

# ─────────────────────────────────────────
# EXCHANGES
# ─────────────────────────────────────────
def get_public_exchange():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})

def get_trade_exchange():
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET,
        "enableRateLimit": True, "options": {"defaultType": "spot"},
        "urls": {"api": {"public": "https://testnet.binance.vision/api/v3",
                         "private": "https://testnet.binance.vision/api/v3"}}
    })

def get_futures_exchange():
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET,
        "enableRateLimit": True, "options": {"defaultType": "future"},
        "urls": {"api": {"public": "https://testnet.binancefuture.com",
                         "private": "https://testnet.binancefuture.com"}}
    })

# ─────────────────────────────────────────
# MARKET REGIME
# ─────────────────────────────────────────
def detect_market_regime(public_ex):
    global _regime
    try:
        df = get_ohlcv(public_ex, "BTC/USDT", timeframe="1d", limit=30)
        df = calculate_indicators(df)
        last  = df.iloc[-1]; prev = df.iloc[-2]
        ema50  = df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = df["close"].ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 50 else ema50
        change_24h = (last["close"] - prev["close"]) / prev["close"]
        if change_24h <= REGIME_CRASH_THRESHOLD:       regime = "crash"
        elif ema50 > ema200 * (1 + REGIME_TREND_THRESHOLD): regime = "bull"
        elif ema50 < ema200 * (1 - REGIME_TREND_THRESHOLD): regime = "bear"
        else:                                           regime = "sideways"
        _regime = regime
        log.info(f"🧭 Régimen: {regime.upper()} | BTC 24h: {change_24h*100:+.2f}%")
        return regime
    except Exception as e:
        log.warning(f"Regime error: {e}")
        return "unknown"

def regime_filter(regime, action):
    if regime == "crash": return False
    return True

# ─────────────────────────────────────────
# RESUMEN DIARIO + BACKTEST SEMANAL
# ─────────────────────────────────────────
_last_daily_report    = None
_last_weekly_backtest = None

def maybe_send_daily_report(fg_value, fg_label):
    global _last_daily_report
    now_arg = datetime.now(ARG_TZ)
    today   = now_arg.date()
    if _last_daily_report == today or now_arg.hour != 9: return
    _last_daily_report = today
    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: trades = json.load(f)
        except: pass
    cutoff = now_arg - timedelta(hours=24)
    today_t = []
    for t in trades:
        try:
            ts = datetime.fromisoformat(t["timestamp"]).replace(tzinfo=timezone.utc).astimezone(ARG_TZ)
            if ts >= cutoff: today_t.append(t)
        except: pass
    closed  = [t for t in today_t if t.get("pnl_pct") is not None]
    winners = [t for t in closed if t.get("pnl_pct", 0) > 0]
    total_pnl = sum((t.get("pnl_pct", 0) * t.get("usd_size", 3)) / 100 for t in closed)
    win_rate  = round(len(winners) / len(closed) * 100) if closed else 0
    state     = load_state()
    weights   = _weights_cache if _weights_cache else DEFAULT_WEIGHTS
    top_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    msg = (
        f"📊 <b>Resumen {today.strftime('%d/%m/%Y')}</b>\n\n"
        f"💰 Capital: ${state.get('capital', 100):.2f}\n"
        f"📈 Trades: {len(today_t)} | ✅ Win rate: {win_rate}%\n"
        f"💵 P&L: {'+'if total_pnl>=0 else ''}${total_pnl:.2f}\n"
        f"🧠 Top señales: {', '.join(f'{k}={v:.1f}x' for k,v in top_w)}\n"
        f"🤖 RL min score: {state.get('rl_min_signals', MIN_SIGNALS_SIDEWAYS)}\n"
        f"{'⚠️ DRAWDOWN MODE' if state.get('drawdown_mode') else '✅ Normal'}\n"
        f"🧭 F&G: {fg_value} — {fg_label} | {_regime.upper()}"
    )
    send_telegram(msg)
    log.info("📊 Resumen diario enviado")

def maybe_run_weekly_backtest():
    global _last_weekly_backtest
    now_arg = datetime.now(ARG_TZ)
    if now_arg.weekday() != 6 or now_arg.hour != 10: return
    today = now_arg.date()
    if _last_weekly_backtest == today: return
    _last_weekly_backtest = today
    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: trades = json.load(f)
        except: pass
    closed = [t for t in trades if t.get("pnl_pct") is not None]
    if len(closed) < 5: return
    log.info("📊 Backtest semanal con Claude...")
    try:
        memory  = load_signal_memory()
        weights = _weights_cache if _weights_cache else DEFAULT_WEIGHTS
        summary = []
        for t in closed[-30:]:
            summary.append({
                "symbol": t.get("symbol"), "pnl_pct": t.get("pnl_pct"),
                "score_weighted": t.get("score_weighted"),
                "signals": {k: t.get(f"{k}_signal", 0) for k in ["tech","macd","bb","ob","vol","funding","news"]},
                "rsi_div": t.get("rsi_div", False), "regime": t.get("regime"),
                "timeframe": t.get("timeframe"), "fg": t.get("fear_greed_value"),
            })
        win_rate = round(len([t for t in closed if t.get("pnl_pct", 0) > 0]) / len(closed) * 100)
        prompt = f"""Sos un quant trader analizando un bot de crypto con aprendizaje automático.

Win rate actual: {win_rate}% ({len(closed)} trades)
Pesos actuales de señales: {json.dumps(weights)}
Memoria de señales: {json.dumps({k: v for k,v in memory.items() if v['count'] >= 3})}
Últimos 20 trades: {json.dumps(summary[:20])}

Analizá:
1. ¿Qué señales tienen mejor win rate real?
2. ¿Qué pesos deberían ajustarse?
3. ¿Hay algún patrón en las pérdidas?
4. Recomendaciones concretas para mejorar el win rate.

Respondé en menos de 250 palabras, en español, accionable."""
        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]},
            timeout=30)
        analysis = resp.json()["content"][0]["text"]
        send_telegram(f"📊 <b>Backtest semanal</b>\n\nWR: {win_rate}% | {len(closed)} trades\n\n{analysis}")
        log.info("📊 Backtest semanal enviado")
    except Exception as e:
        log.error(f"Backtest error: {e}")

# ─────────────────────────────────────────
# ALTCOIN SCANNER
# ─────────────────────────────────────────
def scan_top_altcoins(exchange, max_alts=15):
    global _scanner
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, t in tickers.items():
            if not symbol.endswith("/USDT"): continue
            base = symbol.replace("/USDT", "")
            if base in EXCLUDE_SYMBOLS: continue
            if not base.isascii() or len(base) > 10: continue
            vol = t.get("quoteVolume") or 0
            if vol < 20_000_000: continue
            usdt_pairs.append({"symbol": symbol, "volume": vol})
        usdt_pairs.sort(key=lambda x: x["volume"], reverse=True)
        altcoins = [p["symbol"] for p in usdt_pairs[:max_alts]]
        _scanner = [{"symbol": s, "volume": None} for s in BASE_WATCHLIST] + usdt_pairs[:max_alts]
        log.info(f"🔍 Altcoins: {altcoins}")
        return altcoins
    except Exception as e:
        log.error(f"Scanner error: {e}")
        return []

# ─────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────
def get_ohlcv(exchange, symbol, timeframe="1h", limit=150):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def calculate_indicators(df):
    df = df.copy()

    # ── EMAs ──────────────────────────────────────────────────────────────────
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = df["close"].diff()
    gain  = delta.clip(lower=0); loss = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean() /
                               loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)))

    # ── Stochastic RSI ────────────────────────────────────────────────────────
    rsi_min = df["rsi"].rolling(14).min()
    rsi_max = df["rsi"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi"] - rsi_min) / (rsi_max - rsi_min + 1e-9) * 100
    df["stoch_rsi_k"] = df["stoch_rsi"].rolling(3).mean()
    df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(3).mean()

    # ── Williams %R ───────────────────────────────────────────────────────────
    high14 = df["high"].rolling(14).max()
    low14  = df["low"].rolling(14).min()
    df["williams_r"] = (high14 - df["close"]) / (high14 - low14 + 1e-9) * -100

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Volume ────────────────────────────────────────────────────────────────
    df["vol_ma20"]    = df["volume"].rolling(20).mean()

    # ── VWAP (diario, reseteado cada 24 períodos aprox) ───────────────────────
    df["vwap"] = (df["close"] * df["volume"]).rolling(24).sum() / df["volume"].rolling(24).sum()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df["bb_mid"]   = df["close"].rolling(BB_PERIOD).mean()
    bb_std         = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_width_ma"] = df["bb_width"].rolling(20).mean()

    # ── ATR ───────────────────────────────────────────────────────────────────
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # ── Keltner Channels ─────────────────────────────────────────────────────
    df["kc_mid"]   = df["close"].ewm(span=20, adjust=False).mean()
    df["kc_upper"] = df["kc_mid"] + 1.5 * df["atr"]
    df["kc_lower"] = df["kc_mid"] - 1.5 * df["atr"]

    # ── Volatility Squeeze (BB dentro de KC) ──────────────────────────────────
    df["squeeze"] = (df["bb_upper"] < df["kc_upper"]) & (df["bb_lower"] > df["kc_lower"])

    # ── Supertrend ────────────────────────────────────────────────────────────
    mult = 3.0
    hl2  = (df["high"] + df["low"]) / 2
    df["st_upper"] = hl2 + mult * df["atr"]
    df["st_lower"] = hl2 - mult * df["atr"]
    supertrend = pd.Series(index=df.index, dtype=float)
    direction  = pd.Series(index=df.index, dtype=int)
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["st_upper"].iloc[i-1]:
            direction.iloc[i] = 1   # bullish
            supertrend.iloc[i] = df["st_lower"].iloc[i]
        elif df["close"].iloc[i] < df["st_lower"].iloc[i-1]:
            direction.iloc[i] = -1  # bearish
            supertrend.iloc[i] = df["st_upper"].iloc[i]
        else:
            direction.iloc[i] = direction.iloc[i-1]
            supertrend.iloc[i] = df["st_lower"].iloc[i] if direction.iloc[i] == 1 else df["st_upper"].iloc[i]
    df["supertrend"]     = supertrend
    df["supertrend_dir"] = direction

    # ── CCI (Commodity Channel Index) ─────────────────────────────────────────
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - tp_ma) / (0.015 * tp_mad + 1e-9)

    # ── Support & Resistance (pivots locales 20 velas) ───────────────────────
    df["pivot_high"] = df["high"].rolling(5, center=True).max() == df["high"]
    df["pivot_low"]  = df["low"].rolling(5, center=True).min() == df["low"]
    # Últimos niveles de soporte y resistencia
    recent_highs = df[df["pivot_high"]]["high"].tail(3)
    recent_lows  = df[df["pivot_low"]]["low"].tail(3)
    df["resistance"] = recent_highs.mean() if len(recent_highs) else float("nan")
    df["support"]    = recent_lows.mean()  if len(recent_lows)  else float("nan")

    # ── Higher Highs / Lower Lows (últimas 10 velas) ─────────────────────────
    highs10 = df["high"].tail(10)
    lows10  = df["low"].tail(10)
    df["higher_highs"] = highs10.iloc[-1] > highs10.iloc[0]
    df["lower_lows"]   = lows10.iloc[-1]  < lows10.iloc[0]

    return df

def technical_signal(df):
    last = df.iloc[-1]; prev = df.iloc[-2]
    bull = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
    bear = prev["ema9"] >= prev["ema21"] and last["ema9"] < last["ema21"]
    if bull and last["rsi"] < 65: return +1
    if bear and last["rsi"] > 35: return -1
    return 0

def macd_signal(df):
    last = df.iloc[-1]; prev = df.iloc[-2]
    if prev["macd_hist"] < 0 and last["macd_hist"] > 0: return +1
    if prev["macd_hist"] > 0 and last["macd_hist"] < 0: return -1
    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]: return +1
    if last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]: return -1
    return 0

def volume_signal(df):
    last = df.iloc[-1]
    if pd.isna(last["vol_ma20"]): return 0
    return +1 if last["volume"] > last["vol_ma20"] * 1.2 else 0

# ─────────────────────────────────────────
# NUEVAS SEÑALES TÉCNICAS
# ─────────────────────────────────────────

def vwap_signal(df) -> int:
    """VWAP: precio sobre VWAP = alcista, bajo = bajista."""
    last = df.iloc[-1]
    if pd.isna(last.get("vwap", float("nan"))): return 0
    diff_pct = (float(last["close"]) - float(last["vwap"])) / float(last["vwap"])
    if diff_pct > 0.005:   return +1   # 0.5% sobre VWAP
    if diff_pct < -0.005:  return -1   # 0.5% bajo VWAP
    return 0

def supertrend_signal(df) -> int:
    """Supertrend: +1 alcista, -1 bajista, 0 cambio de dirección reciente."""
    if len(df) < 5: return 0
    last = df.iloc[-1]; prev = df.iloc[-2]
    curr_dir = int(last.get("supertrend_dir", 0))
    prev_dir = int(prev.get("supertrend_dir", 0))
    if curr_dir == 1 and prev_dir == -1:
        log.info("  🟢 Supertrend: cruce BULLISH")
        return +1
    if curr_dir == -1 and prev_dir == 1:
        log.info("  🔴 Supertrend: cruce BEARISH")
        return -1
    if curr_dir == 1:   return +1
    if curr_dir == -1:  return -1
    return 0

def stoch_rsi_signal(df) -> int:
    """Stochastic RSI: sobrevendido (<20) = +1, sobrecomprado (>80) = -1."""
    last = df.iloc[-1]; prev = df.iloc[-2]
    k = float(last.get("stoch_rsi_k", 50))
    k_prev = float(prev.get("stoch_rsi_k", 50))
    if k < 20 and k > k_prev:   return +1   # saliendo de sobrevendido
    if k > 80 and k < k_prev:   return -1   # saliendo de sobrecomprado
    return 0

def williams_r_signal(df) -> int:
    """Williams %R: sobrevendido (<-80) = +1, sobrecomprado (>-20) = -1."""
    last = df.iloc[-1]
    wr = float(last.get("williams_r", -50))
    if wr < -80: return +1
    if wr > -20: return -1
    return 0

def cci_signal(df) -> int:
    """CCI: extremo bajo (<-100) = posible rebote +1, extremo alto (>100) = -1."""
    last = df.iloc[-1]; prev = df.iloc[-2]
    cci = float(last.get("cci", 0))
    cci_prev = float(prev.get("cci", 0))
    if cci < -100 and cci > cci_prev: return +1   # rebotando desde sobrevendido
    if cci > 100  and cci < cci_prev: return -1   # cayendo desde sobrecomprado
    return 0

def squeeze_signal(df) -> int:
    """Volatility Squeeze: detecta explosión de precio inminente."""
    if len(df) < 3: return 0
    last = df.iloc[-1]; prev = df.iloc[-2]
    # Squeeze se rompe (estaba comprimido, ahora no) + dirección MACD
    was_squeezed = bool(prev.get("squeeze", False))
    is_squeezed  = bool(last.get("squeeze", False))
    if was_squeezed and not is_squeezed:
        # Squeeze release — dirección según MACD histogram
        macd_hist = float(last.get("macd_hist", 0))
        if macd_hist > 0:
            log.info("  💥 Squeeze release BULLISH")
            return +1
        elif macd_hist < 0:
            log.info("  💥 Squeeze release BEARISH")
            return -1
    return 0

def support_resistance_signal(df, direction: str) -> int:
    """
    Soporte/Resistencia: solo entrar long cerca de soporte, short cerca de resistencia.
    Retorna +1 si el precio está cerca del soporte (long OK),
            -1 si está cerca de la resistencia (short OK),
             0 si está en medio (neutral).
    """
    last = df.iloc[-1]
    price = float(last["close"])
    support    = float(last.get("support",    float("nan")))
    resistance = float(last.get("resistance", float("nan")))
    if pd.isna(support) or pd.isna(resistance): return 0
    sr_range = resistance - support
    if sr_range <= 0: return 0
    pos = (price - support) / sr_range  # 0=soporte, 1=resistencia
    if pos < 0.25:  return +1   # cerca del soporte → buen long
    if pos > 0.75:  return -1   # cerca de resistencia → buen short
    return 0

def candle_pattern_signal(df) -> int:
    """
    Patrones de velas japonesas:
    - Bullish Engulfing, Hammer, Pinbar bullish → +1
    - Bearish Engulfing, Shooting Star, Pinbar bearish → -1
    - Doji en tendencia → señal débil de reversión
    """
    if len(df) < 3: return 0
    last = df.iloc[-1]; prev = df.iloc[-2]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])
    po, ph, pl, pc = float(prev["open"]), float(prev["high"]), float(prev["low"]), float(prev["close"])
    body      = abs(c - o)
    prev_body = abs(pc - po)
    candle_range = h - l
    if candle_range == 0: return 0

    # Bullish Engulfing: vela alcista que engulle vela bajista previa
    if c > o and pc > po and c > po and o < pc and body > prev_body * 1.1:
        log.info("  🕯️  Bullish Engulfing")
        return +1

    # Bearish Engulfing: vela bajista que engulle vela alcista previa
    if c < o and pc < po and c < po and o > pc and body > prev_body * 1.1:
        log.info("  🕯️  Bearish Engulfing")
        return -1

    # Hammer (martillo): mecha inferior larga, cuerpo pequeño arriba
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
        log.info("  🕯️  Hammer (bullish)")
        return +1

    # Shooting Star: mecha superior larga, cuerpo pequeño abajo
    if upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
        log.info("  🕯️  Shooting Star (bearish)")
        return -1

    # Pinbar bullish: mecha inferior > 60% del rango total, precio cierra arriba
    if lower_wick / candle_range > 0.6 and c > (h + l) / 2:
        log.info("  🕯️  Pinbar bullish")
        return +1

    # Pinbar bearish: mecha superior > 60% del rango total
    if upper_wick / candle_range > 0.6 and c < (h + l) / 2:
        log.info("  🕯️  Pinbar bearish")
        return -1

    # Doji: cuerpo muy pequeño (indecisión)
    if body / candle_range < 0.1:
        # En tendencia bajista un doji es señal de posible reversión alcista
        ema9 = float(last.get("ema9", c))
        ema21 = float(last.get("ema21", c))
        if ema9 < ema21:  return +1   # posible reversión alcista
        if ema9 > ema21:  return -1   # posible reversión bajista
    return 0

def trend_structure_signal(df) -> int:
    """
    Higher Highs / Lower Lows — confirma estructura de tendencia.
    HH = alcista, LL = bajista.
    """
    last = df.iloc[-1]
    hh = bool(last.get("higher_highs", False))
    ll = bool(last.get("lower_lows", False))
    if hh and not ll: return +1
    if ll and not hh: return -1
    return 0

def rsi_divergence_signal(df):
    if len(df) < 10: return 0, False
    window = df.tail(10)
    price_lows  = window["close"].rolling(3).min()
    rsi_lows    = window["rsi"].rolling(3).min()
    price_highs = window["close"].rolling(3).max()
    rsi_highs   = window["rsi"].rolling(3).max()
    if (price_lows.iloc[-1] < price_lows.iloc[-4] and
        rsi_lows.iloc[-1]   > rsi_lows.iloc[-4] and
        window["rsi"].iloc[-1] < 50):
        log.info("  🔄 RSI Divergence BULLISH")
        return +1, True
    if (price_highs.iloc[-1] > price_highs.iloc[-4] and
        rsi_highs.iloc[-1]   < rsi_highs.iloc[-4] and
        window["rsi"].iloc[-1] > 50):
        log.info("  🔄 RSI Divergence BEARISH")
        return -1, True
    return 0, False

def bollinger_signal(df):
    last = df.iloc[-1]; prev = df.iloc[-2]
    squeeze = last["bb_width"] < df["bb_width"].rolling(20).mean().iloc[-1] * 0.75
    if last["close"] <= last["bb_lower"] and prev["close"] > prev["bb_lower"]:
        log.info("  🎯 BB: precio cruza banda inferior")
        return +1
    if last["close"] >= last["bb_upper"] and prev["close"] < prev["bb_upper"]:
        log.info("  🎯 BB: precio cruza banda superior")
        return -1
    if squeeze and last["close"] > last["bb_mid"]: return +1
    if squeeze and last["close"] < last["bb_mid"]: return -1
    return 0

def order_book_signal(exchange, symbol):
    try:
        ob  = exchange.fetch_order_book(symbol, limit=20)
        bid = sum(b[1] for b in ob["bids"])
        ask = sum(a[1] for a in ob["asks"])
        total = bid + ask
        if total == 0: return 0
        ratio = bid / total
        log.info(f"  📖 OB: bids={ratio*100:.1f}%")
        if ratio >= OB_IMBALANCE_THRESHOLD:       return +1
        if ratio <= 1 - OB_IMBALANCE_THRESHOLD:   return -1
        return 0
    except Exception as e:
        log.warning(f"OB error {symbol}: {e}")
        return 0

def timeframe_confirm_signal(exchange, symbol, base_tf):
    confirm_tf = "15m" if base_tf == "3m" else ("30m" if base_tf == "1h" else "4h")
    try:
        df = calculate_indicators(get_ohlcv(exchange, symbol, timeframe=confirm_tf, limit=50))
        # MACD más sensible que EMA cross en mercado lateral
        return macd_signal(df)
    except Exception as e:
        log.warning(f"Confirm error {symbol}: {e}")
        return 0

# ─────────────────────────────────────────
# FEAR & GREED
# ─────────────────────────────────────────
def get_fear_greed():
    global _fear_greed
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        data = resp.json()["data"][0]
        value = int(data["value"])
        _fear_greed = {"value": value, "label": data["value_classification"]}
        return value, data["value_classification"]
    except Exception as e:
        log.warning(f"F&G error: {e}")
        return 50, "Neutral"

def fear_greed_filter(value, action, regime):
    if action == "BUY"  and value < 12: return False   # bajado de 15 — más margen en Extreme Fear
    if action == "SELL" and value > 85: return False
    if regime == "crash" and action == "BUY": return False
    return True

# ─────────────────────────────────────────
# NOTICIAS
# ─────────────────────────────────────────
BULLISH_KW = ["rally","surge","breakout","bullish","adoption","upgrade","all-time high","gains","rises","jumps","soars","recovery"]
BEARISH_KW = ["crash","hack","ban","bearish","lawsuit","sell-off","collapse","drops","falls","plunges","warning","fraud"]
RSS_FEEDS  = ["https://www.coindesk.com/arc/outboundfeeds/rss/","https://cointelegraph.com/rss"]
COIN_NAMES = {
    "btc":["bitcoin","btc"],"eth":["ethereum","eth"],"sol":["solana","sol"],
    "bnb":["bnb","binance"],"xrp":["ripple","xrp"],"doge":["dogecoin","doge"],
    "ada":["cardano","ada"],"avax":["avalanche","avax"],"pepe":["pepe"],
    "trx":["tron","trx"],"link":["chainlink","link"],"aave":["aave"],
}

def get_news_sentiment(symbol):
    coin  = symbol.split("/")[0].lower()
    terms = COIN_NAMES.get(coin, [coin.lower()])
    all_titles = []
    for url in RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
            if not titles: titles = re.findall(r"<title>(.*?)</title>", resp.text)
            all_titles.extend(titles[:20])
        except: pass
    if not all_titles: return 0
    relevant = [t.lower() for t in all_titles if any(term in t.lower() for term in terms)]
    if not relevant: return 0
    text = " ".join(relevant)
    bull = sum(1 for kw in BULLISH_KW if kw in text)
    bear = sum(1 for kw in BEARISH_KW if kw in text)
    log.info(f"  📰 Noticias {coin}: {len(relevant)} específicas | bull={bull} bear={bear}")
    if bull > bear: return +1
    if bear > bull: return -1
    return 0

# ─────────────────────────────────────────
# FUNDING RATES
# ─────────────────────────────────────────
_funding_cache     = {}
_funding_last_fetch = 0

def get_funding_rate(symbol):
    global _funding_cache, _funding_last_fetch
    now = time.time()
    if now - _funding_last_fetch > 900:
        try:
            resp = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=10)
            _funding_cache = {item["symbol"]: float(item.get("lastFundingRate", 0)) for item in resp.json()}
            _funding_last_fetch = now
            log.info(f"  💹 Funding rates actualizados ({len(_funding_cache)} pares)")
        except Exception as e:
            log.warning(f"Funding error: {e}")
            return 0.0
    return _funding_cache.get(symbol.replace("/", ""), 0.0)

def funding_rate_signal(symbol):
    rate = get_funding_rate(symbol)
    if rate == 0.0: return 0, rate
    if rate <= FUNDING_BULLISH_THRESHOLD:
        log.info(f"  💹 Funding: {rate:.4%} → BULLISH")
        return +1, rate
    elif rate >= FUNDING_BEARISH_THRESHOLD:
        log.info(f"  💹 Funding: {rate:.4%} → BEARISH")
        return -1, rate
    log.info(f"  💹 Funding: {rate:.4%} → neutral")
    return 0, rate

# ─────────────────────────────────────────
# CAPITAL ALLOCATOR
# ─────────────────────────────────────────
def get_position_size(effective_capital, score_int, regime):
    abs_score = abs(score_int)
    pct = POSITION_SIZE_MAP.get(min(abs_score, 6), 0.02)
    if regime == "bear": pct = pct * 0.7
    usd = round(effective_capital * pct, 2)
    log.info(f"  💰 Size: {pct*100:.1f}% = ${usd} (score={score_int:+d} regime={regime})")
    return usd, pct

def get_allocated_capital(positions):
    return sum(p.get("usd_size", 0) for p in positions.values())

def can_open_position(positions, new_size, effective_capital):
    allocated   = get_allocated_capital(positions)
    max_allowed = effective_capital * MAX_CAPITAL_EXPOSURE
    available   = max_allowed - allocated
    log.info(f"  💼 Capital: ${allocated:.2f} usado / ${max_allowed:.2f} máx (${available:.2f} disp)")
    if new_size > available:
        log.info(f"  🚫 Capital insuficiente: necesita ${new_size} pero ${available:.2f} disponible")
        return False
    return True

# ─────────────────────────────────────────
# POSICIONES — TRAILING STOP + SALIDA PARCIAL
# ─────────────────────────────────────────
def load_positions():
    if _USE_PG:
        return _pg_get("positions", {})
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_positions(positions):
    if _USE_PG:
        _pg_set("positions", positions)
    else:
        with open(POSITIONS_FILE, "w") as f: json.dump(positions, f, indent=2, default=str)

def open_position(symbol, entry_price, usd_size, pct, action, timeframe, mode, signals_snap, atr_value=None):
    positions = load_positions()
    is_futures = "FUTURES" in mode
    is_short   = action == "SELL"

    # ATR trailing dinámico — si no hay ATR, usar % fijo
    if atr_value and atr_value > 0:
        mult      = ATR_MULTIPLIER_FUT if is_futures else ATR_MULTIPLIER_SPOT
        trail_abs = atr_value * mult
        trail_pct_actual = trail_abs / entry_price
        # Cap: 3% max en 3m altcoins, 2% max en futuros, 4% max en 1h spot
        max_trail = 0.02 if is_futures else (0.03 if timeframe == "3m" else 0.04)
        if trail_pct_actual > max_trail:
            log.info(f"  📐 ATR trailing: {atr_value:.4f} × {mult} = {trail_abs:.4f} ({trail_pct_actual*100:.2f}%) → capped a {max_trail*100:.0f}%")
            trail_pct_actual = max_trail
        else:
            log.info(f"  📐 ATR trailing: {atr_value:.4f} × {mult} = {trail_abs:.4f} ({trail_pct_actual*100:.2f}%)")
    else:
        trail_pct_actual = FUTURES_TRAILING if is_futures else TRAILING_STOP_PCT

    tp_pct = FUTURES_TP if is_futures else TAKE_PROFIT_PCT

    # Para shorts: trail y TP van en dirección opuesta
    if is_short:
        trail_stop  = round(entry_price * (1 + trail_pct_actual), 4)  # sube si el precio sube
        take_profit = round(entry_price * (1 - tp_pct), 4)            # TP debajo del precio
        partial_tp  = round(entry_price * (1 - TAKE_PROFIT_PARTIAL), 4)
        high_price  = entry_price  # para shorts, rastreamos el mínimo
    else:
        trail_stop  = round(entry_price * (1 - trail_pct_actual), 4)
        take_profit = round(entry_price * (1 + tp_pct), 4)
        partial_tp  = round(entry_price * (1 + TAKE_PROFIT_PARTIAL), 4)
        high_price  = entry_price

    positions[symbol] = {
        "symbol": symbol, "action": action, "timeframe": timeframe, "mode": mode,
        "entry_price": entry_price, "current_price": entry_price, "high_price": high_price,
        "trail_stop": trail_stop, "take_profit": take_profit, "partial_tp": partial_tp,
        "partial_closed": False, "atr_value": round(atr_value, 6) if atr_value else None,
        "usd_size": usd_size, "risk_pct": pct,
        "opened_at": datetime.now().isoformat(),
        "signals_snap": signals_snap,
    }
    save_positions(positions)
    log.info(f"  📂 Posición: {symbol} [{mode}] @ {entry_price} | Trail={positions[symbol]['trail_stop']} TP={positions[symbol]['take_profit']}")

def update_trailing_stops(public_ex, state):
    positions = load_positions()
    if not positions: return
    closed = []
    for symbol, pos in positions.items():
        try:
            current   = public_ex.fetch_ticker(symbol)["last"]
            pos["current_price"] = current
            trail_pct = FUTURES_TRAILING if "FUTURES" in pos.get("mode","") else TRAILING_STOP_PCT
            tp_pct    = FUTURES_TP       if "FUTURES" in pos.get("mode","") else TAKE_PROFIT_PCT

            is_short = pos["action"] == "SELL"

            # ATR trail dinámico — recalcular trail_pct si hay ATR guardado
            atr_saved = pos.get("atr_value")
            if atr_saved and atr_saved > 0 and current > 0:
                mult = ATR_MULTIPLIER_FUT if "FUTURES" in pos.get("mode","") else ATR_MULTIPLIER_SPOT
                trail_pct = (atr_saved * mult) / current

            if not is_short:  # ── LONG ──
                if current > pos["high_price"]:
                    pos["high_price"] = current
                    pos["trail_stop"] = round(current * (1 - trail_pct), 4)
                    log.info(f"  📈 Trail {symbol}: {pos['trail_stop']}")

                # SALIDA PARCIAL — cerrar 50% al primer TP parcial
                if not pos.get("partial_closed") and current >= pos.get("partial_tp", float("inf")):
                    partial_pnl = apply_fee((current - pos["entry_price"]) / pos["entry_price"] * 100, pos.get("mode","SPOT"))
                    partial_usd = pos["usd_size"] * PARTIAL_EXIT_PCT
                    log.info(f"  ½ PARTIAL TP {symbol} @ {current} | PnL parcial: +{partial_pnl:.2f}%")
                    pos["partial_closed"] = True
                    pos["usd_size"]      = round(pos["usd_size"] * (1 - PARTIAL_EXIT_PCT), 2)
                    pnl_usd = partial_pnl * partial_usd / 100
                    update_compounding(state, pnl_usd)
                    send_telegram(f"½ <b>Partial TP</b> — {symbol}\n+{partial_pnl:.2f}% — cerrando 50%\nDejando correr el resto con trailing")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"SELL","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":f"Partial TP 50% @ {current}",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(partial_pnl,2),
                        "partial_exit":True,"entry_price":pos["entry_price"],"usd_size":partial_usd})

                # Stop loss absoluto
                stop_loss_price = pos["entry_price"] * (1 - STOP_LOSS_PCT)
                if current <= stop_loss_price and current < pos["entry_price"]:
                    pnl = apply_fee((current - pos["entry_price"]) / pos["entry_price"] * 100, pos.get("mode","SPOT"))
                    log.info(f"  🛑 STOP LOSS {symbol} @ {current} | PnL: {pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    # Guardar en signal memory para aprender
                    if pos.get("signals_snap"):
                        save_signal_to_memory(pos["signals_snap"], pnl, pos.get("regime","sideways"))
                    send_telegram(f"🛑 <b>Stop Loss</b> — {symbol}\n{pnl:+.2f}% ❌")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"SELL","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":f"Stop loss -3%",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "stop_loss":True,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol); continue

                # Profit trailing dinámico
                unrealized_pct = (current - pos["entry_price"]) / pos["entry_price"]
                if unrealized_pct >= PROFIT_TRAIL_TRIGGER:
                    new_tp = round(current * (1 + PROFIT_TRAIL_STEP), 4)
                    if new_tp > pos["take_profit"]:
                        pos["take_profit"] = new_tp
                        log.info(f"  🎯 TP subido {symbol}: {new_tp} (+{unrealized_pct*100:.1f}%)")

                # Trail stop hit
                if current <= pos["trail_stop"]:
                    pnl = apply_fee((current - pos["entry_price"]) / pos["entry_price"] * 100, pos.get("mode","SPOT"))
                    log.info(f"  🔴 TRAIL STOP {symbol} @ {current} | PnL: {pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    if pos.get("signals_snap"):
                        save_signal_to_memory(pos["signals_snap"], pnl, pos.get("regime","sideways"))
                    send_telegram(f"🔴 <b>Trail Stop</b> — {symbol}\n{pnl:+.2f}% {'✅' if pnl>0 else '❌'}")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"SELL","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":f"Trail stop",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "trail_triggered":True,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol); continue

                # Take profit final
                if current >= pos["take_profit"]:
                    pnl = apply_fee((current - pos["entry_price"]) / pos["entry_price"] * 100, pos.get("mode","SPOT"))
                    log.info(f"  🎯 TAKE PROFIT {symbol} @ {current} | PnL: +{pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    if pos.get("signals_snap"):
                        save_signal_to_memory(pos["signals_snap"], pnl, pos.get("regime","sideways"))
                    send_telegram(f"🎯 <b>Take Profit</b> — {symbol}\n+{pnl:.2f}% 🎉")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"SELL","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":f"Take profit",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "trail_triggered":False,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol)

            else:  # ── SHORT ──
                # Para shorts, el trail_stop sube cuando el precio baja
                if current < pos["high_price"]:
                    pos["high_price"] = current   # rastreamos el mínimo
                    pos["trail_stop"] = round(current * (1 + trail_pct), 4)
                    log.info(f"  📉 Short Trail {symbol}: {pos['trail_stop']}")

                # Stop loss short — si el precio SUBE más del 3%
                stop_loss_price = pos["entry_price"] * (1 + STOP_LOSS_PCT)
                if current >= stop_loss_price:
                    pnl = apply_fee((pos["entry_price"] - current) / pos["entry_price"] * 100, pos.get("mode","SPOT"))
                    log.info(f"  🛑 SHORT STOP LOSS {symbol} @ {current} | PnL: {pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    if pos.get("signals_snap"):
                        save_signal_to_memory(pos["signals_snap"], pnl, pos.get("regime","sideways"))
                    send_telegram(f"🛑 <b>Short Stop Loss</b> — {symbol}\n{pnl:+.2f}% ❌")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"BUY","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":"Short stop loss +3%",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "stop_loss":True,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol); continue

                # Trail stop short — si el precio sube por encima del trail
                if current >= pos["trail_stop"]:
                    pnl = apply_fee((pos["entry_price"] - current) / pos["entry_price"] * 100, pos.get("mode","SPOT"))
                    log.info(f"  🔴 SHORT TRAIL {symbol} @ {current} | PnL: {pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    if pos.get("signals_snap"):
                        save_signal_to_memory(pos["signals_snap"], pnl, pos.get("regime","sideways"))
                    send_telegram(f"🔴 <b>Short Trail Stop</b> — {symbol}\n{pnl:+.2f}% {'✅' if pnl>0 else '❌'}")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"BUY","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":"Short trail stop",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "trail_triggered":True,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol); continue

                # Take profit short
                if current <= pos["take_profit"]:
                    pnl = apply_fee((pos["entry_price"] - current) / pos["entry_price"] * 100, pos.get("mode","SPOT"))
                    log.info(f"  🎯 SHORT TP {symbol} @ {current} | PnL: +{pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    if pos.get("signals_snap"):
                        save_signal_to_memory(pos["signals_snap"], pnl, pos.get("regime","sideways"))
                    send_telegram(f"🎯 <b>Short TP</b> — {symbol}\n+{pnl:.2f}% 🎉")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"BUY","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":"Short take profit",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "trail_triggered":False,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol)
        except Exception as e:
            log.error(f"Error trailing {symbol}: {e}")
    for s in closed: del positions[s]
    save_positions(positions)

# ─────────────────────────────────────────
# CLAUDE — con rate limiter y retry
# ─────────────────────────────────────────
def ask_claude(symbol, signals, df, fg_value, usd_size, pct, timeframe, regime, score_float, direction="LONG"):
    global _claude_last_call
    last  = df.iloc[-1]
    rsi_div_info = "YES (strong signal)" if signals.get("rsi_div_val") else "No"
    weights = _weights_cache if _weights_cache else DEFAULT_WEIGHTS

    expected_action = "SELL" if direction == "SHORT" else "BUY"

    prompt = f"""Crypto trading signal validator. Respond ONLY in JSON, no backticks.

DIRECTION: This is a {direction} signal — expected action is {expected_action}.
IMPORTANT RULES:
- The weighted score already filters RSI, EMA, MACD, volume, order book, funding rates and news
- On {timeframe} timeframe, RSI 65-80 is NORMAL momentum, NOT a reason to HOLD
- For SHORT: respond SELL if score supports it. For LONG: respond BUY if score supports it
- Only HOLD if clear contradiction: score is {direction} but majority of signals point opposite direction
- Trust the weighted score — it passed confirmation delay and multi-signal validation

Pair: {symbol} [{timeframe}] | Regime: {regime.upper()} | F&G: {fg_value}
Weighted score: {score_float:+.2f} | Direction: {direction}
Signals: EMA:{signals.get('tech',0):+d} MACD:{signals.get('macd',0):+d} BB:{signals.get('bb',0):+d} OB:{signals.get('ob',0):+d} VOL:{signals.get('vol',0):+d} FR:{signals.get('funding',0):+d} NEWS:{signals.get('news',0):+d} RSI_DIV:{rsi_div_info}
Size: ${usd_size} | SL: {STOP_LOSS_PCT*100}% | Partial TP: {TAKE_PROFIT_PARTIAL*100}%

{{"action":"BUY"|"SELL"|"HOLD","confidence":0.0,"reasoning":"one concise line"}}"""

    with _claude_semaphore:
        now = time.time()
        wait = CLAUDE_RATE_LIMIT_SEC - (now - _claude_last_call)
        if wait > 0: time.sleep(wait)
        for attempt in range(CLAUDE_MAX_RETRIES):
            try:
                resp = requests.post("https://api.anthropic.com/v1/messages",
                    headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
                    json={"model":"claude-haiku-4-5-20251001","max_tokens":200,"messages":[{"role":"user","content":prompt}]},
                    timeout=20)
                _claude_last_call = time.time()
                data = resp.json()
                if "content" not in data:
                    raise ValueError(f"No content: {data.get('error', data)}")
                text = data["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
                return json.loads(text)
            except Exception as e:
                backoff = 2 ** attempt
                log.warning(f"Claude attempt {attempt+1}/{CLAUDE_MAX_RETRIES}: {e} — retry {backoff}s")
                if attempt < CLAUDE_MAX_RETRIES - 1: time.sleep(backoff)
        return {"action":"HOLD","confidence":0,"reasoning":"API unavailable"}

# ─────────────────────────────────────────
# ÓRDENES
# ─────────────────────────────────────────
def execute_spot_trade(trade_ex, symbol, action, usd_size):
    try:
        price = trade_ex.fetch_ticker(symbol)["last"]
        qty   = usd_size / price
        order = trade_ex.create_market_buy_order(symbol, qty) if action == "BUY" else trade_ex.create_market_sell_order(symbol, qty)
        return order, price
    except Exception as e:
        log.error(f"Spot order error {action} {symbol}: {e}")
        return None, None

def execute_futures_trade(futures_ex, symbol, action, usd_size, leverage=FUTURES_LEVERAGE):
    try:
        price    = futures_ex.fetch_ticker(symbol)["last"]
        notional = usd_size * leverage
        qty      = notional / price
        try: futures_ex.set_leverage(leverage, symbol)
        except: pass
        order = futures_ex.create_market_buy_order(symbol, qty) if action == "BUY" else futures_ex.create_market_sell_order(symbol, qty)
        return order, price
    except Exception as e:
        log.error(f"Futures order error {action} {symbol}: {e}")
        return None, None

def save_trade(record):
    if _USE_PG:
        try:
            with _pg.cursor() as cur:
                cur.execute("INSERT INTO trades (data) VALUES (%s)", (json.dumps(record, default=str),))
        except Exception as e:
            log.warning(f"PG save_trade error: {e}")
            # fallback a JSON
            data = []
            if os.path.exists(TRADE_LOG_FILE):
                try:
                    with open(TRADE_LOG_FILE) as f: data = json.load(f)
                except: pass
            data.append(record)
            with open(TRADE_LOG_FILE,"w") as f: json.dump(data, f, indent=2, default=str)
    else:
        data = []
        if os.path.exists(TRADE_LOG_FILE):
            with open(TRADE_LOG_FILE) as f: data = json.load(f)
        data.append(record)
        with open(TRADE_LOG_FILE,"w") as f: json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────
# FILTROS DE CALIDAD DE ENTRADA
# ─────────────────────────────────────────

_btc_macro_cache = {"value": None, "ts": 0}

def btc_macro_filter(exchange, direction: str) -> bool:
    """
    Filtro macro: solo abrir LONG si BTC 4h está por encima de EMA21.
    Solo abrir SHORT si BTC 4h está por debajo de EMA21.
    Evita entrar en contra de la tendencia mayor.
    Cache de 15 minutos para no spammear la API.
    """
    global _btc_macro_cache
    now = time.time()
    if now - _btc_macro_cache["ts"] < 900 and _btc_macro_cache["value"] is not None:
        btc_above_ema = _btc_macro_cache["value"]
    else:
        try:
            df4h = calculate_indicators(get_ohlcv(exchange, "BTC/USDT", "4h", limit=30))
            last = df4h.iloc[-1]
            btc_above_ema = float(last["close"]) > float(last["ema21"])
            _btc_macro_cache = {"value": btc_above_ema, "ts": now}
            trend = "↑ BULL" if btc_above_ema else "↓ BEAR"
            log.info(f"  🌍 BTC 4h macro: {trend} (close={last['close']:.0f} vs EMA21={last['ema21']:.0f})")
        except Exception as e:
            log.warning(f"  BTC macro filter error: {e}")
            return True  # si falla, no bloquear

    if direction == "LONG" and not btc_above_ema:
        log.info("  ⏭️  Macro filter: BTC 4h bajista — no abrir LONG")
        return False
    if direction == "SHORT" and btc_above_ema:
        log.info("  ⏭️  Macro filter: BTC 4h alcista — no abrir SHORT")
        return False
    return True


def pump_dump_filter(df: pd.DataFrame, symbol: str, fr_val: float) -> bool:
    """
    Detecta pumps artificiales y condiciones de baja liquidez.
    Retorna False si NO se debe entrar.

    Condiciones de rechazo:
    - Precio subió/bajó >8% en las últimas 4 velas → pump/dump en curso
    - Funding extremo en altcoin desconocida (>0.5% o <-0.5%)
    - Volatilidad ATR >5% del precio → mercado demasiado errático
    """
    last  = df.iloc[-1]
    prev4 = df.iloc[-5] if len(df) >= 5 else df.iloc[0]

    # Movimiento extremo en 4 velas
    pct_move = abs(float(last["close"]) - float(prev4["close"])) / float(prev4["close"])
    if pct_move > 0.08:
        log.info(f"  ⏭️  Pump/dump filter: movimiento {pct_move*100:.1f}% en 4 velas — skip")
        return False

    # Funding extremo en altcoins (no en majors)
    is_major = any(symbol.startswith(m) for m in ["BTC", "ETH", "SOL", "BNB"])
    if not is_major and abs(fr_val) > 0.005:  # >0.5%
        log.info(f"  ⏭️  Pump/dump filter: funding extremo {fr_val*100:.2f}% en altcoin — skip")
        return False

    # ATR demasiado alto — volatilidad extrema
    if not pd.isna(last.get("atr", float("nan"))):
        atr_pct = float(last["atr"]) / float(last["close"])
        if atr_pct > 0.05:
            log.info(f"  ⏭️  Pump/dump filter: ATR {atr_pct*100:.1f}% — volatilidad extrema, skip")
            return False

    return True


def volume_conviction_filter(df: pd.DataFrame, regime: str = "sideways") -> tuple:
    """
    Volumen como ajuste de size, no como bloqueo binario.
    - < 0.7x: skip (ruido puro)
    - 0.7x-1.2x: entra con size * 0.75
    - >= 1.2x: entra con size completo
    Retorna (ok: bool, size_mult: float)
    """
    last = df.iloc[-1]
    if pd.isna(last.get("vol_ma20", float("nan"))) or last["vol_ma20"] == 0:
        return True, 1.0  # sin datos, no bloquear
    ratio = float(last["volume"]) / float(last["vol_ma20"])
    if ratio < 0.7:
        log.info(f"  ⏭️  Volume muy bajo: {ratio:.1f}x promedio — skip (ruido)")
        return False, 0.0
    elif ratio < 1.2:
        log.info(f"  ⚠️  Volume medio: {ratio:.1f}x promedio — size reducido 25%")
        return True, 0.75
    else:
        log.info(f"  ✅ Volume conviction: {ratio:.1f}x promedio")
        return True, 1.0

# ─────────────────────────────────────────
# ANALIZAR UN PAR — motor principal
# ─────────────────────────────────────────
def analyze_and_trade(symbol, timeframe, public_ex, trade_ex, futures_ex,
                      fg_value, fg_label, open_positions, regime, state):
    if symbol in open_positions:
        log.info(f"  {symbol}: posición ya abierta — skip")
        return

    df     = calculate_indicators(get_ohlcv(public_ex, symbol, timeframe=timeframe))
    t_sig  = technical_signal(df)
    m_sig  = macd_signal(df)
    v_sig  = volume_signal(df)
    h_sig  = timeframe_confirm_signal(public_ex, symbol, timeframe)
    n_sig  = get_news_sentiment(symbol)
    fr_sig, fr_val = funding_rate_signal(symbol)
    bb_sig = bollinger_signal(df)
    ob_sig = order_book_signal(public_ex, symbol)
    rsi_sig, rsi_div = rsi_divergence_signal(df)

    # ── Nuevas señales técnicas ────────────────────────────────────────────
    def _safe_sig(fn, *args):
        """Wrapper seguro — NaN o errores retornan 0."""
        try:
            result = fn(*args)
            if result is None or (isinstance(result, float) and (result != result)):
                return 0
            return int(result)
        except Exception:
            return 0

    direction_hint = "long" if (t_sig + m_sig + bb_sig) >= 0 else "short"
    vwap_sig    = _safe_sig(vwap_signal, df)
    st_sig      = _safe_sig(supertrend_signal, df)
    stoch_sig   = _safe_sig(stoch_rsi_signal, df)
    willy_sig   = _safe_sig(williams_r_signal, df)
    cci_sig     = _safe_sig(cci_signal, df)
    squeeze_sig = _safe_sig(squeeze_signal, df)
    sr_sig      = _safe_sig(support_resistance_signal, df, direction_hint)
    candle_sig  = _safe_sig(candle_pattern_signal, df)
    trend_sig   = _safe_sig(trend_structure_signal, df)

    signals = {
        "tech": t_sig, "macd": m_sig, "vol": v_sig, "tf4h": h_sig,
        "news": n_sig, "funding": fr_sig, "bb": bb_sig, "ob": ob_sig,
        "rsi_div": rsi_sig, "rsi_div_val": rsi_div,
        # señales técnicas (peso 0 en score → usadas como filtros)
        "vwap": vwap_sig, "supertrend": st_sig, "stoch_rsi": stoch_sig,
        "williams_r": willy_sig, "cci": cci_sig, "squeeze": squeeze_sig,
        "support_res": sr_sig, "candle": candle_sig, "trend_struct": trend_sig,
    }

    # ── SCORE: solo señales con peso > 0 (trend + momentum + confirmation) ─
    score_float, score_int = weighted_score(signals, regime)

    # ── Grupos de score para log legible ──────────────────────────────────
    trend_score    = t_sig * 1.0 + st_sig * 1.3
    momentum_score = m_sig * 1.0 + rsi_sig * 1.0
    confirm_score  = vwap_sig * 1.2 + v_sig * 0.8 + trend_sig * 0.9

    # C10: Log estructurado — siempre muestra los 3 grupos + decisión
    log.info(
        f"  [{timeframe}] trend={trend_score:+.1f}(EMA:{t_sig:+d} ST:{st_sig:+d}) "
        f"momentum={momentum_score:+.1f}(MACD:{m_sig:+d} RSI▲:{rsi_sig:+d}) "
        f"confirm={confirm_score:+.1f}(VWAP:{vwap_sig:+d} VOL:{v_sig:+d} TREND:{trend_sig:+d}) "
        f"= {score_float:+.1f}"
    )
    log.info(
        f"  [{timeframe}] ctx: BB:{bb_sig:+d} OB:{ob_sig:+d} FR:{fr_sig:+d} "
        f"NEWS:{n_sig:+d} STOCH:{stoch_sig:+d} WILLY:{willy_sig:+d} "
        f"CCI:{cci_sig:+d} SQZ:{squeeze_sig:+d} SR:{sr_sig:+d} CANDLE:{candle_sig:+d}"
    )

    # C5: News como filtro — noticias muy negativas bloquean, positivas ajustan size
    if n_sig <= -1:
        log.info(f"  ⏭️  News muy negativas ({n_sig}) — skip")
        return
    news_size_mult = 1.1 if n_sig >= 1 else 1.0

    # C6: Funding como ajuste de size — no como señal de entrada
    funding_size_mult = 1.0
    if abs(fr_val) > 0.0005:
        if fr_val > 0.0005:   funding_size_mult = 0.85   # funding alto → longs caros
        elif fr_val < -0.0005: funding_size_mult = 1.1   # funding negativo → longs baratos

    # C4: Order book como ajuste de size — no como voto del score
    ob_bids = getattr(ob_sig, '__ob_bids__', None)
    ob_size_mult = 1.0
    if isinstance(ob_sig, (int, float)):
        # ob_sig -1 = asks dominan, 0 = neutral, +1 = bids dominan
        if ob_sig < 0: ob_size_mult = 0.8  # asks dominan → reducir size

    # ── SCORE mínimo dinámico: RL acotado (C9) ──────────────────────────────
    # RL puede mover el threshold ±0.5 máximo — no domina el sistema
    rl_min = rl_adjust_min_signals(state)
    rl_adjustment = (rl_min - MIN_SIGNALS_SIDEWAYS) * 0.2  # acotado al 20%

    # C7: Sideways ajusta threshold, no cancela el sistema
    if regime == "sideways":
        min_threshold = SIDEWAYS_MIN_FLOAT + rl_adjustment
    else:
        min_threshold = float(max(MIN_SIGNALS, rl_min - 1)) + rl_adjustment

    if abs(score_float) < min_threshold:
        log.info(
            f"  ⏭️  score={score_float:+.1f} < threshold={min_threshold:.1f} "
            f"(régimen={regime} rl_adj={rl_adjustment:+.1f}) — skip"
        )
        return

    # Shorts
    if score_float < 0 and not SHORT_ENABLED:
        log.info("  ⏭️  Bajista y shorts deshabilitados — skip")
        return
    if timeframe == "3m" and score_float < 0 and score_float > SHORT_MIN_SCORE:
        log.info(f"  ⏭️  Short score {score_float:+.1f} insuficiente (mínimo {SHORT_MIN_SCORE}) — skip")
        return

    # C8: Shorts selectivos — permitir short en activos débiles aunque BTC suba
    direction_macro = "LONG" if score_float > 0 else "SHORT"
    if direction_macro == "SHORT" and score_float < 0:
        # Detectar debilidad relativa: si el activo cayó más que BTC en 24h
        try:
            ticker     = public_ex.fetch_ticker(symbol)
            btc_ticker = public_ex.fetch_ticker("BTC/USDT")
            asset_24h  = ticker.get("percentage", 0) or 0
            btc_24h    = btc_ticker.get("percentage", 0) or 0
            is_weak    = asset_24h < btc_24h - 2  # más de 2% peor que BTC
        except:
            is_weak = False

    # Correlación
    base_symbols = set(BASE_WATCHLIST)
    open_alts    = [s for s in open_positions if s not in base_symbols]
    if symbol not in base_symbols and len(open_alts) >= MAX_CORRELATION_ALTS:
        log.info(f"  ⏭️  Correlación: {len(open_alts)} altcoins abiertas — skip")
        return

    # ── Filtro 0: Circuit breakers globales ─────────────────────────────────
    if state.get("daily_drawdown_mode"):
        log.info("  ⏭️  Daily circuit breaker activo — skip")
        return
    if not trading_hours_filter():
        return
    if not losing_positions_filter(open_positions):
        return

    # ── Filtro 1: Macro BTC 4h ──────────────────────────────────────────────
    # C8: Shorts selectivos — activos débiles pueden shortearse aunque BTC suba
    direction_macro = "LONG" if score_float > 0 else "SHORT"
    if direction_macro == "SHORT" and not locals().get("is_weak", False):
        if not btc_macro_filter(public_ex, direction_macro):
            return
    elif direction_macro == "LONG":
        if not btc_macro_filter(public_ex, direction_macro):
            return
    # Si is_weak=True, short permitido aunque BTC esté alcista

    # ── Filtro 2: Pump/dump y liquidez ───────────────────────────────────────
    if not pump_dump_filter(df, symbol, fr_val):
        return

    # ── Filtro 3: Volumen de convicción ──────────────────────────────────────
    vol_ok, vol_size_mult = volume_conviction_filter(df, regime)
    if not vol_ok:
        return

    # Entry confirmation delay
    if not check_entry_confirmation(symbol, signals, df, timeframe):
        return

    effective_cap = get_effective_capital(state)
    # Aplicar multiplicadores contextuales acumulados (C3, C4, C5, C6)
    context_size_mult = vol_size_mult * news_size_mult * funding_size_mult * ob_size_mult
    context_size_mult = max(0.5, min(1.3, context_size_mult))  # acotar entre 0.5x y 1.3x
    if context_size_mult != 1.0:
        log.info(f"  📐 Size mult contextual: {context_size_mult:.2f}x (vol={vol_size_mult:.2f} news={news_size_mult:.2f} fr={funding_size_mult:.2f} ob={ob_size_mult:.2f})")

    usd_size, risk_pct = get_position_size(effective_cap, score_int, regime)
    usd_size = round(usd_size * context_size_mult, 2)

    if not can_open_position(open_positions, usd_size, effective_cap): return
    if not regime_filter(regime, "BUY" if score_int > 0 else "SELL"): return

    log.info("  🧠 Consultando Claude...")
    direction = "SHORT" if score_float < 0 else "LONG"
    analysis = ask_claude(symbol, signals, df, fg_value, usd_size, risk_pct, timeframe, regime, score_float, direction)
    log.info(f"  Claude: {analysis['action']} ({analysis['confidence']:.2f}) — {analysis['reasoning']}")

    if not fear_greed_filter(fg_value, analysis["action"], regime): return

    # Spot vs Futuros
    use_futures = (abs(score_int) >= FUTURES_MIN_SCORE) and (fg_value >= 20) and (regime != "crash")
    mode_label  = f"FUTURES {FUTURES_LEVERAGE}x" if use_futures else "SPOT"
    trail_pct   = FUTURES_TRAILING if use_futures else TRAILING_STOP_PCT
    tp_pct      = FUTURES_TP       if use_futures else TAKE_PROFIT_PCT
    leverage    = FUTURES_LEVERAGE if use_futures else 1

    current_price = df.iloc[-1]["close"]
    atr_value     = df.iloc[-1].get("atr", None)
    trail_pct_atr = (atr_value * (ATR_MULTIPLIER_FUT if use_futures else ATR_MULTIPLIER_SPOT) / current_price) if atr_value else None
    trail_stop    = round(current_price * (1 - (trail_pct_atr or (FUTURES_TRAILING if use_futures else TRAILING_STOP_PCT))), 4)
    take_profit   = round(current_price * (1 + tp_pct), 4)
    partial_tp    = round(current_price * (1 + TAKE_PROFIT_PARTIAL), 4)

    # Snapshot de señales para guardar en memoria
    signals_snap = {k: v for k,v in signals.items() if k != "rsi_div_val"}
    signals_snap["regime"] = regime

    base_record = {
        "timestamp": datetime.now().isoformat(), "symbol": symbol,
        "action": analysis["action"], "confidence": analysis["confidence"],
        "reasoning": analysis["reasoning"], "timeframe": timeframe, "mode": mode_label,
        "tech_signal": t_sig, "macd_signal": m_sig, "vol_signal": v_sig,
        "bb_signal": bb_sig, "ob_signal": ob_sig, "rsi_div": rsi_div,
        "funding_signal": fr_sig, "funding_rate": round(fr_val * 100, 4),
        "news_signal": n_sig, "tf4h_signal": h_sig,
        "fear_greed_value": fg_value, "fear_greed_label": fg_label,
        "price": current_price, "usd_size": usd_size, "risk_pct": risk_pct,
        "leverage": leverage, "signal_score": score_int,
        "score_weighted": round(score_float, 2), "regime": regime,
    }

    if analysis["action"] == "BUY" and analysis["confidence"] >= CONFIDENCE_MIN:
        if PAPER_TRADING:
            save_trade({**base_record, "paper": True, "order_id": None})
            open_position(symbol, current_price, usd_size, risk_pct, "BUY", timeframe, mode_label, signals_snap, atr_value)
            log.info(f"  📝 PAPER {mode_label} BUY @ {current_price} | Trail={trail_stop} TP={take_profit} ATR={atr_value:.4f if atr_value else 'N/A'}")
            allocated_now = get_allocated_capital(open_positions)
            send_telegram(
                f"📝 <b>PAPER {'🚀' if use_futures else '✅'} {mode_label} [{timeframe}]</b>\n"
                f"<b>{symbol}</b> @ {current_price}\n"
                f"½ Partial TP: {partial_tp} | 🎯 TP: {take_profit}\n"
                f"🔴 Trail: {trail_stop} | 🛑 SL: {round(current_price*(1-STOP_LOSS_PCT),4)}\n"
                f"💰 ${usd_size}×{leverage} (score {score_float:+.1f})\n"
                f"💼 ${allocated_now:.1f}/${CAPITAL_TOTAL_USD*MAX_CAPITAL_EXPOSURE:.0f} | Conf: {int(analysis['confidence']*100)}%"
            )
        else:
            fn = execute_futures_trade if use_futures else execute_spot_trade
            args = [futures_ex if use_futures else trade_ex, symbol, "BUY", usd_size]
            order, exec_price = fn(*args)
            if order:
                actual = exec_price or current_price
                save_trade({**base_record, "paper": False, "order_id": order.get("id"), "price": actual})
                open_position(symbol, actual, usd_size, risk_pct, "BUY", timeframe, mode_label, signals_snap)
                send_telegram(f"{'🚀' if use_futures else '✅'} <b>{mode_label}</b> — {symbol} @ {actual}\n💰 ${usd_size}×{leverage}")

    elif analysis["action"] == "SELL" and analysis["confidence"] >= CONFIDENCE_MIN:
        if not use_futures:
            log.info("  ⏭️  SELL requiere futuros — skip")
            return
        if PAPER_TRADING:
            save_trade({**base_record, "paper": True, "order_id": None})
            open_position(symbol, current_price, usd_size, risk_pct, "SELL", timeframe, mode_label, signals_snap, atr_value)
            log.info(f"  📝 PAPER SHORT {mode_label} @ {current_price} | ATR={atr_value:.4f if atr_value else 'N/A'}")
            send_telegram(
                f"📉 <b>PAPER SHORT {mode_label} [{timeframe}]</b>\n"
                f"<b>{symbol}</b> @ {current_price}\n"
                f"Score: {score_float:+.1f} | Conf: {int(analysis['confidence']*100)}%\n"
                f"🛑 SL: {round(current_price*(1+STOP_LOSS_PCT),4)} | 🎯 TP: {round(current_price*(1-FUTURES_TP),4)}"
            )
        else:
            fn = execute_futures_trade if use_futures else execute_spot_trade
            order, _ = fn(futures_ex if use_futures else trade_ex, symbol, "SELL", usd_size)
            if order:
                save_trade({**base_record, "paper": False, "order_id": order.get("id")})

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run_bot():
    log.info("🤖 CryptoBot v10 — Aggressive Self-Learning Edition")
    log.info(f"Mode: {'PAPER' if PAPER_TRADING else 'REAL'} | Capital: ${CAPITAL_TOTAL_USD}")

    send_telegram(
        f"🤖 <b>CryptoBot v10 — Aggressive Self-Learning</b>\n"
        f"Mode: {'📝 PAPER' if PAPER_TRADING else '💰 REAL'}\n"
        f"💾 Persistencia: PostgreSQL\n"
        f"🌍 Filtro macro BTC 4h | 🛡️ Pump/dump filter\n"
        f"📊 Volume conviction | 🛑 Circuit breaker diario\n"
        f"⏰ Sin trading 00-06 UTC | 📉 Max 2 posiciones en pérdida\n"
        f"Spot trail {TRAILING_STOP_PCT*100}% (ATR cap 3%) | Futuros {FUTURES_LEVERAGE}x"
    )

    public_ex  = get_public_exchange()
    trade_ex   = get_trade_exchange()
    futures_ex = get_futures_exchange()
    state      = load_state()
    altcoins   = []
    last_scan  = 0
    last_regime_check = 0
    regime     = "unknown"

    while True:
        now = time.time()
        log.info(f"\n{'='*50}\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        state = load_state()
        check_drawdown(state)
        update_trailing_stops(public_ex, state)

        fg_value, fg_label = get_fear_greed()
        log.info(f"🧭 F&G: {fg_value} — {fg_label} | Régimen: {regime.upper()}")

        maybe_send_daily_report(fg_value, fg_label)
        maybe_run_weekly_backtest()

        # Recalcular pesos dinámicos
        recalculate_weights(regime)

        # Market regime cada 15 min
        if now - last_regime_check > 900:
            regime = detect_market_regime(public_ex)
            last_regime_check = now

        # Scan altcoins cada 10 min
        if now - last_scan > 600:
            log.info("🔍 Escaneando altcoins...")
            altcoins  = scan_top_altcoins(public_ex, max_alts=15)
            last_scan = now

        open_positions = load_positions()
        log.info(f"📂 Posiciones: {list(open_positions.keys()) or 'ninguna'} | Capital: ${state.get('capital',100):.2f} | RL min={state.get('rl_min_signals',MIN_SIGNALS_SIDEWAYS)}")

        log.info("\n--- BASE [1h] ---")
        for symbol in BASE_WATCHLIST:
            try:
                log.info(f"\n📊 {symbol}...")
                analyze_and_trade(symbol, "1h", public_ex, trade_ex, futures_ex,
                                  fg_value, fg_label, open_positions, regime, state)
                open_positions = load_positions()
            except Exception as e:
                log.error(f"Error {symbol}: {e}")

        if altcoins:
            log.info("\n--- ALTCOINS [3m] ---")
            for symbol in altcoins:
                try:
                    log.info(f"\n📊 {symbol} [3m]...")
                    analyze_and_trade(symbol, "3m", public_ex, trade_ex, futures_ex,
                                      fg_value, fg_label, open_positions, regime, state)
                    open_positions = load_positions()
                except Exception as e:
                    log.error(f"Error {symbol}: {e}")

        log.info(f"\n💤 {LOOP_INTERVAL_SEC}s...")
        time.sleep(LOOP_INTERVAL_SEC)

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_dashboard, daemon=True).start()
    run_bot()
