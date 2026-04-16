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
SIGNAL_MEMORY_FILE   = "signal_memory.json"
DYNAMIC_WEIGHTS_FILE = "dynamic_weights.json"
WEIGHTS_UPDATE_INTERVAL = 3600   # recalcular pesos cada 1h
MIN_SAMPLES_FOR_WEIGHT  = 5      # mínimo trades para confiar en un peso
RL_STREAK_THRESHOLD     = 3      # 3 pérdidas seguidas → subir MIN_SIGNALS
ENTRY_CONFIRM_CANDLES   = 1      # esperar N velas de confirmación antes de entrar
PARTIAL_EXIT_PCT        = 0.50   # cerrar 50% al TP parcial

TRADE_LOG_FILE   = "trade_log.json"
POSITIONS_FILE   = "positions.json"
STATE_FILE       = "bot_state.json"

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
    if os.path.exists(SIGNAL_MEMORY_FILE):
        try:
            with open(SIGNAL_MEMORY_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_signal_to_memory(signals_dict, pnl_pct, regime):
    """
    Guarda el resultado de un trade asociado a las señales activas.
    Esto alimenta el sistema de aprendizaje.
    """
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
        if pnl_pct > 0:
            memory[key]["wins"] += 1
        else:
            memory[key]["losses"] += 1
    with open(SIGNAL_MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

# ─────────────────────────────────────────
# DYNAMIC WEIGHTS — ponderación por historial
# ─────────────────────────────────────────
_weights_cache = {}
_weights_last_update = 0

DEFAULT_WEIGHTS = {
    "tech": 1.0, "macd": 1.0, "bb": 1.0, "ob": 1.0,
    "rsi_div": 1.0, "vol": 0.8, "funding": 1.0,
    "news": 0.5,   # bajado: RSS no es confiable para altcoins pequeñas
    "tf4h": 0.6,   # bajado: 15m EMA estaba bajando el score de BIO innecesariamente
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
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f: return json.load(f)
        except: pass
    return {
        "capital": CAPITAL_TOTAL_USD,
        "week_start_capital": CAPITAL_TOTAL_USD,
        "week_start_date": datetime.now(ARG_TZ).strftime("%Y-%m-%d"),
        "drawdown_mode": False,
        "last_backtest": None,
        "rl_min_signals": MIN_SIGNALS_SIDEWAYS,
    }

def save_state(state):
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
    if now_arg.weekday() == 0:
        week_date = now_arg.strftime("%Y-%m-%d")
        if state.get("week_start_date") != week_date:
            state["week_start_capital"] = current
            state["week_start_date"]    = week_date
            state["drawdown_mode"]      = False
            state["rl_min_signals"]     = MIN_SIGNALS_SIDEWAYS
            log.info(f"📅 Reset semanal — capital base: ${current:.2f}")
    if loss_pct > MAX_WEEKLY_LOSS_PCT and not state.get("drawdown_mode"):
        state["drawdown_mode"] = True
        send_telegram(f"⚠️ <b>DRAWDOWN MODE</b>\nPérdida semanal: {loss_pct*100:.1f}%\nSizing reducido 50%")
    elif loss_pct <= MAX_WEEKLY_LOSS_PCT * 0.5 and state.get("drawdown_mode"):
        state["drawdown_mode"] = False
    save_state(state)

# ─────────────────────────────────────────
# FLASK DASHBOARD
# ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CryptoBot v10</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#04080f;--s0:#070d17;--s1:#0b1220;--s2:#101828;
  --border:#162030;--border2:#1e2d42;
  --green:#0dffb0;--red:#ff3366;--yellow:#ffd700;
  --blue:#38bdf8;--orange:#fb923c;--purple:#c084fc;
  --green-dim:rgba(13,255,176,.08);--red-dim:rgba(255,51,102,.08);
  --text:#a8bfd4;--text2:#6b8299;--white:#e2f0ff;
  --font-mono:'IBM Plex Mono',monospace;--font-sans:'IBM Plex Sans',sans-serif;
  --radius:6px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--font-mono);font-size:13px;min-height:100vh}
body::after{content:'';position:fixed;inset:0;
  background-image:radial-gradient(circle at 20% 50%,rgba(13,255,176,.03) 0%,transparent 50%),
    radial-gradient(circle at 80% 20%,rgba(56,189,248,.03) 0%,transparent 40%);
  pointer-events:none;z-index:0}
.wrap{position:relative;z-index:1;max-width:1440px;margin:0 auto;padding:20px 24px}
header{display:flex;align-items:center;justify-content:space-between;padding-bottom:16px;margin-bottom:20px;border-bottom:1px solid var(--border)}
.brand{display:flex;align-items:baseline;gap:10px}
.brand-name{font-family:var(--font-sans);font-weight:700;font-size:18px;color:var(--white);letter-spacing:-.3px}
.brand-name span{color:var(--green)}
.brand-ver{font-size:10px;color:var(--text2);background:var(--s2);border:1px solid var(--border2);padding:2px 6px;border-radius:3px}
.header-right{display:flex;align-items:center;gap:8px}
.pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:100px;font-size:10px;font-weight:500}
.pill-live{color:var(--green);border:1px solid rgba(13,255,176,.3);background:rgba(13,255,176,.05)}
.pill-paper{color:var(--blue);border:1px solid rgba(56,189,248,.3);background:rgba(56,189,248,.05)}
.pill-mode{color:var(--purple);border:1px solid rgba(192,132,252,.3);background:rgba(192,132,252,.05)}
.dot{width:5px;height:5px;border-radius:50%;background:currentColor;animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.refresh-badge{font-size:10px;color:var(--text2);background:var(--s1);border:1px solid var(--border);padding:3px 8px;border-radius:3px}
#cd{color:var(--green)}
.kpis{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-bottom:20px}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(4,1fr)}}
@media(max-width:600px){.kpis{grid-template-columns:repeat(2,1fr)}}
.kpi{background:var(--s1);border:1px solid var(--border);border-radius:var(--radius);padding:12px 14px;position:relative;overflow:hidden}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--k-color,var(--green));opacity:.6}
.kpi-label{font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--text2);margin-bottom:6px}
.kpi-val{font-family:var(--font-sans);font-weight:700;font-size:22px;line-height:1;color:var(--white)}
.kpi-val.g{color:var(--green)}.kpi-val.r{color:var(--red)}.kpi-val.y{color:var(--yellow)}.kpi-val.b{color:var(--blue)}.kpi-val.o{color:var(--orange)}.kpi-val.p{color:var(--purple)}
.kpi-sub{font-size:10px;color:var(--text2);margin-top:3px}
.two-col{display:grid;grid-template-columns:1fr 340px;gap:12px;margin-bottom:16px}
@media(max-width:900px){.two-col{grid-template-columns:1fr}}
.panel{background:var(--s1);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:9px 14px;border-bottom:1px solid var(--border);background:var(--s2)}
.panel-title{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--text2);font-weight:600}
.panel-body{padding:12px 14px}
.pos-list{display:flex;flex-direction:column;gap:6px;padding:10px}
.pos-card{border-radius:5px;padding:10px 12px;border:1px solid var(--border2);background:var(--s2)}
.pos-card.long{border-left:2px solid var(--green)}
.pos-card.short{border-left:2px solid var(--red)}
.pos-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.pos-sym{font-family:var(--font-sans);font-weight:700;font-size:13px;color:var(--white)}
.pos-dir{font-size:9px;padding:1px 5px;border-radius:2px;font-weight:700}
.pos-dir.long{background:rgba(13,255,176,.12);color:var(--green)}
.pos-dir.short{background:rgba(255,51,102,.12);color:var(--red)}
.pos-pnl{font-family:var(--font-sans);font-weight:700;font-size:14px}
.pos-pnl.pos{color:var(--green)}.pos-pnl.neg{color:var(--red)}
.pos-meta{display:grid;grid-template-columns:1fr 1fr;gap:2px 8px;font-size:10px}
.pos-meta .lbl{color:var(--text2)}.pos-meta .val{color:var(--text);text-align:right}
.pos-bar{height:2px;background:var(--border2);border-radius:1px;margin-top:7px;overflow:hidden}
.pos-bar-fill{height:100%;background:linear-gradient(90deg,var(--orange),var(--yellow));border-radius:1px}
.pos-time{font-size:9px;color:var(--text2);margin-top:4px}
.heatmap{display:grid;grid-template-columns:repeat(9,1fr);gap:3px;padding:10px 14px}
.sig-cell{padding:5px 3px;border-radius:4px;text-align:center;border:1px solid var(--border)}
.sig-cell .sig-name{font-size:8px;color:var(--text2);letter-spacing:.05em}
.sig-cell .sig-count{font-family:var(--font-sans);font-weight:700;font-size:15px;margin:2px 0}
.sig-cell .sig-wr{font-size:8px}
.sig-hot{background:rgba(13,255,176,.1);border-color:rgba(13,255,176,.25)}
.sig-cold{background:rgba(255,51,102,.08);border-color:rgba(255,51,102,.2)}
.sig-neutral{background:var(--s2)}
.metrics-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:10px 14px}
.metric{text-align:center;padding:8px 4px}
.metric-val{font-family:var(--font-sans);font-weight:700;font-size:17px;color:var(--white)}
.metric-val.g{color:var(--green)}.metric-val.r{color:var(--red)}
.metric-lbl{font-size:9px;color:var(--text2);margin-top:3px;letter-spacing:.07em;text-transform:uppercase}
.weights-row{display:flex;gap:5px;flex-wrap:wrap;padding:10px 14px}
.w-chip{display:flex;flex-direction:column;align-items:center;padding:5px 8px;border-radius:4px;border:1px solid var(--border);background:var(--s2);min-width:54px}
.w-name{font-size:8px;color:var(--text2);letter-spacing:.08em}
.w-val{font-family:var(--font-sans);font-weight:700;font-size:14px;margin:1px 0}
.w-bar{width:100%;height:2px;background:var(--border2);border-radius:1px;overflow:hidden}
.w-fill{height:100%;border-radius:1px}
.scanner-row{display:flex;flex-wrap:wrap;gap:4px;padding:8px 14px}
.chip{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:3px;font-size:10px;border:1px solid var(--border)}
.chip.base{border-color:rgba(13,255,176,.25);color:var(--green)}
.chip.alt{border-color:rgba(192,132,252,.2);color:var(--purple)}
.tbl-wrap{background:var(--s1);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:20px}
table{width:100%;border-collapse:collapse;font-size:11px}
thead{background:var(--s2);border-bottom:1px solid var(--border)}
th{padding:7px 10px;text-align:left;font-size:8px;letter-spacing:.1em;text-transform:uppercase;color:var(--text2);font-weight:500}
td{padding:7px 10px;border-bottom:1px solid rgba(22,32,48,.6);vertical-align:middle}
tr:last-child td{border-bottom:none}tr:hover td{background:rgba(255,255,255,.015)}
.pair{color:var(--white);font-weight:600;font-size:12px}
.bdg{display:inline-block;padding:1px 6px;border-radius:2px;font-size:10px;font-weight:700}
.bdg-buy{background:var(--green-dim);color:var(--green);border:1px solid rgba(13,255,176,.2)}
.bdg-sell{background:var(--red-dim);color:var(--red);border:1px solid rgba(255,51,102,.2)}
.mini{display:inline-block;padding:1px 4px;border-radius:2px;font-size:8px;margin-left:2px}
.mini-p{background:rgba(56,189,248,.1);color:var(--blue);border:1px solid rgba(56,189,248,.2)}
.mini-f{background:rgba(192,132,252,.1);color:var(--purple);border:1px solid rgba(192,132,252,.2)}
.mini-h{background:rgba(251,146,60,.1);color:var(--orange);border:1px solid rgba(251,146,60,.2)}
.bar-wrap{display:flex;align-items:center;gap:4px}
.bar-bg{width:36px;height:2px;background:var(--border2);border-radius:1px;overflow:hidden}
.bar-fg{height:100%;background:var(--green);border-radius:1px}
.empty-row{text-align:center;padding:40px;color:var(--text2)}
.fg-tag{display:inline-block;padding:1px 5px;border-radius:2px;font-size:9px;font-weight:700}
.fg-f{background:rgba(255,51,102,.15);color:var(--red)}.fg-g{background:rgba(13,255,176,.15);color:var(--green)}.fg-n{background:rgba(255,215,0,.15);color:var(--yellow)}
.ts{color:var(--text2);font-size:10px}
footer{text-align:center;font-size:9px;color:var(--text2);padding-top:12px;border-top:1px solid var(--border);letter-spacing:.05em}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="brand">
    <div class="brand-name">Crypto<span>Bot</span></div>
    <span class="brand-ver">v10 Aggressive</span>
  </div>
  <div class="header-right">
    <span class="pill pill-mode"><div class="dot"></div>SELF-LEARNING</span>
    <span class="pill pill-paper" id="mode-pill"><div class="dot"></div><span id="mode-txt">PAPER</span></span>
    <span class="refresh-badge">&#8635; <span id="cd">15</span>s</span>
  </div>
</header>
<div class="kpis">
  <div class="kpi" style="--k-color:var(--green)"><div class="kpi-label">Capital</div><div class="kpi-val g" id="k-cap">—</div><div class="kpi-sub" id="k-cap-d">—</div></div>
  <div class="kpi" style="--k-color:var(--yellow)"><div class="kpi-label">P&L Total</div><div class="kpi-val y" id="k-pnl">—</div><div class="kpi-sub">en trades cerrados</div></div>
  <div class="kpi" style="--k-color:var(--blue)"><div class="kpi-label">Win Rate</div><div class="kpi-val b" id="k-wr">—</div><div class="kpi-sub" id="k-wr-d">—</div></div>
  <div class="kpi" style="--k-color:var(--orange)"><div class="kpi-label">Trades</div><div class="kpi-val o" id="k-trades">—</div><div class="kpi-sub" id="k-trades-d">—</div></div>
  <div class="kpi" style="--k-color:var(--red)"><div class="kpi-label">Fear & Greed</div><div class="kpi-val" id="k-fg">—</div><div class="kpi-sub" id="k-fg-l">—</div></div>
  <div class="kpi" style="--k-color:var(--purple)"><div class="kpi-label">Régimen</div><div class="kpi-val p" id="k-reg">—</div><div class="kpi-sub" id="k-reg-s">—</div></div>
  <div class="kpi" style="--k-color:var(--orange)"><div class="kpi-label">Posiciones</div><div class="kpi-val o" id="k-pos">0</div><div class="kpi-sub" id="k-cap-u">$0 usado</div></div>
  <div class="kpi" style="--k-color:var(--purple)"><div class="kpi-label">RL Min Score</div><div class="kpi-val p" id="k-rl">—</div><div class="kpi-sub">ajuste automático</div></div>
</div>
<div class="two-col">
  <div>
    <div class="panel" style="margin-bottom:12px">
      <div class="panel-head"><span class="panel-title">Curva de capital</span><span id="chart-stats" style="font-size:10px;color:var(--text2)">—</span></div>
      <div class="panel-body"><canvas id="pnl-chart" style="width:100%;height:120px"></canvas></div>
    </div>
    <div class="panel" style="margin-bottom:12px">
      <div class="panel-head"><span class="panel-title">Performance</span></div>
      <div class="metrics-grid">
        <div class="metric"><div class="metric-val g" id="m-best">—</div><div class="metric-lbl">Mejor trade</div></div>
        <div class="metric"><div class="metric-val r" id="m-worst">—</div><div class="metric-lbl">Peor trade</div></div>
        <div class="metric"><div class="metric-val g" id="m-avgw">—</div><div class="metric-lbl">Avg Win</div></div>
        <div class="metric"><div class="metric-val r" id="m-avgl">—</div><div class="metric-lbl">Avg Loss</div></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title">Señales activas (últimos 20 trades)</span></div>
      <div class="heatmap" id="sig-heatmap"></div>
    </div>
  </div>
  <div>
    <div class="panel" style="margin-bottom:12px">
      <div class="panel-head"><span class="panel-title">Posiciones abiertas</span><span id="pos-count" style="font-size:10px;color:var(--text2)">ninguna</span></div>
      <div id="pos-list" class="pos-list"><div style="padding:20px;text-align:center;color:var(--text2);font-size:11px">Sin posiciones abiertas</div></div>
    </div>
    <div class="panel" style="margin-bottom:12px">
      <div class="panel-head"><span class="panel-title">Pesos dinámicos</span></div>
      <div class="weights-row" id="weights-row"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><span class="panel-title">Scanner activo</span></div>
      <div class="scanner-row" id="scanner-row"></div>
    </div>
  </div>
</div>
<div style="background:var(--s2);border:1px solid var(--border);border-radius:6px 6px 0 0;padding:9px 14px">
  <span style="font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--text2);font-weight:600">Historial de trades</span>
</div>
<div class="tbl-wrap" style="border-top:none;border-radius:0 0 6px 6px">
  <table>
    <thead><tr><th>Par</th><th>P&L</th><th>TF</th><th>Precio</th><th>Size</th><th>Tipo</th><th>Score</th><th>Conf</th><th>Señales</th><th>Razón</th><th>Hora</th></tr></thead>
    <tbody id="tbody"></tbody>
  </table>
</div>
<footer>CryptoBot v10 · Aggressive Self-Learning · ATR Trailing · Shorts · Dynamic Weights · Entry Confirmation</footer>
</div>
<script>
let cd=15;
const ICAP=1000;
function fmt(n,d=2){return(+n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d})}
function fmtP(n){return(n>=0?'+':'')+fmt(n)+'%'}
function drawChart(canvas,pts){
  if(!pts||pts.length<2)return;
  const W=canvas.clientWidth||600,H=120;
  canvas.width=W;canvas.height=H;
  const ctx=canvas.getContext('2d');
  const pad=10;
  const mn=Math.min(...pts),mx=Math.max(...pts),rng=mx-mn||1;
  const x=i=>pad+(W-pad*2)*i/(pts.length-1);
  const y=v=>H-pad-(H-pad*2)*(v-mn)/rng;
  ctx.clearRect(0,0,W,H);
  const last=pts[pts.length-1];
  const col=last>=ICAP?'13,255,176':'255,51,102';
  const grad=ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'rgba('+col+',.18)');grad.addColorStop(1,'rgba('+col+',0)');
  ctx.beginPath();ctx.moveTo(x(0),y(pts[0]));
  for(let i=1;i<pts.length;i++)ctx.lineTo(x(i),y(pts[i]));
  ctx.lineTo(x(pts.length-1),H);ctx.lineTo(x(0),H);ctx.closePath();
  ctx.fillStyle=grad;ctx.fill();
  ctx.strokeStyle='#'+((col==='13,255,176')?'0dffb0':'ff3366');
  ctx.lineWidth=1.5;ctx.lineJoin='round';
  ctx.beginPath();ctx.moveTo(x(0),y(pts[0]));
  for(let i=1;i<pts.length;i++)ctx.lineTo(x(i),y(pts[i]));
  ctx.stroke();
  ctx.fillStyle=ctx.strokeStyle;
  ctx.beginPath();ctx.arc(x(pts.length-1),y(last),3,0,Math.PI*2);ctx.fill();
}
async function load(){
  try{
    const d=await fetch('/api/trades').then(r=>r.json());
    const st=d.state||{};
    const trades=(d.trades||[]).filter(t=>t.action!=='HOLD');
    const closed=trades.filter(t=>t.pnl_pct!==undefined&&!t.partial_exit);
    const wins=closed.filter(t=>t.pnl_pct>0);
    const cap=st.capital||ICAP;
    const diff=cap-ICAP;
    document.getElementById('k-cap').textContent='$'+fmt(cap);
    document.getElementById('k-cap').className='kpi-val '+(cap>=ICAP?'g':'r');
    document.getElementById('k-cap-d').textContent=(diff>=0?'+':'')+fmt(diff)+' desde inicio';
    const tpnl=closed.reduce((s,t)=>s+(t.pnl_pct||0)*(t.usd_size||20)/100,0);
    document.getElementById('k-pnl').textContent=(tpnl>=0?'+':'')+'$'+fmt(tpnl);
    document.getElementById('k-pnl').className='kpi-val '+(tpnl>=0?'g':'r');
    const wr=closed.length?Math.round(wins.length/closed.length*100):null;
    document.getElementById('k-wr').textContent=wr!==null?wr+'%':'—';
    document.getElementById('k-wr').className='kpi-val '+(wr===null?'b':wr>=50?'b':'r');
    document.getElementById('k-wr-d').textContent=closed.length?wins.length+'/'+closed.length+' trades':'sin datos';
    document.getElementById('k-trades').textContent=trades.length;
    document.getElementById('k-trades-d').textContent=closed.length+' cerrados';
    if(d.fear_greed){const fg=d.fear_greed,fv=+fg.value;document.getElementById('k-fg').textContent=fv;document.getElementById('k-fg').className='kpi-val '+(fv<35?'r':fv>65?'g':'y');document.getElementById('k-fg-l').textContent=fg.label;}
    const rm={bull:'BULL 📈',bear:'BEAR 📉',sideways:'SW ↔',crash:'CRASH 💥'};
    const reg=d.regime||'?';
    document.getElementById('k-reg').textContent=rm[reg]||reg.toUpperCase();
    document.getElementById('k-reg-s').textContent=reg;
    const pos=d.positions||[];
    document.getElementById('k-pos').textContent=pos.length;
    const alloc=pos.reduce((s,p)=>s+(p.usd_size||0),0);
    document.getElementById('k-cap-u').textContent='$'+fmt(alloc)+' usado';
    document.getElementById('k-rl').textContent=st.rl_min_signals||3;
    const hasReal=trades.some(t=>!t.paper);
    document.getElementById('mode-txt').textContent=hasReal?'REAL':'PAPER';
    document.getElementById('mode-pill').className='pill '+(hasReal?'pill-live':'pill-paper');
    // Chart
    const curve=[ICAP];let run=ICAP;
    [...closed].sort((a,b)=>new Date(a.timestamp)-new Date(b.timestamp)).forEach(t=>{run+=(t.pnl_pct||0)*(t.usd_size||20)/100;curve.push(run);});
    const cv=document.getElementById('pnl-chart');
    drawChart(cv,curve);
    if(curve.length>1)document.getElementById('chart-stats').textContent='min $'+fmt(Math.min(...curve))+' · max $'+fmt(Math.max(...curve))+' · '+closed.length+' pts';
    // Metrics
    if(closed.length){
      const pnls=closed.map(t=>t.pnl_pct||0);
      const wP=closed.filter(t=>t.pnl_pct>0).map(t=>t.pnl_pct);
      const lP=closed.filter(t=>t.pnl_pct<=0).map(t=>t.pnl_pct);
      document.getElementById('m-best').textContent=fmtP(Math.max(...pnls));
      document.getElementById('m-worst').textContent=fmtP(Math.min(...pnls));
      document.getElementById('m-avgw').textContent=wP.length?fmtP(wP.reduce((a,b)=>a+b,0)/wP.length):'—';
      document.getElementById('m-avgl').textContent=lP.length?fmtP(lP.reduce((a,b)=>a+b,0)/lP.length):'—';
    }
    // Heatmap
    const rec=closed.slice(-20);
    const sdefs=[{k:'tech',n:'EMA'},{k:'macd',n:'MACD'},{k:'bb',n:'BB'},{k:'ob',n:'OB'},{k:'vol',n:'VOL'},{k:'funding',n:'FR'},{k:'news',n:'NEWS'},{k:'rsi_div',n:'RSI▲'},{k:'tf4h',n:'CONF'}];
    document.getElementById('sig-heatmap').innerHTML=sdefs.map(s=>{
      const hits=rec.filter(t=>t[s.k+'_signal']===1||t[s.k+'_signal']===-1||t[s.k]);
      const ws=hits.filter(t=>(t.pnl_pct||0)>0);
      const sr=hits.length?Math.round(ws.length/hits.length*100):null;
      const cls=hits.length>=5?'sig-hot':hits.length<=1?'sig-cold':'sig-neutral';
      const wc=sr===null?'var(--text2)':sr>=60?'var(--green)':sr<40?'var(--red)':'var(--yellow)';
      return '<div class="sig-cell '+cls+'"><div class="sig-name">'+s.n+'</div><div class="sig-count" style="color:'+(hits.length>=5?'var(--green)':'var(--text)')+'">'+hits.length+'</div><div class="sig-wr" style="color:'+wc+'">'+(sr!==null?sr+'%':'—')+'</div></div>';
    }).join('');
    // Positions
    document.getElementById('pos-count').textContent=pos.length||'ninguna';
    document.getElementById('pos-list').innerHTML=pos.length?pos.map(p=>{
      const isS=p.action==='SELL';
      const pp=isS?((p.entry_price-p.current_price)/p.entry_price*100):((p.current_price-p.entry_price)/p.entry_price*100);
      const isp=pp>=0;
      const oa=p.opened_at?new Date(p.opened_at):null;
      const el=oa?Math.round((Date.now()-oa)/60000):null;
      const dt=isS?((p.trail_stop-p.current_price)/p.current_price*100):((p.current_price-p.trail_stop)/p.current_price*100);
      const bw=Math.min(100,Math.max(0,(1-dt/5)*100));
      return '<div class="pos-card '+(isS?'short':'long')+'"><div class="pos-top"><span class="pos-sym">'+p.symbol.replace('/USDT','')+' <span class="pos-dir '+(isS?'short':'long')+'">'+(isS?'SHORT':'LONG')+'</span></span><span class="pos-pnl '+(isp?'pos':'neg')+'">'+(isp?'+':'')+pp.toFixed(2)+'%</span></div><div class="pos-meta"><span class="lbl">Entrada</span><span class="val">'+fmt(p.entry_price,4)+'</span><span class="lbl">Actual</span><span class="val">'+fmt(p.current_price,4)+'</span><span class="lbl">Trail</span><span class="val" style="color:var(--orange)">'+fmt(p.trail_stop,4)+'</span><span class="lbl">TP</span><span class="val" style="color:var(--green)">'+fmt(p.take_profit,4)+'</span><span class="lbl">Size</span><span class="val">$'+p.usd_size+'</span><span class="lbl">Modo</span><span class="val" style="color:var(--purple)">'+(p.mode||'SPOT')+'</span></div>'+(p.partial_closed?'<div style="font-size:9px;color:var(--orange);margin-top:4px">½ partial exit ejecutado</div>':'')+'<div class="pos-bar"><div class="pos-bar-fill" style="width:'+bw+'%"></div></div>'+(el!==null?'<div class="pos-time">Hace '+(el<60?el+'m':(el/60).toFixed(1)+'h')+'</div>':'')+'</div>';
    }).join(''):'<div style="padding:20px;text-align:center;color:var(--text2);font-size:11px">Sin posiciones abiertas</div>';
    // Weights
    const wn={tech:'EMA',macd:'MACD',bb:'BB',ob:'OB',rsi_div:'RSI▲',vol:'VOL',funding:'FR',news:'NEWS',tf4h:'CONF'};
    const wts=d.weights||{};
    document.getElementById('weights-row').innerHTML=Object.entries(wn).map(([k,n])=>{const w=wts[k]||1;const c=w>=1.3?'var(--green)':w<=0.7?'var(--red)':'var(--text)';const p=Math.min(100,w/2*100);return '<div class="w-chip"><div class="w-name">'+n+'</div><div class="w-val" style="color:'+c+'">'+w.toFixed(1)+'x</div><div class="w-bar"><div class="w-fill" style="width:'+p+'%;background:'+c+'"></div></div></div>';}).join('');
    // Scanner
    const sc=d.scanner||[];
    document.getElementById('scanner-row').innerHTML=sc.map(s=>{const b=['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT'].includes(s.symbol);const v=s.volume?'<span style="color:var(--text2);font-size:9px"> $'+Math.round(s.volume/1e6)+'M</span>':'';return '<span class="chip '+(b?'base':'alt')+'">'+s.symbol.replace('/USDT','')+v+'</span>';}).join('');
    // Table
    const tb=document.getElementById('tbody');
    if(!trades.length){tb.innerHTML='<tr><td colspan="11"><div class="empty-row">🤖 Sin trades aún. El bot está aprendiendo...</div></td></tr>';return;}
    tb.innerHTML=[...trades].reverse().map(t=>{
      const ts=new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const conf=Math.round((t.confidence||0)*100);
      const sigs=[];
      if(t.tech_signal===1)sigs.push('<span style="color:var(--green)">EMA+</span>');else if(t.tech_signal===-1)sigs.push('<span style="color:var(--red)">EMA-</span>');
      if(t.macd_signal===1)sigs.push('<span style="color:var(--green)">MACD+</span>');else if(t.macd_signal===-1)sigs.push('<span style="color:var(--red)">MACD-</span>');
      if(t.rsi_div)sigs.push('<span style="color:var(--yellow)">RSI▲</span>');
      if(t.ob_signal===1)sigs.push('<span style="color:var(--blue)">OB+</span>');
      if(t.vol_signal===1)sigs.push('<span style="color:var(--purple)">VOL+</span>');
      if(t.funding_signal===1)sigs.push('<span style="color:var(--green)">FR+</span>');
      const fg=t.fear_greed_value||'—';
      const pc=t.pnl_pct>=0?'var(--green)':'var(--red)';
      const ps=t.pnl_pct!==undefined?'<span style="color:'+pc+';font-weight:700"> '+(t.pnl_pct>=0?'+':'')+t.pnl_pct+'%</span>':'<span style="color:var(--text2)">open</span>';
      const sc_v=(+(t.score_weighted||0)).toFixed(1);
      const sc_c=+sc_v>0?'var(--green)':+sc_v<0?'var(--red)':'var(--text2)';
      return '<tr><td class="pair">'+t.symbol+'</td><td>'+ps+'</td><td style="color:'+(t.timeframe==='3m'?'var(--purple)':'var(--text2)')+'">'+t.timeframe+'</td><td>'+(t.price?(+t.price).toLocaleString('en-US',{maximumFractionDigits:4}):'—')+'</td><td style="color:var(--orange)">$'+(t.usd_size||'—')+'</td><td><span class="bdg bdg-'+(t.action||'buy').toLowerCase()+'">'+t.action+'</span>'+(t.paper?'<span class="mini mini-p">P</span>':'')+(t.mode&&t.mode.includes('FUT')?'<span class="mini mini-f">F</span>':'')+(t.partial_exit?'<span class="mini mini-h">½</span>':'')+'</td><td style="color:'+sc_c+'">'+sc_v+'</td><td><div class="bar-wrap"><div class="bar-bg"><div class="bar-fg" style="width:'+conf+'%"></div></div><span style="font-size:10px">'+conf+'%</span></div></td><td style="font-size:10px">'+sigs.join(' ')+'</td><td style="color:var(--text2);max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="'+(t.reasoning||'')+'">'+( t.reasoning||'—')+'</td><td class="ts">'+ts+'</td></tr>';
    }).join('');
  }catch(e){console.error(e);}
}
function tick(){cd--;document.getElementById('cd').textContent=cd;if(cd<=0){cd=15;load();}}
load();setInterval(tick,1000);
window.addEventListener('resize',load);
</script>
</body>
</html>"""

flask_app   = Flask(__name__)
_fear_greed = {"value": 50, "label": "Neutral"}
_scanner    = []
_regime     = "unknown"

@flask_app.route("/")
def index(): return render_template_string(DASHBOARD_HTML)

@flask_app.route("/api/trades")
def api_trades():
    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: trades = json.load(f)
        except: pass
    positions = load_positions()
    state     = load_state()
    weights   = _weights_cache if _weights_cache else DEFAULT_WEIGHTS
    return jsonify({
        "trades": trades, "count": len(trades),
        "fear_greed": _fear_greed,
        "positions": list(positions.values()),
        "scanner": _scanner,
        "regime": _regime,
        "state": state,
        "weights": weights,
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
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0); loss = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean() /
                               loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["vol_ma20"]    = df["volume"].rolling(20).mean()
    df["bb_mid"]      = df["close"].rolling(BB_PERIOD).mean()
    bb_std = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"]    = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"]    = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    # ATR para trailing dinámico por volatilidad
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
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
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_positions(positions):
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
                    partial_pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
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
                    pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
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
                    pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
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
                    pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
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
                    pnl = (pos["entry_price"] - current) / pos["entry_price"] * 100
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
                    pnl = (pos["entry_price"] - current) / pos["entry_price"] * 100
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
                    pnl = (pos["entry_price"] - current) / pos["entry_price"] * 100
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
    data = []
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE) as f: data = json.load(f)
    data.append(record)
    with open(TRADE_LOG_FILE,"w") as f: json.dump(data, f, indent=2, default=str)

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

    signals = {
        "tech": t_sig, "macd": m_sig, "vol": v_sig, "tf4h": h_sig,
        "news": n_sig, "funding": fr_sig, "bb": bb_sig, "ob": ob_sig,
        "rsi_div": rsi_sig, "rsi_div_val": rsi_div,
    }

    # Score ponderado por aprendizaje
    score_float, score_int = weighted_score(signals, regime)
    log.info(f"  [{timeframe}] EMA:{t_sig:+d} MACD:{m_sig:+d} BB:{bb_sig:+d} OB:{ob_sig:+d} RSIDiv:{rsi_sig:+d} VOL:{v_sig:+d} FR:{fr_sig:+d} NEWS:{n_sig:+d} = {score_float:+.1f} (raw int={score_int:+d})")

    # Score mínimo dinámico (régimen + RL)
    # Sideways: threshold float 2.5 — ADA +2.7 entra, DOGE +1.7 no entra
    # Bull/Bear: threshold int 2 — más permisivo
    rl_min = rl_adjust_min_signals(state)
    if regime == "sideways":
        min_float = SIDEWAYS_MIN_FLOAT
        if abs(score_float) < min_float:
            log.info(f"  ⏭️  Score {score_float:+.1f} < {min_float} (sideways) — skip")
            return
    else:
        min_score = max(MIN_SIGNALS, rl_min - 1)
        if abs(score_int) < min_score:
            log.info(f"  ⏭️  Score {score_int:+d} < {min_score} ({regime}) — skip")
            return

    # Shorts habilitados en futuros — solo bloquear si SHORT_ENABLED=False
    if score_float < 0 and not SHORT_ENABLED:
        log.info("  ⏭️  Bajista y shorts deshabilitados — skip")
        return
    # Para shorts en altcoins 3m, requerir score ≤ SHORT_MIN_SCORE
    if timeframe == "3m" and score_float < 0 and score_float > SHORT_MIN_SCORE:
        log.info(f"  ⏭️  Short score {score_float:+.1f} insuficiente (mínimo {SHORT_MIN_SCORE}) — skip")
        return

    # Correlación
    base_symbols = set(BASE_WATCHLIST)
    open_alts    = [s for s in open_positions if s not in base_symbols]
    if symbol not in base_symbols and len(open_alts) >= MAX_CORRELATION_ALTS:
        log.info(f"  ⏭️  Correlación: {len(open_alts)} altcoins abiertas — skip")
        return

    # Entry confirmation delay
    if not check_entry_confirmation(symbol, signals, df, timeframe):
        return

    effective_cap = get_effective_capital(state)
    usd_size, risk_pct = get_position_size(effective_cap, score_int, regime)

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
        f"🤖 <b>CryptoBot v9 — Self-Learning</b>\n"
        f"Mode: {'📝 PAPER' if PAPER_TRADING else '💰 REAL'}\n"
        f"✨ Dynamic Weights — aprende de cada trade\n"
        f"⏳ Entry confirmation delay: {ENTRY_CONFIRM_CANDLES} vela(s)\n"
        f"½ Partial exit al {TAKE_PROFIT_PARTIAL*100}% | 🛑 SL {STOP_LOSS_PCT*100}%\n"
        f"🤖 RL: ajuste automático de min_signals\n"
        f"Spot trail {TRAILING_STOP_PCT*100}% | Futuros {FUTURES_LEVERAGE}x (score≥{FUTURES_MIN_SCORE})"
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
