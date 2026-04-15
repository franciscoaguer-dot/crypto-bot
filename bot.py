"""
CryptoBot v7 — Full Featured
- Noticias neutral por defecto (fix)
- RSI Divergence
- Bollinger Bands
- Order Book Imbalance
- Auto-backtesting semanal con Claude
- Market Regime Detector
- Compounding automático
- Anti-drawdown (circuit breaker)
- Spot + Futuros 2x
- 15 altcoins dinámicas
- Trailing stop + position sizing dinámico
- Funding rates + Capital allocator
- Resumen diario 9 AM
- Paper trading mode
"""

import os
import re
import time
import json
import logging
import requests
import threading
from datetime import datetime, timezone, timedelta
import ccxt
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

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

# Position sizing dinámico
POSITION_SIZE_MAP  = {2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.06}

# Spot
TRAILING_STOP_PCT  = 0.010
TAKE_PROFIT_PCT    = 0.025

# Futuros
FUTURES_TRAILING   = 0.008
FUTURES_TP         = 0.020
FUTURES_MIN_SCORE  = 4
FUTURES_LEVERAGE   = 2

# Operación
MIN_SIGNALS        = 2
CONFIDENCE_MIN     = 0.50
LOOP_INTERVAL_SEC  = 60
MAX_CAPITAL_EXPOSURE = 0.20

# Anti-drawdown
MAX_WEEKLY_LOSS_PCT  = 0.10   # si perdemos >10% en la semana, reducir posiciones
DRAWDOWN_REDUCE_PCT  = 0.50   # reducir position sizing a 50%

# Funding Rate
FUNDING_BULLISH_THRESHOLD = -0.0001
FUNDING_BEARISH_THRESHOLD =  0.0015

# Order Book
OB_IMBALANCE_THRESHOLD = 0.60   # >60% bids = bullish, <40% = bearish

# Bollinger Bands
BB_PERIOD = 20
BB_STD    = 2.0

# Market Regime
REGIME_TREND_THRESHOLD  = 0.02   # diferencia EMA50/EMA200 > 2% = tendencia
REGIME_CRASH_THRESHOLD  = -0.05  # caída >5% en 24h = crash

# Archivos
TRADE_LOG_FILE    = "trade_log.json"
POSITIONS_FILE    = "positions.json"
STATE_FILE        = "bot_state.json"

# Watchlist base
BASE_WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

EXCLUDE_SYMBOLS = {
    "USDT","USDC","BUSD","DAI","TUSD","FDUSD","USDP","USD1","RLUSD",
    "WBTC","WETH","STETH","BETH","BTC","ETH","SOL","BNB","LDUSDT","XAUT","PAXG"
}

ARG_TZ = timezone(timedelta(hours=-3))

# ─────────────────────────────────────────
# STATE — capital dinámico y anti-drawdown
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
    }

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2, default=str)

def get_effective_capital(state):
    """Capital ajustado por compounding y anti-drawdown."""
    cap = state.get("capital", CAPITAL_TOTAL_USD)
    if state.get("drawdown_mode"):
        cap = cap * DRAWDOWN_REDUCE_PCT
        log.info(f"  ⚠️ DRAWDOWN MODE: capital reducido a ${cap:.2f}")
    return cap

def update_compounding(state, pnl_usd):
    """Actualiza capital con ganancias/pérdidas reales."""
    state["capital"] = round(state["capital"] + pnl_usd, 2)
    save_state(state)
    log.info(f"  💰 Capital actualizado: ${state['capital']:.2f} ({'+' if pnl_usd>=0 else ''}{pnl_usd:.2f})")

def check_drawdown(state):
    """Activa/desactiva modo anti-drawdown."""
    week_start = state.get("week_start_capital", CAPITAL_TOTAL_USD)
    current    = state.get("capital", CAPITAL_TOTAL_USD)
    loss_pct   = (week_start - current) / week_start

    now_arg = datetime.now(ARG_TZ)
    # Reset semanal los lunes
    if now_arg.weekday() == 0:
        week_date = now_arg.strftime("%Y-%m-%d")
        if state.get("week_start_date") != week_date:
            state["week_start_capital"] = current
            state["week_start_date"]    = week_date
            state["drawdown_mode"]      = False
            log.info(f"📅 Reset semanal — capital base: ${current:.2f}")

    if loss_pct > MAX_WEEKLY_LOSS_PCT and not state.get("drawdown_mode"):
        state["drawdown_mode"] = True
        save_state(state)
        send_telegram(
            f"⚠️ <b>DRAWDOWN MODE activado</b>\n"
            f"Pérdida semanal: {loss_pct*100:.1f}%\n"
            f"Position sizing reducido al 50% hasta el lunes"
        )
    elif loss_pct <= MAX_WEEKLY_LOSS_PCT * 0.5 and state.get("drawdown_mode"):
        state["drawdown_mode"] = False
        save_state(state)
        log.info("✅ Drawdown mode desactivado")

    save_state(state)

# ─────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CryptoBot v7</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root{--bg:#080c10;--surface:#0d1117;--border:#1a2332;--green:#00ff88;--red:#ff3355;--yellow:#ffcc00;--blue:#00aaff;--orange:#ff9900;--purple:#aa55ff;--muted:#3d5166;--text:#c9d8e8;--mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);min-height:100vh}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,136,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
  .container{position:relative;z-index:1;max-width:1300px;margin:0 auto;padding:32px 20px}
  header{display:flex;align-items:center;justify-content:space-between;margin-bottom:32px;padding-bottom:18px;border-bottom:1px solid var(--border)}
  .logo{font-family:var(--sans);font-weight:800;font-size:1.4rem;color:#fff}.logo span{color:var(--green)}
  .logo small{font-size:.65rem;color:var(--muted);margin-left:8px}
  .badges{display:flex;gap:6px;flex-wrap:wrap}
  .pill{display:flex;align-items:center;gap:5px;font-size:.68rem;padding:4px 10px;border-radius:100px}
  .pill-live{color:var(--green);border:1px solid rgba(0,255,136,.3)}
  .pill-paper{color:var(--blue);border:1px solid rgba(0,170,255,.3)}
  .pill-warn{color:var(--orange);border:1px solid rgba(255,153,0,.3)}
  .dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:24px}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;position:relative;overflow:hidden}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,var(--green));opacity:.7}
  .stat-label{font-size:.57rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
  .stat-value{font-family:var(--sans);font-weight:800;font-size:1.5rem;line-height:1;color:#fff}
  .stat-value.green{color:var(--green)}.stat-value.red{color:var(--red)}.stat-value.yellow{color:var(--yellow)}
  .stat-value.blue{color:var(--blue)}.stat-value.orange{color:var(--orange)}.stat-value.purple{color:var(--purple)}
  .stat-sub{font-size:.58rem;color:var(--muted);margin-top:4px}
  .regime-badge{display:inline-block;padding:3px 8px;border-radius:4px;font-size:.65rem;font-weight:700;margin-top:4px}
  .regime-bull{background:rgba(0,255,136,.15);color:var(--green)}
  .regime-bear{background:rgba(255,51,85,.15);color:var(--red)}
  .regime-side{background:rgba(255,204,0,.15);color:var(--yellow)}
  .regime-crash{background:rgba(255,51,85,.3);color:var(--red);animation:pulse 1s infinite}
  .section-title{font-family:var(--sans);font-size:.68rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}
  .pos-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;margin-bottom:20px}
  .pos-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px}
  .pos-card.profit{border-color:rgba(0,255,136,.3)}.pos-card.loss{border-color:rgba(255,51,85,.3)}
  .pos-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
  .pos-symbol{font-family:var(--sans);font-weight:700;font-size:.9rem;color:#fff}
  .pos-pnl{font-family:var(--sans);font-weight:700;font-size:.82rem}
  .pos-pnl.pos{color:var(--green)}.pos-pnl.neg{color:var(--red)}
  .pos-row{display:flex;justify-content:space-between;font-size:.65rem;color:var(--muted);margin-top:2px}
  .pos-row span:last-child{color:var(--text)}
  .trail-bar-bg{height:3px;background:var(--border);border-radius:2px;margin-top:7px;overflow:hidden}
  .trail-bar-fill{height:100%;background:var(--orange);border-radius:2px}
  .table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:24px}
  table{width:100%;border-collapse:collapse;font-size:.72rem}
  thead tr{border-bottom:1px solid var(--border)}
  th{padding:9px 12px;text-align:left;font-size:.57rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:400}
  td{padding:9px 12px;border-bottom:1px solid rgba(26,35,50,.5);vertical-align:middle}
  tr:last-child td{border-bottom:none}tr:hover td{background:rgba(255,255,255,.02)}
  .badge{display:inline-block;padding:2px 7px;border-radius:3px;font-size:.63rem;font-weight:700}
  .badge-buy{background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.2)}
  .badge-sell{background:rgba(255,51,85,.12);color:var(--red);border:1px solid rgba(255,51,85,.2)}
  .badge-mini{font-size:.5rem;padding:1px 4px;border-radius:2px;margin-left:3px}
  .badge-paper{background:rgba(0,170,255,.1);color:var(--blue);border:1px solid rgba(0,170,255,.2)}
  .badge-fut{background:rgba(170,85,255,.1);color:var(--purple);border:1px solid rgba(170,85,255,.2)}
  .badge-trail{background:rgba(255,153,0,.1);color:var(--orange);border:1px solid rgba(255,153,0,.2)}
  .conf-bar{display:flex;align-items:center;gap:5px}
  .bar-bg{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
  .bar-fill{height:100%;background:var(--green);border-radius:2px}
  .pair{color:#fff;font-weight:600}.ts{color:var(--muted);font-size:.63rem}
  .empty{text-align:center;padding:40px 20px;color:var(--muted)}
  .empty-icon{font-size:1.8rem;margin-bottom:8px}.empty-text{font-size:.78rem;line-height:1.6}
  footer{text-align:center;font-size:.62rem;color:var(--muted);padding-top:16px;border-top:1px solid var(--border)}
  .refresh-info{font-size:.62rem;color:var(--muted);text-align:right;margin-bottom:8px}
  #countdown{color:var(--green)}
  .fg-pill{display:inline-block;padding:2px 6px;border-radius:3px;font-size:.58rem;font-weight:700}
  .fg-fear{background:rgba(255,51,85,.15);color:var(--red)}
  .fg-greed{background:rgba(0,255,136,.15);color:var(--green)}
  .fg-neutral{background:rgba(255,204,0,.15);color:var(--yellow)}
  .scanner-grid{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:20px}
  .chip{background:var(--surface);border:1px solid var(--border);border-radius:5px;padding:4px 10px;font-size:.65rem}
  .chip.base{border-color:rgba(0,255,136,.25);color:var(--green)}
  .chip.alt{border-color:rgba(170,85,255,.25);color:var(--purple)}
  .chip-vol{color:var(--muted);font-size:.58rem;margin-left:4px}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">Crypto<span>Bot</span><small>v7</small></div>
    <div class="badges">
      <div class="pill pill-paper" id="mode-pill"><div class="dot"></div><span id="mode-text">PAPER</span></div>
      <div class="pill pill-live" id="drawdown-pill" style="display:none"><div class="dot"></div>⚠️ DRAWDOWN</div>
      <div class="pill pill-live"><div class="dot"></div>LIVE</div>
    </div>
  </header>

  <div class="stats">
    <div class="stat-card" style="--accent:var(--green)"><div class="stat-label">Capital</div><div class="stat-value green" id="capital">—</div><div class="stat-sub" id="capital-change">—</div></div>
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Total Trades</div><div class="stat-value" id="total">—</div><div class="stat-sub">ejecutados</div></div>
    <div class="stat-card" style="--accent:var(--green)"><div class="stat-label">Win Rate</div><div class="stat-value green" id="winrate">—</div><div class="stat-sub">trades cerrados</div></div>
    <div class="stat-card" style="--accent:var(--yellow)"><div class="stat-label">P&L Total</div><div class="stat-value yellow" id="pnl">—</div><div class="stat-sub">paper USD</div></div>
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Fear & Greed</div><div class="stat-value blue" id="fg-val">—</div><div class="stat-sub" id="fg-lbl">—</div></div>
    <div class="stat-card" style="--accent:var(--orange)"><div class="stat-label">Posiciones</div><div class="stat-value orange" id="open-pos">—</div><div class="stat-sub">abiertas</div></div>
    <div class="stat-card" style="--accent:var(--purple)"><div class="stat-label">Régimen</div><div class="stat-value purple" id="regime-val">—</div><div class="stat-sub" id="regime-sub">mercado</div></div>
    <div class="stat-card" style="--accent:var(--orange)"><div class="stat-label">Capital usado</div><div class="stat-value orange" id="cap-used">—</div><div class="stat-sub">de $20 máx</div></div>
  </div>

  <div class="section-title">Scanner activo</div>
  <div class="scanner-grid" id="scanner-grid"><span style="color:var(--muted);font-size:.72rem">Cargando...</span></div>

  <div id="positions-section" style="display:none;margin-bottom:20px">
    <div class="section-title">Posiciones abiertas</div>
    <div class="pos-grid" id="pos-grid"></div>
  </div>

  <div class="refresh-info">Auto-refresh en <span id="countdown">30</span>s</div>
  <div class="section-title">Historial de operaciones</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Par</th><th>TF</th><th>Tipo</th><th>Precio</th><th>Size</th><th>Conf</th><th>Señales</th><th>P&L</th><th>Razonamiento</th><th>Hora</th></tr></thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>
  <footer>CryptoBot v7 · RSI Div · BB · Order Book · Regime · Compounding · Anti-Drawdown</footer>
</div>
<script>
let countdown=30;
async function loadData(){
  try{
    const res=await fetch('/api/trades');
    const data=await res.json();
    const state=data.state||{};
    const trades=(data.trades||[]).filter(t=>t.action!=='HOLD');
    const closed=trades.filter(t=>t.pnl_pct!==undefined);
    const winners=closed.filter(t=>t.pnl_pct>0);

    // Capital
    const cap=state.capital||100;
    const capEl=document.getElementById('capital');
    capEl.textContent='$'+cap.toFixed(2);
    const diff=cap-100;
    document.getElementById('capital-change').textContent=(diff>=0?'+':'')+diff.toFixed(2)+' desde inicio';
    capEl.className='stat-value '+(cap>=100?'green':'red');

    // Drawdown badge
    document.getElementById('drawdown-pill').style.display=state.drawdown_mode?'flex':'none';

    document.getElementById('total').textContent=trades.length;
    document.getElementById('winrate').textContent=closed.length?(Math.round(winners.length/closed.length*100)+'%'):'—';
    document.getElementById('winrate').className='stat-value '+(closed.length&&winners.length/closed.length>=0.5?'green':'red');

    const totalPnl=closed.reduce((s,t)=>s+(t.pnl_pct||0)*(t.usd_size||3)/100,0);
    const pnlEl=document.getElementById('pnl');
    pnlEl.textContent=(totalPnl>=0?'+':'')+'$'+totalPnl.toFixed(2);
    pnlEl.className='stat-value '+(totalPnl>=0?'green':'red');

    if(data.fear_greed){
      const fg=data.fear_greed;
      const el=document.getElementById('fg-val');
      el.textContent=fg.value;
      document.getElementById('fg-lbl').textContent=fg.label;
      el.className='stat-value '+(fg.value<35?'red':fg.value>65?'green':'yellow');
    }

    // Regime
    const regime=data.regime||'unknown';
    const regimeEl=document.getElementById('regime-val');
    const regMap={'bull':'📈','bear':'📉','sideways':'↔️','crash':'💥'};
    regimeEl.textContent=regMap[regime]||'?';
    document.getElementById('regime-sub').textContent=regime.toUpperCase();
    regimeEl.className='stat-value purple';

    // Scanner
    const scanner=data.scanner||[];
    document.getElementById('scanner-grid').innerHTML=scanner.map(s=>{
      const isBase=['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT'].includes(s.symbol);
      const vol=s.volume?'$'+Math.round(s.volume/1e6)+'M':'';
      return `<div class="chip ${isBase?'base':'alt'}">${s.symbol.replace('/USDT','')}<span class="chip-vol">${vol}</span></div>`;
    }).join('')||'<span style="color:var(--muted)">Sin datos</span>';

    // Capital usado
    const positions=data.positions||[];
    document.getElementById('open-pos').textContent=positions.length;
    const allocated=positions.reduce((s,p)=>s+(p.usd_size||0),0);
    const capUsed=document.getElementById('cap-used');
    capUsed.textContent='$'+allocated.toFixed(1);
    capUsed.className='stat-value '+(allocated>=20?'red':allocated>=12?'yellow':'green');

    // Posiciones
    const posSection=document.getElementById('positions-section');
    if(positions.length>0){
      posSection.style.display='block';
      document.getElementById('pos-grid').innerHTML=positions.map(p=>{
        const pnlPct=((p.current_price-p.entry_price)/p.entry_price*100).toFixed(2);
        const isProfit=pnlPct>=0;
        const distToTrail=(p.current_price-p.trail_stop)/p.current_price*100;
        const barWidth=Math.min(100,Math.max(0,(1-distToTrail/5)*100));
        const futBadge=p.mode&&p.mode.includes('FUTURES')?`<span style="color:var(--purple);font-size:.58rem"> ${p.mode}</span>`:'';
        return `<div class="pos-card ${isProfit?'profit':'loss'}">
          <div class="pos-header"><span class="pos-symbol">${p.symbol}${futBadge}</span><span class="pos-pnl ${isProfit?'pos':'neg'}">${isProfit?'+':''}${pnlPct}%</span></div>
          <div class="pos-row"><span>Entrada</span><span>${p.entry_price}</span></div>
          <div class="pos-row"><span>Actual</span><span>${p.current_price}</span></div>
          <div class="pos-row"><span>🔴 Trail</span><span style="color:var(--orange)">${p.trail_stop}</span></div>
          <div class="pos-row"><span>🎯 TP</span><span style="color:var(--green)">${p.take_profit}</span></div>
          <div class="trail-bar-bg"><div class="trail-bar-fill" style="width:${barWidth}%"></div></div>
        </div>`;
      }).join('');
    } else { posSection.style.display='none'; }

    // Mode
    const hasReal=trades.some(t=>!t.paper);
    document.getElementById('mode-text').textContent=hasReal?'REAL':'PAPER';
    document.getElementById('mode-pill').className='pill '+(hasReal?'pill-live':'pill-paper');

    // Trades table
    const tbody=document.getElementById('trades-body');
    if(!trades.length){
      tbody.innerHTML='<tr><td colspan="10"><div class="empty"><div class="empty-icon">🤖</div><div class="empty-text">Analizando señales...<br>Las operaciones aparecen aquí.</div></div></td></tr>';
      return;
    }
    tbody.innerHTML=[...trades].reverse().map(t=>{
      const ts=new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const conf=Math.round((t.confidence||0)*100);
      const sigs=[];
      if(t.tech_signal===1)sigs.push('📈');else if(t.tech_signal===-1)sigs.push('📉');
      if(t.macd_signal===1)sigs.push('📊+');else if(t.macd_signal===-1)sigs.push('📊-');
      if(t.rsi_div)sigs.push('🔄RSI');
      if(t.bb_signal===1)sigs.push('🎯BB');
      if(t.ob_signal===1)sigs.push('📖OB+');
      if(t.vol_signal===1)sigs.push('📦');
      if(t.funding_signal===1)sigs.push('💹+');else if(t.funding_signal===-1)sigs.push('💹-');
      if(t.news_signal===1)sigs.push('📰+');else if(t.news_signal===-1)sigs.push('📰-');
      const fgVal=t.fear_greed_value||'—';
      const fgCls=fgVal<35?'fg-fear':fgVal>65?'fg-greed':'fg-neutral';
      const paperTag=t.paper?'<span class="badge-mini badge-paper">P</span>':'';
      const futTag=t.mode&&t.mode.includes('FUT')?'<span class="badge-mini badge-fut">F</span>':'';
      const trailTag=t.trail_triggered?'<span class="badge-mini badge-trail">T</span>':'';
      const price=t.price?(+t.price).toLocaleString('en-US',{maximumFractionDigits:4}):'—';
      const pnlStr=t.pnl_pct!==undefined?`<span style="color:${t.pnl_pct>=0?'var(--green)':'var(--red)'}"> ${t.pnl_pct>=0?'+':''}${t.pnl_pct}%</span>`:'';
      return `<tr>
        <td class="pair">${t.symbol||'—'}${pnlStr}</td>
        <td style="color:${t.timeframe==='3m'?'var(--purple)':'var(--muted)'}">${t.timeframe||'1h'}</td>
        <td><span class="badge badge-${(t.action||'').toLowerCase()}">${t.action||'—'}</span>${paperTag}${futTag}${trailTag}</td>
        <td>${price}</td>
        <td style="color:var(--orange);font-size:.65rem">${t.usd_size?'$'+t.usd_size+(t.leverage&&t.leverage>1?'×'+t.leverage:''):'—'}</td>
        <td><div class="conf-bar"><div class="bar-bg"><div class="bar-fill" style="width:${conf}%"></div></div><span style="font-size:.65rem;min-width:26px">${conf}%</span></div></td>
        <td style="font-size:.62rem">${sigs.join(' ')}</td>
        <td><span class="fg-pill ${fgCls}">${fgVal}</span></td>
        <td style="color:var(--muted);max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${t.reasoning||''}">${t.reasoning||'—'}</td>
        <td class="ts">${ts}</td>
      </tr>`;
    }).join('');
  }catch(e){console.error(e);}
}
function tick(){countdown--;document.getElementById('countdown').textContent=countdown;if(countdown<=0){countdown=30;loadData();}}
loadData();setInterval(tick,1000);
</script>
</body>
</html>"""

# ─────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────
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
    return jsonify({
        "trades": trades, "count": len(trades),
        "fear_greed": _fear_greed,
        "positions": list(positions.values()),
        "scanner": _scanner,
        "regime": _regime,
        "state": state,
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
        "urls": {"api": {
            "public":  "https://testnet.binance.vision/api/v3",
            "private": "https://testnet.binance.vision/api/v3",
        }}
    })

def get_futures_exchange():
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET,
        "enableRateLimit": True, "options": {"defaultType": "future"},
        "urls": {"api": {
            "public":  "https://testnet.binancefuture.com",
            "private": "https://testnet.binancefuture.com",
        }}
    })

# ─────────────────────────────────────────
# MARKET REGIME DETECTOR
# ─────────────────────────────────────────
def detect_market_regime(public_ex):
    """
    Detecta el régimen del mercado usando BTC como referencia:
    bull / bear / sideways / crash
    """
    global _regime
    try:
        df = get_ohlcv(public_ex, "BTC/USDT", timeframe="1d", limit=30)
        df = calculate_indicators(df)

        last  = df.iloc[-1]
        prev  = df.iloc[-2]

        # EMA largo plazo
        ema50  = df["close"].ewm(span=50,  adjust=False).mean().iloc[-1]
        ema200 = df["close"].ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 50 else ema50

        # Cambio 24h
        change_24h = (last["close"] - prev["close"]) / prev["close"]

        if change_24h <= REGIME_CRASH_THRESHOLD:
            regime = "crash"
        elif ema50 > ema200 * (1 + REGIME_TREND_THRESHOLD):
            regime = "bull"
        elif ema50 < ema200 * (1 - REGIME_TREND_THRESHOLD):
            regime = "bear"
        else:
            regime = "sideways"

        _regime = regime
        log.info(f"🧭 Régimen: {regime.upper()} | BTC 24h: {change_24h*100:+.2f}% | EMA50/200: {ema50:.0f}/{ema200:.0f}")
        return regime
    except Exception as e:
        log.warning(f"Regime detect error: {e}")
        return "unknown"

def regime_filter(regime, action):
    """Ajusta estrategia según régimen."""
    if regime == "crash":
        log.info("  ⚠️ CRASH — solo operaciones muy fuertes (score ≥ 4)")
        return False  # bloqueamos en crash, solo futuros score ≥ 4 pasan por otra vía
    if regime == "bear" and action == "BUY":
        log.info("  📉 BEAR regime — operando con cautela")
        # En bear permitimos pero con score más alto (se maneja en el score)
    return True

# ─────────────────────────────────────────
# RESUMEN DIARIO + AUTO-BACKTEST
# ─────────────────────────────────────────
_last_daily_report  = None
_last_weekly_backtest = None

def maybe_send_daily_report(fg_value, fg_label):
    global _last_daily_report
    now_arg = datetime.now(ARG_TZ)
    today   = now_arg.date()
    if _last_daily_report == today: return
    if now_arg.hour != 9: return
    _last_daily_report = today

    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: trades = json.load(f)
        except: pass

    cutoff = now_arg - timedelta(hours=24)
    today_trades = []
    for t in trades:
        try:
            ts = datetime.fromisoformat(t["timestamp"]).replace(tzinfo=timezone.utc).astimezone(ARG_TZ)
            if ts >= cutoff: today_trades.append(t)
        except: pass

    buys    = [t for t in today_trades if t.get("action") == "BUY"]
    closed  = [t for t in today_trades if t.get("pnl_pct") is not None]
    winners = [t for t in closed if t.get("pnl_pct", 0) > 0]
    total_pnl = sum((t.get("pnl_pct", 0) * t.get("usd_size", 3)) / 100 for t in closed)
    win_rate  = round(len(winners) / len(closed) * 100) if closed else 0
    state     = load_state()

    best  = max(closed, key=lambda t: t.get("pnl_pct", 0), default=None)
    worst = min(closed, key=lambda t: t.get("pnl_pct", 0), default=None)

    msg = (
        f"📊 <b>Resumen — {today.strftime('%d/%m/%Y')}</b>\n\n"
        f"💰 Capital: ${state.get('capital', 100):.2f}\n"
        f"📈 Compras: {len(buys)} | ✅ Cerrados: {len(closed)}\n"
        f"🎯 Win rate: {win_rate}% | P&L: {'+'if total_pnl>=0 else ''}${total_pnl:.2f}\n"
        f"{'⚠️ DRAWDOWN MODE activo' if state.get('drawdown_mode') else '✅ Operación normal'}\n\n"
    )
    if best: msg += f"🏆 Mejor: {best['symbol']} {'+' if best['pnl_pct']>=0 else ''}{best['pnl_pct']}%\n"
    if worst and worst != best: msg += f"💀 Peor: {worst['symbol']} {worst['pnl_pct']}%\n"
    msg += f"\n🧭 F&G: {fg_value} — {fg_label} | Régimen: {_regime.upper()}"

    send_telegram(msg)
    log.info("📊 Resumen diario enviado")

def maybe_run_weekly_backtest():
    """Cada domingo analiza el historial con Claude y ajusta sugerencias."""
    global _last_weekly_backtest
    now_arg = datetime.now(ARG_TZ)
    if now_arg.weekday() != 6: return  # solo domingo
    if now_arg.hour != 10: return      # a las 10 AM
    today = now_arg.date()
    if _last_weekly_backtest == today: return
    _last_weekly_backtest = today

    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE) as f: trades = json.load(f)
        except: pass

    closed = [t for t in trades if t.get("pnl_pct") is not None]
    if len(closed) < 5:
        log.info("📊 Backtest: menos de 5 trades cerrados, saltando")
        return

    log.info("📊 Ejecutando backtest semanal con Claude...")
    try:
        summary = []
        for t in closed[-30:]:  # últimos 30 trades
            summary.append({
                "symbol": t.get("symbol"),
                "action": t.get("action"),
                "pnl_pct": t.get("pnl_pct"),
                "signal_score": t.get("signal_score"),
                "tech": t.get("tech_signal", 0),
                "macd": t.get("macd_signal", 0),
                "rsi_div": t.get("rsi_div", False),
                "bb": t.get("bb_signal", 0),
                "ob": t.get("ob_signal", 0),
                "funding": t.get("funding_signal", 0),
                "news": t.get("news_signal", 0),
                "timeframe": t.get("timeframe"),
                "fg": t.get("fear_greed_value"),
            })

        winners = [t for t in closed if t.get("pnl_pct", 0) > 0]
        win_rate = round(len(winners) / len(closed) * 100) if closed else 0

        prompt = f"""Sos un analista de trading cuantitativo. Analizá estos {len(closed)} trades y dá recomendaciones concretas.

Win rate actual: {win_rate}%
Trades recientes: {json.dumps(summary[:20], indent=2)}

Analizá:
1. ¿Qué combinación de señales tuvo mejor win rate?
2. ¿Qué señales son más confiables?
3. ¿En qué timeframe operamos mejor?
4. ¿Hay un F&G óptimo para entrar?
5. Recomienda 3 ajustes concretos para mejorar el win rate.

Respondé en menos de 300 palabras, en español, de forma directa y accionable."""

        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]},
            timeout=30)
        analysis = resp.json()["content"][0]["text"]

        send_telegram(f"📊 <b>Backtest semanal</b>\n\nWin rate: {win_rate}% ({len(closed)} trades)\n\n{analysis}")
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
        _scanner = (
            [{"symbol": s, "volume": None} for s in BASE_WATCHLIST] +
            usdt_pairs[:max_alts]
        )
        log.info(f"🔍 Altcoins: {altcoins}")
        return altcoins
    except Exception as e:
        log.error(f"Scanner error: {e}")
        return []

# ─────────────────────────────────────────
# EXCHANGES HELPER
# ─────────────────────────────────────────
def get_ohlcv(exchange, symbol, timeframe="1h", limit=150):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ─────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────
def calculate_indicators(df):
    df = df.copy()
    # EMA
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0); loss = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean() / loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)))
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    # Volume
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    # Bollinger Bands
    df["bb_mid"]   = df["close"].rolling(BB_PERIOD).mean()
    bb_std         = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"] = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
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
    """
    RSI Divergence — una de las señales más confiables.
    Bullish: precio hace mínimos más bajos pero RSI hace mínimos más altos
    Bearish: precio hace máximos más altos pero RSI hace máximos más bajos
    """
    if len(df) < 10: return 0, False
    window = df.tail(10)
    price_lows  = window["close"].rolling(3).min()
    rsi_lows    = window["rsi"].rolling(3).min()
    price_highs = window["close"].rolling(3).max()
    rsi_highs   = window["rsi"].rolling(3).max()

    # Bullish divergence: precio bajando, RSI subiendo
    if (price_lows.iloc[-1] < price_lows.iloc[-4] and
        rsi_lows.iloc[-1]   > rsi_lows.iloc[-4] and
        window["rsi"].iloc[-1] < 50):
        log.info("  🔄 RSI Divergence BULLISH detectada")
        return +1, True

    # Bearish divergence: precio subiendo, RSI bajando
    if (price_highs.iloc[-1] > price_highs.iloc[-4] and
        rsi_highs.iloc[-1]   < rsi_highs.iloc[-4] and
        window["rsi"].iloc[-1] > 50):
        log.info("  🔄 RSI Divergence BEARISH detectada")
        return -1, True

    return 0, False

def bollinger_signal(df):
    """
    Bollinger Bands:
    +1 = precio toca/cruza la banda inferior (potencial rebote)
    -1 = precio toca/cruza la banda superior (potencial caída)
     0 = dentro de las bandas o compresión
    """
    last = df.iloc[-1]; prev = df.iloc[-2]
    # Squeeze (bandas comprimiéndose) — inminente movimiento fuerte
    squeeze = last["bb_width"] < df["bb_width"].rolling(20).mean().iloc[-1] * 0.75

    if last["close"] <= last["bb_lower"] and prev["close"] > prev["bb_lower"]:
        log.info(f"  🎯 BB: precio cruza banda inferior — señal bullish")
        return +1
    if last["close"] >= last["bb_upper"] and prev["close"] < prev["bb_upper"]:
        log.info(f"  🎯 BB: precio cruza banda superior — señal bearish")
        return -1
    if squeeze and last["close"] > last["bb_mid"]:
        return +1  # squeeze + precio arriba del medio = posible breakout alcista
    if squeeze and last["close"] < last["bb_mid"]:
        return -1
    return 0

def order_book_signal(exchange, symbol):
    """
    Order Book Imbalance — señal en tiempo real.
    >60% bids = compradores dominan = bullish
    <40% bids = vendedores dominan = bearish
    """
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        bid_vol = sum(b[1] for b in ob["bids"])
        ask_vol = sum(a[1] for a in ob["asks"])
        total   = bid_vol + ask_vol
        if total == 0: return 0
        bid_ratio = bid_vol / total
        log.info(f"  📖 Order Book: bids={bid_ratio*100:.1f}%")
        if bid_ratio >= OB_IMBALANCE_THRESHOLD:  return +1
        if bid_ratio <= 1 - OB_IMBALANCE_THRESHOLD: return -1
        return 0
    except Exception as e:
        log.warning(f"Order book error {symbol}: {e}")
        return 0

def timeframe_confirm_signal(exchange, symbol, base_tf):
    confirm_tf = "15m" if base_tf == "3m" else ("30m" if base_tf == "1h" else "4h")
    try:
        df = calculate_indicators(get_ohlcv(exchange, symbol, timeframe=confirm_tf, limit=50))
        return technical_signal(df)
    except Exception as e:
        log.warning(f"Confirm signal error {symbol}: {e}")
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
        label = data["value_classification"]
        _fear_greed = {"value": value, "label": label}
        return value, label
    except Exception as e:
        log.warning(f"F&G error: {e}")
        return 50, "Neutral"

def fear_greed_filter(value, action, regime):
    if action == "BUY"  and value < 15: return False
    if action == "SELL" and value > 85: return False
    if regime == "crash" and action == "BUY": return False
    return True

# ─────────────────────────────────────────
# NOTICIAS — neutral por defecto
# ─────────────────────────────────────────
BULLISH_KW = ["rally","surge","breakout","bullish","adoption","partnership","upgrade","all-time high","ath","gains","rises","jumps","soars","recovery","bullrun"]
BEARISH_KW = ["crash","hack","ban","bearish","lawsuit","regulation","sell-off","collapse","fear","drops","falls","plunges","warning","risk","dump","fraud"]
RSS_FEEDS  = ["https://www.coindesk.com/arc/outboundfeeds/rss/","https://cointelegraph.com/rss"]
COIN_NAMES = {
    "btc":["bitcoin","btc"],"eth":["ethereum","eth"],"sol":["solana","sol"],
    "bnb":["bnb","binance coin"],"xrp":["ripple","xrp"],"doge":["dogecoin","doge"],
    "ada":["cardano","ada"],"avax":["avalanche","avax"],"pepe":["pepe"],"trx":["tron","trx"],
    "matic":["polygon","matic"],"link":["chainlink","link"],"dot":["polkadot","dot"],
}

def get_news_sentiment(symbol):
    """
    FIX: Si no hay noticias específicas del coin, retorna 0 (neutral).
    Ya no propaga sentimiento general al coin.
    """
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

    # Solo noticias específicas del coin
    relevant = [t.lower() for t in all_titles if any(term in t.lower() for term in terms)]

    # FIX: Si no hay noticias específicas → neutral (no contaminar con sentimiento general)
    if not relevant:
        return 0

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
            data = resp.json()
            _funding_cache = {item["symbol"]: float(item.get("lastFundingRate", 0)) for item in data}
            _funding_last_fetch = now
            log.info(f"  💹 Funding rates actualizados ({len(_funding_cache)} pares)")
        except Exception as e:
            log.warning(f"Funding error: {e}")
            return 0.0
    futures_symbol = symbol.replace("/", "")
    return _funding_cache.get(futures_symbol, 0.0)

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
def get_position_size(effective_capital, signal_score, regime):
    abs_score = abs(signal_score)
    pct = POSITION_SIZE_MAP.get(min(abs_score, 6), 0.02)
    # En bear reducir sizing
    if regime == "bear": pct = pct * 0.7
    usd = round(effective_capital * pct, 2)
    log.info(f"  💰 Size: {pct*100:.1f}% = ${usd} (score={signal_score:+d} regime={regime})")
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
# POSICIONES — TRAILING STOP
# ─────────────────────────────────────────
def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_positions(positions):
    with open(POSITIONS_FILE, "w") as f: json.dump(positions, f, indent=2, default=str)

def open_position(symbol, entry_price, usd_size, pct, action, timeframe, mode="SPOT"):
    positions = load_positions()
    trail_pct = FUTURES_TRAILING if "FUTURES" in mode else TRAILING_STOP_PCT
    tp_pct    = FUTURES_TP       if "FUTURES" in mode else TAKE_PROFIT_PCT
    positions[symbol] = {
        "symbol": symbol, "action": action, "timeframe": timeframe, "mode": mode,
        "entry_price": entry_price, "current_price": entry_price, "high_price": entry_price,
        "trail_stop":  round(entry_price * (1 - trail_pct), 4),
        "take_profit": round(entry_price * (1 + tp_pct), 4),
        "usd_size": usd_size, "risk_pct": pct,
        "opened_at": datetime.now().isoformat(),
    }
    save_positions(positions)
    log.info(f"  📂 Posición: {symbol} [{mode}] @ {entry_price} | Trail={positions[symbol]['trail_stop']} TP={positions[symbol]['take_profit']}")

def update_trailing_stops(public_ex, state):
    positions = load_positions()
    if not positions: return
    closed = []
    for symbol, pos in positions.items():
        try:
            current  = public_ex.fetch_ticker(symbol)["last"]
            pos["current_price"] = current
            trail_pct = FUTURES_TRAILING if "FUTURES" in pos.get("mode","") else TRAILING_STOP_PCT
            tp_pct    = FUTURES_TP       if "FUTURES" in pos.get("mode","") else TAKE_PROFIT_PCT

            if pos["action"] == "BUY":
                if current > pos["high_price"]:
                    pos["high_price"] = current
                    pos["trail_stop"] = round(current * (1 - trail_pct), 4)
                    log.info(f"  📈 Trail {symbol}: {pos['trail_stop']}")

                if current <= pos["trail_stop"]:
                    pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
                    log.info(f"  🔴 TRAIL STOP {symbol} @ {current} | PnL: {pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    send_telegram(f"🔴 <b>Trail Stop</b> — {symbol}\nEntrada: {pos['entry_price']} → Salida: {current}\nPnL: {pnl:+.2f}% {'✅' if pnl>0 else '❌'}")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"SELL","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":f"Trail stop (entrada {pos['entry_price']})",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "trail_triggered":True,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol); continue

                if current >= pos["take_profit"]:
                    pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
                    log.info(f"  🎯 TAKE PROFIT {symbol} @ {current} | PnL: +{pnl:.2f}%")
                    pnl_usd = pnl * pos["usd_size"] / 100
                    update_compounding(state, pnl_usd)
                    send_telegram(f"🎯 <b>Take Profit</b> — {symbol}\nEntrada: {pos['entry_price']} → Salida: {current}\nPnL: +{pnl:.2f}% 🎉")
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action":"SELL","price":current,"timeframe":pos.get("timeframe","1h"),
                        "mode":pos.get("mode","SPOT"),"reasoning":f"Take profit (entrada {pos['entry_price']})",
                        "confidence":1.0,"paper":PAPER_TRADING,"pnl_pct":round(pnl,2),
                        "trail_triggered":False,"entry_price":pos["entry_price"],"usd_size":pos["usd_size"]})
                    closed.append(symbol)
        except Exception as e:
            log.error(f"Error trailing {symbol}: {e}")

    for s in closed: del positions[s]
    save_positions(positions)

# ─────────────────────────────────────────
# CLAUDE
# ─────────────────────────────────────────
def ask_claude(symbol, signals, df, fg_value, usd_size, pct, timeframe, regime):
    last  = df.iloc[-1]
    total = sum(v for k,v in signals.items() if k != "rsi_div_val")
    direction = "BULLISH" if total > 0 else "BEARISH"
    rsi_div_info = "Sí (señal fuerte)" if signals.get("rsi_div_val") else "No"

    prompt = f"""Trading crypto analyst. Respond ONLY in JSON, no backticks.

Pair: {symbol} [{timeframe}] | Price: {last['close']:.4f} | Regime: {regime.upper()}
EMA9/21: {last['ema9']:.4f}/{last['ema21']:.4f} | RSI: {last['rsi']:.1f}
MACD hist: {last['macd_hist']:.4f} | BB: {last['bb_lower']:.4f}/{last['bb_mid']:.4f}/{last['bb_upper']:.4f}
Vol/MA20: {last['volume']:.0f}/{last['vol_ma20']:.0f} | Fear&Greed: {fg_value}

Signals: EMA:{signals.get('tech',0):+d} MACD:{signals.get('macd',0):+d} RSI_DIV:{rsi_div_info} BB:{signals.get('bb',0):+d} OB:{signals.get('ob',0):+d} VOL:{signals.get('vol',0):+d} FR:{signals.get('funding',0):+d} NEWS:{signals.get('news',0):+d}
Total: {total:+d} | Size: ${usd_size} ({pct*100:.0f}%) | Trail: {FUTURES_TRAILING*100 if 'FUT' in str(signals) else TRAILING_STOP_PCT*100}%

{{"action":"BUY"|"SELL"|"HOLD","confidence":0.0,"reasoning":"one concise line"}}"""

    try:
        resp = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-haiku-4-5-20251001","max_tokens":200,"messages":[{"role":"user","content":prompt}]},
            timeout=15)
        text = resp.json()["content"][0]["text"].strip().replace("```json","").replace("```","").strip()
        return json.loads(text)
    except Exception as e:
        log.error(f"Claude error: {e}")
        return {"action":"HOLD","confidence":0,"reasoning":"API error"}

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
# ANALIZAR UN PAR
# ─────────────────────────────────────────
def analyze_and_trade(symbol, timeframe, public_ex, trade_ex, futures_ex, fg_value, fg_label, open_positions, regime, state):
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
    total = sum(v for k,v in signals.items() if k != "rsi_div_val")

    log.info(f"  [{timeframe}] EMA:{t_sig:+d} MACD:{m_sig:+d} BB:{bb_sig:+d} OB:{ob_sig:+d} RSIDiv:{rsi_sig:+d} VOL:{v_sig:+d} FR:{fr_sig:+d} NEWS:{n_sig:+d} = {total:+d}")

    if abs(total) < MIN_SIGNALS:
        log.info("  ⏭️  Señales insuficientes — skip")
        return

    if timeframe == "3m" and total < 0:
        log.info("  ⏭️  Bajista en 3m — solo long en altcoins")
        return

    effective_cap = get_effective_capital(state)
    usd_size, risk_pct = get_position_size(effective_cap, total, regime)

    if not can_open_position(open_positions, usd_size, effective_cap): return
    if not regime_filter(regime, "BUY" if total > 0 else "SELL"): return

    log.info("  🧠 Consultando Claude...")
    analysis = ask_claude(symbol, signals, df, fg_value, usd_size, risk_pct, timeframe, regime)
    log.info(f"  Claude: {analysis['action']} ({analysis['confidence']:.2f}) — {analysis['reasoning']}")

    if not fear_greed_filter(fg_value, analysis["action"], regime): return

    # Decidir spot vs futuros
    use_futures = (abs(total) >= FUTURES_MIN_SCORE) and (fg_value >= 20) and (regime != "crash")
    mode_label  = f"FUTURES {FUTURES_LEVERAGE}x" if use_futures else "SPOT"
    trail_pct   = FUTURES_TRAILING if use_futures else TRAILING_STOP_PCT
    tp_pct      = FUTURES_TP       if use_futures else TAKE_PROFIT_PCT
    leverage    = FUTURES_LEVERAGE if use_futures else 1

    current_price = df.iloc[-1]["close"]
    trail_stop    = round(current_price * (1 - trail_pct), 4)
    take_profit   = round(current_price * (1 + tp_pct), 4)

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
        "leverage": leverage, "signal_score": total, "regime": regime,
    }

    if analysis["action"] == "BUY" and analysis["confidence"] >= CONFIDENCE_MIN:
        if PAPER_TRADING:
            save_trade({**base_record, "paper": True, "order_id": None})
            open_position(symbol, current_price, usd_size, risk_pct, "BUY", timeframe, mode_label)
            log.info(f"  📝 PAPER {mode_label} BUY @ {current_price} | Trail={trail_stop} TP={take_profit}")
            send_telegram(
                f"📝 <b>PAPER {'🚀' if use_futures else '✅'} {mode_label} [{timeframe}]</b>\n"
                f"<b>{symbol}</b> @ {current_price}\n"
                f"🔴 Trail: {trail_stop} | 🎯 TP: {take_profit}\n"
                f"💰 ${usd_size}×{leverage} (score {total:+d} | {regime})\n"
                f"🔄RSI:{rsi_div} 🎯BB:{bb_sig:+d} 📖OB:{ob_sig:+d} | Conf: {int(analysis['confidence']*100)}%"
            )
        else:
            if use_futures:
                order, exec_price = execute_futures_trade(futures_ex, symbol, "BUY", usd_size)
            else:
                order, exec_price = execute_spot_trade(trade_ex, symbol, "BUY", usd_size)
            if order:
                actual = exec_price or current_price
                save_trade({**base_record, "paper": False, "order_id": order.get("id"), "price": actual})
                open_position(symbol, actual, usd_size, risk_pct, "BUY", timeframe, mode_label)
                send_telegram(f"{'🚀' if use_futures else '✅'} <b>{mode_label}</b> — {symbol} @ {actual}\n💰 ${usd_size}×{leverage}")

    elif analysis["action"] == "SELL" and analysis["confidence"] >= CONFIDENCE_MIN:
        if PAPER_TRADING:
            save_trade({**base_record, "paper": True, "order_id": None})
            log.info(f"  📝 PAPER SELL @ {current_price}")
        else:
            fn = execute_futures_trade if use_futures else execute_spot_trade
            order, exec_price = fn(trade_ex if not use_futures else futures_ex, symbol, "SELL", usd_size)
            if order:
                save_trade({**base_record, "paper": False, "order_id": order.get("id")})

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run_bot():
    log.info("🤖 CryptoBot v7 iniciado")
    log.info(f"Mode: {'PAPER' if PAPER_TRADING else 'REAL'} | Capital: ${CAPITAL_TOTAL_USD}")

    send_telegram(
        f"🤖 <b>CryptoBot v7 iniciado</b>\n"
        f"Mode: {'📝 PAPER' if PAPER_TRADING else '💰 REAL'}\n"
        f"Señales: EMA+MACD+RSI_Div+BB+OB+Funding+Noticias\n"
        f"Spot: Trail {TRAILING_STOP_PCT*100}% | TP {TAKE_PROFIT_PCT*100}%\n"
        f"Futuros {FUTURES_LEVERAGE}x (score≥{FUTURES_MIN_SCORE}): Trail {FUTURES_TRAILING*100}%\n"
        f"Anti-drawdown: {MAX_WEEKLY_LOSS_PCT*100}% | Compounding: ON\n"
        f"Resumen diario: 9 AM | Backtest: domingos 10 AM"
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
        log.info(f"\n{'='*50}\n⏰ Ciclo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # State y anti-drawdown
        state = load_state()
        check_drawdown(state)

        # Trailing stops
        update_trailing_stops(public_ex, state)

        # Fear & Greed
        fg_value, fg_label = get_fear_greed()
        log.info(f"🧭 F&G: {fg_value} — {fg_label} | Régimen: {regime.upper()}")

        # Resumen diario y backtest
        maybe_send_daily_report(fg_value, fg_label)
        maybe_run_weekly_backtest()

        # Market Regime (cada 15 minutos)
        if now - last_regime_check > 900:
            regime = detect_market_regime(public_ex)
            last_regime_check = now

        # Altcoin scan (cada 10 minutos)
        if now - last_scan > 600:
            log.info("🔍 Escaneando altcoins...")
            altcoins  = scan_top_altcoins(public_ex, max_alts=15)
            last_scan = now

        open_positions = load_positions()
        log.info(f"📂 Posiciones: {list(open_positions.keys()) or 'ninguna'} | Capital: ${state.get('capital',100):.2f}")

        # Base watchlist [1h]
        log.info("\n--- BASE [1h] ---")
        for symbol in BASE_WATCHLIST:
            try:
                log.info(f"\n📊 {symbol}...")
                analyze_and_trade(symbol, "1h", public_ex, trade_ex, futures_ex, fg_value, fg_label, open_positions, regime, state)
                open_positions = load_positions()  # refrescar tras cada trade
            except Exception as e:
                log.error(f"Error {symbol}: {e}")

        # Altcoins [3m]
        if altcoins:
            log.info("\n--- ALTCOINS [3m] ---")
            for symbol in altcoins:
                try:
                    log.info(f"\n📊 {symbol} [3m]...")
                    analyze_and_trade(symbol, "3m", public_ex, trade_ex, futures_ex, fg_value, fg_label, open_positions, regime, state)
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
