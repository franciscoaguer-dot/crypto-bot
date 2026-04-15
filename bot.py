"""
CryptoBot v6
- BTC/ETH/SOL/BNB en 1h (base)
- Top altcoins por volumen en 3m (scalping)
- Funding Rates — señal de posicionamiento del mercado
- Capital Allocator — máximo 15% capital en posiciones abiertas
- Trailing stop + position sizing dinámico
- Resumen diario a las 9 AM (Argentina)
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

POSITION_SIZE_MAP  = {2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05}
TRAILING_STOP_PCT  = 0.010   # spot scalping: 1%
TAKE_PROFIT_PCT    = 0.025   # spot: 2.5%
FUTURES_TRAILING   = 0.008   # futuros: 0.8% (más ajustado)
FUTURES_TP         = 0.020   # futuros: 2%
MIN_SIGNALS        = 2
CONFIDENCE_MIN     = 0.50    # confianza mínima para ejecutar
LOOP_INTERVAL_SEC  = 60
TRADE_LOG_FILE     = "trade_log.json"
POSITIONS_FILE     = "positions.json"

# Capital Allocator
MAX_CAPITAL_EXPOSURE = 0.20   # máximo 20% del capital en posiciones abiertas

# Futuros — solo cuando señal es fuerte
FUTURES_MIN_SCORE  = 4        # score mínimo para operar futuros
FUTURES_LEVERAGE   = 2        # apalancamiento 2x (conservador)

# Funding Rate thresholds
FUNDING_BULLISH_THRESHOLD  = -0.0001
FUNDING_BEARISH_THRESHOLD  =  0.0015

# Watchlist base — siempre monitoreada en 1h
BASE_WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# Stablecoins y tokens a excluir del scanner
EXCLUDE_SYMBOLS = {
    "USDT","USDC","BUSD","DAI","TUSD","FDUSD","USDP","USD1",
    "WBTC","WETH","STETH","BETH","BTC","ETH","SOL","BNB",
    "LDUSDT","XAUT","PAXG"  # gold tokens — muy illiquidos para scalping
}

# Solo altcoins con nombre ASCII (filtra tokens basura con caracteres chinos/especiales)
def is_valid_symbol(symbol):
    base = symbol.replace("/USDT", "")
    if not base.isascii(): return False           # excluir caracteres no ASCII
    if len(base) > 10: return False               # nombres muy largos = tokens raros
    if any(c in base for c in ["1","2","3"] if base.endswith(c)): pass  # ok
    return True

# Timezone Argentina (UTC-3)
ARG_TZ = timezone(timedelta(hours=-3))

# ─────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CryptoBot v4</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root{--bg:#080c10;--surface:#0d1117;--border:#1a2332;--green:#00ff88;--red:#ff3355;--yellow:#ffcc00;--blue:#00aaff;--orange:#ff9900;--purple:#aa55ff;--muted:#3d5166;--text:#c9d8e8;--mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);min-height:100vh}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,136,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
  .container{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:36px 20px}
  header{display:flex;align-items:center;justify-content:space-between;margin-bottom:36px;padding-bottom:20px;border-bottom:1px solid var(--border)}
  .logo{font-family:var(--sans);font-weight:800;font-size:1.4rem;color:#fff}.logo span{color:var(--green)}
  .logo small{font-size:.65rem;color:var(--muted);margin-left:8px}
  .badges{display:flex;gap:8px;flex-wrap:wrap}
  .pill{display:flex;align-items:center;gap:5px;font-size:.7rem;padding:4px 11px;border-radius:100px}
  .pill-live{color:var(--green);border:1px solid rgba(0,255,136,.3)}
  .pill-paper{color:var(--blue);border:1px solid rgba(0,170,255,.3)}
  .pill-alt{color:var(--purple);border:1px solid rgba(170,85,255,.3)}
  .dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:28px}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;position:relative;overflow:hidden}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,var(--green));opacity:.7}
  .stat-label{font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:7px}
  .stat-value{font-family:var(--sans);font-weight:800;font-size:1.6rem;line-height:1;color:#fff}
  .stat-value.green{color:var(--green)}.stat-value.red{color:var(--red)}.stat-value.yellow{color:var(--yellow)}
  .stat-value.blue{color:var(--blue)}.stat-value.orange{color:var(--orange)}.stat-value.purple{color:var(--purple)}
  .stat-sub{font-size:.6rem;color:var(--muted);margin-top:5px}

  /* Scanner */
  .scanner-section{margin-bottom:28px}
  .scanner-grid{display:flex;flex-wrap:wrap;gap:8px}
  .scanner-chip{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:.7rem;display:flex;align-items:center;gap:6px}
  .scanner-chip.base{border-color:rgba(0,255,136,.25);color:var(--green)}
  .scanner-chip.alt{border-color:rgba(170,85,255,.25);color:var(--purple)}
  .chip-vol{color:var(--muted);font-size:.62rem}

  /* Posiciones */
  .pos-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin-bottom:28px}
  .pos-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px}
  .pos-card.profit{border-color:rgba(0,255,136,.3)}.pos-card.loss{border-color:rgba(255,51,85,.3)}
  .pos-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
  .pos-symbol{font-family:var(--sans);font-weight:700;font-size:.95rem;color:#fff}
  .pos-tf{font-size:.6rem;color:var(--muted);margin-left:6px}
  .pos-pnl{font-family:var(--sans);font-weight:700;font-size:.85rem}
  .pos-pnl.pos{color:var(--green)}.pos-pnl.neg{color:var(--red)}
  .pos-row{display:flex;justify-content:space-between;font-size:.68rem;color:var(--muted);margin-top:3px}
  .pos-row span:last-child{color:var(--text)}
  .trail-bar-bg{height:3px;background:var(--border);border-radius:2px;margin-top:8px;overflow:hidden}
  .trail-bar-fill{height:100%;background:var(--orange);border-radius:2px}

  .section-title{font-family:var(--sans);font-size:.7rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}
  .table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:28px}
  table{width:100%;border-collapse:collapse;font-size:.74rem}
  thead tr{border-bottom:1px solid var(--border)}
  th{padding:10px 13px;text-align:left;font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:400}
  td{padding:10px 13px;border-bottom:1px solid rgba(26,35,50,.5);vertical-align:middle}
  tr:last-child td{border-bottom:none}tr:hover td{background:rgba(255,255,255,.02)}
  .badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:.65rem;font-weight:700}
  .badge-buy{background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.2)}
  .badge-sell{background:rgba(255,51,85,.12);color:var(--red);border:1px solid rgba(255,51,85,.2)}
  .badge-paper{font-size:.52rem;background:rgba(0,170,255,.1);color:var(--blue);border:1px solid rgba(0,170,255,.2);padding:1px 4px;border-radius:3px;margin-left:3px}
  .badge-trail{font-size:.52rem;background:rgba(255,153,0,.1);color:var(--orange);border:1px solid rgba(255,153,0,.2);padding:1px 4px;border-radius:3px;margin-left:3px}
  .badge-alt{font-size:.52rem;background:rgba(170,85,255,.1);color:var(--purple);border:1px solid rgba(170,85,255,.2);padding:1px 4px;border-radius:3px;margin-left:3px}
  .conf-bar{display:flex;align-items:center;gap:6px}
  .bar-bg{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden}
  .bar-fill{height:100%;background:var(--green);border-radius:2px}
  .pair{color:#fff;font-weight:600}.ts{color:var(--muted);font-size:.65rem}
  .empty{text-align:center;padding:48px 20px;color:var(--muted)}
  .empty-icon{font-size:2rem;margin-bottom:10px}.empty-text{font-size:.8rem;line-height:1.6}
  footer{text-align:center;font-size:.65rem;color:var(--muted);padding-top:18px;border-top:1px solid var(--border)}
  .refresh-info{font-size:.65rem;color:var(--muted);text-align:right;margin-bottom:10px}
  #countdown{color:var(--green)}
  .fg-pill{display:inline-block;padding:2px 6px;border-radius:3px;font-size:.6rem;font-weight:700}
  .fg-fear{background:rgba(255,51,85,.15);color:var(--red)}
  .fg-greed{background:rgba(0,255,136,.15);color:var(--green)}
  .fg-neutral{background:rgba(255,204,0,.15);color:var(--yellow)}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">Crypto<span>Bot</span><small>v4</small></div>
    <div class="badges">
      <div class="pill pill-paper" id="mode-pill"><div class="dot"></div><span id="mode-text">PAPER</span></div>
      <div class="pill pill-alt"><div class="dot"></div>ALTCOIN SCANNER</div>
      <div class="pill pill-live"><div class="dot"></div>LIVE</div>
    </div>
  </header>

  <div class="stats">
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Total Trades</div><div class="stat-value" id="total">—</div><div class="stat-sub">paper + real</div></div>
    <div class="stat-card" style="--accent:var(--green)"><div class="stat-label">Compras</div><div class="stat-value green" id="buys">—</div><div class="stat-sub">BUY</div></div>
    <div class="stat-card" style="--accent:var(--red)"><div class="stat-label">Win Rate</div><div class="stat-value red" id="winrate">—</div><div class="stat-sub">trades cerrados</div></div>
    <div class="stat-card" style="--accent:var(--yellow)"><div class="stat-label">P&L Total</div><div class="stat-value yellow" id="pnl">—</div><div class="stat-sub">paper USD</div></div>
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Fear & Greed</div><div class="stat-value blue" id="fg-val">—</div><div class="stat-sub" id="fg-lbl">—</div></div>
    <div class="stat-card" style="--accent:var(--orange)"><div class="stat-label">Posiciones</div><div class="stat-value orange" id="open-pos">—</div><div class="stat-sub">abiertas</div></div>
    <div class="stat-card" style="--accent:var(--purple)"><div class="stat-label">Altcoins</div><div class="stat-value purple" id="altcount">—</div><div class="stat-sub">en scanner</div></div>
    <div class="stat-card" style="--accent:var(--orange)"><div class="stat-label">Capital usado</div><div class="stat-value orange" id="cap-used">—</div><div class="stat-sub">de $15 máx</div></div>
  </div>

  <div class="scanner-section">
    <div class="section-title">Scanner activo</div>
    <div class="scanner-grid" id="scanner-grid"><span style="color:var(--muted);font-size:.75rem">Cargando...</span></div>
  </div>

  <div id="positions-section" style="display:none;margin-bottom:28px">
    <div class="section-title">Posiciones abiertas</div>
    <div class="pos-grid" id="pos-grid"></div>
  </div>

  <div class="refresh-info">Auto-refresh en <span id="countdown">30</span>s</div>
  <div class="section-title">Historial de operaciones</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Par</th><th>TF</th><th>Acción</th><th>Precio</th><th>Size</th><th>Confianza</th><th>Señales</th><th>F&G</th><th>Razonamiento</th><th>Timestamp</th></tr></thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>
  <footer>CryptoBot v4 · Altcoin Scanner · Trailing Stop · Position Sizing · Resumen Diario</footer>
</div>
<script>
let countdown=30;
async function loadData(){
  try{
    const res=await fetch('/api/trades');
    const data=await res.json();
    const trades=(data.trades||[]).filter(t=>t.action!=='HOLD');
    const closed=trades.filter(t=>t.pnl_pct!==undefined);
    const winners=closed.filter(t=>t.pnl_pct>0);

    document.getElementById('total').textContent=trades.length;
    document.getElementById('buys').textContent=trades.filter(t=>t.action==='BUY').length;
    document.getElementById('winrate').textContent=closed.length?(Math.round(winners.length/closed.length*100)+'%'):'—';
    document.getElementById('winrate').className='stat-value '+(closed.length&&winners.length/closed.length>=0.5?'green':'red');

    const totalPnl=closed.reduce((s,t)=>s+(t.pnl_pct||0)*(t.usd_size||3)/100,0);
    const pnlEl=document.getElementById('pnl');
    pnlEl.textContent=(totalPnl>=0?'+':'')+'$'+totalPnl.toFixed(2);
    pnlEl.className='stat-value '+(totalPnl>=0?'green':'red');

    // Capital exposure
    const positions2=data.positions||[];
    const allocated=positions2.reduce((s,p)=>s+(p.usd_size||0),0);
    const capEl=document.getElementById('cap-used');
    if(capEl){
      capEl.textContent='$'+allocated.toFixed(1);
      capEl.className='stat-value '+(allocated>=15?'red':allocated>=10?'yellow':'green');
    }

    if(data.fear_greed){
      const fg=data.fear_greed;
      const el=document.getElementById('fg-val');
      el.textContent=fg.value;
      document.getElementById('fg-lbl').textContent=fg.label;
      el.className='stat-value '+(fg.value<35?'red':fg.value>65?'green':'yellow');
    }

    // Scanner
    const scanner=data.scanner||[];
    document.getElementById('altcount').textContent=scanner.filter(s=>!['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT'].includes(s.symbol)).length;
    document.getElementById('scanner-grid').innerHTML=scanner.map(s=>{
      const isBase=['BTC/USDT','ETH/USDT','SOL/USDT','BNB/USDT'].includes(s.symbol);
      const vol=s.volume?'$'+Math.round(s.volume/1e6)+'M':'';
      return `<div class="scanner-chip ${isBase?'base':'alt'}">${s.symbol.replace('/USDT','')}<span class="chip-vol">${vol}</span></div>`;
    }).join('')||'<span style="color:var(--muted);font-size:.75rem">Sin datos</span>';

    // Mode pill
    const hasReal=trades.some(t=>!t.paper);
    document.getElementById('mode-text').textContent=hasReal?'REAL':'PAPER';
    document.getElementById('mode-pill').className='pill '+(hasReal?'pill-live':'pill-paper');

    // Posiciones
    const positions=data.positions||[];
    document.getElementById('open-pos').textContent=positions.length;
    const posSection=document.getElementById('positions-section');
    if(positions.length>0){
      posSection.style.display='block';
      document.getElementById('pos-grid').innerHTML=positions.map(p=>{
        const pnlPct=((p.current_price-p.entry_price)/p.entry_price*100).toFixed(2);
        const isProfit=pnlPct>=0;
        const distToTrail=(p.current_price-p.trail_stop)/p.current_price*100;
        const barWidth=Math.min(100,Math.max(0,(1-distToTrail/5)*100));
        const tfBadge=p.timeframe==='15m'?'<span style="color:var(--purple);font-size:.6rem">15m</span>':'<span style="color:var(--green);font-size:.6rem">1h</span>';
        return `<div class="pos-card ${isProfit?'profit':'loss'}">
          <div class="pos-header"><span class="pos-symbol">${p.symbol}${tfBadge}</span><span class="pos-pnl ${isProfit?'pos':'neg'}">${isProfit?'+':''}${pnlPct}%</span></div>
          <div class="pos-row"><span>Entrada</span><span>${p.entry_price}</span></div>
          <div class="pos-row"><span>Actual</span><span>${p.current_price}</span></div>
          <div class="pos-row"><span>🔴 Trail</span><span style="color:var(--orange)">${p.trail_stop}</span></div>
          <div class="pos-row"><span>🎯 TP</span><span style="color:var(--green)">${p.take_profit}</span></div>
          <div class="pos-row"><span>Size</span><span>$${p.usd_size}</span></div>
          <div class="trail-bar-bg"><div class="trail-bar-fill" style="width:${barWidth}%"></div></div>
        </div>`;
      }).join('');
    } else { posSection.style.display='none'; }

    // Trades table
    const tbody=document.getElementById('trades-body');
    if(!trades.length){
      tbody.innerHTML='<tr><td colspan="10"><div class="empty"><div class="empty-icon">🤖</div><div class="empty-text">El bot está analizando señales...<br>Las operaciones aparecerán aquí cuando se ejecuten.</div></div></td></tr>';
      return;
    }
    tbody.innerHTML=[...trades].reverse().map(t=>{
      const ts=new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const conf=Math.round((t.confidence||0)*100);
      const sigs=[];
      if(t.tech_signal===1)sigs.push('📈');else if(t.tech_signal===-1)sigs.push('📉');
      if(t.macd_signal===1)sigs.push('📊+');else if(t.macd_signal===-1)sigs.push('📊-');
      if(t.vol_signal===1)sigs.push('📦');
      if(t.tf4h_signal===1)sigs.push('⏱+');else if(t.tf4h_signal===-1)sigs.push('⏱-');
      if(t.news_signal===1)sigs.push('📰+');else if(t.news_signal===-1)sigs.push('📰-');
      const fgVal=t.fear_greed_value||'—';
      const fgCls=fgVal<35?'fg-fear':fgVal>65?'fg-greed':'fg-neutral';
      const paperTag=t.paper?'<span class="badge-paper">PAPER</span>':'';
      const trailTag=t.trail_triggered?'<span class="badge-trail">TRAIL</span>':'';
      const altTag=t.timeframe==='15m'?'<span class="badge-alt">ALT</span>':'';
      const price=t.price?(+t.price).toLocaleString('en-US',{maximumFractionDigits:4}):'—';
      const pnlStr=t.pnl_pct!==undefined?`<span style="color:${t.pnl_pct>=0?'var(--green)':'var(--red)'}"> ${t.pnl_pct>=0?'+':''}${t.pnl_pct}%</span>`:'';
      return `<tr>
        <td class="pair">${t.symbol||'—'}${pnlStr}</td>
        <td style="color:${t.timeframe==='15m'?'var(--purple)':'var(--muted)'}">${t.timeframe||'1h'}</td>
        <td><span class="badge badge-${(t.action||'').toLowerCase()}">${t.action||'—'}</span>${paperTag}${trailTag}${altTag}</td>
        <td>${price}</td>
        <td style="color:var(--orange);font-size:.68rem">${t.usd_size?'$'+t.usd_size:'—'}</td>
        <td><div class="conf-bar"><div class="bar-bg"><div class="bar-fill" style="width:${conf}%"></div></div><span style="font-size:.68rem;min-width:28px">${conf}%</span></div></td>
        <td style="font-size:.65rem">${sigs.join(' ')}</td>
        <td><span class="fg-pill ${fgCls}">${fgVal}</span></td>
        <td style="color:var(--muted);max-width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${t.reasoning||''}">${t.reasoning||'—'}</td>
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
flask_app    = Flask(__name__)
_fear_greed  = {"value": 50, "label": "Neutral"}
_scanner     = []

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
    return jsonify({
        "trades": trades, "count": len(trades),
        "fear_greed": _fear_greed,
        "positions": list(positions.values()),
        "scanner": _scanner
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
# RESUMEN DIARIO
# ─────────────────────────────────────────
_last_daily_report = None

def maybe_send_daily_report(fg_value, fg_label):
    """Envía resumen diario a las 9 AM Argentina si no se mandó hoy."""
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

    # Filtrar trades de las últimas 24h
    cutoff = now_arg - timedelta(hours=24)
    today_trades = []
    for t in trades:
        try:
            ts = datetime.fromisoformat(t["timestamp"]).replace(tzinfo=timezone.utc).astimezone(ARG_TZ)
            if ts >= cutoff: today_trades.append(t)
        except: pass

    buys   = [t for t in today_trades if t.get("action") == "BUY"]
    sells  = [t for t in today_trades if t.get("action") == "SELL"]
    closed = [t for t in today_trades if t.get("pnl_pct") is not None]
    winners = [t for t in closed if t.get("pnl_pct", 0) > 0]
    total_pnl = sum((t.get("pnl_pct", 0) * t.get("usd_size", 3)) / 100 for t in closed)
    win_rate  = round(len(winners) / len(closed) * 100) if closed else 0

    best  = max(closed, key=lambda t: t.get("pnl_pct", 0), default=None)
    worst = min(closed, key=lambda t: t.get("pnl_pct", 0), default=None)

    msg = (
        f"📊 <b>Resumen diario — {today.strftime('%d/%m/%Y')}</b>\n\n"
        f"📈 Compras: {len(buys)} | 📉 Ventas: {len(sells)}\n"
        f"✅ Cerrados: {len(closed)} | Win rate: {win_rate}%\n"
        f"💰 P&L del día: {'+'if total_pnl>=0 else ''}${total_pnl:.2f}\n\n"
    )
    if best:
        msg += f"🏆 Mejor: {best['symbol']} {'+' if best['pnl_pct']>=0 else ''}{best['pnl_pct']}%\n"
    if worst and worst != best:
        msg += f"💀 Peor: {worst['symbol']} {worst['pnl_pct']}%\n"
    msg += f"\n🧭 F&G ahora: {fg_value} — {fg_label}\n"
    msg += f"{'📝 PAPER MODE' if PAPER_TRADING else '💰 REAL MODE'}"

    send_telegram(msg)
    log.info(f"📊 Resumen diario enviado")

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
    """Binance Futures (testnet) — para trades con apalancamiento."""
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
        "urls": {"api": {
            "public":  "https://testnet.binancefuture.com",
            "private": "https://testnet.binancefuture.com",
        }}
    })

# ─────────────────────────────────────────
# ALTCOIN SCANNER — top por volumen 24h
# ─────────────────────────────────────────
def scan_top_altcoins(exchange, max_alts=8):
    """Devuelve top altcoins por volumen en USDT, excluyendo stables y base watchlist."""
    global _scanner
    try:
        tickers = exchange.fetch_tickers()
        usdt_pairs = []
        for symbol, t in tickers.items():
            if not symbol.endswith("/USDT"): continue
            base = symbol.replace("/USDT", "")
            if base in EXCLUDE_SYMBOLS: continue
            if not is_valid_symbol(symbol): continue   # filtrar tokens basura
            vol = t.get("quoteVolume") or 0
            if vol < 20_000_000: continue  # mínimo $20M de volumen (más estricto)
            usdt_pairs.append({"symbol": symbol, "volume": vol})

        usdt_pairs.sort(key=lambda x: x["volume"], reverse=True)
        altcoins = [p["symbol"] for p in usdt_pairs[:max_alts]]

        # Scanner list para el dashboard (base + alts)
        _scanner = (
            [{"symbol": s, "volume": None} for s in BASE_WATCHLIST] +
            usdt_pairs[:max_alts]
        )

        log.info(f"🔍 Altcoins seleccionadas: {altcoins}")
        return altcoins
    except Exception as e:
        log.error(f"Scanner error: {e}")
        return []

# ─────────────────────────────────────────
# POSITION SIZING
# ─────────────────────────────────────────
def get_position_size(capital, signal_score):
    abs_score = abs(signal_score)
    pct = POSITION_SIZE_MAP.get(abs_score, 0.02)
    usd = round(capital * pct, 2)
    log.info(f"  💰 Size: {pct*100:.0f}% = ${usd} (score={signal_score:+d})")
    return usd, pct

# ─────────────────────────────────────────
# TRAILING STOP
# ─────────────────────────────────────────
def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_positions(positions):
    with open(POSITIONS_FILE, "w") as f: json.dump(positions, f, indent=2, default=str)

def open_position(symbol, entry_price, usd_size, pct, action, timeframe):
    positions = load_positions()
    positions[symbol] = {
        "symbol": symbol, "action": action, "timeframe": timeframe,
        "entry_price": entry_price, "current_price": entry_price, "high_price": entry_price,
        "trail_stop":  round(entry_price * (1 - TRAILING_STOP_PCT), 4),
        "take_profit": round(entry_price * (1 + TAKE_PROFIT_PCT), 4),
        "usd_size": usd_size, "risk_pct": pct,
        "opened_at": datetime.now().isoformat(),
    }
    save_positions(positions)
    log.info(f"  📂 Posición abierta: {symbol} [{timeframe}] @ {entry_price} | Trail={positions[symbol]['trail_stop']} TP={positions[symbol]['take_profit']}")

def update_trailing_stops(public_ex):
    positions = load_positions()
    if not positions: return
    closed = []
    for symbol, pos in positions.items():
        try:
            current = public_ex.fetch_ticker(symbol)["last"]
            pos["current_price"] = current
            if pos["action"] == "BUY":
                if current > pos["high_price"]:
                    pos["high_price"] = current
                    pos["trail_stop"] = round(current * (1 - TRAILING_STOP_PCT), 4)
                    log.info(f"  📈 Trail actualizado {symbol}: {pos['trail_stop']}")

                if current <= pos["trail_stop"]:
                    pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
                    log.info(f"  🔴 TRAIL STOP {symbol} @ {current} | PnL: {pnl:.2f}%")
                    send_telegram(
                        f"🔴 <b>Trail Stop</b> — {symbol}\n"
                        f"Entrada: {pos['entry_price']} → Salida: {current}\n"
                        f"PnL: {pnl:+.2f}% {'✅' if pnl>0 else '❌'}"
                    )
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action": "SELL", "price": current, "timeframe": pos.get("timeframe","1h"),
                        "reasoning": f"Trail stop (entrada {pos['entry_price']})",
                        "confidence": 1.0, "paper": PAPER_TRADING,
                        "pnl_pct": round(pnl,2), "trail_triggered": True,
                        "entry_price": pos["entry_price"], "usd_size": pos["usd_size"]})
                    closed.append(symbol); continue

                if current >= pos["take_profit"]:
                    pnl = (current - pos["entry_price"]) / pos["entry_price"] * 100
                    log.info(f"  🎯 TAKE PROFIT {symbol} @ {current} | PnL: +{pnl:.2f}%")
                    send_telegram(
                        f"🎯 <b>Take Profit</b> — {symbol}\n"
                        f"Entrada: {pos['entry_price']} → Salida: {current}\n"
                        f"PnL: +{pnl:.2f}% 🎉"
                    )
                    save_trade({"timestamp": datetime.now().isoformat(), "symbol": symbol,
                        "action": "SELL", "price": current, "timeframe": pos.get("timeframe","1h"),
                        "reasoning": f"Take profit (entrada {pos['entry_price']})",
                        "confidence": 1.0, "paper": PAPER_TRADING,
                        "pnl_pct": round(pnl,2), "trail_triggered": False,
                        "entry_price": pos["entry_price"], "usd_size": pos["usd_size"]})
                    closed.append(symbol)
        except Exception as e:
            log.error(f"Error trailing {symbol}: {e}")

    for s in closed: del positions[s]
    save_positions(positions)

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
    df["rsi"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean() / loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["vol_ma20"]    = df["volume"].rolling(20).mean()
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

def timeframe_confirm_signal(exchange, symbol, base_tf):
    """Para 3m usa 15m como confirmación. Para 1h usa 30m."""
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
        log.warning(f"Fear & Greed error: {e}")
        return 50, "Neutral"

def fear_greed_filter(value, action):
    if action == "BUY"  and value < 15: return False  # solo bloquear Extreme Fear severo
    if action == "SELL" and value > 85: return False
    return True

# ─────────────────────────────────────────
# NOTICIAS
# ─────────────────────────────────────────
BULLISH_KW = ["rally","surge","breakout","bullish","adoption","partnership","upgrade","all-time high","ath","gains","rises","jumps","soars","recovery"]
BEARISH_KW = ["crash","hack","ban","bearish","lawsuit","regulation","sell-off","collapse","fear","drops","falls","plunges","warning","risk"]
RSS_FEEDS  = ["https://www.coindesk.com/arc/outboundfeeds/rss/","https://cointelegraph.com/rss"]
COIN_NAMES = {"btc":["bitcoin","btc"],"eth":["ethereum","eth"],"sol":["solana","sol"],"bnb":["bnb","binance"]}

def get_news_sentiment(symbol):
    coin  = symbol.split("/")[0].lower()
    terms = COIN_NAMES.get(coin, [coin.lower()])
    all_titles = []
    for url in RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
            if not titles: titles = re.findall(r"<title>(.*?)</title>", resp.text)
            all_titles.extend(titles[:15])
        except: pass
    if not all_titles: return 0
    relevant = [t.lower() for t in all_titles if any(term in t.lower() for term in terms)]
    if not relevant: relevant = [t.lower() for t in all_titles]
    text = " ".join(relevant)
    bull = sum(1 for kw in BULLISH_KW if kw in text)
    bear = sum(1 for kw in BEARISH_KW if kw in text)
    if bull > bear: return +1
    if bear > bull: return -1
    return 0

# ─────────────────────────────────────────
# CLAUDE
# ─────────────────────────────────────────
def ask_claude(symbol, signals, df, fg_value, usd_size, pct, timeframe):
    last  = df.iloc[-1]
    total = sum(signals.values())
    direction = "BULLISH" if total > 0 else "BEARISH"
    prompt = f"""Trading crypto analyst. Respond ONLY in JSON, no backticks.

Pair: {symbol} [{timeframe}] | Price: {last['close']:.4f} USDT
EMA9: {last['ema9']:.4f} | EMA21: {last['ema21']:.4f} | RSI: {last['rsi']:.1f}
MACD hist: {last['macd_hist']:.4f} | Vol/MA20: {last['volume']:.0f}/{last['vol_ma20']:.0f}
Fear&Greed: {fg_value} | Timeframe: {timeframe}
Signals (EMA:{signals['tech']:+d} MACD:{signals['macd']:+d} VOL:{signals['vol']:+d} CONF:{signals['tf4h']:+d} NEWS:{signals['news']:+d}) = {total:+d}
Position: ${usd_size} ({pct*100:.0f}%) | Trail: {TRAILING_STOP_PCT*100}% | TP: {TAKE_PROFIT_PCT*100}%
Funding rate: {signals.get("funding", 0):+d} (positivo=longs sobrecargados, negativo=oportunidad)

{{"action":"BUY"|"SELL"|"HOLD","confidence":0.0,"reasoning":"one line"}}"""
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
def execute_trade(trade_exchange, symbol, action, usd_size):
    try:
        price = trade_exchange.fetch_ticker(symbol)["last"]
        qty   = usd_size / price
        order = trade_exchange.create_market_buy_order(symbol, qty) if action == "BUY" else trade_exchange.create_market_sell_order(symbol, qty)
        return order, price
    except Exception as e:
        log.error(f"Error orden {action} {symbol}: {e}")
        send_telegram(f"⚠️ Error orden {action} {symbol}\n{e}")
        return None, None


# ─────────────────────────────────────────
# FUTUROS — EJECUTAR CON APALANCAMIENTO
# ─────────────────────────────────────────
def execute_futures_trade(futures_ex, symbol, action, usd_size, leverage=FUTURES_LEVERAGE):
    """Ejecuta orden en futuros con apalancamiento."""
    try:
        if PAPER_TRADING:
            price = futures_ex.fetch_ticker(symbol)["last"] if futures_ex else 0
            return None, price
        # Set leverage
        try:
            futures_ex.set_leverage(leverage, symbol)
        except Exception:
            pass  # algunos pares no soportan set_leverage directo
        price    = futures_ex.fetch_ticker(symbol)["last"]
        notional = usd_size * leverage
        qty      = notional / price
        if action == "BUY":
            order = futures_ex.create_market_buy_order(symbol, qty, {"reduceOnly": False})
        elif action == "SELL":
            order = futures_ex.create_market_sell_order(symbol, qty, {"reduceOnly": False})
        return order, price
    except Exception as e:
        log.error(f"Error futuros {action} {symbol}: {e}")
        return None, None

def save_trade(record):
    data = []
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE) as f: data = json.load(f)
    data.append(record)
    with open(TRADE_LOG_FILE,"w") as f: json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────
# FUNDING RATES
# ─────────────────────────────────────────
_funding_cache = {}
_funding_last_fetch = 0

def get_funding_rate(symbol):
    """
    Obtiene el funding rate de Binance Futuros para un par.
    Positivo = longs pagan a shorts (mercado sobrecargado de longs)
    Negativo = shorts pagan a longs (oportunidad de compra)
    Cache de 15 minutos para no sobrecargar la API.
    """
    global _funding_cache, _funding_last_fetch
    now = time.time()

    # Refrescar cache cada 15 minutos
    if now - _funding_last_fetch > 900:
        try:
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            _funding_cache = {
                item["symbol"]: float(item.get("lastFundingRate", 0))
                for item in data
            }
            _funding_last_fetch = now
            log.info(f"  💹 Funding rates actualizados ({len(_funding_cache)} pares)")
        except Exception as e:
            log.warning(f"Funding rate fetch error: {e}")
            return 0.0

    # Buscar el símbolo (BTCUSDT, ETHUSDT, etc.)
    futures_symbol = symbol.replace("/", "")
    rate = _funding_cache.get(futures_symbol, None)
    if rate is None:
        return 0.0
    return rate

def funding_rate_signal(symbol):
    """
    Retorna señal basada en funding rate:
    +1 = funding negativo → shorts pagando → buen momento para comprar
     0 = neutral
    -1 = funding muy positivo → longs sobrecargados → evitar compras
    """
    rate = get_funding_rate(symbol)
    if rate == 0.0:
        return 0, rate

    if rate <= FUNDING_BULLISH_THRESHOLD:
        log.info(f"  💹 Funding: {rate:.4%} → BULLISH (shorts pagando)")
        return +1, rate
    elif rate >= FUNDING_BEARISH_THRESHOLD:
        log.info(f"  💹 Funding: {rate:.4%} → BEARISH (mercado sobrecargado)")
        return -1, rate
    else:
        log.info(f"  💹 Funding: {rate:.4%} → neutral")
        return 0, rate

# ─────────────────────────────────────────
# CAPITAL ALLOCATOR
# ─────────────────────────────────────────
def get_allocated_capital(positions):
    """Calcula cuánto capital está actualmente en posiciones abiertas."""
    return sum(p.get("usd_size", 0) for p in positions.values())

def can_open_position(positions, new_size):
    """
    Verifica si hay capital disponible para abrir una nueva posición.
    Límite: MAX_CAPITAL_EXPOSURE % del capital total.
    """
    allocated    = get_allocated_capital(positions)
    max_allowed  = CAPITAL_TOTAL_USD * MAX_CAPITAL_EXPOSURE
    available    = max_allowed - allocated

    log.info(f"  💼 Capital: ${allocated:.2f} usado / ${max_allowed:.2f} máx (${available:.2f} disp)")

    if new_size > available:
        log.info(f"  🚫 Capital insuficiente: necesita ${new_size} pero solo ${available:.2f} disponible")
        return False
    return True

# ─────────────────────────────────────────
# ANALIZAR UN PAR
# ─────────────────────────────────────────
def analyze_and_trade(symbol, timeframe, public_ex, trade_ex, futures_ex, fg_value, fg_label, open_positions):
    """Analiza un par y ejecuta/registra trade si hay señal.
    Score 2-3: spot sin apalancamiento
    Score 4-5: futuros con 2x
    """
    if symbol in open_positions:
        log.info(f"  {symbol}: posición ya abierta — skip")
        return

    df    = calculate_indicators(get_ohlcv(public_ex, symbol, timeframe=timeframe))
    t_sig = technical_signal(df)
    m_sig = macd_signal(df)
    v_sig = volume_signal(df)
    h_sig = timeframe_confirm_signal(public_ex, symbol, timeframe)
    n_sig = get_news_sentiment(symbol)

    # Funding rate — señal de posicionamiento
    fr_sig, fr_value = funding_rate_signal(symbol)

    signals = {"tech": t_sig, "macd": m_sig, "vol": v_sig, "tf4h": h_sig, "news": n_sig, "funding": fr_sig}
    total   = sum(signals.values())

    log.info(f"  [{timeframe}] EMA:{t_sig:+d} MACD:{m_sig:+d} VOL:{v_sig:+d} CONF:{h_sig:+d} NEWS:{n_sig:+d} FR:{fr_sig:+d} = {total:+d}")

    if abs(total) < MIN_SIGNALS:
        log.info("  ⏭️  Señales insuficientes — skip")
        return

    # Para altcoins en 3m solo buscar compras (scalping long)
    if timeframe == "3m" and total < 0:
        log.info("  ⏭️  Señal bajista en 3m — solo long en altcoins")
        return

    usd_size, risk_pct = get_position_size(CAPITAL_TOTAL_USD, total)

    # Capital Allocator — verificar que hay capital disponible
    if not can_open_position(open_positions, usd_size):
        send_telegram(f"💼 <b>Capital límite alcanzado</b>\nNo se puede abrir {symbol} — máximo {MAX_CAPITAL_EXPOSURE*100:.0f}% expuesto")
        return

    log.info("  🧠 Consultando Claude...")
    analysis = ask_claude(symbol, signals, df, fg_value, usd_size, risk_pct, timeframe)
    log.info(f"  Claude: {analysis['action']} ({analysis['confidence']:.2f}) — {analysis['reasoning']}")

    if not fear_greed_filter(fg_value, analysis["action"]): return

    current_price = df.iloc[-1]["close"]
    base_record = {
        "timestamp": datetime.now().isoformat(), "symbol": symbol,
        "action": analysis["action"], "confidence": analysis["confidence"],
        "reasoning": analysis["reasoning"], "timeframe": timeframe,
        "tech_signal": t_sig, "macd_signal": m_sig, "vol_signal": v_sig,
        "tf4h_signal": h_sig, "news_signal": n_sig,
        "funding_signal": fr_sig, "funding_rate": round(fr_value * 100, 4),
        "fear_greed_value": fg_value, "fear_greed_label": fg_label,
        "price": current_price, "usd_size": usd_size, "risk_pct": risk_pct,
        "signal_score": total,
    }

    if analysis["action"] == "BUY" and analysis["confidence"] >= CONFIDENCE_MIN:
        # Decidir spot vs futuros según score
        use_futures   = (abs(total) >= FUTURES_MIN_SCORE) and (fg_value >= 20)
        trail_pct     = FUTURES_TRAILING if use_futures else TRAILING_STOP_PCT
        tp_pct        = FUTURES_TP       if use_futures else TAKE_PROFIT_PCT
        leverage      = FUTURES_LEVERAGE if use_futures else 1
        trail_stop    = round(current_price * (1 - trail_pct), 4)
        take_profit   = round(current_price * (1 + tp_pct), 4)
        mode_label    = f"FUTURES {leverage}x" if use_futures else "SPOT"

        log.info(f"  Mode: {mode_label} | Trail={trail_pct*100}% TP={tp_pct*100}%")

        if PAPER_TRADING:
            save_trade({**base_record, "paper": True, "order_id": None,
                "mode": mode_label, "leverage": leverage,
                "trail_stop": trail_stop, "take_profit": take_profit})
            open_position(symbol, current_price, usd_size, risk_pct, "BUY", timeframe)
            log.info(f"  📝 PAPER {mode_label} BUY @ {current_price} | Trail={trail_stop} TP={take_profit}")
            allocated_now = get_allocated_capital(open_positions)
            send_telegram(
                f"📝 <b>PAPER {mode_label} BUY [{timeframe}]</b>\n"
                f"Par: <b>{symbol}</b> @ {current_price}\n"
                f"🔴 Trail: {trail_stop} | 🎯 TP: {take_profit}\n"
                f"💰 ${usd_size}×{leverage} = ${usd_size*leverage:.0f} ({risk_pct*100:.0f}% — score {total:+d})\n"
                f"💹 FR: {fr_sig:+d} | F&G: {fg_value}\n"
                f"💼 Capital: ${allocated_now:.1f}/${CAPITAL_TOTAL_USD*MAX_CAPITAL_EXPOSURE:.0f}\n"
                f"Confianza: {int(analysis['confidence']*100)}%"
            )
        else:
            if use_futures:
                order, exec_price = execute_futures_trade(futures_ex, symbol, "BUY", usd_size)
            else:
                order, exec_price = execute_trade(trade_ex, symbol, "BUY", usd_size)
            if order:
                actual = exec_price or current_price
                save_trade({**base_record, "paper": False, "order_id": order.get("id"),
                    "price": actual, "mode": mode_label, "leverage": leverage})
                open_position(symbol, actual, usd_size, risk_pct, "BUY", timeframe)
                send_telegram(
                    f"{'🚀' if use_futures else '✅'} <b>{mode_label} [{timeframe}]</b> — {symbol} @ {actual}\n"
                    f"💰 ${usd_size}×{leverage} | Trail: {round(actual*(1-trail_pct),4)}"
                )

    elif analysis["action"] == "SELL" and analysis["confidence"] >= CONFIDENCE_MIN:
        if PAPER_TRADING:
            save_trade({**base_record, "paper": True, "order_id": None})
            log.info(f"  📝 PAPER SELL @ {current_price}")
        else:
            order, exec_price = execute_trade(trade_ex, symbol, "SELL", usd_size)
            if order:
                save_trade({**base_record, "paper": False, "order_id": order.get("id")})

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run_bot():
    log.info("🤖 CryptoBot v6 iniciado — Spot + Futuros 2x + 15 Altcoins")
    log.info(f"Mode: {'PAPER' if PAPER_TRADING else 'REAL'} | Capital: ${CAPITAL_TOTAL_USD}")
    log.info(f"Base: {BASE_WATCHLIST} [1h] + Top altcoins [3m scalping]")

    send_telegram(
        f"🤖 <b>CryptoBot v6 iniciado</b>\n"
        f"Mode: {'📝 PAPER' if PAPER_TRADING else '💰 REAL'}\n"
        f"Base 1h: {', '.join(s.replace('/USDT','') for s in BASE_WATCHLIST)}\n"
        f"+ Top altcoins 3m (scalping)\n"
        f"Spot: Trail {TRAILING_STOP_PCT*100}% | TP {TAKE_PROFIT_PCT*100}%\n"
        f"Futuros {FUTURES_LEVERAGE}x (score≥{FUTURES_MIN_SCORE}): Trail {FUTURES_TRAILING*100}% | TP {FUTURES_TP*100}%\n"
        f"F&G block: <15 | Confianza: >{int(CONFIDENCE_MIN*100)}% | Resumen: 9 AM"
    )

    public_ex  = get_public_exchange()
    trade_ex   = get_trade_exchange()
    futures_ex = get_futures_exchange()
    altcoins   = []
    last_scan  = 0
    log.info(f"💹 Futuros: {FUTURES_LEVERAGE}x apalancamiento (score ≥{FUTURES_MIN_SCORE})")

    while True:
        now = time.time()
        log.info(f"\n{'='*50}\n⏰ Ciclo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Actualizar trailing stops
        update_trailing_stops(public_ex)

        # 2. Fear & Greed
        fg_value, fg_label = get_fear_greed()
        log.info(f"🧭 Fear & Greed: {fg_value} — {fg_label}")

        # 3. Resumen diario
        maybe_send_daily_report(fg_value, fg_label)

        # 4. Scan altcoins cada 30 minutos
        if now - last_scan > 600:  # re-scan cada 10 minutos
            log.info("🔍 Escaneando top altcoins por volumen...")
            altcoins  = scan_top_altcoins(public_ex, max_alts=15)
            last_scan = now

        open_positions = load_positions()
        log.info(f"📂 Posiciones abiertas: {list(open_positions.keys()) or 'ninguna'}")

        # 5. Analizar base watchlist en 1h
        log.info("\n--- BASE WATCHLIST [1h] ---")
        for symbol in BASE_WATCHLIST:
            try:
                log.info(f"\n📊 {symbol}...")
                analyze_and_trade(symbol, "1h", public_ex, trade_ex, futures_ex, fg_value, fg_label, open_positions)
            except Exception as e:
                log.error(f"Error {symbol}: {e}")

        # 6. Analizar altcoins en 15m
        if altcoins:
            log.info("\n--- ALTCOIN SCANNER [3m scalping] ---")
            for symbol in altcoins:
                try:
                    log.info(f"\n📊 {symbol} [3m]...")
                    analyze_and_trade(symbol, "3m", public_ex, trade_ex, futures_ex, fg_value, fg_label, open_positions)
                except Exception as e:
                    log.error(f"Error {symbol}: {e}")

        log.info(f"\n💤 Esperando {LOOP_INTERVAL_SEC}s...")
        time.sleep(LOOP_INTERVAL_SEC)

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_dashboard, daemon=True).start()
    run_bot()
