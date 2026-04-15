"""
CryptoBot v3
- Trailing stop loss dinámico
- Position sizing por conviction (score de señales)
- Paper trading mode
- EMA + MACD + Volume + 4h + Fear&Greed + Noticias
"""

import os
import re
import time
import json
import logging
import requests
import threading
from datetime import datetime
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

# Position sizing dinámico por score de señales
POSITION_SIZE_MAP  = {2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05}  # score → % capital
TRAILING_STOP_PCT  = 0.015   # trailing stop: 1.5% por debajo del máximo
TAKE_PROFIT_PCT    = 0.04    # take profit: 4%
MIN_SIGNALS        = 2
LOOP_INTERVAL_SEC  = 300
WATCHLIST          = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TRADE_LOG_FILE     = "trade_log.json"
POSITIONS_FILE     = "positions.json"  # posiciones abiertas para trailing stop

# ─────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CryptoBot v3</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root{--bg:#080c10;--surface:#0d1117;--border:#1a2332;--green:#00ff88;--red:#ff3355;--yellow:#ffcc00;--blue:#00aaff;--orange:#ff9900;--muted:#3d5166;--text:#c9d8e8;--mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);min-height:100vh}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,136,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
  .container{position:relative;z-index:1;max-width:1200px;margin:0 auto;padding:40px 24px}
  header{display:flex;align-items:center;justify-content:space-between;margin-bottom:40px;padding-bottom:24px;border-bottom:1px solid var(--border)}
  .logo{font-family:var(--sans);font-weight:800;font-size:1.4rem;color:#fff}.logo span{color:var(--green)}
  .logo small{font-size:.7rem;color:var(--muted);font-weight:400;margin-left:8px}
  .badges{display:flex;gap:8px}
  .pill{display:flex;align-items:center;gap:6px;font-size:.72rem;padding:5px 12px;border-radius:100px}
  .pill-live{color:var(--green);border:1px solid rgba(0,255,136,.3)}
  .pill-paper{color:var(--blue);border:1px solid rgba(0,170,255,.3)}
  .dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin-bottom:32px}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px;position:relative;overflow:hidden}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,var(--green));opacity:.7}
  .stat-label{font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
  .stat-value{font-family:var(--sans);font-weight:800;font-size:1.7rem;line-height:1;color:#fff}
  .stat-value.green{color:var(--green)}.stat-value.red{color:var(--red)}.stat-value.yellow{color:var(--yellow)}.stat-value.blue{color:var(--blue)}.stat-value.orange{color:var(--orange)}
  .stat-sub{font-size:.62rem;color:var(--muted);margin-top:5px}

  /* Posiciones abiertas */
  .positions-section{margin-bottom:32px}
  .pos-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
  .pos-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px}
  .pos-card.profit{border-color:rgba(0,255,136,.3)}
  .pos-card.loss{border-color:rgba(255,51,85,.3)}
  .pos-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
  .pos-symbol{font-family:var(--sans);font-weight:700;font-size:1rem;color:#fff}
  .pos-pnl{font-family:var(--sans);font-weight:700;font-size:.9rem}
  .pos-pnl.pos{color:var(--green)}.pos-pnl.neg{color:var(--red)}
  .pos-row{display:flex;justify-content:space-between;font-size:.72rem;color:var(--muted);margin-top:4px}
  .pos-row span:last-child{color:var(--text)}
  .trail-bar-bg{height:3px;background:var(--border);border-radius:2px;margin-top:10px;overflow:hidden}
  .trail-bar-fill{height:100%;background:var(--orange);border-radius:2px;transition:width .5s}

  .section-title{font-family:var(--sans);font-size:.72rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
  .table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin-bottom:32px}
  table{width:100%;border-collapse:collapse;font-size:.76rem}
  thead tr{border-bottom:1px solid var(--border)}
  th{padding:11px 14px;text-align:left;font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:400}
  td{padding:11px 14px;border-bottom:1px solid rgba(26,35,50,.5);vertical-align:middle}
  tr:last-child td{border-bottom:none}tr:hover td{background:rgba(255,255,255,.02)}
  .badge{display:inline-block;padding:2px 9px;border-radius:4px;font-size:.68rem;font-weight:700}
  .badge-buy{background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.2)}
  .badge-sell{background:rgba(255,51,85,.12);color:var(--red);border:1px solid rgba(255,51,85,.2)}
  .badge-hold{background:rgba(61,81,102,.3);color:var(--muted);border:1px solid rgba(61,81,102,.4)}
  .badge-paper{font-size:.54rem;background:rgba(0,170,255,.1);color:var(--blue);border:1px solid rgba(0,170,255,.2);padding:1px 5px;border-radius:3px;margin-left:4px;vertical-align:middle}
  .badge-trail{font-size:.54rem;background:rgba(255,153,0,.1);color:var(--orange);border:1px solid rgba(255,153,0,.2);padding:1px 5px;border-radius:3px;margin-left:4px;vertical-align:middle}
  .conf-bar{display:flex;align-items:center;gap:8px}
  .bar-bg{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
  .bar-fill{height:100%;background:var(--green);border-radius:2px}
  .pair{color:#fff;font-weight:600}.ts{color:var(--muted);font-size:.68rem}
  .empty{text-align:center;padding:50px 20px;color:var(--muted)}
  .empty-icon{font-size:2rem;margin-bottom:10px}.empty-text{font-size:.82rem;line-height:1.6}
  footer{text-align:center;font-size:.68rem;color:var(--muted);padding-top:20px;border-top:1px solid var(--border)}
  .refresh-info{font-size:.68rem;color:var(--muted);text-align:right;margin-bottom:10px}
  #countdown{color:var(--green)}
  .fg-pill{display:inline-block;padding:2px 7px;border-radius:3px;font-size:.62rem;font-weight:700}
  .fg-fear{background:rgba(255,51,85,.15);color:var(--red)}
  .fg-greed{background:rgba(0,255,136,.15);color:var(--green)}
  .fg-neutral{background:rgba(255,204,0,.15);color:var(--yellow)}
  .size-badge{font-size:.62rem;color:var(--orange);font-weight:700}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">Crypto<span>Bot</span><small>v3</small></div>
    <div class="badges">
      <div class="pill pill-paper" id="mode-pill"><div class="dot"></div><span id="mode-text">PAPER</span></div>
      <div class="pill pill-live"><div class="dot"></div>LIVE</div>
    </div>
  </header>

  <div class="stats">
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Total Trades</div><div class="stat-value" id="total">—</div><div class="stat-sub">paper + real</div></div>
    <div class="stat-card" style="--accent:var(--green)"><div class="stat-label">Compras</div><div class="stat-value green" id="buys">—</div><div class="stat-sub">BUY</div></div>
    <div class="stat-card" style="--accent:var(--red)"><div class="stat-label">Ventas</div><div class="stat-value red" id="sells">—</div><div class="stat-sub">SELL</div></div>
    <div class="stat-card" style="--accent:var(--yellow)"><div class="stat-label">Confianza</div><div class="stat-value yellow" id="avg-conf">—</div><div class="stat-sub">promedio Claude</div></div>
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Fear & Greed</div><div class="stat-value blue" id="fg-val">—</div><div class="stat-sub" id="fg-lbl">cargando...</div></div>
    <div class="stat-card" style="--accent:var(--orange)"><div class="stat-label">Posiciones</div><div class="stat-value orange" id="open-pos">—</div><div class="stat-sub">abiertas</div></div>
  </div>

  <div class="positions-section" id="positions-section" style="display:none">
    <div class="section-title">Posiciones abiertas (trailing stop activo)</div>
    <div class="pos-grid" id="pos-grid"></div>
  </div>

  <div class="refresh-info">Auto-refresh en <span id="countdown">30</span>s</div>
  <div class="section-title">Historial de operaciones</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Par</th><th>Acción</th><th>Precio</th><th>Size</th><th>Confianza</th><th>Señales</th><th>F&G</th><th>Razonamiento</th><th>Timestamp</th></tr></thead>
      <tbody id="trades-body"></tbody>
    </table>
  </div>
  <footer>CryptoBot v3 · Trailing Stop · Position Sizing Dinámico · Paper Trading</footer>
</div>
<script>
let countdown=30;
async function loadData(){
  try{
    const res=await fetch('/api/trades');
    const data=await res.json();
    const trades=(data.trades||[]).filter(t=>t.action!=='HOLD');

    document.getElementById('total').textContent=trades.length;
    document.getElementById('buys').textContent=trades.filter(t=>t.action==='BUY').length;
    document.getElementById('sells').textContent=trades.filter(t=>t.action==='SELL').length;
    const avgConf=trades.length?(trades.reduce((s,t)=>s+(t.confidence||0),0)/trades.length*100).toFixed(0)+'%':'—';
    document.getElementById('avg-conf').textContent=avgConf;

    if(data.fear_greed){
      const fg=data.fear_greed;
      const el=document.getElementById('fg-val');
      el.textContent=fg.value;
      document.getElementById('fg-lbl').textContent=fg.label;
      el.className='stat-value '+(fg.value<35?'red':fg.value>65?'green':'yellow');
    }

    // Paper mode indicator
    const hasPaper=trades.some(t=>t.paper);
    const hasReal=trades.some(t=>!t.paper);
    const pill=document.getElementById('mode-pill');
    document.getElementById('mode-text').textContent=hasReal?'REAL':'PAPER';
    pill.className='pill '+(hasReal?'pill-live':'pill-paper');

    // Posiciones abiertas
    const positions=data.positions||[];
    document.getElementById('open-pos').textContent=positions.length;
    const posSection=document.getElementById('positions-section');
    if(positions.length>0){
      posSection.style.display='block';
      document.getElementById('pos-grid').innerHTML=positions.map(p=>{
        const pnlPct=((p.current_price-p.entry_price)/p.entry_price*100).toFixed(2);
        const trailPct=((p.current_price-p.trail_stop)/p.current_price*100).toFixed(1);
        const isProfit=pnlPct>=0;
        const distToTrail=((p.current_price-p.trail_stop)/p.current_price*100);
        const barWidth=Math.min(100,Math.max(0,(1-distToTrail/5)*100));
        return `<div class="pos-card ${isProfit?'profit':'loss'}">
          <div class="pos-header">
            <span class="pos-symbol">${p.symbol}</span>
            <span class="pos-pnl ${isProfit?'pos':'neg'}">${isProfit?'+':''}${pnlPct}%</span>
          </div>
          <div class="pos-row"><span>Entrada</span><span>${p.entry_price}</span></div>
          <div class="pos-row"><span>Precio actual</span><span>${p.current_price}</span></div>
          <div class="pos-row"><span>🔴 Trail Stop</span><span style="color:var(--orange)">${p.trail_stop}</span></div>
          <div class="pos-row"><span>🎯 Take Profit</span><span style="color:var(--green)">${p.take_profit}</span></div>
          <div class="pos-row"><span>Size</span><span>$${p.usd_size}</span></div>
          <div class="trail-bar-bg"><div class="trail-bar-fill" style="width:${barWidth}%"></div></div>
        </div>`;
      }).join('');
    } else { posSection.style.display='none'; }

    // Trades table
    const tbody=document.getElementById('trades-body');
    if(!trades.length){
      tbody.innerHTML='<tr><td colspan="9"><div class="empty"><div class="empty-icon">🤖</div><div class="empty-text">El bot está analizando señales...<br>Las operaciones aparecerán aquí cuando se ejecuten.</div></div></td></tr>';
      return;
    }
    tbody.innerHTML=[...trades].reverse().map(t=>{
      const ts=new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const conf=Math.round((t.confidence||0)*100);
      const signals=[];
      if(t.tech_signal===1)signals.push('📈EMA');else if(t.tech_signal===-1)signals.push('📉EMA');
      if(t.macd_signal===1)signals.push('📊+');else if(t.macd_signal===-1)signals.push('📊-');
      if(t.vol_signal===1)signals.push('📦VOL');
      if(t.tf4h_signal===1)signals.push('⏱4H+');else if(t.tf4h_signal===-1)signals.push('⏱4H-');
      if(t.news_signal===1)signals.push('📰+');else if(t.news_signal===-1)signals.push('📰-');
      const fgVal=t.fear_greed_value||'—';
      const fgCls=fgVal<35?'fg-fear':fgVal>65?'fg-greed':'fg-neutral';
      const paperTag=t.paper?'<span class="badge-paper">PAPER</span>':'';
      const trailTag=t.trail_triggered?'<span class="badge-trail">TRAIL</span>':'';
      const price=t.price?t.price.toLocaleString('en-US',{maximumFractionDigits:4}):'—';
      const sizeUSD=t.usd_size?`$${t.usd_size}`:(t.risk_pct?`${(t.risk_pct*100).toFixed(0)}%`:'—');
      return `<tr>
        <td class="pair">${t.symbol||'—'}</td>
        <td><span class="badge badge-${(t.action||'').toLowerCase()}">${t.action||'—'}</span>${paperTag}${trailTag}</td>
        <td style="color:var(--text)">${price}</td>
        <td class="size-badge">${sizeUSD}</td>
        <td><div class="conf-bar"><div class="bar-bg"><div class="bar-fill" style="width:${conf}%"></div></div><span style="font-size:.7rem;min-width:30px">${conf}%</span></div></td>
        <td style="color:var(--muted);font-size:.68rem">${signals.join(' ')}</td>
        <td><span class="fg-pill ${fgCls}">${fgVal}</span></td>
        <td style="color:var(--muted);max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${t.reasoning||''}">${t.reasoning||'—'}</td>
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
    # Enriquecer posiciones con precio actual (en memoria)
    return jsonify({"trades": trades, "count": len(trades), "fear_greed": _fear_greed, "positions": list(positions.values())})

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

# ─────────────────────────────────────────
# POSITION SIZING DINÁMICO
# ─────────────────────────────────────────
def get_position_size(capital, signal_score):
    """Más señales alineadas = más capital en juego."""
    abs_score = abs(signal_score)
    pct = POSITION_SIZE_MAP.get(abs_score, 0.02)
    usd = round(capital * pct, 2)
    log.info(f"  💰 Position size: {pct*100:.0f}% = ${usd} (score={signal_score:+d})")
    return usd, pct

# ─────────────────────────────────────────
# TRAILING STOP — GESTIÓN DE POSICIONES
# ─────────────────────────────────────────
def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE) as f: return json.load(f)
        except: pass
    return {}

def save_positions(positions):
    with open(POSITIONS_FILE, "w") as f: json.dump(positions, f, indent=2, default=str)

def open_position(symbol, entry_price, usd_size, pct, action):
    positions = load_positions()
    positions[symbol] = {
        "symbol":       symbol,
        "action":       action,
        "entry_price":  entry_price,
        "current_price": entry_price,
        "high_price":   entry_price,
        "trail_stop":   round(entry_price * (1 - TRAILING_STOP_PCT), 4),
        "take_profit":  round(entry_price * (1 + TAKE_PROFIT_PCT), 4),
        "usd_size":     usd_size,
        "risk_pct":     pct,
        "opened_at":    datetime.now().isoformat(),
    }
    save_positions(positions)
    log.info(f"  📂 Posición abierta: {symbol} @ {entry_price} | Trail={positions[symbol]['trail_stop']} TP={positions[symbol]['take_profit']}")

def update_trailing_stops(public_ex):
    """Actualiza trailing stops con precios actuales y cierra posiciones si corresponde."""
    positions = load_positions()
    if not positions: return

    closed = []
    for symbol, pos in positions.items():
        try:
            current = public_ex.fetch_ticker(symbol)["last"]
            pos["current_price"] = current

            if pos["action"] == "BUY":
                # Actualizar trailing stop si el precio subió
                if current > pos["high_price"]:
                    pos["high_price"]  = current
                    pos["trail_stop"]  = round(current * (1 - TRAILING_STOP_PCT), 4)
                    log.info(f"  📈 Trail stop actualizado {symbol}: {pos['trail_stop']} (precio: {current})")

                # Verificar si se activó el trailing stop
                if current <= pos["trail_stop"]:
                    pnl_pct = (current - pos["entry_price"]) / pos["entry_price"] * 100
                    log.info(f"  🔴 TRAIL STOP activado {symbol} @ {current} | PnL: {pnl_pct:.2f}%")
                    send_telegram(
                        f"🔴 <b>Trail Stop activado</b>\n"
                        f"Par: <b>{symbol}</b>\n"
                        f"Entrada: {pos['entry_price']} | Salida: {current}\n"
                        f"PnL: {pnl_pct:+.2f}%\n"
                        f"{'✅ Ganancia' if pnl_pct > 0 else '❌ Pérdida'}"
                    )
                    save_trade({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol, "action": "SELL",
                        "price": current, "reasoning": f"Trail stop activado (entrada: {pos['entry_price']})",
                        "confidence": 1.0, "paper": PAPER_TRADING,
                        "pnl_pct": round(pnl_pct, 2), "trail_triggered": True,
                        "entry_price": pos["entry_price"], "usd_size": pos["usd_size"],
                    })
                    closed.append(symbol)
                    continue

                # Verificar take profit
                if current >= pos["take_profit"]:
                    pnl_pct = (current - pos["entry_price"]) / pos["entry_price"] * 100
                    log.info(f"  🎯 TAKE PROFIT {symbol} @ {current} | PnL: +{pnl_pct:.2f}%")
                    send_telegram(
                        f"🎯 <b>Take Profit alcanzado</b>\n"
                        f"Par: <b>{symbol}</b>\n"
                        f"Entrada: {pos['entry_price']} | Salida: {current}\n"
                        f"PnL: +{pnl_pct:.2f}% 🎉"
                    )
                    save_trade({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol, "action": "SELL",
                        "price": current, "reasoning": f"Take profit alcanzado (entrada: {pos['entry_price']})",
                        "confidence": 1.0, "paper": PAPER_TRADING,
                        "pnl_pct": round(pnl_pct, 2), "trail_triggered": False,
                        "entry_price": pos["entry_price"], "usd_size": pos["usd_size"],
                    })
                    closed.append(symbol)

        except Exception as e:
            log.error(f"Error actualizando trailing {symbol}: {e}")

    # Cerrar posiciones alcanzadas
    for symbol in closed:
        del positions[symbol]

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
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
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

def timeframe_4h_signal(exchange, symbol):
    try:
        df4h = calculate_indicators(get_ohlcv(exchange, symbol, timeframe="4h", limit=50))
        return technical_signal(df4h)
    except Exception as e:
        log.warning(f"4h signal error {symbol}: {e}")
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
        log.info(f"  Fear & Greed: {value} ({label})")
        return value, label
    except Exception as e:
        log.warning(f"Fear & Greed error: {e}")
        return 50, "Neutral"

def fear_greed_filter(value, action):
    if action == "BUY"  and value < 25:
        log.info(f"  🚫 Bloqueado por Extreme Fear ({value})")
        return False
    if action == "SELL" and value > 75:
        log.info(f"  🚫 Bloqueado por Extreme Greed ({value})")
        return False
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
    terms = COIN_NAMES.get(coin, [coin])
    all_titles = []
    for url in RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
            if not titles: titles = re.findall(r"<title>(.*?)</title>", resp.text)
            all_titles.extend(titles[:15])
        except Exception as e: log.warning(f"RSS error: {e}")
    if not all_titles: return 0
    relevant  = [t.lower() for t in all_titles if any(term in t.lower() for term in terms)]
    if not relevant: relevant = [t.lower() for t in all_titles]
    text = " ".join(relevant)
    bull = sum(1 for kw in BULLISH_KW if kw in text)
    bear = sum(1 for kw in BEARISH_KW if kw in text)
    log.info(f"  Noticias: {len(relevant)} | bull={bull} bear={bear}")
    if bull > bear: return +1
    if bear > bull: return -1
    return 0

# ─────────────────────────────────────────
# CLAUDE
# ─────────────────────────────────────────
def ask_claude(symbol, signals, df, fg_value, usd_size, pct):
    last  = df.iloc[-1]
    total = sum(signals.values())
    direction = "BULLISH" if total > 0 else "BEARISH"
    prompt = f"""Sos un analista de trading crypto experto. Respondé SOLO en JSON sin backticks.

Par: {symbol} | Precio: {last['close']:.4f} USDT
EMA9: {last['ema9']:.4f} | EMA21: {last['ema21']:.4f} | RSI: {last['rsi']:.1f}
MACD hist: {last['macd_hist']:.4f} | Volumen vs MA20: {last['volume']:.0f}/{last['vol_ma20']:.0f}
Fear & Greed: {fg_value}

Señales (EMA:{signals['tech']:+d} MACD:{signals['macd']:+d} VOL:{signals['vol']:+d} 4H:{signals['tf4h']:+d} NEWS:{signals['news']:+d}) = {total:+d}
Position size: ${usd_size} ({pct*100:.0f}% capital) — Trailing stop: {TRAILING_STOP_PCT*100:.1f}%

{{"action":"BUY"|"SELL"|"HOLD","confidence":0.0,"reasoning":"una línea"}}"""
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
        if action == "BUY":
            order = trade_exchange.create_market_buy_order(symbol, qty)
        elif action == "SELL":
            order = trade_exchange.create_market_sell_order(symbol, qty)
        return order, price
    except Exception as e:
        log.error(f"Error orden {action} {symbol}: {e}")
        send_telegram(f"⚠️ Error orden {action} {symbol}\n{e}")
        return None, None

def save_trade(record):
    data = []
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE) as f: data = json.load(f)
    data.append(record)
    with open(TRADE_LOG_FILE,"w") as f: json.dump(data, f, indent=2, default=str)

# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run_bot():
    log.info("🤖 CryptoBot v3 iniciado")
    log.info(f"Mode: {'PAPER TRADING' if PAPER_TRADING else 'REAL'}")
    log.info(f"Trailing stop: {TRAILING_STOP_PCT*100}% | Take profit: {TAKE_PROFIT_PCT*100}%")
    log.info(f"Position sizing: {POSITION_SIZE_MAP}")

    send_telegram(
        f"🤖 <b>CryptoBot v3 iniciado</b>\n"
        f"Mode: {'📝 PAPER TRADING' if PAPER_TRADING else '💰 REAL'}\n"
        f"Capital: ${CAPITAL_TOTAL_USD}\n"
        f"Trailing stop: {TRAILING_STOP_PCT*100}% | TP: {TAKE_PROFIT_PCT*100}%\n"
        f"Position sizing: 2-5% según señales\n"
        f"Pares: {', '.join(WATCHLIST)}"
    )

    public_ex = get_public_exchange()
    trade_ex  = get_trade_exchange()

    while True:
        log.info(f"\n{'='*50}\n⏰ Ciclo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Actualizar trailing stops de posiciones abiertas
        update_trailing_stops(public_ex)

        # 2. Fear & Greed
        fg_value, fg_label = get_fear_greed()
        log.info(f"🧭 Fear & Greed: {fg_value} — {fg_label}")

        # 3. Analizar cada par
        open_positions = load_positions()

        for symbol in WATCHLIST:
            try:
                # No abrir nueva posición si ya hay una abierta en este par
                if symbol in open_positions:
                    log.info(f"\n📊 {symbol}... ya tiene posición abierta, skip")
                    continue

                log.info(f"\n📊 {symbol}...")
                df    = calculate_indicators(get_ohlcv(public_ex, symbol))
                t_sig = technical_signal(df)
                m_sig = macd_signal(df)
                v_sig = volume_signal(df)
                h_sig = timeframe_4h_signal(public_ex, symbol)
                n_sig = get_news_sentiment(symbol)

                signals = {"tech": t_sig, "macd": m_sig, "vol": v_sig, "tf4h": h_sig, "news": n_sig}
                total   = sum(signals.values())

                log.info(f"  EMA:{t_sig:+d} MACD:{m_sig:+d} VOL:{v_sig:+d} 4H:{h_sig:+d} NEWS:{n_sig:+d} = {total:+d}")

                if abs(total) < MIN_SIGNALS:
                    log.info("  ⏭️  Señales insuficientes — skip")
                    continue

                # Position sizing dinámico
                usd_size, risk_pct = get_position_size(CAPITAL_TOTAL_USD, total)

                # Consultar Claude
                log.info("  🧠 Consultando Claude...")
                analysis = ask_claude(symbol, signals, df, fg_value, usd_size, risk_pct)
                log.info(f"  Claude: {analysis['action']} ({analysis['confidence']:.2f}) — {analysis['reasoning']}")

                # Filtro Fear & Greed
                if not fear_greed_filter(fg_value, analysis["action"]):
                    continue

                current_price = df.iloc[-1]["close"]
                trail_stop    = round(current_price * (1 - TRAILING_STOP_PCT), 4)
                take_profit   = round(current_price * (1 + TAKE_PROFIT_PCT), 4)

                base_record = {
                    "timestamp":        datetime.now().isoformat(),
                    "symbol":           symbol,
                    "action":           analysis["action"],
                    "confidence":       analysis["confidence"],
                    "reasoning":        analysis["reasoning"],
                    "tech_signal":      t_sig,
                    "macd_signal":      m_sig,
                    "vol_signal":       v_sig,
                    "tf4h_signal":      h_sig,
                    "news_signal":      n_sig,
                    "fear_greed_value": fg_value,
                    "fear_greed_label": fg_label,
                    "price":            current_price,
                    "trail_stop":       trail_stop,
                    "take_profit":      take_profit,
                    "usd_size":         usd_size,
                    "risk_pct":         risk_pct,
                    "signal_score":     total,
                }

                if analysis["action"] == "BUY" and analysis["confidence"] >= 0.6:
                    if PAPER_TRADING:
                        save_trade({**base_record, "paper": True, "order_id": None})
                        open_position(symbol, current_price, usd_size, risk_pct, "BUY")
                        log.info(f"  📝 PAPER BUY @ {current_price} | Trail={trail_stop} TP={take_profit} Size=${usd_size}")
                        send_telegram(
                            f"📝 <b>PAPER BUY</b>\n"
                            f"Par: <b>{symbol}</b> @ {current_price} USDT\n"
                            f"🔴 Trail Stop: {trail_stop}\n"
                            f"🎯 Take Profit: {take_profit}\n"
                            f"💰 Size: ${usd_size} ({risk_pct*100:.0f}% — score {total:+d})\n"
                            f"Confianza: {int(analysis['confidence']*100)}%\n"
                            f"F&G: {fg_value} — {analysis['reasoning']}"
                        )
                    else:
                        order, exec_price = execute_trade(trade_ex, symbol, "BUY", usd_size)
                        if order:
                            actual_price = exec_price or current_price
                            save_trade({**base_record, "paper": False, "order_id": order.get("id"), "price": actual_price})
                            open_position(symbol, actual_price, usd_size, risk_pct, "BUY")
                            send_telegram(
                                f"✅ <b>COMPRA ejecutada</b>\n"
                                f"Par: <b>{symbol}</b> @ {actual_price} USDT\n"
                                f"🔴 Trail Stop: {round(actual_price*(1-TRAILING_STOP_PCT),4)}\n"
                                f"🎯 Take Profit: {round(actual_price*(1+TAKE_PROFIT_PCT),4)}\n"
                                f"💰 Size: ${usd_size} ({risk_pct*100:.0f}%)"
                            )

                elif analysis["action"] == "SELL" and analysis["confidence"] >= 0.6:
                    if PAPER_TRADING:
                        save_trade({**base_record, "paper": True, "order_id": None})
                        log.info(f"  📝 PAPER SELL @ {current_price}")
                        send_telegram(f"📝 <b>PAPER SELL</b>\nPar: <b>{symbol}</b> @ {current_price}\nConfianza: {int(analysis['confidence']*100)}%")
                    else:
                        order, exec_price = execute_trade(trade_ex, symbol, "SELL", usd_size)
                        if order:
                            save_trade({**base_record, "paper": False, "order_id": order.get("id")})
                else:
                    log.info("  🚫 HOLD")

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
