"""
Crypto Trading Bot v2
Señales: EMA crossover + MACD + Volume + Timeframe 1h/4h + Fear&Greed + Noticias
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

BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN     = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
CAPITAL_TOTAL_USD  = float(os.environ.get("CAPITAL_USD", "100"))

RISK_PER_TRADE    = 0.03
STOP_LOSS_PCT     = 0.02
TAKE_PROFIT_PCT   = 0.04
MIN_SIGNALS       = 2
LOOP_INTERVAL_SEC = 300
WATCHLIST         = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TRADE_LOG_FILE    = "trade_log.json"

# ─────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CryptoBot v2 Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root{--bg:#080c10;--surface:#0d1117;--border:#1a2332;--green:#00ff88;--red:#ff3355;--yellow:#ffcc00;--blue:#00aaff;--muted:#3d5166;--text:#c9d8e8;--mono:'Share Tech Mono',monospace;--sans:'Syne',sans-serif}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);min-height:100vh}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,255,136,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,255,136,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
  .container{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:40px 24px}
  header{display:flex;align-items:center;justify-content:space-between;margin-bottom:48px;padding-bottom:24px;border-bottom:1px solid var(--border)}
  .logo{font-family:var(--sans);font-weight:800;font-size:1.4rem;color:#fff}.logo span{color:var(--green)}
  .logo small{font-size:.7rem;color:var(--muted);font-weight:400;margin-left:8px}
  .status-pill{display:flex;align-items:center;gap:8px;font-size:.75rem;color:var(--green);border:1px solid rgba(0,255,136,.3);padding:6px 14px;border-radius:100px}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
  .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:40px}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;position:relative;overflow:hidden}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,var(--green));opacity:.6}
  .stat-label{font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
  .stat-value{font-family:var(--sans);font-weight:800;font-size:1.8rem;line-height:1;color:#fff}
  .stat-value.green{color:var(--green)}.stat-value.red{color:var(--red)}.stat-value.yellow{color:var(--yellow)}.stat-value.blue{color:var(--blue)}
  .stat-sub{font-size:.65rem;color:var(--muted);margin-top:6px}
  .section-title{font-family:var(--sans);font-size:.75rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--muted);margin-bottom:16px}
  .table-wrap{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:40px}
  table{width:100%;border-collapse:collapse;font-size:.78rem}
  thead tr{border-bottom:1px solid var(--border)}
  th{padding:12px 16px;text-align:left;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);font-weight:400}
  td{padding:12px 16px;border-bottom:1px solid rgba(26,35,50,.5);vertical-align:middle}
  tr:last-child td{border-bottom:none}tr:hover td{background:rgba(255,255,255,.02)}
  .badge{display:inline-block;padding:3px 10px;border-radius:4px;font-size:.68rem;font-weight:700}
  .badge-buy{background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.2)}
  .badge-sell{background:rgba(255,51,85,.12);color:var(--red);border:1px solid rgba(255,51,85,.2)}
  .confidence-bar{display:flex;align-items:center;gap:8px}
  .bar-bg{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
  .bar-fill{height:100%;background:var(--green);border-radius:2px}
  .pair{color:#fff;font-weight:600}.ts{color:var(--muted);font-size:.7rem}
  .empty{text-align:center;padding:60px 20px;color:var(--muted)}
  .empty-icon{font-size:2rem;margin-bottom:12px}.empty-text{font-size:.85rem;line-height:1.6}
  footer{text-align:center;font-size:.7rem;color:var(--muted);padding-top:24px;border-top:1px solid var(--border)}
  .refresh-info{font-size:.7rem;color:var(--muted);text-align:right;margin-bottom:12px}
  #countdown{color:var(--green)}
  .fg-pill{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.65rem;font-weight:700}
  .fg-fear{background:rgba(255,51,85,.15);color:var(--red)}
  .fg-greed{background:rgba(0,255,136,.15);color:var(--green)}
  .fg-neutral{background:rgba(255,204,0,.15);color:var(--yellow)}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">Crypto<span>Bot</span><small>v2</small></div>
    <div class="status-pill"><div class="dot"></div>LIVE</div>
  </header>
  <div class="stats">
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Total Trades</div><div class="stat-value" id="total">—</div><div class="stat-sub">operaciones</div></div>
    <div class="stat-card" style="--accent:var(--green)"><div class="stat-label">Compras</div><div class="stat-value green" id="buys">—</div><div class="stat-sub">BUY</div></div>
    <div class="stat-card" style="--accent:var(--red)"><div class="stat-label">Ventas</div><div class="stat-value red" id="sells">—</div><div class="stat-sub">SELL</div></div>
    <div class="stat-card" style="--accent:var(--yellow)"><div class="stat-label">Confianza</div><div class="stat-value yellow" id="avg-conf">—</div><div class="stat-sub">promedio Claude</div></div>
    <div class="stat-card" style="--accent:var(--blue)"><div class="stat-label">Fear & Greed</div><div class="stat-value blue" id="fg-value">—</div><div class="stat-sub" id="fg-label">cargando...</div></div>
  </div>
  <div class="refresh-info">Auto-refresh en <span id="countdown">30</span>s</div>
  <div class="section-title">Historial de operaciones</div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Par</th><th>Acción</th><th>Confianza</th><th>Señales</th><th>F&G</th><th>Razonamiento</th><th>Timestamp</th></tr></thead>
      <tbody id="trades-body"><tr><td colspan="7"><div class="empty"><div class="empty-icon">⏳</div><div class="empty-text">Cargando...</div></div></td></tr></tbody>
    </table>
  </div>
  <footer>CryptoBot v2 · EMA + MACD + Volume + 1h/4h + Fear&Greed + Noticias</footer>
</div>
<script>
let countdown=30;
async function loadData(){
  try{
    const res=await fetch('/api/trades');
    const data=await res.json();
    const trades=data.trades||[];
    document.getElementById('total').textContent=trades.length;
    document.getElementById('buys').textContent=trades.filter(t=>t.action==='BUY').length;
    document.getElementById('sells').textContent=trades.filter(t=>t.action==='SELL').length;
    const avgConf=trades.length?(trades.reduce((s,t)=>s+(t.confidence||0),0)/trades.length*100).toFixed(0)+'%':'—';
    document.getElementById('avg-conf').textContent=avgConf;
    if(data.fear_greed){
      const fg=data.fear_greed;
      document.getElementById('fg-value').textContent=fg.value;
      const lbl=document.getElementById('fg-label');
      lbl.textContent=fg.label;
      const el=document.getElementById('fg-value');
      el.className='stat-value blue';
      if(fg.value<35)el.className='stat-value red';
      else if(fg.value>65)el.className='stat-value green';
    }
    const tbody=document.getElementById('trades-body');
    if(!trades.length){tbody.innerHTML='<tr><td colspan="7"><div class="empty"><div class="empty-icon">🤖</div><div class="empty-text">El bot está analizando señales...<br>Las operaciones aparecerán aquí cuando se ejecuten.</div></div></td></tr>';return;}
    tbody.innerHTML=[...trades].reverse().map(t=>{
      const ts=new Date(t.timestamp).toLocaleString('es-AR',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const conf=Math.round((t.confidence||0)*100);
      const signals=[];
      if(t.tech_signal===1)signals.push('📈EMA');else if(t.tech_signal===-1)signals.push('📉EMA');
      if(t.macd_signal===1)signals.push('📊MACD+');else if(t.macd_signal===-1)signals.push('📊MACD-');
      if(t.vol_signal===1)signals.push('📦VOL+');
      if(t.tf4h_signal===1)signals.push('⏱4H+');else if(t.tf4h_signal===-1)signals.push('⏱4H-');
      if(t.news_signal===1)signals.push('📰+');else if(t.news_signal===-1)signals.push('📰-');
      const fgVal=t.fear_greed_value||'—';
      const fgCls=fgVal<35?'fg-fear':fgVal>65?'fg-greed':'fg-neutral';
      return `<tr>
        <td class="pair">${t.symbol||'—'}</td>
        <td><span class="badge badge-${(t.action||'').toLowerCase()}">${t.action||'—'}</span></td>
        <td><div class="confidence-bar"><div class="bar-bg"><div class="bar-fill" style="width:${conf}%"></div></div><span style="font-size:.7rem;min-width:32px">${conf}%</span></div></td>
        <td style="color:var(--muted);font-size:.7rem">${signals.join(' ')}</td>
        <td><span class="fg-pill ${fgCls}">${fgVal}</span></td>
        <td style="color:var(--muted);max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${t.reasoning||''}">${t.reasoning||'—'}</td>
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
    return jsonify({"trades": trades, "count": len(trades), "fear_greed": _fear_greed})

@flask_app.route("/health")
def health(): return jsonify({"status": "ok"})

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
# INDICADORES
# ─────────────────────────────────────────
def get_ohlcv(exchange, symbol, timeframe="1h", limit=150):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def calculate_indicators(df):
    df = df.copy()
    # EMA
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean() / loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)))
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    # Volume
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    return df

def technical_signal(df):
    """EMA crossover + RSI confirmación."""
    last = df.iloc[-1]; prev = df.iloc[-2]
    bull = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
    bear = prev["ema9"] >= prev["ema21"] and last["ema9"] < last["ema21"]
    rsi  = last["rsi"]
    if bull and rsi < 65: return +1
    if bear and rsi > 35: return -1
    return 0

def macd_signal(df):
    """MACD histogram cambia de negativo a positivo (bull) o viceversa (bear)."""
    last = df.iloc[-1]; prev = df.iloc[-2]
    if prev["macd_hist"] < 0 and last["macd_hist"] > 0: return +1
    if prev["macd_hist"] > 0 and last["macd_hist"] < 0: return -1
    # Sin cruce — pero histograma creciendo/cayendo consistentemente
    if last["macd_hist"] > 0 and last["macd_hist"] > prev["macd_hist"]: return +1
    if last["macd_hist"] < 0 and last["macd_hist"] < prev["macd_hist"]: return -1
    return 0

def volume_signal(df):
    """Volumen actual > promedio 20 velas = confirma el movimiento."""
    last = df.iloc[-1]
    if pd.isna(last["vol_ma20"]): return 0
    return +1 if last["volume"] > last["vol_ma20"] * 1.2 else 0

def timeframe_4h_signal(exchange, symbol):
    """Señal técnica en timeframe 4h — confirmación de tendencia mayor."""
    try:
        df4h = calculate_indicators(get_ohlcv(exchange, symbol, timeframe="4h", limit=50))
        return technical_signal(df4h)
    except Exception as e:
        log.warning(f"4h signal error {symbol}: {e}")
        return 0

# ─────────────────────────────────────────
# FEAR & GREED INDEX
# ─────────────────────────────────────────
def get_fear_greed():
    """API pública de alternative.me — sin key."""
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
    """
    Filtra operaciones según el sentimiento del mercado:
    - Extreme Fear (<25): no comprar
    - Extreme Greed (>75): no vender
    """
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
def ask_claude(symbol, signals_dict, df, fg_value):
    last = df.iloc[-1]
    total = sum(signals_dict.values())
    direction = "BULLISH" if total > 0 else "BEARISH"
    prompt = f"""Sos un analista de trading crypto experto. Analizá y respondé SOLO en JSON sin backticks.

Par: {symbol} | Precio: {last['close']:.4f} USDT
EMA9: {last['ema9']:.4f} | EMA21: {last['ema21']:.4f} | RSI: {last['rsi']:.1f}
MACD: {last['macd']:.4f} | MACD Signal: {last['macd_signal']:.4f} | Hist: {last['macd_hist']:.4f}
Volumen actual vs MA20: {last['volume']:.0f} vs {last['vol_ma20']:.0f}
Fear & Greed Index: {fg_value}

Señales:
- EMA crossover: {'+1' if signals_dict.get('tech')==1 else '-1' if signals_dict.get('tech')==-1 else '0'}
- MACD: {'+1' if signals_dict.get('macd')==1 else '-1' if signals_dict.get('macd')==-1 else '0'}
- Volumen: {'+1' if signals_dict.get('vol')==1 else '0'}
- Timeframe 4h: {'+1' if signals_dict.get('tf4h')==1 else '-1' if signals_dict.get('tf4h')==-1 else '0'}
- Noticias: {'+1' if signals_dict.get('news')==1 else '-1' if signals_dict.get('news')==-1 else '0'}
Dirección total: {direction} (score: {total})

{{"action":"BUY"|"SELL"|"HOLD","confidence":0.0,"reasoning":"una línea concisa"}}"""

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
def execute_trade(trade_exchange, symbol, action, capital):
    try:
        price    = trade_exchange.fetch_ticker(symbol)["last"]
        usd_size = round(capital * RISK_PER_TRADE, 2)
        qty      = usd_size / price
        if action == "BUY":
            order = trade_exchange.create_market_buy_order(symbol, qty)
            sl = round(price*(1-STOP_LOSS_PCT),4); tp = round(price*(1+TAKE_PROFIT_PCT),4)
            log.info(f"✅ COMPRA {symbol} | precio={price} | SL={sl} | TP={tp}")
            send_telegram(f"✅ <b>COMPRA ejecutada</b>\nPar: <b>{symbol}</b>\nPrecio: <b>{price} USDT</b>\nSL: {sl} | TP: {tp}\nCapital: ${usd_size}")
        elif action == "SELL":
            order = trade_exchange.create_market_sell_order(symbol, qty)
            log.info(f"✅ VENTA {symbol} | precio={price}")
            send_telegram(f"🔴 <b>VENTA ejecutada</b>\nPar: <b>{symbol}</b>\nPrecio: <b>{price} USDT</b>")
        return order
    except Exception as e:
        log.error(f"Error orden {action} {symbol}: {e}")
        send_telegram(f"⚠️ Error en orden {action} — {symbol}\n{e}")
        return None

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
    log.info("🤖 CryptoBot v2 iniciado")
    log.info(f"Señales: EMA + MACD + Volume + 4h + Fear&Greed + Noticias")
    log.info(f"Capital: ${CAPITAL_TOTAL_USD} | Watchlist: {WATCHLIST}")
    send_telegram(
        f"🤖 <b>CryptoBot v2 iniciado</b>\n"
        f"Capital: ${CAPITAL_TOTAL_USD}\n"
        f"Señales: EMA + MACD + Volume + 4h + F&G + Noticias\n"
        f"Pares: {', '.join(WATCHLIST)}"
    )

    public_ex = get_public_exchange()
    trade_ex  = get_trade_exchange()

    while True:
        log.info(f"\n{'='*50}\n⏰ Ciclo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Fear & Greed — se obtiene una vez por ciclo
        fg_value, fg_label = get_fear_greed()
        log.info(f"🧭 Fear & Greed: {fg_value} — {fg_label}")

        for symbol in WATCHLIST:
            try:
                log.info(f"\n📊 {symbol}...")

                # Datos e indicadores
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

                # Consultar Claude
                log.info("  🧠 Consultando Claude...")
                send_telegram(f"🧠 <b>Señales alineadas: {symbol}</b> (score: {total:+d})\nF&G: {fg_value} {fg_label}")
                analysis = ask_claude(symbol, signals, df, fg_value)
                log.info(f"  Claude: {analysis['action']} ({analysis['confidence']}) — {analysis['reasoning']}")

                # Filtro Fear & Greed
                if not fear_greed_filter(fg_value, analysis["action"]):
                    continue

                if analysis["action"] in ("BUY","SELL") and analysis["confidence"] >= 0.6:
                    order = execute_trade(trade_ex, symbol, analysis["action"], CAPITAL_TOTAL_USD)
                    if order:
                        save_trade({
                            "timestamp":      datetime.now().isoformat(),
                            "symbol":         symbol,
                            "action":         analysis["action"],
                            "confidence":     analysis["confidence"],
                            "reasoning":      analysis["reasoning"],
                            "tech_signal":    t_sig,
                            "macd_signal":    m_sig,
                            "vol_signal":     v_sig,
                            "tf4h_signal":    h_sig,
                            "news_signal":    n_sig,
                            "fear_greed_value": fg_value,
                            "fear_greed_label": fg_label,
                            "order_id":       order.get("id")
                        })
                else:
                    log.info("  🚫 HOLD")
                    send_telegram(f"🚫 <b>HOLD {symbol}</b>\n{analysis['reasoning']}\nF&G: {fg_value}")

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
