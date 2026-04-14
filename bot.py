"""
Crypto Trading Bot — Binance Testnet
Estrategia: Multi-señal momentum (EMA crossover + RSI + Sentimiento noticias)
Dashboard web integrado en thread separado.
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

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
BINANCE_API_KEY    = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN     = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

CAPITAL_TOTAL_USD  = float(os.environ.get("CAPITAL_USD", "100"))
RISK_PER_TRADE     = 0.03
STOP_LOSS_PCT      = 0.02
TAKE_PROFIT_PCT    = 0.04
MIN_SIGNALS        = 2
LOOP_INTERVAL_SEC  = 300

WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TRADE_LOG_FILE = "trade_log.json"

# ─────────────────────────────────────────
# DASHBOARD HTML
# ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CryptoBot Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080c10;
    --surface: #0d1117;
    --border: #1a2332;
    --green: #00ff88;
    --red: #ff3355;
    --yellow: #ffcc00;
    --blue: #00aaff;
    --muted: #3d5166;
    --text: #c9d8e8;
    --mono: 'Share Tech Mono', monospace;
    --sans: 'Syne', sans-serif;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    min-height: 100vh;
    overflow-x: hidden;
  }
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(0,255,136,0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,255,136,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }
  .container { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 40px 24px; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 48px; padding-bottom: 24px; border-bottom: 1px solid var(--border);
  }
  .logo { font-family: var(--sans); font-weight: 800; font-size: 1.4rem; letter-spacing: -0.02em; color: #fff; }
  .logo span { color: var(--green); }
  .status-pill {
    display: flex; align-items: center; gap: 8px; font-size: 0.75rem;
    color: var(--green); border: 1px solid rgba(0,255,136,0.3); padding: 6px 14px; border-radius: 100px;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 40px; }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
    padding: 24px; position: relative; overflow: hidden; transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: var(--muted); }
  .stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--accent, var(--green)); opacity: 0.6;
  }
  .stat-label { font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
  .stat-value { font-family: var(--sans); font-weight: 800; font-size: 2rem; line-height: 1; color: #fff; }
  .stat-value.green { color: var(--green); }
  .stat-value.red { color: var(--red); }
  .stat-value.yellow { color: var(--yellow); }
  .stat-sub { font-size: 0.7rem; color: var(--muted); margin-top: 8px; }
  .section-title { font-family: var(--sans); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }
  .table-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; margin-bottom: 40px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  thead tr { border-bottom: 1px solid var(--border); }
  th { padding: 14px 20px; text-align: left; font-size: 0.65rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 400; }
  td { padding: 14px 20px; border-bottom: 1px solid rgba(26,35,50,0.5); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; }
  .badge-buy { background: rgba(0,255,136,0.12); color: var(--green); border: 1px solid rgba(0,255,136,0.2); }
  .badge-sell { background: rgba(255,51,85,0.12); color: var(--red); border: 1px solid rgba(255,51,85,0.2); }
  .confidence-bar { display: flex; align-items: center; gap: 8px; }
  .bar-bg { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .bar-fill { height: 100%; background: var(--green); border-radius: 2px; }
  .pair { color: #fff; font-weight: 600; }
  .ts { color: var(--muted); font-size: 0.72rem; }
  .empty { text-align: center; padding: 60px 20px; color: var(--muted); }
  .empty-icon { font-size: 2rem; margin-bottom: 12px; }
  .empty-text { font-size: 0.85rem; line-height: 1.6; }
  footer { text-align: center; font-size: 0.7rem; color: var(--muted); padding-top: 24px; border-top: 1px solid var(--border); }
  .refresh-info { font-size: 0.7rem; color: var(--muted); text-align: right; margin-bottom: 12px; }
  #countdown { color: var(--green); }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo">Crypto<span>Bot</span></div>
    <div class="status-pill"><div class="dot"></div>TESTNET ACTIVO</div>
  </header>
  <div class="stats">
    <div class="stat-card" style="--accent:var(--blue)">
      <div class="stat-label">Total Trades</div>
      <div class="stat-value" id="total">—</div>
      <div class="stat-sub">operaciones ejecutadas</div>
    </div>
    <div class="stat-card" style="--accent:var(--green)">
      <div class="stat-label">Compras</div>
      <div class="stat-value green" id="buys">—</div>
      <div class="stat-sub">órdenes de compra</div>
    </div>
    <div class="stat-card" style="--accent:var(--red)">
      <div class="stat-label">Ventas</div>
      <div class="stat-value red" id="sells">—</div>
      <div class="stat-sub">órdenes de venta</div>
    </div>
    <div class="stat-card" style="--accent:var(--yellow)">
      <div class="stat-label">Confianza promedio</div>
      <div class="stat-value yellow" id="avg-conf">—</div>
      <div class="stat-sub">score de Claude</div>
    </div>
  </div>
  <div class="refresh-info">Auto-refresh en <span id="countdown">30</span>s</div>
  <div class="section-title">Historial de operaciones</div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Par</th><th>Acción</th><th>Confianza</th><th>Señales</th><th>Razonamiento</th><th>Timestamp</th>
        </tr>
      </thead>
      <tbody id="trades-body">
        <tr><td colspan="6"><div class="empty"><div class="empty-icon">⏳</div><div class="empty-text">Cargando...</div></div></td></tr>
      </tbody>
    </table>
  </div>
  <footer>CryptoBot Dashboard · Binance Testnet · Datos en tiempo real</footer>
</div>
<script>
let countdown = 30;
async function loadData() {
  try {
    const res = await fetch('/api/trades');
    const data = await res.json();
    const trades = data.trades || [];
    document.getElementById('total').textContent = trades.length;
    const buys  = trades.filter(t => t.action === 'BUY').length;
    const sells = trades.filter(t => t.action === 'SELL').length;
    document.getElementById('buys').textContent  = buys;
    document.getElementById('sells').textContent = sells;
    const avgConf = trades.length
      ? (trades.reduce((s,t) => s + (t.confidence||0), 0) / trades.length * 100).toFixed(0) + '%'
      : '—';
    document.getElementById('avg-conf').textContent = avgConf;
    const tbody = document.getElementById('trades-body');
    if (trades.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6"><div class="empty"><div class="empty-icon">🤖</div><div class="empty-text">El bot está analizando señales...<br>Las operaciones aparecerán aquí cuando se ejecuten.</div></div></td></tr>`;
      return;
    }
    const sorted = [...trades].reverse();
    tbody.innerHTML = sorted.map(t => {
      const ts = new Date(t.timestamp).toLocaleString('es-AR', {day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'});
      const conf = Math.round((t.confidence||0)*100);
      const tSig = t.tech_signal===1?'📈 BULL':t.tech_signal===-1?'📉 BEAR':'—';
      const nSig = t.news_signal===1?'📰+':t.news_signal===-1?'📰-':'—';
      return `<tr>
        <td class="pair">${t.symbol||'—'}</td>
        <td><span class="badge badge-${(t.action||'').toLowerCase()}">${t.action||'—'}</span></td>
        <td><div class="confidence-bar"><div class="bar-bg"><div class="bar-fill" style="width:${conf}%"></div></div><span style="font-size:.72rem;color:var(--text);min-width:32px">${conf}%</span></div></td>
        <td style="color:var(--muted)">${tSig} ${nSig}</td>
        <td style="color:var(--muted);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${t.reasoning||''}">${t.reasoning||'—'}</td>
        <td class="ts">${ts}</td>
      </tr>`;
    }).join('');
  } catch(e) { console.error(e); }
}
function tick() {
  countdown--;
  document.getElementById('countdown').textContent = countdown;
  if (countdown <= 0) { countdown = 30; loadData(); }
}
loadData();
setInterval(tick, 1000);
</script>
</body>
</html>"""

# ─────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@flask_app.route("/api/trades")
def api_trades():
    trades = []
    if os.path.exists(TRADE_LOG_FILE):
        try:
            with open(TRADE_LOG_FILE, "r") as f:
                trades = json.load(f)
        except Exception:
            trades = []
    return jsonify({"trades": trades, "count": len(trades)})

@flask_app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})

def run_dashboard():
    port = int(os.environ.get("PORT", 8080))
    log.info(f"🌐 Dashboard en puerto {port}")
    flask_app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ─────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
        log.info("📱 Telegram enviado")
    except Exception as e:
        log.warning(f"Telegram error: {e}")


# ─────────────────────────────────────────
# EXCHANGE
# ─────────────────────────────────────────
def get_exchange() -> ccxt.binance:
    return ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
        "urls": {"api": {
            "public":  "https://testnet.binance.vision/api/v3",
            "private": "https://testnet.binance.vision/api/v3",
        }}
    })


# ─────────────────────────────────────────
# INDICADORES TÉCNICOS
# ─────────────────────────────────────────
def get_ohlcv(exchange, symbol, timeframe="1h", limit=100):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def calculate_indicators(df):
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def technical_signal(df):
    last = df.iloc[-1]; prev = df.iloc[-2]
    bull = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
    bear = prev["ema9"] >= prev["ema21"] and last["ema9"] < last["ema21"]
    rsi  = last["rsi"]
    if bull and rsi < 65: return +1
    if bear and rsi > 35: return -1
    return 0


# ─────────────────────────────────────────
# SENTIMIENTO DE NOTICIAS
# ─────────────────────────────────────────
BULLISH_KW = ["rally","surge","breakout","bullish","adoption","partnership","upgrade","all-time high","ath","gains","rises","jumps","soars","recovery"]
BEARISH_KW = ["crash","hack","ban","bearish","lawsuit","regulation","sell-off","collapse","fear","drops","falls","plunges","warning","risk"]
RSS_FEEDS  = ["https://www.coindesk.com/arc/outboundfeeds/rss/","https://cointelegraph.com/rss"]
COIN_NAMES = {"btc":["bitcoin","btc"],"eth":["ethereum","eth"],"sol":["solana","sol"],"bnb":["bnb","binance"]}

def get_news_sentiment(symbol):
    coin = symbol.split("/")[0].lower()
    terms = COIN_NAMES.get(coin, [coin])
    all_titles = []
    for url in RSS_FEEDS:
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
            if not titles: titles = re.findall(r"<title>(.*?)</title>", resp.text)
            all_titles.extend(titles[:15])
        except Exception as e:
            log.warning(f"RSS error: {e}")
    if not all_titles: return 0
    relevant = [t.lower() for t in all_titles if any(term in t.lower() for term in terms)]
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
def ask_claude(symbol, t_sig, n_sig, df):
    last = df.iloc[-1]
    direction = "BULLISH" if t_sig + n_sig > 0 else "BEARISH"
    prompt = f"""Sos un analista de trading crypto. Respondé SOLO en JSON.
Par: {symbol} | Precio: {last['close']:.4f} USDT
EMA9: {last['ema9']:.4f} | EMA21: {last['ema21']:.4f} | RSI: {last['rsi']:.1f}
Señal técnica: {"COMPRA" if t_sig==1 else "VENTA" if t_sig==-1 else "NEUTRAL"}
Señal noticias: {"POSITIVA" if n_sig==1 else "NEGATIVA" if n_sig==-1 else "NEUTRAL"}
Dirección: {direction}
Respondé SOLO con este JSON sin backticks:
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
# EJECUCIÓN DE ÓRDENES
# ─────────────────────────────────────────
def execute_trade(exchange, symbol, action, capital):
    try:
        price    = exchange.fetch_ticker(symbol)["last"]
        usd_size = round(capital * RISK_PER_TRADE, 2)
        qty      = usd_size / price
        if action == "BUY":
            order = exchange.create_market_buy_order(symbol, qty)
            sl = round(price*(1-STOP_LOSS_PCT),4); tp = round(price*(1+TAKE_PROFIT_PCT),4)
            log.info(f"✅ COMPRA {symbol} | precio={price} | SL={sl} | TP={tp}")
            send_telegram(f"✅ <b>COMPRA ejecutada</b>\nPar: <b>{symbol}</b>\nPrecio: <b>{price} USDT</b>\nSL: {sl} | TP: {tp}\nCapital: ${usd_size}")
        elif action == "SELL":
            order = exchange.create_market_sell_order(symbol, qty)
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
        with open(TRADE_LOG_FILE,"r") as f: data = json.load(f)
    data.append(record)
    with open(TRADE_LOG_FILE,"w") as f: json.dump(data, f, indent=2, default=str)


# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run_bot():
    log.info("🤖 Bot iniciado — Binance Testnet")
    send_telegram(f"🤖 <b>Bot iniciado</b>\nCapital: ${CAPITAL_TOTAL_USD}\nPares: {', '.join(WATCHLIST)}\nCiclo: cada {LOOP_INTERVAL_SEC//60} min")
    exchange = get_exchange()

    while True:
        log.info(f"\n{'='*50}\n⏰ Ciclo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for symbol in WATCHLIST:
            try:
                log.info(f"\n📊 {symbol}...")
                df    = calculate_indicators(get_ohlcv(exchange, symbol))
                t_sig = technical_signal(df)
                n_sig = get_news_sentiment(symbol)
                log.info(f"  Técnica: {'+1'if t_sig==1 else'-1'if t_sig==-1 else'0'} | Noticias: {'+1'if n_sig==1 else'-1'if n_sig==-1 else'0'}")

                if abs(t_sig + n_sig) < MIN_SIGNALS:
                    log.info(f"  ⏭️  Señales insuficientes — skip"); continue

                send_telegram(f"🧠 <b>Señales alineadas: {symbol}</b>\nConsultando Claude...")
                analysis = ask_claude(symbol, t_sig, n_sig, df)
                log.info(f"  Claude: {analysis['action']} ({analysis['confidence']}) — {analysis['reasoning']}")

                if analysis["action"] in ("BUY","SELL") and analysis["confidence"] >= 0.6:
                    order = execute_trade(exchange, symbol, analysis["action"], CAPITAL_TOTAL_USD)
                    if order:
                        save_trade({"timestamp":datetime.now().isoformat(),"symbol":symbol,
                            "action":analysis["action"],"confidence":analysis["confidence"],
                            "reasoning":analysis["reasoning"],"tech_signal":t_sig,
                            "news_signal":n_sig,"order_id":order.get("id")})
                else:
                    log.info("  🚫 HOLD")
                    send_telegram(f"🚫 <b>HOLD {symbol}</b>\n{analysis['reasoning']}")
            except Exception as e:
                log.error(f"Error {symbol}: {e}")

        log.info(f"\n💤 Esperando {LOOP_INTERVAL_SEC}s...")
        time.sleep(LOOP_INTERVAL_SEC)


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Dashboard en thread separado
    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()
    # Bot en main thread
    run_bot()
