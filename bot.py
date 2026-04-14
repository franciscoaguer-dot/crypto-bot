"""
Crypto Trading Bot — Binance Testnet
Estrategia: Multi-señal momentum (EMA crossover + RSI + Sentimiento noticias)
El bot opera solo cuando 2+ señales coinciden en la misma dirección.
"""

import os
import re
import time
import json
import logging
import requests
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np

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
RISK_PER_TRADE     = 0.03        # 3% del capital por operación
STOP_LOSS_PCT      = 0.02        # 2% stop loss
TAKE_PROFIT_PCT    = 0.04        # 4% take profit (ratio 1:2)
MIN_SIGNALS        = 2           # mínimo de señales para operar
LOOP_INTERVAL_SEC  = 300         # cada 5 minutos

WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# ─────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────
def send_telegram(msg: str):
    """Envía mensaje a Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }, timeout=10)
        log.info("📱 Telegram enviado")
    except Exception as e:
        log.warning(f"Telegram error: {e}")


# ─────────────────────────────────────────
# EXCHANGE
# ─────────────────────────────────────────
def get_exchange() -> ccxt.binance:
    exchange = ccxt.binance({
        "apiKey": BINANCE_API_KEY,
        "secret": BINANCE_API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
        "urls": {
            "api": {
                "public":  "https://testnet.binance.vision/api/v3",
                "private": "https://testnet.binance.vision/api/v3",
            }
        }
    })
    return exchange


# ─────────────────────────────────────────
# INDICADORES TÉCNICOS
# ─────────────────────────────────────────
def get_ohlcv(exchange: ccxt.binance, symbol: str, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def technical_signal(df: pd.DataFrame) -> int:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    ema_cross_bull = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
    ema_cross_bear = prev["ema9"] >= prev["ema21"] and last["ema9"] < last["ema21"]
    rsi = last["rsi"]
    if ema_cross_bull and rsi < 65:
        return +1
    if ema_cross_bear and rsi > 35:
        return -1
    return 0


# ─────────────────────────────────────────
# SENTIMIENTO DE NOTICIAS (RSS gratuito)
# ─────────────────────────────────────────
BULLISH_KEYWORDS = ["rally", "surge", "breakout", "bullish", "adoption", "partnership", "upgrade", "all-time high", "ath", "gains", "rises", "jumps", "soars", "recovery"]
BEARISH_KEYWORDS = ["crash", "hack", "ban", "bearish", "lawsuit", "regulation", "sell-off", "collapse", "fear", "drops", "falls", "plunges", "warning", "risk"]

RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
]

COIN_NAMES = {
    "btc": ["bitcoin", "btc"],
    "eth": ["ethereum", "eth"],
    "sol": ["solana", "sol"],
    "bnb": ["bnb", "binance"],
}

def get_news_sentiment(symbol: str) -> int:
    coin = symbol.split("/")[0].lower()
    search_terms = COIN_NAMES.get(coin, [coin])
    all_titles = []
    for feed_url in RSS_FEEDS:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(feed_url, timeout=10, headers=headers)
            titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", resp.text)
            if not titles:
                titles = re.findall(r"<title>(.*?)</title>", resp.text)
            all_titles.extend(titles[:15])
        except Exception as e:
            log.warning(f"RSS feed error ({feed_url}): {e}")
            continue

    if not all_titles:
        return 0

    relevant = [t.lower() for t in all_titles if any(term in t.lower() for term in search_terms)]
    if not relevant:
        relevant = [t.lower() for t in all_titles]

    titles_text = " ".join(relevant)
    bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in titles_text)
    bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in titles_text)
    log.info(f"  Noticias relevantes: {len(relevant)} | bull={bull_count} bear={bear_count}")

    if bull_count > bear_count:
        return +1
    if bear_count > bull_count:
        return -1
    return 0


# ─────────────────────────────────────────
# ANÁLISIS CON CLAUDE
# ─────────────────────────────────────────
def ask_claude(symbol: str, tech_signal: int, news_signal: int, df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    direction = "BULLISH" if tech_signal + news_signal > 0 else "BEARISH"

    prompt = f"""Sos un analista de trading crypto. Analizá esta situación y respondé SOLO en JSON.

Par: {symbol}
Precio actual: {last['close']:.4f} USDT
EMA9: {last['ema9']:.4f} | EMA21: {last['ema21']:.4f}
RSI: {last['rsi']:.1f}
Señal técnica: {"COMPRA" if tech_signal == 1 else "VENTA" if tech_signal == -1 else "NEUTRAL"}
Señal noticias: {"POSITIVA" if news_signal == 1 else "NEGATIVA" if news_signal == -1 else "NEUTRAL"}
Dirección combinada: {direction}

Respondé ÚNICAMENTE con este JSON (sin texto extra, sin backticks):
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 a 1.0,
  "reasoning": "una línea explicando la decisión"
}}"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=body, timeout=15)
        text = resp.json()["content"][0]["text"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        log.error(f"Claude API error: {e}")
        return {"action": "HOLD", "confidence": 0, "reasoning": "API error"}


# ─────────────────────────────────────────
# GESTIÓN DE POSICIONES
# ─────────────────────────────────────────
def get_position_size(capital: float) -> float:
    return round(capital * RISK_PER_TRADE, 2)


def execute_trade(exchange: ccxt.binance, symbol: str, action: str, capital: float):
    try:
        ticker   = exchange.fetch_ticker(symbol)
        price    = ticker["last"]
        usd_size = get_position_size(capital)
        qty      = usd_size / price

        if action == "BUY":
            order    = exchange.create_market_buy_order(symbol, qty)
            sl_price = round(price * (1 - STOP_LOSS_PCT), 4)
            tp_price = round(price * (1 + TAKE_PROFIT_PCT), 4)
            log.info(f"✅ COMPRA: {symbol} | qty={qty:.6f} | precio={price} | SL={sl_price} | TP={tp_price}")
            send_telegram(
                f"✅ <b>COMPRA ejecutada</b>\n"
                f"Par: <b>{symbol}</b>\n"
                f"Precio: <b>{price} USDT</b>\n"
                f"Cantidad: {qty:.6f}\n"
                f"Stop Loss: {sl_price}\n"
                f"Take Profit: {tp_price}\n"
                f"Capital usado: ${usd_size}"
            )
        elif action == "SELL":
            order = exchange.create_market_sell_order(symbol, qty)
            log.info(f"✅ VENTA: {symbol} | qty={qty:.6f} | precio={price}")
            send_telegram(
                f"🔴 <b>VENTA ejecutada</b>\n"
                f"Par: <b>{symbol}</b>\n"
                f"Precio: <b>{price} USDT</b>\n"
                f"Cantidad: {qty:.6f}"
            )

        return order

    except Exception as e:
        log.error(f"Error ejecutando orden {action} en {symbol}: {e}")
        send_telegram(f"⚠️ <b>Error en orden {action}</b>\nPar: {symbol}\nError: {e}")
        return None


# ─────────────────────────────────────────
# MEMORIA / LOG DE OPERACIONES
# ─────────────────────────────────────────
TRADE_LOG_FILE = "trade_log.json"

def save_trade(record: dict):
    log_data = []
    if os.path.exists(TRADE_LOG_FILE):
        with open(TRADE_LOG_FILE, "r") as f:
            log_data = json.load(f)
    log_data.append(record)
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2, default=str)


# ─────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────
def run():
    log.info("🤖 Bot iniciado — Binance Testnet")
    log.info(f"Capital: ${CAPITAL_TOTAL_USD} | Riesgo/op: {RISK_PER_TRADE*100}% | Watchlist: {WATCHLIST}")

    send_telegram(
        f"🤖 <b>Bot iniciado</b>\n"
        f"Capital: ${CAPITAL_TOTAL_USD}\n"
        f"Pares: {', '.join(WATCHLIST)}\n"
        f"Ciclo: cada {LOOP_INTERVAL_SEC//60} minutos"
    )

    exchange = get_exchange()

    while True:
        log.info(f"\n{'='*50}")
        log.info(f"⏰ Ciclo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for symbol in WATCHLIST:
            try:
                log.info(f"\n📊 Analizando {symbol}...")
                df    = get_ohlcv(exchange, symbol)
                df    = calculate_indicators(df)
                t_sig = technical_signal(df)
                n_sig = get_news_sentiment(symbol)

                log.info(f"  Señal técnica : {'+1 (BULL)' if t_sig==1 else '-1 (BEAR)' if t_sig==-1 else '0 (neutral)'}")
                log.info(f"  Señal noticias: {'+1 (POS)' if n_sig==1 else '-1 (NEG)' if n_sig==-1 else '0 (neutral)'}")

                total_signals = t_sig + n_sig
                if abs(total_signals) < MIN_SIGNALS:
                    log.info(f"  ⏭️  Señales insuficientes ({total_signals}) — skip")
                    continue

                log.info(f"  🧠 Consultando Claude (señales alineadas: {total_signals})...")
                send_telegram(
                    f"🧠 <b>Señales alineadas en {symbol}</b>\n"
                    f"Técnica: {'+1' if t_sig==1 else '-1'} | Noticias: {'+1' if n_sig==1 else '-1'}\n"
                    f"Consultando Claude..."
                )

                analysis = ask_claude(symbol, t_sig, n_sig, df)
                log.info(f"  Claude: {analysis['action']} (confianza: {analysis['confidence']}) — {analysis['reasoning']}")

                if analysis["action"] in ("BUY", "SELL") and analysis["confidence"] >= 0.6:
                    order = execute_trade(exchange, symbol, analysis["action"], CAPITAL_TOTAL_USD)
                    if order:
                        save_trade({
                            "timestamp": datetime.now().isoformat(),
                            "symbol": symbol,
                            "action": analysis["action"],
                            "confidence": analysis["confidence"],
                            "reasoning": analysis["reasoning"],
                            "tech_signal": t_sig,
                            "news_signal": n_sig,
                            "order_id": order.get("id")
                        })
                else:
                    log.info(f"  🚫 Claude no confirmó — HOLD")
                    send_telegram(
                        f"🚫 <b>HOLD en {symbol}</b>\n"
                        f"Claude: {analysis['action']} (confianza: {analysis['confidence']})\n"
                        f"{analysis['reasoning']}"
                    )

            except Exception as e:
                log.error(f"Error procesando {symbol}: {e}")
                continue

        log.info(f"\n💤 Esperando {LOOP_INTERVAL_SEC}s hasta próximo ciclo...")
        time.sleep(LOOP_INTERVAL_SEC)


if __name__ == "__main__":
    run()
