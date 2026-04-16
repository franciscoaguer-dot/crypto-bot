"""
CryptoBot v10 — Backtester
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usa la MISMA lógica del bot en tiempo real pero sobre datos históricos.
Corre localmente o en Railway como one-shot job.

Uso:
    pip install ccxt pandas numpy tabulate
    python backtest.py

    # Con opciones:
    python backtest.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 180 --tf 1h
    python backtest.py --symbols BTC/USDT --days 90 --tf 3m
"""

import argparse
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import ccxt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS (mismos que bot.py)
# ─────────────────────────────────────────────────────────────────────────────
TRAILING_STOP_PCT    = 0.010   # 1% spot
FUTURES_TRAILING     = 0.008   # 0.8% futuros
TAKE_PROFIT_PCT      = 0.025   # 2.5% spot TP
FUTURES_TP           = 0.020   # 2% futuros TP
TAKE_PROFIT_PARTIAL  = 0.015   # 1.5% partial TP
PARTIAL_EXIT_PCT     = 0.50    # cerrar 50%
STOP_LOSS_PCT        = 0.03    # 3% stop loss absoluto
PROFIT_TRAIL_TRIGGER = 0.015
PROFIT_TRAIL_STEP    = 0.010
ATR_PERIOD           = 14
ATR_MULT_SPOT        = 1.5
ATR_MULT_FUT         = 1.0
BB_PERIOD            = 20
BB_STD               = 2.0
OB_IMBALANCE         = 0.60    # no disponible en backtest, se ignora
SIDEWAYS_MIN_FLOAT   = 2.0
MIN_SIGNALS          = 2
FUTURES_MIN_SCORE    = 4
COMMISSION           = 0.001   # 0.1% por side (Binance taker)
INITIAL_CAPITAL      = 1000.0
POSITION_SIZE_MAP    = {2: 0.02, 3: 0.03, 4: 0.04, 5: 0.05, 6: 0.06}

# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Descarga datos históricos de Binance (público, sin API key)."""
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    )
    print(f"  ↓ Descargando {symbol} [{timeframe}] últimos {days} días...")
    all_ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"    → {len(df)} velas ({df.iloc[0]['timestamp'].date()} → {df.iloc[-1]['timestamp'].date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICADORES (copiados de bot.py)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    df["rsi"] = 100 - (100 / (
        1 + gain.ewm(com=13, adjust=False).mean() /
        loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)
    ))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["vol_ma20"]    = df["volume"].rolling(20).mean()
    df["bb_mid"]      = df["close"].rolling(BB_PERIOD).mean()
    bb_std            = df["close"].rolling(BB_PERIOD).std()
    df["bb_upper"]    = df["bb_mid"] + BB_STD * bb_std
    df["bb_lower"]    = df["bb_mid"] - BB_STD * bb_std
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SEÑALES (copiadas de bot.py — sin OB/Funding/News: no disponibles históricamente)
# ─────────────────────────────────────────────────────────────────────────────
def sig_ema(row, prev) -> int:
    bull = prev["ema9"] <= prev["ema21"] and row["ema9"] > row["ema21"]
    bear = prev["ema9"] >= prev["ema21"] and row["ema9"] < row["ema21"]
    if bull and row["rsi"] < 65: return +1
    if bear and row["rsi"] > 35: return -1
    return 0

def sig_macd(row, prev) -> int:
    if prev["macd_hist"] < 0 and row["macd_hist"] > 0: return +1
    if prev["macd_hist"] > 0 and row["macd_hist"] < 0: return -1
    if row["macd_hist"] > 0 and row["macd_hist"] > prev["macd_hist"]: return +1
    if row["macd_hist"] < 0 and row["macd_hist"] < prev["macd_hist"]: return -1
    return 0

def sig_volume(row) -> int:
    if pd.isna(row["vol_ma20"]): return 0
    return +1 if row["volume"] > row["vol_ma20"] * 1.2 else 0

def sig_bb(row, prev) -> int:
    squeeze = row["bb_width"] < row.get("bb_width_ma", row["bb_width"]) * 0.75
    if row["close"] <= row["bb_lower"] and prev["close"] > prev["bb_lower"]: return +1
    if row["close"] >= row["bb_upper"] and prev["close"] < prev["bb_upper"]: return -1
    if squeeze and row["close"] > row["bb_mid"]: return +1
    if squeeze and row["close"] < row["bb_mid"]: return -1
    return 0

def sig_rsi_div(window_df) -> int:
    if len(window_df) < 10: return 0
    w = window_df.tail(10)
    pl = w["close"].rolling(3).min()
    rl = w["rsi"].rolling(3).min()
    ph = w["close"].rolling(3).max()
    rh = w["rsi"].rolling(3).max()
    if (pl.iloc[-1] < pl.iloc[-4] and rl.iloc[-1] > rl.iloc[-4] and w["rsi"].iloc[-1] < 50):
        return +1
    if (ph.iloc[-1] > ph.iloc[-4] and rh.iloc[-1] < rh.iloc[-4] and w["rsi"].iloc[-1] > 50):
        return -1
    return 0

def compute_score(t_sig, m_sig, v_sig, bb_sig, rsid_sig) -> float:
    """Score sin OB/Funding/News (no disponibles históricamente).
    Nota: el bot real tiene +3 señales extra, así que los scores aquí
    son más bajos que en live. Usamos threshold ajustado."""
    return float(t_sig + m_sig + v_sig * 0.8 + bb_sig + rsid_sig)

def detect_regime(df_slice: pd.DataFrame) -> str:
    """Detecta régimen a partir de los últimos 24 períodos."""
    if len(df_slice) < 24: return "sideways"
    pct = (df_slice["close"].iloc[-1] - df_slice["close"].iloc[-24]) / df_slice["close"].iloc[-24]
    atr_ratio = df_slice["atr"].iloc[-1] / df_slice["close"].iloc[-1]
    if pct < -0.05 and atr_ratio > 0.03: return "crash"
    if pct < -0.02: return "bear"
    if pct > 0.02:  return "bull"
    return "sideways"


# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN DE POSICIÓN
# ─────────────────────────────────────────────────────────────────────────────
class Position:
    def __init__(self, symbol, entry_price, usd_size, action, atr, regime):
        self.symbol       = symbol
        self.entry        = entry_price
        self.size         = usd_size
        self.action       = action  # "BUY" or "SELL"
        self.regime       = regime
        self.partial_done = False
        is_fut            = abs(usd_size) >= FUTURES_MIN_SCORE  # heuristic
        mult              = ATR_MULT_FUT if is_fut else ATR_MULT_SPOT
        trail_pct         = (atr * mult / entry_price) if atr > 0 else (
            FUTURES_TRAILING if is_fut else TRAILING_STOP_PCT)
        tp_pct            = FUTURES_TP if is_fut else TAKE_PROFIT_PCT
        if action == "BUY":
            self.trail_stop = entry_price * (1 - trail_pct)
            self.take_profit = entry_price * (1 + tp_pct)
            self.partial_tp  = entry_price * (1 + TAKE_PROFIT_PARTIAL)
            self.high        = entry_price
        else:
            self.trail_stop  = entry_price * (1 + trail_pct)
            self.take_profit = entry_price * (1 - tp_pct)
            self.partial_tp  = entry_price * (1 - TAKE_PROFIT_PARTIAL)
            self.high        = entry_price
        self.trail_pct    = trail_pct

    def update(self, price) -> tuple[bool, float, str]:
        """Actualiza posición con precio actual.
        Retorna (closed, pnl_pct, reason)."""
        if self.action == "BUY":
            # Trailing
            if price > self.high:
                self.high = price
                self.trail_stop = price * (1 - self.trail_pct)
            # Partial TP
            if not self.partial_done and price >= self.partial_tp:
                self.partial_done = True
                self.size *= (1 - PARTIAL_EXIT_PCT)
            # Stop loss absoluto
            if price <= self.entry * (1 - STOP_LOSS_PCT):
                return True, (price - self.entry) / self.entry * 100, "stop_loss"
            # Profit trail boost
            unreal = (price - self.entry) / self.entry
            if unreal >= PROFIT_TRAIL_TRIGGER:
                new_tp = price * (1 + PROFIT_TRAIL_STEP)
                if new_tp > self.take_profit:
                    self.take_profit = new_tp
            # Trail stop hit
            if price <= self.trail_stop:
                return True, (price - self.entry) / self.entry * 100, "trail"
            # TP
            if price >= self.take_profit:
                return True, (price - self.entry) / self.entry * 100, "take_profit"
        else:  # SHORT
            if price < self.high:
                self.high = price
                self.trail_stop = price * (1 + self.trail_pct)
            if not self.partial_done and price <= self.partial_tp:
                self.partial_done = True
                self.size *= (1 - PARTIAL_EXIT_PCT)
            if price >= self.entry * (1 + STOP_LOSS_PCT):
                return True, (self.entry - price) / self.entry * 100, "stop_loss"
            if price >= self.trail_stop:
                return True, (self.entry - price) / self.entry * 100, "trail"
            if price <= self.take_profit:
                return True, (self.entry - price) / self.entry * 100, "take_profit"
        return False, 0.0, ""


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(symbol: str, df: pd.DataFrame, timeframe: str) -> dict:
    df = calculate_indicators(df)
    df["bb_width_ma"] = df["bb_width"].rolling(20).mean()

    capital  = INITIAL_CAPITAL
    position = None
    trades   = []
    equity   = [capital]
    warmup   = 50  # velas necesarias para indicadores

    # Ajustar threshold: sin OB/Funding/News el score máximo es ~5 vs 8 en live
    # Usamos threshold de 1.5 en vez de 2.0
    SCORE_THRESHOLD = 1.5

    for i in range(warmup, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        price = float(row["close"])
        regime = detect_regime(df.iloc[max(0, i-24):i+1])

        # ── Actualizar posición abierta ──
        if position:
            closed, pnl_pct, reason = position.update(price)
            if closed:
                commission = COMMISSION * 2  # entrada + salida
                net_pnl_pct = pnl_pct - commission * 100
                usd_profit  = net_pnl_pct * position.size / 100
                capital += usd_profit
                trades.append({
                    "symbol":    symbol,
                    "action":    position.action,
                    "regime":    position.regime,
                    "entry":     position.entry,
                    "exit":      price,
                    "pnl_pct":   round(net_pnl_pct, 3),
                    "usd":       round(usd_profit, 2),
                    "reason":    reason,
                    "partial":   position.partial_done,
                    "timestamp": row["timestamp"],
                })
                position = None
                equity.append(capital)
            continue  # solo 1 posición a la vez

        # ── Calcular señales ──
        t_sig   = sig_ema(row, prev)
        m_sig   = sig_macd(row, prev)
        v_sig   = sig_volume(row)
        bb_sig  = sig_bb(row, prev)
        rsid    = sig_rsi_div(df.iloc[max(0, i-10):i+1])
        score   = compute_score(t_sig, m_sig, v_sig, bb_sig, rsid)

        # ── Filtro de régimen ──
        if regime == "crash": continue
        if abs(score) < SCORE_THRESHOLD: continue
        # En 3m no hacer shorts (igual que el bot)
        if timeframe == "3m" and score < 0: continue

        # ── Determinar dirección ──
        action = "BUY" if score > 0 else "SELL"
        if action == "SELL" and timeframe != "1h": continue  # shorts solo en 1h

        # ── Tamaño de posición ──
        abs_score = abs(int(round(score)))
        size_pct  = POSITION_SIZE_MAP.get(min(abs_score, 6), 0.02)
        if regime == "bear": size_pct *= 0.7
        usd_size  = round(capital * size_pct, 2)
        if usd_size < 5: continue  # posición demasiado chica

        # ── Abrir posición ──
        atr = float(row["atr"]) if not pd.isna(row["atr"]) else 0
        position = Position(symbol, price, usd_size, action, atr, regime)

    # Cerrar posición abierta al final (mark-to-market)
    if position and len(df) > 0:
        last_price = float(df.iloc[-1]["close"])
        if position.action == "BUY":
            pnl = (last_price - position.entry) / position.entry * 100
        else:
            pnl = (position.entry - last_price) / position.entry * 100
        pnl -= COMMISSION * 2 * 100
        capital += pnl * position.size / 100
        trades.append({
            "symbol":    symbol, "action": position.action,
            "regime":    position.regime, "entry": position.entry,
            "exit":      last_price, "pnl_pct": round(pnl, 3),
            "usd":       round(pnl * position.size / 100, 2),
            "reason":    "end_of_data", "partial": False,
            "timestamp": df.iloc[-1]["timestamp"],
        })
        equity.append(capital)

    return {"symbol": symbol, "capital": capital, "trades": trades, "equity": equity}


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
def analyze(results: list[dict], days: int):
    all_trades = []
    for r in results:
        all_trades.extend(r["trades"])

    if not all_trades:
        print("\n⚠️  Sin trades — bajá el SCORE_THRESHOLD o usá más días de datos.")
        return

    df = pd.DataFrame(all_trades)
    closed = df[df["reason"] != "end_of_data"]
    wins   = closed[closed["pnl_pct"] > 0]
    losses = closed[closed["pnl_pct"] <= 0]

    # Capital final (todos los símbolos combinados en capital único es simplificación)
    # Para multi-symbol mostramos por símbolo
    total_return = sum(r["capital"] - INITIAL_CAPITAL for r in results)
    avg_capital  = INITIAL_CAPITAL + total_return / len(results)

    print("\n" + "═" * 65)
    print(f"  📊  BACKTEST RESULTS — últimos {days} días")
    print("═" * 65)

    # ── Por símbolo ──
    print(f"\n{'SYM':<12} {'TRADES':>6} {'WR':>6} {'P&L $':>8} {'BEST':>7} {'WORST':>8} {'AVG WIN':>8} {'AVG LOSS':>9}")
    print("─" * 65)
    for r in results:
        t  = pd.DataFrame(r["trades"])
        if t.empty: continue
        c  = t[t["reason"] != "end_of_data"]
        w  = c[c["pnl_pct"] > 0]
        l  = c[c["pnl_pct"] <= 0]
        wr = f"{len(w)/len(c)*100:.0f}%" if len(c) else "—"
        pnl_usd = r["capital"] - INITIAL_CAPITAL
        best  = c["pnl_pct"].max() if len(c) else 0
        worst = c["pnl_pct"].min() if len(c) else 0
        avgw  = w["pnl_pct"].mean() if len(w) else 0
        avgl  = l["pnl_pct"].mean() if len(l) else 0
        print(f"{r['symbol']:<12} {len(c):>6} {wr:>6} {pnl_usd:>+8.2f} {best:>+7.2f}% {worst:>+8.2f}% {avgw:>+8.2f}% {avgl:>+9.2f}%")

    # ── Totales ──
    c = closed
    w = wins; l = losses
    wr = len(w)/len(c)*100 if len(c) else 0
    print("─" * 65)
    print(f"{'TOTAL':<12} {len(c):>6} {wr:>5.0f}% {total_return/len(results):>+8.2f}", end="")
    if len(c): print(f" {c['pnl_pct'].max():>+7.2f}% {c['pnl_pct'].min():>+8.2f}% {w['pnl_pct'].mean() if len(w) else 0:>+8.2f}% {l['pnl_pct'].mean() if len(l) else 0:>+9.2f}%")
    else: print()

    # ── Por régimen ──
    print(f"\n{'RÉGIMEN':<12} {'TRADES':>6} {'WR':>6} {'AVG P&L':>8} {'TOTAL $':>8}")
    print("─" * 45)
    for regime in ["bull", "sideways", "bear", "crash"]:
        sub = closed[closed["regime"] == regime]
        if sub.empty: continue
        sw  = sub[sub["pnl_pct"] > 0]
        wr  = len(sw)/len(sub)*100
        avg = sub["pnl_pct"].mean()
        tot = sub["usd"].sum()
        print(f"{regime:<12} {len(sub):>6} {wr:>5.0f}% {avg:>+8.2f}% {tot:>+8.2f}")

    # ── Por motivo de salida ──
    print(f"\n{'SALIDA':<14} {'TRADES':>6} {'WR':>6} {'AVG P&L':>8}")
    print("─" * 38)
    for reason in ["trail", "take_profit", "stop_loss"]:
        sub = closed[closed["reason"] == reason]
        if sub.empty: continue
        sw  = sub[sub["pnl_pct"] > 0]
        wr  = len(sw)/len(sub)*100
        avg = sub["pnl_pct"].mean()
        print(f"{reason:<14} {len(sub):>6} {wr:>5.0f}% {avg:>+8.2f}%")

    # ── Drawdown máximo ──
    eq = results[0]["equity"] if len(results) == 1 else None
    if eq and len(eq) > 1:
        peak = eq[0]; max_dd = 0
        for v in eq:
            peak = max(peak, v)
            dd   = (peak - v) / peak * 100
            max_dd = max(max_dd, dd)
        print(f"\n  Max drawdown:    {max_dd:.2f}%")

    # ── Métricas clave ──
    total_months = days / 30
    monthly_ret  = (total_return / len(results)) / total_months if total_months > 0 else 0
    print(f"  Return mensual:  {monthly_ret:+.2f} USD/mes (aprox)")
    print(f"  Profit factor:   {abs(w['usd'].sum()/l['usd'].sum()):.2f}x" if len(w) and len(l) else "")

    # ── Advertencia si pocos trades ──
    if len(closed) < 30:
        print(f"\n  ⚠️  Solo {len(closed)} trades cerrados — resultados no estadísticamente significativos.")
        print("     Usá --days 180 o --days 365 para más datos.")
    elif wr >= 55 and total_return/len(results) > 0:
        print(f"\n  ✅ Sistema muestra edge estadístico ({wr:.0f}% WR, {total_return/len(results):+.2f} P&L)")
        print("     Recomendado: validar en período fuera de muestra antes de pasar a real.")
    else:
        print(f"\n  ⚠️  Win rate {wr:.0f}% — sistema necesita ajustes antes de operar real.")

    print("\n" + "═" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CryptoBot v10 Backtester")
    parser.add_argument("--symbols", nargs="+",
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"],
        help="Símbolos a testear")
    parser.add_argument("--days",   type=int, default=180,
        help="Días de historia (default: 180)")
    parser.add_argument("--tf",     default="1h",
        choices=["3m", "15m", "1h", "4h"],
        help="Timeframe (default: 1h)")
    args = parser.parse_args()

    print(f"\n🤖 CryptoBot v10 — Backtester")
    print(f"   Símbolos: {', '.join(args.symbols)}")
    print(f"   Período:  últimos {args.days} días")
    print(f"   TF:       {args.tf}")
    print(f"   Capital:  ${INITIAL_CAPITAL:,.0f}\n")

    results = []
    for symbol in args.symbols:
        try:
            df = fetch_ohlcv(symbol, args.tf, args.days)
            result = run_backtest(symbol, df, args.tf)
            n_closed = len([t for t in result["trades"] if t["reason"] != "end_of_data"])
            final_ret = result["capital"] - INITIAL_CAPITAL
            print(f"  {symbol:<14} → {n_closed} trades | P&L: {final_ret:+.2f} USD")
            results.append(result)
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")

    if results:
        analyze(results, args.days)


if __name__ == "__main__":
    main()
