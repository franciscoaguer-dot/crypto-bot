"""
CryptoBot v10 — Optimizador de parámetros
Corre múltiples configuraciones de trailing stop / TP y compara resultados.
Uso: python optimize.py
"""
import sys, itertools
sys.path.insert(0, '.')

# Importar el motor de backtest
from backtest import fetch_ohlcv, calculate_indicators, run_backtest, INITIAL_CAPITAL
import pandas as pd

# ── Parámetros a optimizar ──────────────────────────────────────────────────
PARAM_GRID = {
    "TRAILING_STOP_PCT": [0.010, 0.015, 0.020, 0.025],   # 1% → 2.5%
    "TAKE_PROFIT_PCT":   [0.025, 0.030, 0.040, 0.050],   # 2.5% → 5%
    "SCORE_THRESHOLD":   [1.5,   2.0,   2.5],            # más selectivo
    "ATR_MULT_SPOT":     [1.5,   2.0,   2.5],            # ATR más amplio
}

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DAYS    = 180   # 6 meses para optimización, guardamos 6 para validación

import backtest as bt

def run_with_params(dfs, trail, tp, score_thresh, atr_mult):
    """Corre backtest con parámetros modificados."""
    # Override parámetros globales
    bt.TRAILING_STOP_PCT  = trail
    bt.TAKE_PROFIT_PCT    = tp
    bt.FUTURES_TRAILING   = trail * 0.8
    bt.FUTURES_TP         = tp * 0.8
    bt.ATR_MULT_SPOT      = atr_mult
    bt.ATR_MULT_FUT       = atr_mult * 0.7

    results = []
    for sym, df in dfs.items():
        r = run_backtest(sym, df, "1h", score_threshold=score_thresh)
        results.append(r)

    all_trades = []
    for r in results: all_trades.extend(r["trades"])
    if not all_trades: return None

    df_t = pd.DataFrame(all_trades)
    closed = df_t[df_t["reason"] != "end_of_data"]
    if len(closed) < 10: return None

    wins = closed[closed["pnl_pct"] > 0]
    losses = closed[closed["pnl_pct"] <= 0]
    wr = len(wins) / len(closed) * 100
    total_pnl = sum(r["capital"] - INITIAL_CAPITAL for r in results) / len(results)
    pf = abs(wins["usd"].sum() / losses["usd"].sum()) if len(losses) and losses["usd"].sum() != 0 else 0

    return {
        "trail": trail,
        "tp": tp,
        "score": score_thresh,
        "atr_mult": atr_mult,
        "trades": len(closed),
        "wr": round(wr, 1),
        "pnl": round(total_pnl, 2),
        "pf": round(pf, 2),
        "avg_win": round(wins["pnl_pct"].mean(), 3) if len(wins) else 0,
        "avg_loss": round(losses["pnl_pct"].mean(), 3) if len(losses) else 0,
    }

def main():
    print("\n🔬 CryptoBot v10 — Optimizador de parámetros")
    print(f"   Descargando {DAYS} días de datos...\n")

    # Descargar datos una sola vez
    dfs = {}
    for sym in SYMBOLS:
        dfs[sym] = fetch_ohlcv(sym, "1h", DAYS)

    # Grid search
    combos = list(itertools.product(
        PARAM_GRID["TRAILING_STOP_PCT"],
        PARAM_GRID["TAKE_PROFIT_PCT"],
        PARAM_GRID["SCORE_THRESHOLD"],
        PARAM_GRID["ATR_MULT_SPOT"],
    ))
    print(f"   Testeando {len(combos)} combinaciones...\n")

    results = []
    for i, (trail, tp, score, atr) in enumerate(combos):
        r = run_with_params(dfs, trail, tp, score, atr)
        if r:
            results.append(r)
        if (i+1) % 10 == 0:
            print(f"   {i+1}/{len(combos)} combinaciones...")

    if not results:
        print("Sin resultados válidos")
        return

    df_r = pd.DataFrame(results)

    # Top 10 por profit factor con WR > 45%
    viable = df_r[df_r["wr"] >= 45].sort_values("pf", ascending=False)

    print("\n" + "═" * 80)
    print("  🏆  TOP CONFIGURACIONES (WR ≥ 45%, ordenado por Profit Factor)")
    print("═" * 80)
    print(f"{'TRAIL':>7} {'TP':>6} {'SCORE':>6} {'ATR':>6} | {'TRADES':>6} {'WR':>5} {'P&L':>7} {'PF':>5} {'AVG W':>7} {'AVG L':>7}")
    print("─" * 80)

    if viable.empty:
        print("  ⚠️  Ninguna combinación alcanzó 45% WR.")
        print("  Mostrando top 10 por menor pérdida:\n")
        top = df_r.sort_values("pnl", ascending=False).head(10)
    else:
        top = viable.head(10)

    for _, row in top.iterrows():
        print(f"{row['trail']*100:>6.1f}% {row['tp']*100:>5.1f}% {row['score']:>6.1f} {row['atr_mult']:>6.1f} | "
              f"{row['trades']:>6} {row['wr']:>4.0f}% {row['pnl']:>+7.2f} {row['pf']:>5.2f}x "
              f"{row['avg_win']:>+7.3f}% {row['avg_loss']:>+7.3f}%")

    print("─" * 80)

    # Mejor configuración
    best = top.iloc[0] if not top.empty else df_r.sort_values("pnl", ascending=False).iloc[0]
    print(f"\n  ✅ MEJOR CONFIGURACIÓN:")
    print(f"     TRAILING_STOP_PCT = {best['trail']:.3f}  ({best['trail']*100:.1f}%)")
    print(f"     TAKE_PROFIT_PCT   = {best['tp']:.3f}  ({best['tp']*100:.1f}%)")
    print(f"     SCORE_THRESHOLD   = {best['score']:.1f}")
    print(f"     ATR_MULT_SPOT     = {best['atr_mult']:.1f}")
    print(f"     Win Rate:          {best['wr']}%")
    print(f"     P&L promedio:      ${best['pnl']:+.2f}")
    print(f"     Profit Factor:     {best['pf']}x")
    print("\n  ⚠️  Estos parámetros fueron optimizados en los últimos 180 días.")
    print("     Validalos corriendo backtest.py con --days 365 antes de usar en real.\n")
    print("═" * 80 + "\n")

if __name__ == "__main__":
    main()
