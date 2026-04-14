# 🤖 Crypto Trading Bot — Binance Testnet

Bot de trading automático con estrategia multi-señal:
**EMA crossover + RSI + Sentimiento noticias → validación final con Claude AI**

Opera solo cuando **2 o más señales coinciden** en la misma dirección.

---

## Arquitectura

```
Precio OHLCV (ccxt)
        ↓
  Indicadores técnicos
  EMA9 / EMA21 / RSI14  →  Señal técnica (+1 / 0 / -1)
        +
  Noticias CryptoPanic  →  Señal sentimiento (+1 / 0 / -1)
        ↓
  ¿2+ señales alinean?
        ↓ SÍ
  Claude AI (validación final)
        ↓ BUY/SELL + confianza > 0.6
  Ejecutar orden en Binance Testnet
        ↓
  Guardar en trade_log.json
```

---

## Setup

### 1. Binance Testnet
1. Ir a https://testnet.binance.vision/
2. Login con GitHub
3. Generar API Key y Secret
4. El testnet te da USDT ficticio para operar

### 2. Variables de entorno en Railway
```
BINANCE_API_KEY     = (de Binance Testnet)
BINANCE_API_SECRET  = (de Binance Testnet)
ANTHROPIC_API_KEY   = (tu key de Anthropic)
CAPITAL_USD         = 100
```

### 3. Deploy en Railway
```bash
# Subir a GitHub
git init
git add .
git commit -m "crypto bot inicial"
git remote add origin https://github.com/tu-usuario/crypto-bot
git push -u origin main

# En Railway:
# New Project → Deploy from GitHub → seleccionar repo
# Agregar variables de entorno
# El Procfile ya configura el worker automáticamente
```

---

## Parámetros configurables (en bot.py)

| Parámetro | Valor default | Descripción |
|-----------|--------------|-------------|
| `CAPITAL_TOTAL_USD` | 100 | Capital total |
| `RISK_PER_TRADE` | 0.03 | 3% por operación = $3 |
| `STOP_LOSS_PCT` | 0.02 | Stop loss 2% |
| `TAKE_PROFIT_PCT` | 0.04 | Take profit 4% |
| `MIN_SIGNALS` | 2 | Mínimo señales para operar |
| `LOOP_INTERVAL_SEC` | 300 | Ciclo cada 5 minutos |
| `WATCHLIST` | BTC, ETH, SOL, BNB | Pares a monitorear |

---

## Logs de ejemplo

```
2024-01-15 10:00:00 [INFO] 🤖 Bot iniciado — Binance Testnet
2024-01-15 10:00:01 [INFO] 📊 Analizando BTC/USDT...
2024-01-15 10:00:02 [INFO]   Señal técnica : +1 (BULL)
2024-01-15 10:00:03 [INFO]   Señal noticias: +1 (POS)
2024-01-15 10:00:03 [INFO]   🧠 Consultando Claude (señales alineadas: 2)...
2024-01-15 10:00:04 [INFO]   Claude dice: BUY (confianza: 0.78) — EMA crossover confirmado con sentimiento positivo
2024-01-15 10:00:05 [INFO]   ✅ COMPRA ejecutada: BTC/USDT | qty=0.000065 | precio=46150 | SL=45227 | TP=48196
```

---

## Costos estimados (API)

- **ccxt + CryptoPanic**: gratis
- **Claude (Haiku)**: ~$0.0003 por consulta. Solo se invoca cuando 2+ señales alinean → estimado $0.50–2/mes

---

## ⚠️ Disclaimer

Este bot es experimental y educativo. Siempre probá en testnet antes de usar dinero real. El trading de crypto conlleva riesgo de pérdida total del capital.
