from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# SETTINGS
SYMBOL = "GC=F"  # Gold Futures
FAST_EMA = 9
SLOW_EMA = 21
INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.02

def run_backtest():
    df = yf.download(SYMBOL, period="30d", interval="1h")

    df['ema_fast'] = df['Close'].ewm(span=FAST_EMA, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=SLOW_EMA, adjust=False).mean()

    balance = INITIAL_BALANCE
    position = None
    entry_price = 0
    trades = []

    for i in range(1, len(df)):

        # BUY crossover
        if df['ema_fast'].iloc[i-1] < df['ema_slow'].iloc[i-1] and \
           df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]:

            if position is None:
                position = "BUY"
                entry_price = df['Close'].iloc[i]

        # SELL crossover
        elif df['ema_fast'].iloc[i-1] > df['ema_slow'].iloc[i-1] and \
             df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i]:

            if position == "BUY":
                exit_price = df['Close'].iloc[i]
                profit = (exit_price - entry_price) * 10  # simple multiplier
                balance += profit
                trades.append(profit)
                position = None

    win_rate = 0
    if trades:
        win_rate = len([t for t in trades if t > 0]) / len(trades) * 100

    return {
        "initial_balance": INITIAL_BALANCE,
        "final_balance": round(balance, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 2)
    }

@app.route("/")
def home():
    return jsonify({"message": "XAUUSD EMA Bot Running"})

@app.route("/simulate")
def simulate():
    result = run_backtest()
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
