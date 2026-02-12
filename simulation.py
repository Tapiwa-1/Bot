import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

class TradingSimulation:
    def __init__(self, symbol="GC=F", initial_balance=200, fast_ema=9, slow_ema=21):
        self.symbol = symbol
        self.initial_balance = initial_balance
        # Start trading from Today Now (Harare Time: UTC+2)
        # Current time is roughly 2026-02-12 20:40:00 UTC (22:40 Harare)
        self.simulation_start = datetime(2026, 2, 12, 20, 40, 0, tzinfo=timezone.utc)
        self.balance = initial_balance
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.data = None
        self.trades = []
        self.equity_curve = []

    def fetch_data(self, period="1mo", interval="5m"):
        # Fetch data
        df = yf.download(self.symbol, period=period, interval=interval, progress=False)

        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            # If we have (Price, Ticker), we can just drop the Ticker level if it's the only one
            if len(df.columns.levels) > 1:
                 df.columns = df.columns.droplevel(1)

        # Ensure we have a clean DataFrame with single-level columns
        # Calculate Indicators
        df['EMA_Fast'] = df['Close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.slow_ema, adjust=False).mean()

        self.data = df.dropna()

    def run(self):
        if self.data is None:
            self.fetch_data()

        df = self.data.copy()

        # Identify crossovers
        # specific logic:
        # Buy Signal: Fast crosses above Slow
        # Sell Signal: Fast crosses below Slow

        prev_fast = df['EMA_Fast'].shift(1)
        prev_slow = df['EMA_Slow'].shift(1)

        df['Signal'] = 0

        # Buy Signal
        buy_condition = (df['EMA_Fast'] > df['EMA_Slow']) & (prev_fast <= prev_slow)
        df.loc[buy_condition, 'Signal'] = 1

        # Sell Signal
        sell_condition = (df['EMA_Fast'] < df['EMA_Slow']) & (prev_fast >= prev_slow)
        df.loc[sell_condition, 'Signal'] = -1

        position = None
        entry_price = 0.0
        entry_date = None

        self.trades = []
        self.equity_curve = []
        self.balance = self.initial_balance

        # Iterate through rows to simulate trading
        for index, row in df.iterrows():
            # Filter by start date
            if index < self.simulation_start:
                continue

            price = float(row['Close'])
            signal = int(row['Signal'])
            date_str = str(index)

            # Execute Strategy
            if position is None:
                if signal == 1:
                    position = 'LONG'
                    entry_price = price
                    entry_date = date_str
            elif position == 'LONG':
                if signal == -1:
                    # Close Long
                    exit_price = price
                    # Profit calculation: (Exit - Entry) * Contract Size
                    # Assuming standard lot size multiplier of 10 for simplicity (mini lot) or whatever user used.
                    # User used 10. Let's stick to 10.
                    multiplier = 10
                    profit = (exit_price - entry_price) * multiplier

                    self.balance += profit
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date_str,
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'profit': round(profit, 2),
                        'type': 'LONG'
                    })
                    position = None

            self.equity_curve.append({
                'date': date_str,
                'balance': round(self.balance, 2),
                'price': round(price, 2)
            })

        # Calculate metrics
        win_rate = 0
        if self.trades:
            winning_trades = [t for t in self.trades if t['profit'] > 0]
            win_rate = (len(winning_trades) / len(self.trades)) * 100

        total_profit = self.balance - self.initial_balance

        return {
            'initial_balance': self.initial_balance,
            'final_balance': round(self.balance, 2),
            'total_profit': round(total_profit, 2),
            'win_rate': round(win_rate, 2),
            'total_trades': len(self.trades),
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

if __name__ == "__main__":
    sim = TradingSimulation()
    result = sim.run()
    print(f"Simulation Complete. Final Balance: {result['final_balance']}")
    print(f"Total Trades: {result['total_trades']}")
