import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

class TradingSimulation:
    def __init__(self, symbol="GC=F", initial_balance=200, fast_ema=9, slow_ema=21):
        self.symbol = symbol
        self.initial_balance = initial_balance
        # Start trading from 2 hours ago to show recent activity
        self.simulation_start = datetime.now(timezone.utc) - timedelta(hours=2)
        self.balance = initial_balance
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.data = None
        self.trades = []
        self.equity_curve = []

        # MT5 Parameters
        self.lot_size = 0.01
        self.contract_size = 100 # Standard lot for XAUUSD is 100 oz
        self.leverage = 100 # Common leverage for Gold
        self.stop_out_level = 50.0 # Stop out at 50% margin level

    def fetch_data(self, period="5d", interval="1m"):
        # Fetch data
        try:
            df = yf.download(self.symbol, period=period, interval=interval, progress=False)

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                # If we have (Price, Ticker), we can just drop the Ticker level if it's the only one
                if len(df.columns.levels) > 1:
                     df.columns = df.columns.droplevel(1)

            # Ensure we have a clean DataFrame with single-level columns
            # Calculate Indicators
            if not df.empty:
                df['EMA_Fast'] = df['Close'].ewm(span=self.fast_ema, adjust=False).mean()
                df['EMA_Slow'] = df['Close'].ewm(span=self.slow_ema, adjust=False).mean()

            self.data = df.dropna()
        except Exception as e:
            print(f"Error fetching data: {e}")
            self.data = pd.DataFrame()

    def run(self):
        if self.data is None or self.data.empty:
            self.fetch_data()

        if self.data.empty:
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'total_profit': 0,
                'win_rate': 0,
                'total_trades': 0,
                'trades': [],
                'equity_curve': [],
                'current_equity': self.initial_balance,
                'current_margin': 0,
                'current_free_margin': self.initial_balance,
                'current_margin_level': 0
            }

        df = self.data.copy()

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

        current_equity = self.balance
        current_margin = 0.0
        current_free_margin = self.balance
        current_margin_level = 0.0

        # Iterate through rows to simulate trading
        for index, row in df.iterrows():
            # Filter by start date
            if index < self.simulation_start:
                continue

            price = float(row['Close'])
            signal = int(row['Signal'])
            date_str = str(index)

            # Calculations for current step
            floating_profit = 0.0
            margin_used = 0.0

            # Check Stop Out first if position is open
            if position == 'SHORT':
                # For SHORT: Profit = (Entry - Price) * Lots * Contract
                floating_profit = (entry_price - price) * self.lot_size * self.contract_size
                # Margin calculation is same for Short: (Price * Lots * Contract) / Leverage
                # Usually based on Entry Price
                margin_used = (entry_price * self.lot_size * self.contract_size) / self.leverage

                equity_check = self.balance + floating_profit
                if margin_used > 0:
                    margin_level_check = (equity_check / margin_used) * 100
                    if margin_level_check < self.stop_out_level:
                        # Stop Out triggered
                        profit = floating_profit
                        self.balance += profit
                        self.trades.append({
                            'entry_date': entry_date,
                            'exit_date': date_str,
                            'entry_price': round(entry_price, 2),
                            'exit_price': round(price, 2),
                            'profit': round(profit, 2),
                            'type': 'SHORT (STOP OUT)'
                        })
                        position = None
                        floating_profit = 0.0
                        margin_used = 0.0

            # Execute Strategy (only if still in position or looking to enter)
            if position is None:
                if signal == -1: # Enter SHORT on Signal -1
                    # Check if enough free margin
                    required_margin = (price * self.lot_size * self.contract_size) / self.leverage
                    if self.balance > required_margin:
                        position = 'SHORT'
                        entry_price = price
                        entry_date = date_str
            elif position == 'SHORT':
                if signal == 1: # Close SHORT on Signal 1
                    exit_price = price
                    profit = (entry_price - exit_price) * self.lot_size * self.contract_size

                    self.balance += profit
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': date_str,
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(exit_price, 2),
                        'profit': round(profit, 2),
                        'type': 'SHORT'
                    })
                    position = None

            # Recalculate metrics for this step
            if position == 'SHORT':
                 floating_profit = (entry_price - price) * self.lot_size * self.contract_size
                 margin_used = (entry_price * self.lot_size * self.contract_size) / self.leverage
            else:
                 floating_profit = 0.0
                 margin_used = 0.0

            current_equity = self.balance + floating_profit
            current_margin = margin_used
            current_free_margin = current_equity - current_margin
            current_margin_level = (current_equity / current_margin * 100) if current_margin > 0 else 0.0

            self.equity_curve.append({
                'date': date_str,
                'balance': round(self.balance, 2),
                'equity': round(current_equity, 2),
                'margin': round(current_margin, 2),
                'free_margin': round(current_free_margin, 2),
                'margin_level': round(current_margin_level, 2),
                'price': round(price, 2),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2)
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
            'equity_curve': self.equity_curve,
            'current_equity': round(current_equity, 2),
            'current_margin': round(current_margin, 2),
            'current_free_margin': round(current_free_margin, 2),
            'current_margin_level': round(current_margin_level, 2)
        }

if __name__ == "__main__":
    sim = TradingSimulation()
    result = sim.run()
    print(f"Simulation Complete. Final Balance: {result['final_balance']}")
    print(f"Total Trades: {result['total_trades']}")
