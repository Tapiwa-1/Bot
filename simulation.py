import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestRegressor

class TradingSimulation:
    def __init__(self, symbol="EURGBP=X", initial_balance=200, fast_ema=9, slow_ema=21):
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

        # MT5 Parameters for Forex (EURGBP)
        # Account is assumed to be in GBP for simplicity (Quote Currency)
        self.lot_size = 0.01 # This will be overridden by dynamic sizing
        self.contract_size = 100000 # Standard Forex lot is 100,000 units
        self.leverage = 100 # Common leverage
        self.stop_out_level = 50.0 # Stop out at 50% margin level

        # New Strategy Parameters
        self.risk_per_trade = 0.02 # 2% of account
        self.risk_reward = 2 # 1:2 ratio

        # ML Model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.prediction_horizon = 30 # Predict 30 minutes into the future

    def fetch_data(self, period="5d", interval="1m"):
        # Fetch data
        try:
            df = yf.download(self.symbol, period=period, interval=interval, progress=False)

            # Handle MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                if len(df.columns.levels) > 1:
                     df.columns = df.columns.droplevel(1)

            # Ensure we have a clean DataFrame with single-level columns
            if not df.empty:
                df['EMA_Fast'] = df['Close'].ewm(span=self.fast_ema, adjust=False).mean()
                df['EMA_Slow'] = df['Close'].ewm(span=self.slow_ema, adjust=False).mean()

                # ML Features (kept for visualization)
                df['Returns'] = df['Close'].pct_change()
                df['Volatility'] = df['Close'].rolling(window=20).std()
                df['RSI'] = self.calculate_rsi(df['Close'], 14)
                df['Momentum'] = df['Close'] - df['Close'].shift(self.prediction_horizon)
                df['Target'] = df['Close'].shift(-self.prediction_horizon)

            self.data = df # Keep NaNs for now to allow feature calculation, drop before training/sim
        except Exception as e:
            print(f"Error fetching data: {e}")
            self.data = pd.DataFrame()

    def calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train_model(self, df):
        train_df = df.dropna()
        features = ['EMA_Fast', 'EMA_Slow', 'Returns', 'Volatility', 'RSI', 'Momentum']
        X = train_df[features]
        y = train_df['Target']
        if len(X) > 100:
            self.model.fit(X, y)
            return True
        return False

    def predict_future(self, row, current_price):
        features = ['EMA_Fast', 'EMA_Slow', 'Returns', 'Volatility', 'RSI', 'Momentum']
        if row[features].isnull().any():
            return None
        X_pred = row[features].to_frame().T
        predicted_price = self.model.predict(X_pred)[0]
        return predicted_price

    # --- Candlestick Pattern Helper Functions ---

    def is_hammer(self, candle):
        body = abs(candle['Close'] - candle['Open'])
        upper_wick = candle['High'] - max(candle['Close'], candle['Open'])
        lower_wick = min(candle['Close'], candle['Open']) - candle['Low']
        # Typical Hammer: small body, long lower wick (2x body), short upper wick
        return lower_wick > 2 * body and upper_wick < 0.5 * body

    def is_inverted_hammer(self, candle):
        body = abs(candle['Close'] - candle['Open'])
        upper_wick = candle['High'] - max(candle['Close'], candle['Open'])
        lower_wick = min(candle['Close'], candle['Open']) - candle['Low']
        return upper_wick > 2 * body and lower_wick < 0.5 * body

    def is_shooting_star(self, candle):
        return self.is_inverted_hammer(candle) # Same shape, context matters

    def is_hanging_man(self, candle):
        return self.is_hammer(candle) # Same shape, context matters

    def is_bullish_engulfing(self, prev, curr):
        if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open']: # Prev red, Curr green
            return curr['Open'] <= prev['Close'] and curr['Close'] >= prev['Open']
        return False

    def is_bearish_engulfing(self, prev, curr):
        if prev['Close'] > prev['Open'] and curr['Close'] < curr['Open']: # Prev green, Curr red
            return curr['Open'] >= prev['Close'] and curr['Close'] <= prev['Open']
        return False

    def is_morning_star(self, c1, c2, c3):
        # 1: Long red
        # 2: Small body (gap down usually, but simplified here)
        # 3: Long green closing well into 1
        is_c1_bearish = c1['Close'] < c1['Open'] and abs(c1['Close'] - c1['Open']) > (c1['High'] - c1['Low']) * 0.5
        is_c2_small = abs(c2['Close'] - c2['Open']) < (c2['High'] - c2['Low']) * 0.3
        is_c3_bullish = c3['Close'] > c3['Open'] and c3['Close'] > (c1['Open'] + c1['Close']) / 2
        return is_c1_bearish and is_c2_small and is_c3_bullish

    def is_evening_star(self, c1, c2, c3):
        # 1: Long green
        # 2: Small body
        # 3: Long red closing well into 1
        is_c1_bullish = c1['Close'] > c1['Open'] and abs(c1['Close'] - c1['Open']) > (c1['High'] - c1['Low']) * 0.5
        is_c2_small = abs(c2['Close'] - c2['Open']) < (c2['High'] - c2['Low']) * 0.3
        is_c3_bearish = c3['Close'] < c3['Open'] and c3['Close'] < (c1['Open'] + c1['Close']) / 2
        return is_c1_bullish and is_c2_small and is_c3_bearish

    def is_doji(self, candle):
        body = abs(candle['Close'] - candle['Open'])
        total_range = candle['High'] - candle['Low']
        return body < 0.1 * total_range

    def detect_pattern(self, candles):
        if len(candles) < 3: return None, None
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        # Check Trend (using EMA_Slow of c2/c1)
        trend_c1 = 'down' if c1['Close'] < c1['EMA_Slow'] else 'up'
        trend_c2 = 'down' if c2['Close'] < c2['EMA_Slow'] else 'up'

        # Hammer / Inverted Hammer (Buy)
        if self.is_hammer(c2) and trend_c2 == 'down': return 'buy', 'Hammer'
        if self.is_inverted_hammer(c2) and trend_c2 == 'down': return 'buy', 'Inverted Hammer'

        # Shooting Star / Hanging Man (Sell)
        if self.is_shooting_star(c2) and trend_c2 == 'up': return 'sell', 'Shooting Star'
        if self.is_hanging_man(c2) and trend_c2 == 'up': return 'sell', 'Hanging Man'

        # Engulfing (c1, c2)
        if self.is_bullish_engulfing(c1, c2) and trend_c1 == 'down': return 'buy', 'Bullish Engulfing'
        if self.is_bearish_engulfing(c1, c2) and trend_c1 == 'up': return 'sell', 'Bearish Engulfing'

        # Morning/Evening Star (c1, c2, c3)
        if self.is_morning_star(c1, c2, c3) and trend_c1 == 'down': return 'buy', 'Morning Star'
        if self.is_evening_star(c1, c2, c3) and trend_c1 == 'up': return 'sell', 'Evening Star'

        # Doji
        if self.is_doji(c2): return 'wait_for_confirmation', 'Doji'

        return None, None

    def confirm_signal(self, pattern, next_candle):
        if pattern == 'buy' and next_candle['Close'] > next_candle['Open']: return True
        if pattern == 'sell' and next_candle['Close'] < next_candle['Open']: return True
        return False

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
                'future_forecast': [],
                'ml_recommendation': 'NEUTRAL',
                'current_equity': self.initial_balance,
                'current_margin': 0,
                'current_free_margin': self.initial_balance,
                'current_margin_level': 0,
                'current_advice': {'action': 'WAIT', 'reason': 'No Data', 'entry': 0, 'sl': 0, 'tp': 0}
            }

        # Train ML Model (keep it running for visualization)
        model_trained = self.train_model(self.data)

        df = self.data.copy()

        # Ensure sufficient history for pattern detection
        if len(df) < 4:
            return {'error': 'Not enough data'}

        self.trades = []
        self.equity_curve = []
        self.balance = self.initial_balance

        current_equity = self.balance
        current_margin = 0.0
        current_free_margin = self.balance
        current_margin_level = 0.0

        ml_recommendation = "NEUTRAL"
        future_forecast = []

        open_trade = None # {type, entry, sl, tp, lots, entry_time}
        current_advice = {}

        candles_list = df.to_dict('records')
        index_list = df.index.to_list()

        for i in range(3, len(df)):
            current_time = index_list[i]

            # Skip if before simulation start
            if current_time < self.simulation_start:
                continue

            current_candle = candles_list[i]
            # Add date to candle dict for easier access if needed
            current_candle['Date'] = str(current_time)

            price = current_candle['Close']

            # --- 1. Manage Open Trade ---
            if open_trade:
                # Check SL/TP
                close_reason = None
                exit_price = 0.0

                if open_trade['type'] == 'LONG':
                    if current_candle['Low'] <= open_trade['sl']:
                        close_reason = 'SL'
                        exit_price = open_trade['sl']
                    elif current_candle['High'] >= open_trade['tp']:
                        close_reason = 'TP'
                        exit_price = open_trade['tp']
                elif open_trade['type'] == 'SHORT':
                    if current_candle['High'] >= open_trade['sl']:
                        close_reason = 'SL'
                        exit_price = open_trade['sl']
                    elif current_candle['Low'] <= open_trade['tp']:
                        close_reason = 'TP'
                        exit_price = open_trade['tp']

                # Execute Close
                if close_reason:
                    profit = 0
                    if open_trade['type'] == 'LONG':
                        profit = (exit_price - open_trade['entry']) * open_trade['lots'] * self.contract_size
                    else:
                        profit = (open_trade['entry'] - exit_price) * open_trade['lots'] * self.contract_size

                    self.balance += profit
                    self.trades.append({
                        'entry_date': open_trade['entry_time'],
                        'exit_date': str(current_time),
                        'entry_price': round(open_trade['entry'], 5), # 5 decimals for Forex
                        'exit_price': round(exit_price, 5),
                        'profit': round(profit, 2),
                        'type': f"{open_trade['type']} ({close_reason})"
                    })
                    open_trade = None

            # --- 2. Check for New Trade (if no open trade) ---
            if open_trade is None:
                pattern_window = [candles_list[i-3], candles_list[i-2], candles_list[i-1]]
                pattern, pattern_name = self.detect_pattern(pattern_window)

                if pattern in ['buy', 'sell']:
                    # Confirm with current candle (i)
                    if self.confirm_signal(pattern, current_candle):
                        # Generate Trade
                        entry_price = current_candle['Close']
                        sl = 0.0
                        tp = 0.0
                        trade_type = ''

                        if pattern == 'buy':
                            trade_type = 'LONG'
                            sl = current_candle['Low']
                            risk = entry_price - sl
                            if risk <= 0: risk = 0.0001 # Safety
                            tp = entry_price + (risk * self.risk_reward)

                        elif pattern == 'sell':
                            trade_type = 'SHORT'
                            sl = current_candle['High']
                            risk = sl - entry_price
                            if risk <= 0: risk = 0.0001
                            tp = entry_price - (risk * self.risk_reward)

                        # Calculate Position Size (Risk 2%)
                        risk_amount = self.balance * self.risk_per_trade
                        risk_per_unit = abs(entry_price - sl) # In Price terms
                        # Risk Value = Risk per unit * Contract Size * Lots
                        # Lots = Risk Amount / (Risk per unit * Contract Size)

                        if risk_per_unit > 0:
                            # Formula for Forex:
                            # Risk per lot = Risk per unit * Contract Size
                            risk_per_lot = risk_per_unit * self.contract_size
                            lots = risk_amount / risk_per_lot
                            lots = round(lots, 2)
                            if lots < 0.01: lots = 0.01
                        else:
                            lots = 0.01

                        # Execute Entry
                        required_margin = (entry_price * lots * self.contract_size) / self.leverage
                        # Wait, Margin for EURGBP is in Base Currency (EUR).
                        # Need to convert to Account Currency (GBP).
                        # EUR/GBP rate is entry_price.
                        # So Margin in GBP = (Lots * Contract Size * Rate) / Leverage * Rate? No.
                        # Margin in Base (EUR) = Lots * Contract Size / Leverage.
                        # Convert EUR to GBP -> Multiply by EURGBP rate (entry_price).
                        # So Required Margin (GBP) = (Lots * Contract * EntryPrice) / Leverage.
                        # This matches the previous logic exactly, so no change needed.

                        if self.balance > required_margin:
                            open_trade = {
                                'type': trade_type,
                                'entry': entry_price,
                                'sl': sl,
                                'tp': tp,
                                'lots': lots,
                                'entry_time': str(current_time)
                            }

            # --- 3. Update Metrics/Equity Curve ---
            floating_profit = 0.0
            margin_used = 0.0

            if open_trade:
                if open_trade['type'] == 'LONG':
                    floating_profit = (price - open_trade['entry']) * open_trade['lots'] * self.contract_size
                    margin_used = (open_trade['entry'] * open_trade['lots'] * self.contract_size) / self.leverage
                else:
                    floating_profit = (open_trade['entry'] - price) * open_trade['lots'] * self.contract_size
                    margin_used = (open_trade['entry'] * open_trade['lots'] * self.contract_size) / self.leverage

            current_equity = self.balance + floating_profit
            current_margin = margin_used
            current_free_margin = current_equity - current_margin
            current_margin_level = (current_equity / current_margin * 100) if current_margin > 0 else 0.0

            # ML Prediction (visual)
            predicted_future_price = None
            if model_trained:
                predicted_future_price = self.predict_future(pd.Series(current_candle), price)

            self.equity_curve.append({
                'date': str(current_time),
                'balance': round(self.balance, 2),
                'equity': round(current_equity, 2),
                'margin': round(current_margin, 2),
                'free_margin': round(current_free_margin, 2),
                'margin_level': round(current_margin_level, 2),
                'price': round(price, 5),
                'open': round(current_candle['Open'], 5),
                'high': round(current_candle['High'], 5),
                'low': round(current_candle['Low'], 5),
                'close': round(current_candle['Close'], 5),
                'ml_prediction': round(predicted_future_price, 5) if predicted_future_price else None
            })

        # --- Finalize Forecast & Summary ---
        if not df.empty and model_trained:
            last_row = df.iloc[-1]
            last_price = float(last_row['Close'])
            predicted_price_30m = self.predict_future(last_row, last_price)
            if predicted_price_30m:
                last_time = df.index[-1]
                steps = 30
                price_step = (predicted_price_30m - last_price) / steps
                for i in range(1, steps + 1):
                    future_time = last_time + timedelta(minutes=i)
                    future_price = last_price + (price_step * i)
                    future_forecast.append({
                        'date': str(future_time),
                        'price': round(future_price, 5)
                    })
                if predicted_price_30m > last_price:
                    ml_recommendation = "BUY (Bullish Trend)"
                else:
                    ml_recommendation = "SELL (Bearish Trend)"

        # --- Generate Current Advice ---
        if open_trade:
             current_advice = {
                 'action': 'HOLD',
                 'reason': f"Active {open_trade['type']} Position",
                 'entry': round(open_trade['entry'], 5),
                 'sl': round(open_trade['sl'], 5),
                 'tp': round(open_trade['tp'], 5),
                 'type': open_trade['type']
             }
        else:
            last_window = [candles_list[-3], candles_list[-2], candles_list[-1]]
            pattern, pattern_name = self.detect_pattern(last_window)

            if pattern:
                current_advice = {
                    'action': 'WATCH',
                    'reason': f"{pattern_name} detected. Wait for confirmation.",
                    'entry': 'Next Candle Close',
                    'sl': 'Pending',
                    'tp': 'Pending'
                }
            else:
                current_advice = {
                    'action': 'WAIT',
                    'reason': 'No clear signal',
                    'entry': '-',
                    'sl': '-',
                    'tp': '-'
                }


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
            'future_forecast': future_forecast,
            'ml_recommendation': ml_recommendation,
            'current_equity': round(current_equity, 2),
            'current_margin': round(current_margin, 2),
            'current_free_margin': round(current_free_margin, 2),
            'current_margin_level': round(current_margin_level, 2),
            'current_advice': current_advice
        }

if __name__ == "__main__":
    sim = TradingSimulation()
    result = sim.run()
    print(f"Simulation Complete. Final Balance: {result['final_balance']}")
    print(f"Total Trades: {result['total_trades']}")
    print(f"Advice: {result['current_advice']}")
