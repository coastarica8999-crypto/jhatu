#!/usr/bin/env python3
"""
Market Prediction Framework
Combines technical analysis, LSTM prediction, and options chain analysis
"""

import logging
from config import DATA_CONFIG, LSTM_CONFIG, SIGNALS_CONFIG
from data.fetch_data import DataFetcher, OptionsChainFetcher
from indicators.moving_averages import MovingAverages
from indicators.oscillators import Oscillators
from indicators.volatility import VolatilityIndicators
from indicators.trend import TrendIndicators
from models.lstm_predictor import LSTMPredictor
from analysis.signal_generator import SignalGenerator
from analysis.options_analysis import OptionsAnalysis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketPredictor:
    """Main prediction engine"""
    
    def __init__(self, symbol=DATA_CONFIG['symbol'], 
                 start_date=DATA_CONFIG['start_date'],
                 end_date=DATA_CONFIG['end_date']):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data_fetcher = DataFetcher(symbol, start_date, end_date)
        self.data = None
        self.signals = None
        self.lstm_predictor = None
        self.options_analysis = None
    
    def prepare_data(self):
        """Fetch and preprocess data"""
        logger.info("=" * 50)
        logger.info("STEP 1: Preparing Data")
        logger.info("=" * 50)
        
        self.data_fetcher.fetch_data()
        self.data = self.data_fetcher.preprocess_data()
        logger.info(f"Data prepared: {len(self.data)} candles")
        return self.data
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 2: Calculating Technical Indicators")
        logger.info("=" * 50)
        
        # Moving Averages
        self.data['SMA20'] = MovingAverages.sma(self.data, 20)
        self.data['SMA50'] = MovingAverages.sma(self.data, 50)
        self.data['SMA200'] = MovingAverages.sma(self.data, 200)
        self.data['EMA12'] = MovingAverages.ema(self.data, 12)
        self.data['EMA26'] = MovingAverages.ema(self.data, 26)
        
        # Oscillators
        self.data['RSI'] = Oscillators.rsi(self.data, 14)
        macd, signal, hist = Oscillators.macd(self.data)
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = signal
        self.data['MACD_Hist'] = hist
        
        # Volatility
        bb_up, bb_mid, bb_low = VolatilityIndicators.bollinger_bands(self.data)
        self.data['BB_Upper'] = bb_up
        self.data['BB_Middle'] = bb_mid
        self.data['BB_Lower'] = bb_low
        self.data['ATR'] = VolatilityIndicators.atr(self.data)
        
        # Trend
        adx, plus_di, minus_di = TrendIndicators.adx(self.data)
        self.data['ADX'] = adx
        
        logger.info("Technical indicators calculated")
        return self.data
    
    def generate_signals(self):
        """Generate trading signals"""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 3: Generating Trading Signals")
        logger.info("=" * 50)
        
        signal_gen = SignalGenerator(self.data)
        self.signals = signal_gen.generate_all_signals()
        
        logger.info("Trading signals generated")
        logger.info(f"\nLatest Signal Summary:")
        summary = signal_gen.get_signal_summary()
        for key, value in summary['signals'].items():
            logger.info(f"  {key}: {value}")
        
        return self.signals
    
    def train_lstm_model(self, epochs=LSTM_CONFIG['epochs']):
        """Train LSTM prediction model"""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 4: Training LSTM Model")
        logger.info("=" * 50)
        
        close_prices = self.data['Close'].values
        
        self.lstm_predictor = LSTMPredictor(
            sequence_length=LSTM_CONFIG['sequence_length'],
            lookforward=LSTM_CONFIG['lookforward']
        )
        
        X_train, X_test, y_train, y_test = self.lstm_predictor.prepare_data(
            close_prices, 
            test_split=LSTM_CONFIG['test_split']
        )
        
        self.lstm_predictor.build_model(
            layers=LSTM_CONFIG['layers'],
            dropout=LSTM_CONFIG['dropout']
        )
        
        self.lstm_predictor.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs,
            batch_size=LSTM_CONFIG['batch_size']
        )
        
        # Evaluate
        loss, mae = self.lstm_predictor.evaluate(X_test, y_test)
        
        return self.lstm_predictor
    
    def analyze_options_chain(self):
        """Analyze options chain data"""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 5: Analyzing Options Chain")
        logger.info("=" * 50)
        
        self.options_analysis = OptionsAnalysis(self.symbol)
        analysis = self.options_analysis.fetch_and_analyze()
        
        if analysis:
            logger.info(f"\nOptions Analysis Results:")
            logger.info(f"  Put/Call Ratio: {analysis['put_call_ratio']:.2f}")
            logger.info(f"  IV Skew: {analysis['iv_skew']:.4f}")
            if analysis['sentiment']:
                logger.info(f"  Market Sentiment: {analysis['sentiment']['sentiment']}")
        
        return analysis
    
    def predict_next_candle(self):
        """Predict next day's candle"""
        logger.info("\n" + "=" * 50)
        logger.info("STEP 6: Predicting Next Candle")
        logger.info("=" * 50)
        
        if self.lstm_predictor is None:
            logger.warning("LSTM model not trained yet")
            return None
        
        last_close = self.data['Close'].iloc[-1]
        predicted_close = self.lstm_predictor.predict_next_candle(self.data['Close'])
        
        # Get signal direction
        combined_signal = self.signals['Combined'].iloc[-1] if 'Combined' in self.signals else 0
        
        # Estimate high and low based on ATR
        atr = self.data['ATR'].iloc[-1]
        predicted_high = predicted_close + (atr * 0.7)
        predicted_low = predicted_close - (atr * 0.7)
        
        prediction = {
            'open': last_close,
            'close': predicted_close,
            'high': predicted_high,
            'low': predicted_low,
            'signal': 'Bullish' if combined_signal > 0 else ('Bearish' if combined_signal < 0 else 'Neutral'),
            'change_percent': ((predicted_close - last_close) / last_close) * 100
        }
        
        logger.info(f"\nNext Candle Prediction:")
        logger.info(f"  Current Close: {last_close:.2f}")
        logger.info(f"  Predicted Close: {predicted_close:.2f}")
        logger.info(f"  Predicted High: {predicted_high:.2f}")
        logger.info(f"  Predicted Low: {predicted_low:.2f}")
        logger.info(f"  Signal: {prediction['signal']}")
        logger.info(f"  Expected Change: {prediction['change_percent']:.2f}%")
        
        return prediction
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        logger.info("\n\n")
        logger.info("╔" + "=" * 48 + "╗")
        logger.info("║  MARKET PREDICTION FRAMEWORK - FULL ANALYSIS     ║")
        logger.info("╚" + "=" * 48 + "╝")
        
        self.prepare_data()
        self.calculate_indicators()
        self.generate_signals()
        self.train_lstm_model()
        self.analyze_options_chain()
        prediction = self.predict_next_candle()
        
        logger.info("\n" + "=" * 50)
        logger.info("Analysis Complete")
        logger.info("=" * 50)
        
        return prediction


if __name__ == "__main__":
    predictor = MarketPredictor()
    prediction = predictor.run_full_analysis()
