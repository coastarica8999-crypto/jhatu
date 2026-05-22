"""
Signal Generator Module
Combines multiple indicators to generate trading signals
"""

import pandas as pd
import numpy as np
from config import SIGNALS_CONFIG, SIGNAL_WEIGHTS
from indicators.moving_averages import MovingAverages
from indicators.oscillators import Oscillators
from indicators.volatility import VolatilityIndicators
from indicators.trend import TrendIndicators


class SignalGenerator:
    """Generate trading signals from technical indicators"""
    
    def __init__(self, data):
        self.data = data
        self.signals = pd.DataFrame(index=data.index)
    
    def generate_rsi_signal(self):
        """Generate RSI signal"""
        rsi = Oscillators.rsi(self.data, SIGNALS_CONFIG['rsi_period'])
        
        signal = np.where(rsi < SIGNALS_CONFIG['rsi_oversold'], 1, 
                 np.where(rsi > SIGNALS_CONFIG['rsi_overbought'], -1, 0))
        
        self.signals['RSI'] = signal
        return signal
    
    def generate_macd_signal(self):
        """Generate MACD signal"""
        macd, signal_line, histogram = Oscillators.macd(
            self.data,
            SIGNALS_CONFIG['macd_fast'],
            SIGNALS_CONFIG['macd_slow'],
            SIGNALS_CONFIG['macd_signal']
        )
        
        signal = np.where(histogram > 0, 1, -1)
        
        self.signals['MACD'] = signal
        return signal
    
    def generate_sma_cross_signal(self):
        """Generate SMA crossover signal"""
        sma_fast, sma_slow, cross = MovingAverages.golden_cross(
            self.data,
            SIGNALS_CONFIG['sma_fast'],
            SIGNALS_CONFIG['sma_slow']
        )
        
        self.signals['SMA_Cross'] = cross
        return cross
    
    def generate_bollinger_signal(self):
        """Generate Bollinger Bands signal"""
        upper, middle, lower = VolatilityIndicators.bollinger_bands(
            self.data,
            SIGNALS_CONFIG['bollinger_period'],
            SIGNALS_CONFIG['bollinger_std']
        )
        
        signal = np.where(self.data['close'] < lower, 1,
                 np.where(self.data['close'] > upper, -1, 0))
        
        self.signals['Bollinger'] = signal
        return signal
    
    def generate_stochastic_signal(self):
        """Generate Stochastic signal"""
        k, d = Oscillators.stochastic(
            self.data,
            SIGNALS_CONFIG['stochastic_period'],
            SIGNALS_CONFIG['stochastic_smooth']
        )
        
        signal = np.where(k < 20, 1,
                 np.where(k > 80, -1, 0))
        
        self.signals['Stochastic'] = signal
        return signal
    
    def generate_adx_signal(self):
        """Generate ADX trend signal"""
        adx, plus_di, minus_di = TrendIndicators.adx(
            self.data,
            SIGNALS_CONFIG['adx_period']
        )
        
        signal = np.where((adx > SIGNALS_CONFIG['adx_threshold']) & (plus_di > minus_di), 1,
                 np.where((adx > SIGNALS_CONFIG['adx_threshold']) & (minus_di > plus_di), -1, 0))
        
        self.signals['ADX'] = signal
        return signal
    
    def generate_volume_signal(self):
        """Generate volume-based signal"""
        if 'volume' not in self.data.columns:
            self.signals['Volume'] = 0
            return np.zeros(len(self.data))
        
        volume_ma = self.data['volume'].rolling(window=20).mean()
        signal = np.where(self.data['volume'] > volume_ma * 1.5, 1, 0)
        
        self.signals['Volume'] = signal
        return signal
    
    def generate_all_signals(self):
        """Generate all signals"""
        self.generate_rsi_signal()
        self.generate_macd_signal()
        self.generate_sma_cross_signal()
        self.generate_bollinger_signal()
        self.generate_stochastic_signal()
        self.generate_adx_signal()
        self.generate_volume_signal()
        
        # Calculate combined signal using weighted average
        weights = SIGNAL_WEIGHTS
        combined = (
            self.signals['RSI'] * weights['rsi'] +
            self.signals['MACD'] * weights['macd'] +
            self.signals['SMA_Cross'] * weights['sma_cross'] +
            self.signals['Bollinger'] * weights['bollinger'] +
            self.signals['Stochastic'] * weights['stochastic'] +
            self.signals['ADX'] * weights['adx'] +
            self.signals['Volume'] * weights['volume']
        )
        
        self.signals['Combined'] = combined
        
        return self.signals
    
    def get_signal_summary(self):
        """Get latest signal summary"""
        latest = self.signals.iloc[-1]
        
        bullish_count = (latest[:-1] > 0).sum()
        bearish_count = (latest[:-1] < 0).sum()
        neutral_count = (latest[:-1] == 0).sum()
        
        return {
            'signals': {
                'RSI': 'Bullish' if latest['RSI'] > 0 else ('Bearish' if latest['RSI'] < 0 else 'Neutral'),
                'MACD': 'Bullish' if latest['MACD'] > 0 else 'Bearish',
                'SMA_Cross': 'Bullish' if latest['SMA_Cross'] > 0 else 'Bearish',
                'Bollinger': 'Bullish' if latest['Bollinger'] > 0 else ('Bearish' if latest['Bollinger'] < 0 else 'Neutral'),
                'Stochastic': 'Bullish' if latest['Stochastic'] > 0 else ('Bearish' if latest['Stochastic'] < 0 else 'Neutral'),
                'ADX': 'Bullish' if latest['ADX'] > 0 else ('Bearish' if latest['ADX'] < 0 else 'Neutral'),
            },
            'counts': {
                'bullish': bullish_count,
                'bearish': bearish_count,
                'neutral': neutral_count
            },
            'combined_signal': 'Bullish' if latest['Combined'] > 0 else ('Bearish' if latest['Combined'] < 0 else 'Neutral'),
            'confidence': abs(latest['Combined'])
        }
