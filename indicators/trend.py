"""
Trend Indicators Module
Implements trend-following indicators
"""

import pandas as pd
import numpy as np


class TrendIndicators:
    """Calculate trend indicators"""
    
    @staticmethod
    def adx(data, period=14):
        """Average Directional Index"""
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff() * -1
        
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([
            pd.Series(high_low),
            pd.Series(high_close),
            pd.Series(low_close)
        ], axis=1).max(axis=1)
        
        atr = true_range.rolling(window=period).mean()
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        
        dx = 100 * (di_diff / di_sum)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def ichimoku(data, fast=9, slow=26, ahead=52):
        """Ichimoku Cloud"""
        high_9 = data['high'].rolling(window=fast).max()
        low_9 = data['low'].rolling(window=fast).min()
        
        high_26 = data['high'].rolling(window=slow).max()
        low_26 = data['low'].rolling(window=slow).min()
        
        tenkan = (high_9 + low_9) / 2
        kijun = (high_26 + low_26) / 2
        
        senkou_a = ((tenkan + kijun) / 2).shift(ahead)
        
        high_52 = data['high'].rolling(window=ahead).max()
        low_52 = data['low'].rolling(window=ahead).min()
        senkou_b = ((high_52 + low_52) / 2).shift(ahead)
        
        chikou = data['close'].shift(-ahead)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }
    
    @staticmethod
    def pivot_points(data):
        """Pivot Points Support & Resistance"""
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        close = data['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            's1': s1,
            's2': s2
        }
    
    @staticmethod
    def obv(data):
        """On-Balance Volume"""
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def vpt(data):
        """Volume Price Trend"""
        returns = data['close'].pct_change()
        vpt = (returns * data['volume']).cumsum()
        return vpt
    
    @staticmethod
    def aroon(data, period=25):
        """Aroon Indicator"""
        high_idx = data['high'].rolling(window=period).apply(lambda x: x.argmax())
        low_idx = data['low'].rolling(window=period).apply(lambda x: x.argmin())
        
        aroon_up = ((period - high_idx) / period) * 100
        aroon_down = ((period - low_idx) / period) * 100
        
        return aroon_up, aroon_down
