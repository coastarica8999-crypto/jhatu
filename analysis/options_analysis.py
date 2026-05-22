"""
Options Chain Analysis Module
Analyzes options data for market sentiment
"""

import pandas as pd
import numpy as np
from data.fetch_data import OptionsChainFetcher
import logging

logger = logging.getLogger(__name__)


class OptionsAnalysis:
    """Analyze options chain data"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.fetcher = OptionsChainFetcher(symbol)
        self.options_data = None
    
    def fetch_and_analyze(self):
        """Fetch and analyze options data"""
        try:
            self.options_data = self.fetcher.fetch_options_data()
            
            if self.options_data is None:
                logger.warning("No options data available")
                return None
            
            analysis = {
                'put_call_ratio': self.fetcher.calculate_put_call_ratio(self.options_data),
                'iv_skew': self.fetcher.calculate_iv_skew(self.options_data),
                'support_resistance': self.fetcher.get_support_resistance(self.options_data),
                'sentiment': self.get_sentiment_score()
            }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing options chain: {e}")
            return None
    
    def get_sentiment_score(self):
        """Generate sentiment score from options data"""
        if self.options_data is None:
            return None
        
        calls = self.options_data['calls']
        puts = self.options_data['puts']
        
        # Put/Call OI ratio
        pc_ratio = puts['openInterest'].sum() / calls['openInterest'].sum()
        
        # IV Ratio
        iv_ratio = puts['impliedVolatility'].mean() / calls['impliedVolatility'].mean()
        
        # Volume ratio
        vol_ratio = puts['volume'].sum() / calls['volume'].sum()
        
        # Composite sentiment
        sentiment_score = (pc_ratio + iv_ratio + vol_ratio) / 3
        
        if sentiment_score > 1.2:
            sentiment = 'Bearish'
        elif sentiment_score < 0.8:
            sentiment = 'Bullish'
        else:
            sentiment = 'Neutral'
        
        return {
            'pc_ratio': pc_ratio,
            'iv_ratio': iv_ratio,
            'vol_ratio': vol_ratio,
            'sentiment_score': sentiment_score,
            'sentiment': sentiment
        }
    
    def get_max_pain(self):
        """Calculate max pain level"""
        if self.options_data is None:
            return None
        
        calls = self.options_data['calls']
        puts = self.options_data['puts']
        
        # Merge calls and puts
        all_strikes = pd.concat([
            calls[['strike', 'openInterest']].rename(columns={'openInterest': 'call_oi'}),
            puts[['strike', 'openInterest']].rename(columns={'openInterest': 'put_oi'})
        ], axis=1)
        
        all_strikes['call_oi'] = all_strikes['call_oi'].fillna(0)
        all_strikes['put_oi'] = all_strikes['put_oi'].fillna(0)
        
        # Calculate profit/loss at each strike
        all_strikes['total_oi'] = all_strikes['call_oi'] + all_strikes['put_oi']
        
        # Max pain is where total loss is highest
        max_pain_strike = all_strikes.loc[all_strikes['total_oi'].idxmax(), 'strike']
        
        return max_pain_strike
