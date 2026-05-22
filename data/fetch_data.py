"""
Data Fetching Module
Handles fetching market data and options chain data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and preprocess market data"""
    
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def fetch_data(self):
        """Fetch OHLCV data from yfinance"""
        try:
            logger.info(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
            self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            logger.info(f"Downloaded {len(self.data)} candles")
            return self.data
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def preprocess_data(self):
        """Clean and preprocess data"""
        if self.data is None:
            self.fetch_data()
        
        df = self.data.copy()
        
        # Reset index to make Date a column
        if not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index()
        
        # Rename columns
        df.columns = [col.lower() for col in df.columns]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate returns
        df['Returns'] = df['close'].pct_change()
        
        # Calculate log returns
        df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate typical price
        df['Typical_Price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate price change
        df['Price_Change'] = df['close'] - df['open']
        
        # Calculate volume ratio
        if 'volume' in df.columns:
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        
        # Calculate volatility
        df['Volatility'] = df['returns'].rolling(window=20).std()
        
        return df
    
    def get_latest_data(self, n=60):
        """Get latest n candles"""
        if self.data is None:
            self.fetch_data()
        return self.preprocess_data().tail(n)


class OptionsChainFetcher:
    """Fetch and analyze options chain data"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def fetch_options_data(self):
        """Fetch options chain data"""
        try:
            expirations = self.ticker.options
            if not expirations:
                logger.warning(f"No options data available for {self.symbol}")
                return None
            
            # Get nearest expiration
            nearest_expiry = expirations[0]
            options_data = self.ticker.option_chain(nearest_expiry)
            
            return {
                'expiration': nearest_expiry,
                'calls': options_data.calls,
                'puts': options_data.puts
            }
        except Exception as e:
            logger.error(f"Error fetching options data: {e}")
            return None
    
    def calculate_put_call_ratio(self, options_data):
        """Calculate put/call ratio"""
        if options_data is None:
            return None
        
        calls_oi = options_data['calls']['openInterest'].sum()
        puts_oi = options_data['puts']['openInterest'].sum()
        
        if calls_oi == 0:
            return np.nan
        
        return puts_oi / calls_oi
    
    def calculate_iv_skew(self, options_data):
        """Calculate implied volatility skew"""
        if options_data is None:
            return None
        
        calls_iv = options_data['calls']['impliedVolatility'].mean()
        puts_iv = options_data['puts']['impliedVolatility'].mean()
        
        return puts_iv - calls_iv
    
    def get_support_resistance(self, options_data, moneyness=0.02):
        """Extract support/resistance from options data"""
        if options_data is None:
            return None
        
        current_price = self.ticker.info.get('currentPrice', 0)
        
        # Find ATM and near-ATM strikes
        calls = options_data['calls']
        puts = options_data['puts']
        
        # Support from put open interest
        support_strikes = puts[
            (puts['strike'] > current_price * (1 - moneyness)) & 
            (puts['strike'] < current_price)
        ].sort_values('openInterest', ascending=False)
        
        support = support_strikes['strike'].iloc[0] if len(support_strikes) > 0 else None
        
        # Resistance from call open interest
        resistance_strikes = calls[
            (calls['strike'] > current_price) & 
            (calls['strike'] < current_price * (1 + moneyness))
        ].sort_values('openInterest', ascending=False)
        
        resistance = resistance_strikes['strike'].iloc[0] if len(resistance_strikes) > 0 else None
        
        return {'support': support, 'resistance': resistance}
