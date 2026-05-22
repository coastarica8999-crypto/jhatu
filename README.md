# Market Prediction Framework - Jhatu

**A comprehensive Python framework for market prediction using technical analysis, machine learning, and options chain data.**

## 🎯 Overview

Jhatu combines cutting-edge technologies to predict stock market movements:

- **15+ Technical Indicators** - Moving averages, oscillators, volatility, and trend indicators
- **Deep Learning (LSTM)** - Neural network trained on historical price data
- **Options Chain Analysis** - Market sentiment from put/call ratios and implied volatility
- **Multi-Indicator Signals** - Weighted composite signals from 7 different indicators
- **BSE Sensex Ready** - Configured for Indian stock market analysis

## 📦 Installation

```bash
git clone https://github.com/coastarica8999-crypto/jhatu.git
cd jhatu
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from main import MarketPredictor

# Initialize predictor
predictor = MarketPredictor(symbol="^BSESN")

# Run complete analysis pipeline
prediction = predictor.run_full_analysis()

# Output includes:
# - Predicted candle (Open, High, Low, Close)
# - Signal direction (Bullish/Bearish/Neutral)
# - Confidence score
# - Support & Resistance levels
# - Market sentiment
```

## 🔧 Framework Components

### 1. Data Pipeline (`data/`)
- **fetch_data.py**: Market data and options chain fetching
  - Automatic data cleaning and preprocessing
  - Feature engineering (returns, volatility, volume ratios)
  - Options data fetcher for sentiment analysis

### 2. Technical Indicators (`indicators/`)

#### Moving Averages (`moving_averages.py`)
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Weighted Moving Average (WMA)
- Golden Cross / Death Cross
- Double EMA (DEMA)
- Triple EMA (TEMA)

#### Oscillators (`oscillators.py`)
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Rate of Change (ROC)
- Momentum
- Williams %R
- Commodity Channel Index (CCI)

#### Volatility (`volatility.py`)
- Bollinger Bands
- Average True Range (ATR)
- Keltner Channel
- SuperTrend
- Historical Volatility

#### Trend (`trend.py`)
- Average Directional Index (ADX)
- Ichimoku Cloud
- Pivot Points
- On-Balance Volume (OBV)
- Volume Price Trend (VPT)
- Aroon Indicator

### 3. Machine Learning (`models/`)
- **lstm_predictor.py**: LSTM neural network
  - 3-layer architecture
  - Dropout regularization
  - Early stopping
  - Multi-step ahead forecasting

### 4. Signal Generation (`analysis/`)
- **signal_generator.py**: Multi-indicator signal generation
  - 7 individual indicator signals
  - Weighted composite scoring
  - Confidence metrics
  
- **options_analysis.py**: Options chain sentiment
  - Put/Call ratio analysis
  - Implied Volatility skew
  - Max Pain detection
  - Market sentiment classification

### 5. Configuration (`config.py`)
- Customizable parameters for all indicators
- LSTM hyperparameters
- Signal weights
- Backtesting settings

## 📊 Analysis Pipeline (6 Steps)

```
Step 1: Prepare Data
  ↓ Fetch 273+ historical candles

Step 2: Calculate Indicators
  ↓ 15+ technical indicators

Step 3: Generate Signals
  ↓ 7-indicator composite signals

Step 4: Train LSTM
  ↓ Deep learning prediction model

Step 5: Analyze Options
  ↓ Put/call sentiment, IV skew

Step 6: Predict Next Candle
  ↓ Open, High, Low, Close + Signal + Confidence
```

## 📈 Output

```
╔════════════════════════════════════════════╗
║     NEXT CANDLE PREDICTION RESULTS         ║
╚════════════════════════════════════════════╝

Current Close: 75,415.35
Predicted Close: 75,650.22
Predicted High: 75,850.50
Predicted Low: 75,200.00

Signal: BULLISH
Confidence: 0.75 (75%)
Expected Change: +0.31%

Support Level: 75,200.00
Resistance Level: 75,850.00

Individual Signals:
  RSI: Bullish
  MACD: Bullish
  SMA Cross: Bullish
  Bollinger: Neutral
  Stochastic: Bullish
  ADX: Bullish

Options Sentiment: BULLISH
  Put/Call Ratio: 0.85
  IV Skew: -0.02
```

## 🎨 Customization

### Change Indicator Parameters

Edit `config.py`:

```python
SIGNALS_CONFIG = {
    'rsi_period': 14,  # Change RSI period
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    # ... other parameters
}
```

### Adjust Signal Weights

```python
SIGNAL_WEIGHTS = {
    'rsi': 0.25,  # Increase RSI importance
    'macd': 0.20,
    'sma_cross': 0.20,
    # ... other weights
}
```

### Modify LSTM Settings

```python
LSTM_CONFIG = {
    'sequence_length': 90,  # Look back 90 days
    'epochs': 50,  # More training iterations
    'layers': 4,  # Add more layers
    'dropout': 0.3,  # Increase regularization
}
```

## 📁 Project Structure

```
jhatu/
├── main.py                          # Main execution pipeline
├── config.py                        # Configuration file
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── data/
│   ├── __init__.py
│   └── fetch_data.py               # Data fetching
├── indicators/
│   ├── __init__.py
│   ├── moving_averages.py
│   ├── oscillators.py
│   ├── volatility.py
│   └── trend.py
├── models/
│   ├── __init__.py
│   └── lstm_predictor.py           # LSTM model
└── analysis/
    ├── __init__.py
    ├── signal_generator.py         # Signal generation
    └── options_analysis.py         # Options analysis
```

## 🔄 Usage Examples

### Basic Prediction

```python
from main import MarketPredictor

predictor = MarketPredictor()
prediction = predictor.run_full_analysis()

print(f"Next Close: {prediction['close']:.2f}")
print(f"Signal: {prediction['signal']}")
print(f"Change: {prediction['change_percent']:.2f}%")
```

### Step-by-Step Analysis

```python
predictor = MarketPredictor()

# Step 1: Prepare data
predictor.prepare_data()

# Step 2: Calculate indicators
predictor.calculate_indicators()

# Step 3: Generate signals
predictor.generate_signals()

# Step 4: Train model
predictor.train_lstm_model(epochs=50)

# Step 5: Analyze options
predictor.analyze_options_chain()

# Step 6: Get prediction
prediction = predictor.predict_next_candle()
```

### Access Individual Components

```python
from data.fetch_data import DataFetcher
from indicators.moving_averages import MovingAverages
from analysis.signal_generator import SignalGenerator

# Fetch data
data_fetcher = DataFetcher('^BSESN', '2025-09-01', '2026-05-22')
data = data_fetcher.prepare_data()

# Calculate indicators
sma20 = MovingAverages.sma(data, 20)
ema12 = MovingAverages.ema(data, 12)

# Generate signals
signal_gen = SignalGenerator(data)
signals = signal_gen.generate_all_signals()
```

## ⚙️ Technical Details

### LSTM Architecture

```
Input Layer: (60, 1) - 60 days of price data
    ↓
LSTM Layer 1: 50 units + Dropout(0.2)
    ↓
LSTM Layer 2: 50 units + Dropout(0.2)
    ↓
LSTM Layer 3: 50 units + Dropout(0.2)
    ↓
Dense Layer: 1 unit - Predicted price
```

### Signal Calculation

```
Combined Signal = 
    0.20 × RSI_Signal +
    0.20 × MACD_Signal +
    0.20 × SMA_Signal +
    0.15 × Bollinger_Signal +
    0.10 × Stochastic_Signal +
    0.10 × ADX_Signal +
    0.05 × Volume_Signal
```

### Options Sentiment

```
Market Sentiment Score = (PC_Ratio + IV_Ratio + Vol_Ratio) / 3

If Score > 1.2: BEARISH
If Score < 0.8: BULLISH
Else: NEUTRAL
```

## ⚠️ Disclaimer

**This framework is for educational and research purposes only.** It is NOT financial advice. Trading and investing involve significant risk. Past performance does not guarantee future results. Always:

- Conduct your own research
- Use proper risk management
- Never invest more than you can afford to lose
- Consult with a financial advisor

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests
- Add new indicators

## 📞 Support

For issues or questions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error logs and reproducible steps

## 🔗 References

- [Technical Analysis Library (TA-Lib)](https://github.com/mrjbq7/ta-lib)
- [Keras/TensorFlow Documentation](https://keras.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

---

**Made with ❤️ for traders and developers**

*Last Updated: May 22, 2026*
