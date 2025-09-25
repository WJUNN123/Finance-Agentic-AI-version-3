# Crypto Investment Analyzer

A comprehensive cryptocurrency analysis tool powered by machine learning and technical analysis. This application provides investment recommendations, price predictions, and risk assessments for major cryptocurrencies.

## Features

- **Real-time Market Data**: Fetches current prices, market cap, and trading volume
- **Technical Analysis**: RSI, MACD, Bollinger Bands, moving averages, and trend detection
- **Sentiment Analysis**: News analysis using FinBERT for market sentiment assessment
- **LSTM Price Prediction**: 7-day price forecasts using trained neural networks
- **Risk Assessment**: Comprehensive risk analysis with stop-loss and take-profit recommendations
- **Interactive Dashboard**: Clean, professional Streamlit interface

## Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- Binance Coin (BNB)
- Ripple (XRP)
- Cardano (ADA)
- Dogecoin (DOGE)
- Avalanche (AVAX)
- Chainlink (LINK)
- Polkadot (DOT)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-analyzer.git
cd crypto-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

## Usage

### Streamlit Web App

Run the web application:
```bash
streamlit run streamlit_app.py
```

### Python Script

```python
from crypto_analyzer import CryptoAnalyzer

# Initialize analyzer
analyzer = CryptoAnalyzer()

# Analyze a cryptocurrency
result = analyzer.analyze("ethereum", "ETH price analysis")
print(result)
```

## Project Structure

```
crypto-analyzer/
├── config.py                 # Configuration settings
├── data_fetcher.py           # Data fetching from APIs/CSV
├── sentiment_analyzer.py     # News and sentiment analysis
├── technical_analyzer.py     # Technical indicators
├── prediction_engine.py      # LSTM price prediction
├── analysis_engine.py        # Decision making and confidence
├── crypto_analyzer.py        # Main analysis orchestrator
├── streamlit_app.py          # Streamlit web interface
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## API Keys Required

- **Google Generative AI**: For advanced analysis (optional)
  - Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
  - Set as environment variable: `GOOGLE_API_KEY`

## Data Sources

- **Market Data**: CoinGecko API (free tier)
- **News Data**: Multiple RSS feeds from crypto news sites
- **Historical Data**: CSV upload support or API fallback

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Set environment variables in Streamlit secrets
4. Deploy

### Local Docker (Optional)

```bash
# Build image
docker build -t crypto-analyzer .

# Run container
docker run -p 8501:8501 -e GOOGLE_API_KEY="your_key" crypto-analyzer
```

## Disclaimer

**This application is for educational purposes only and does not constitute financial advice. Cryptocurrency investments carry high risk and you should always do your own research before making any investment decisions.**

## Features Overview

### Technical Analysis
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence) 
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Support/Resistance levels
- Volatility metrics
- Market regime detection

### Sentiment Analysis
- Real-time news aggregation
- FinBERT sentiment scoring
- Crypto-specific keyword analysis
- Multiple news source integration

### Price Prediction
- LSTM neural networks
- Multi-timeframe analysis
- Model persistence and caching
- Confidence-adjusted forecasts

### Risk Management
- Dynamic stop-loss calculation
- Take-profit recommendations
- Position sizing suggestions
- Comprehensive risk scoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the code comments

## Changelog

### v1.0.0
- Initial release
- Basic technical analysis
- LSTM price prediction
- Streamlit interface
- Multi-cryptocurrency support