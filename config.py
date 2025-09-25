"""
Configuration settings for the Crypto Analysis Application
"""
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict

@dataclass
class Config:
    # API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    COINGECKO_API_URL: str = "https://api.coingecko.com/api/v3"
    
    # Cache and Performance
    MAX_RETRIES: int = 3
    CACHE_DURATION: int = 300  # 5 minutes
    REQUEST_TIMEOUT: int = 10
    
    # Analysis Parameters
    CONFIDENCE_THRESHOLD: float = 70.0
    RISK_FREE_RATE: float = 0.05  # 5% annual risk-free rate
    LSTM_LOOKBACK_DAYS: int = 60
    HISTORICAL_DAYS: int = 365
    FORECAST_DAYS: int = 7
    
    # UI Configuration
    PAGE_TITLE: str = "🚀 Crypto Investment Analyzer"
    PAGE_ICON: str = "🔮"
    LAYOUT: str = "wide"

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    EXTREME = "Extreme"

class Action(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"
    WAIT = "Wait"

# Supported cryptocurrencies mapping
SUPPORTED_CRYPTOS = {
    # Bitcoin
    'btc': 'bitcoin',
    'bitcoin': 'bitcoin',
    # Ethereum
    'eth': 'ethereum',
    'ethereum': 'ethereum',
    # Solana
    'sol': 'solana',
    'solana': 'solana',
    # Binance Coin
    'bnb': 'binancecoin',
    'binance coin': 'binancecoin',
    'binancecoin': 'binancecoin',
    # Ripple
    'xrp': 'ripple',
    'ripple': 'ripple',
    # Cardano
    'ada': 'cardano',
    'cardano': 'cardano',
    # Dogecoin
    'doge': 'dogecoin',
    'dogecoin': 'dogecoin',
    # Avalanche
    'avax': 'avalanche-2',
    'avalanche': 'avalanche-2',
    # Chainlink
    'link': 'chainlink',
    'chainlink': 'chainlink',
    # Polkadot
    'dot': 'polkadot',
    'polkadot': 'polkadot',
}

# Market cap rankings for major cryptocurrencies
MARKET_RANKS = {
    'BITCOIN': 1,
    'ETHEREUM': 2,
    'BINANCECOIN': 4,
    'SOLANA': 5,
    'RIPPLE': 7,
    'CARDANO': 8,
    'DOGECOIN': 9,
    'AVALANCHE-2': 10,
    'CHAINLINK': 15,
    'POLKADOT': 12
}

# RSS feeds for crypto news
RSS_FEEDS = [
    "https://newsbtc.com/feed/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptonews.com/news/feed/",
    "https://www.theblockcrypto.com/rss.xml"
]