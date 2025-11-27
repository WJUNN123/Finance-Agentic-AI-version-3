"""
Data fetching utilities for cryptocurrency market data (API only)
"""
import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional
from config import Config, SUPPORTED_CRYPTOS, MARKET_RANKS

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.cache = {}
        self.realtime_cache = {}

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        return (datetime.now() - self.cache[key]['timestamp']).seconds < Config.CACHE_DURATION

    def _get_realtime_data(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time data from CoinGecko API"""
        coingecko_id = SUPPORTED_CRYPTOS.get(symbol.lower(), symbol.lower())
        cache_key = f"realtime_{coingecko_id}"

        # Use cache if available
        if cache_key in self.realtime_cache:
            cached_time = self.realtime_cache[cache_key]['timestamp']
            if (datetime.now() - cached_time).seconds < 300:  # 5 minutes
                return self.realtime_cache[cache_key]['data']

        try:
            url = f"{Config.COINGECKO_API_URL}/simple/price"
            params = {
                'ids': coingecko_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if coingecko_id in data:
                realtime_data = {
                    'current_price': data[coingecko_id]['usd'],
                    'price_change_24h': data[coingecko_id].get('usd_24h_change', 0),
                    'market_cap': data[coingecko_id].get('usd_market_cap', 0),
                    'volume_24h': data[coingecko_id].get('usd_24h_vol', 0)
                }

                self.realtime_cache[cache_key] = {
                    'data': realtime_data,
                    'timestamp': datetime.now()
                }
                return realtime_data

        except Exception as e:
            logger.warning(f"Failed to fetch real-time data: {e}")
        return None

    def fetch_crypto_data(self, symbol: str = "ethereum") -> Dict:
        """Fetch latest crypto data from API"""
        cache_key = f"crypto_{symbol}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        try:
            coingecko_id = SUPPORTED_CRYPTOS.get(symbol.lower(), symbol.lower())
            url = f"{Config.COINGECKO_API_URL}/coins/{coingecko_id}"
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            # Cache
            self.cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
            return data

        except Exception as e:
            logger.error(f"Failed to fetch crypto data for {symbol}: {e}")
            return {}

    def fetch_historical_data(self, symbol: str = "ethereum", days: int = 90) -> pd.DataFrame:
        """Fetch historical price data from CoinGecko API"""
        cache_key = f"historical_{symbol}_{days}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        try:
            coingecko_id = SUPPORTED_CRYPTOS.get(symbol.lower(), symbol.lower())
            url = f"{Config.COINGECKO_API_URL}/coins/{coingecko_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': days}
            response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame({
                'price': [p[1] for p in data['prices']],
                'volume': [v[1] for v in data['total_volumes']],
                'market_cap': [m[1] for m in data['market_caps']]
            })
            df.index = pd.to_datetime([p[0] for p in data['prices']], unit='ms')
            df.index.name = 'date'

            # Add simplified OHLC
            df['open'] = df['price']
            df['high'] = df['price']
            df['low'] = df['price']

            # Cache
            self.cache[cache_key] = {'data': df, 'timestamp': datetime.now()}
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()
