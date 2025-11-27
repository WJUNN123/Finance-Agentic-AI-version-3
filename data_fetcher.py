"""
Data fetching utilities for cryptocurrency market data (API-only)
"""
import requests
import pandas as pd
import logging
from datetime import datetime
from typing import Dict

from config import Config, SUPPORTED_CRYPTOS, MARKET_RANKS

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.cache = {}
        self.realtime_cache = {}

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self.cache:
            return False
        return (datetime.now() - self.cache[key]['timestamp']).seconds < Config.CACHE_DURATION

    def _get_realtime_data(self, symbol: str) -> Dict:
        """Fetch live data from CoinGecko API"""
        coingecko_id = SUPPORTED_CRYPTOS.get(symbol.lower(), symbol.lower())
        cache_key = f"realtime_{coingecko_id}"

        # Check cache
        if cache_key in self.realtime_cache:
            if (datetime.now() - self.realtime_cache[cache_key]['timestamp']).seconds < 300:
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
            resp = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if coingecko_id in data:
                realtime = {
                    'current_price': data[coingecko_id]['usd'],
                    'price_change_24h': data[coingecko_id].get('usd_24h_change', 0),
                    'market_cap': data[coingecko_id].get('usd_market_cap', 0),
                    'volume_24h': data[coingecko_id].get('usd_24h_vol', 0),
                    'market_cap_rank': MARKET_RANKS.get(symbol.lower())
                }
                self.realtime_cache[cache_key] = {'data': realtime, 'timestamp': datetime.now()}
                return realtime
        except Exception as e:
            logger.warning(f"Failed to fetch realtime data for {symbol}: {e}")

        return {}

    def fetch_crypto_data(self, symbol: str) -> Dict:
        cache_key = f"crypto_{symbol}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        data = self._get_realtime_data(symbol)
        if data:
            self.cache[cache_key] = {'data': {'market_data': data, 'market_cap_rank': data.get('market_cap_rank')}, 'timestamp': datetime.now()}
        return self.cache.get(cache_key, {}).get('data', {})

    def fetch_historical_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical price data via CoinGecko API"""
        cache_key = f"historical_{symbol}_{days}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        try:
            coingecko_id = SUPPORTED_CRYPTOS.get(symbol.lower(), symbol.lower())
            url = f"{Config.COINGECKO_API_URL}/coins/{coingecko_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': days}
            resp = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            df = pd.DataFrame({
                'price': [p[1] for p in data['prices']],
                'volume': [v[1] for v in data['total_volumes']],
                'market_cap': [m[1] for m in data['market_caps']]
            })
            df.index = pd.to_datetime([p[0] for p in data['prices']], unit='ms')
            df.index.name = 'date'
            df['high'] = df['price']
            df['low'] = df['price']
            df['open'] = df['price']

            self.cache[cache_key] = {'data': df, 'timestamp': datetime.now()}
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()
