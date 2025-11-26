"""
Data fetching utilities for cryptocurrency market data
"""
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import Config, SUPPORTED_CRYPTOS, MARKET_RANKS

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, csv_file_path: Optional[str] = None):
        self.csv_file_path = csv_file_path
        self.cache = {}
        self.df_cache = None
        self.realtime_cache = {}

    def _load_csv_data(self) -> pd.DataFrame:
        """Load and cache the CSV data"""
        if self.df_cache is None and self.csv_file_path:
            try:
                self.df_cache = pd.read_csv(self.csv_file_path)
                
                # Convert Date column to datetime
                if 'Date' in self.df_cache.columns:
                    self.df_cache['Date'] = pd.to_datetime(
                        self.df_cache['Date'], dayfirst=True
                    )
                
                # Standardize symbol names
                if 'Symbol' in self.df_cache.columns:
                    self.df_cache['Symbol_lower'] = self.df_cache['Symbol'].str.lower()
                
                logger.info(f"Loaded CSV data with {len(self.df_cache)} rows")
                
            except Exception as e:
                logger.error(f"Failed to load CSV file: {e}")
                self.df_cache = pd.DataFrame()
        
        return self.df_cache if self.df_cache is not None else pd.DataFrame()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        return (datetime.now() - self.cache[key]['timestamp']).seconds < Config.CACHE_DURATION

    def _get_realtime_data(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time data from CoinGecko API"""
        coingecko_mapping = {
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'solana': 'solana',
            'binancecoin': 'binancecoin',
            'ripple': 'ripple',
            'cardano': 'cardano',
            'dogecoin': 'dogecoin',
            'avalanche-2': 'avalanche-2',
            'chainlink': 'chainlink',
            'polkadot': 'polkadot'
        }

        coingecko_id = coingecko_mapping.get(symbol.lower(), symbol.lower())
        cache_key = f"realtime_{coingecko_id}"
        
        # Check cache first
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

            response = requests.get(
                url, params=params, timeout=Config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            if coingecko_id in data:
                realtime_data = {
                    'current_price': data[coingecko_id]['usd'],
                    'price_change_24h': data[coingecko_id].get('usd_24h_change', 0),
                    'market_cap': data[coingecko_id].get('usd_market_cap', 0),
                    'volume_24h': data[coingecko_id].get('usd_24h_vol', 0)
                }

                # Cache the result
                self.realtime_cache[cache_key] = {
                    'data': realtime_data,
                    'timestamp': datetime.now()
                }
                
                return realtime_data

        except Exception as e:
            logger.warning(f"Failed to fetch real-time data: {e}")

        return None

    def fetch_crypto_data(self, symbol: str = "ethereum") -> Dict:
        """Fetch cryptocurrency data from CSV or API"""
        cache_key = f"crypto_{symbol}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        # Try CSV first if available
        if self.csv_file_path:
            return self._fetch_from_csv(symbol)
        
        # Fallback to API
        return self._fetch_from_api(symbol)

    def _fetch_from_csv(self, symbol: str) -> Dict:
        """Fetch data from CSV file"""
        df = self._load_csv_data()
        if df.empty:
            logger.error("CSV data is empty or failed to load")
            return {}

        symbol_mappings = {
            'ethereum': 'ETHEREUM',
            'bitcoin': 'BITCOIN',
            'btc': 'BITCOIN',
            'eth': 'ETHEREUM',
            'solana': 'SOLANA',
            'sol': 'SOLANA',
            'binancecoin': 'BINANCECOIN',
            'bnb': 'BINANCECOIN',
            'ripple': 'RIPPLE',
            'xrp': 'RIPPLE'
        }

        search_symbol = symbol_mappings.get(symbol, symbol.upper())
        symbol_data = df[df['Symbol'] == search_symbol]

        if symbol_data.empty:
            symbol_data = df[df['Symbol_lower'] == symbol.lower()]

        if symbol_data.empty:
            logger.warning(f"No data found for symbol: {symbol}")
            return {}

        # Get the most recent data point
        latest_data = symbol_data.sort_values('Date').iloc[-1]

        # Calculate 24h change
        price_change_24h = 0.0
        if len(symbol_data) >= 2:
            previous_data = symbol_data.sort_values('Date').iloc[-2]
            price_change_24h = (
                (latest_data['Close'] - previous_data['Close']) 
                / previous_data['Close'] * 100
            )

        # Format data
        market_data = {
            'market_data': {
                'current_price': {'usd': latest_data['Close']},
                'market_cap': {'usd': latest_data['Marketcap']},
                'price_change_percentage_24h': price_change_24h,
                'total_volume': {'usd': latest_data['Volume']},
                'market_cap_rank': MARKET_RANKS.get(search_symbol, None)
            },
            'market_cap_rank': MARKET_RANKS.get(search_symbol, None)
        }

        # Cache the result
        cache_key = f"crypto_{symbol}"
        self.cache[cache_key] = {
            'data': market_data,
            'timestamp': datetime.now()
        }

        return market_data

    def _fetch_from_api(self, symbol: str) -> Dict:
        """Fetch data from CoinGecko API"""
        try:
            coingecko_id = SUPPORTED_CRYPTOS.get(symbol, symbol)
            url = f"{Config.COINGECKO_API_URL}/coins/{coingecko_id}"
            
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            cache_key = f"crypto_{symbol}"
            self.cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch API data for {symbol}: {e}")
            return {}

    def fetch_historical_data(self, symbol: str = "ethereum", days: int = 90) -> pd.DataFrame:
        """Fetch historical price data"""
        cache_key = f"historical_{symbol}_{days}"

        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']

        if self.csv_file_path:
            return self._fetch_historical_from_csv(symbol, days)
        
        return self._fetch_historical_from_api(symbol, days)

    def _fetch_historical_from_csv(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch historical data from CSV"""
        df = self._load_csv_data()
        if df.empty:
            return pd.DataFrame()

        symbol_mappings = {
            'ethereum': 'ETHEREUM',
            'bitcoin': 'BITCOIN',
            'btc': 'BITCOIN',
            'eth': 'ETHEREUM',
            'solana': 'SOLANA',
            'sol': 'SOLANA',
            'binancecoin': 'BINANCECOIN',
            'bnb': 'BINANCECOIN',
            'ripple': 'RIPPLE',
            'xrp': 'RIPPLE'
        }

        search_symbol = symbol_mappings.get(symbol, symbol.upper())
        symbol_data = df[df['Symbol'] == search_symbol]

        if symbol_data.empty:
            return pd.DataFrame()

        try:
            symbol_data = symbol_data.sort_values('Date')
            if days > 0:
                symbol_data = symbol_data.tail(days)

            # Create DataFrame with expected column names
            historical_df = pd.DataFrame({
                'price': symbol_data['Close'].values,
                'volume': symbol_data['Volume'].values,
                'market_cap': symbol_data['Marketcap'].values,
                'high': symbol_data['High'].values,
                'low': symbol_data['Low'].values,
                'open': symbol_data['Open'].values
            }, index=pd.to_datetime(symbol_data['Date'].values))

            # Try to add real-time data if CSV is old
            latest_csv_date = symbol_data['Date'].max()
            days_old = (datetime.now() - pd.to_datetime(latest_csv_date)).days

            if days_old > 1:
                realtime_data = self._get_realtime_data(symbol)
                if realtime_data:
                    today = pd.Timestamp.now().normalize()
                    latest_price = historical_df['price'].iloc[-1]

                    today_data = pd.DataFrame({
                        'price': [realtime_data['current_price']],
                        'volume': [realtime_data['volume_24h']],
                        'market_cap': [realtime_data['market_cap']],
                        'high': [max(latest_price, realtime_data['current_price'])],
                        'low': [min(latest_price, realtime_data['current_price'])],
                        'open': [latest_price]
                    }, index=[today])

                    historical_df = pd.concat([historical_df, today_data])

            historical_df.index.name = 'date'

            # Cache the result
            cache_key = f"historical_{symbol}_{days}"
            self.cache[cache_key] = {
                'data': historical_df,
                'timestamp': datetime.now()
            }

            return historical_df

        except Exception as e:
            logger.error(f"Error processing historical data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_historical_from_api(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch historical data from API"""
        try:
            coingecko_id = SUPPORTED_CRYPTOS.get(symbol, symbol)
            url = f"{Config.COINGECKO_API_URL}/coins/{coingecko_id}/market_chart"
            params = {'vs_currency': 'usd', 'days': days}
            
            response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']
            
            df = pd.DataFrame({
                'price': [p[1] for p in prices],
                'volume': [v[1] for v in volumes],
                'market_cap': [m[1] for m in market_caps]
            })
            
            df.index = pd.to_datetime([p[0] for p in prices], unit='ms')
            df.index.name = 'date'
            
            # Add OHLC data (simplified)
            df['high'] = df['price']
            df['low'] = df['price']
            df['open'] = df['price']
            
            cache_key = f"historical_{symbol}_{days}"
            self.cache[cache_key] = {
                'data': df,
                'timestamp': datetime.now()
            }
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data from API: {e}")
            return pd.DataFrame()
