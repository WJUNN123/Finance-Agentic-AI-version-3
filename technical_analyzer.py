"""
Technical analysis indicators and pattern recognition
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict
from config import MarketRegime

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Comprehensive technical analysis"""
        if df.empty or len(df) < 10:
            logger.warning("Insufficient data for technical analysis")
            return self._get_default_technical_data()

        try:
            results = {}
            prices = df['price']

            results.update(self._calculate_moving_averages(prices))
            results.update(self._calculate_rsi(prices))
            results.update(self._calculate_macd(prices))
            results.update(self._calculate_bollinger_bands(prices))
            results.update(self._calculate_volatility(prices))
            results.update(self._calculate_support_resistance(prices))
            results.update(self._detect_patterns(prices))
            results.update(self._calculate_momentum_indicators(prices))
            results['market_regime'] = self._detect_market_regime(prices)
            results['technical_signal'] = self._calculate_technical_signal(results)

            return results
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            return self._get_default_technical_data()

    def _get_default_technical_data(self) -> Dict:
        return {
            'rsi': 50.0, 'rsi_signal': 'Neutral', 'rsi_trend': 'Unknown',
            'sma_7': None, 'sma_21': None, 'sma_50': None, 
            'ema_12': None, 'ema_26': None,
            'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'macd_crossover': False, 'macd_crossunder': False,
            'bb_upper': None, 'bb_lower': None, 'bb_middle': None, 
            'bb_position': 0.5, 'bb_squeeze': False,
            'volatility_7d': 0.5, 'volatility_30d': 0.5, 'avg_true_range': 0.0,
            'support_level': None, 'resistance_level': None,
            'distance_to_support': 0.0, 'distance_to_resistance': 0.0,
            'trend_pattern': 'Unknown', 'consecutive_gains': 0, 
            'consecutive_losses': 0,
            'roc_12d': 0.0, 'momentum_10d': 0.0, 'price_velocity': 0.0,
            'market_regime': 'sideways', 'technical_signal': 0
        }

    def _calculate_moving_averages(self, prices: pd.Series) -> Dict:
        """Calculate various moving averages"""
        return {
            'sma_7': prices.rolling(7).mean().iloc[-1] if len(prices) >= 7 else None,
            'sma_21': prices.rolling(21).mean().iloc[-1] if len(prices) >= 21 else None,
            'sma_50': prices.rolling(50).mean().iloc[-1] if len(prices) >= 50 else None,
            'ema_12': prices.ewm(span=12).mean().iloc[-1] if len(prices) >= 12 else None,
            'ema_26': prices.ewm(span=26).mean().iloc[-1] if len(prices) >= 26 else None,
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Dict:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return {'rsi': 50.0, 'rsi_signal': 'Neutral', 'rsi_trend': 'Unknown'}
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if current_rsi > 80: 
            rsi_signal = "Extremely Overbought"
        elif current_rsi > 70: 
            rsi_signal = "Overbought"
        elif current_rsi < 20: 
            rsi_signal = "Extremely Oversold"
        elif current_rsi < 30: 
            rsi_signal = "Oversold"
        else: 
            rsi_signal = "Neutral"

        rsi_trend = "Rising" if len(rsi) > 1 and rsi.iloc[-1] > rsi.iloc[-2] else "Falling"

        return {
            'rsi': current_rsi, 
            'rsi_signal': rsi_signal,
            'rsi_trend': rsi_trend
        }

    def _calculate_macd(self, prices: pd.Series) -> Dict:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return {
                'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
                'macd_crossover': False, 'macd_crossunder': False
            }
            
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal

        crossover = (len(macd) > 1 and 
                    macd.iloc[-1] > signal.iloc[-1] and 
                    macd.iloc[-2] <= signal.iloc[-2])
        crossunder = (len(macd) > 1 and 
                     macd.iloc[-1] < signal.iloc[-1] and 
                     macd.iloc[-2] >= signal.iloc[-2])

        return {
            'macd': macd.iloc[-1], 
            'macd_signal': signal.iloc[-1], 
            'macd_histogram': histogram.iloc[-1],
            'macd_crossover': crossover,
            'macd_crossunder': crossunder
        }

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {
                'bb_upper': None, 'bb_lower': None, 'bb_middle': None,
                'bb_position': 0.5, 'bb_squeeze': False
            }
            
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        current_price = prices.iloc[-1]

        bb_position = ((current_price - lower.iloc[-1]) / 
                      (upper.iloc[-1] - lower.iloc[-1])) if upper.iloc[-1] != lower.iloc[-1] else 0.5
        bb_squeeze = ((upper.iloc[-1] - lower.iloc[-1]) / sma.iloc[-1]) < 0.1

        return {
            'bb_upper': upper.iloc[-1], 
            'bb_lower': lower.iloc[-1], 
            'bb_middle': sma.iloc[-1],
            'bb_position': bb_position,
            'bb_squeeze': bb_squeeze
        }

    def _calculate_volatility(self, prices: pd.Series) -> Dict:
        """Calculate volatility metrics"""
        returns = prices.pct_change().dropna()
        
        vol_7d = returns.tail(7).std() * np.sqrt(365) if len(returns) >= 7 else 0.5
        vol_30d = returns.tail(30).std() * np.sqrt(365) if len(returns) >= 30 else 0.5
        atr = self._calculate_atr(prices)
        
        return {
            'volatility_7d': vol_7d,
            'volatility_30d': vol_30d,
            'avg_true_range': atr,
        }

    def _calculate_atr(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < period + 1:
            return prices.iloc[-1] * 0.02 if len(prices) > 0 else 0.0
            
        high = prices.rolling(2).max()
        low = prices.rolling(2).min()
        close_prev = prices.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean().iloc[-1]

    def _calculate_support_resistance(self, prices: pd.Series) -> Dict:
        """Calculate support and resistance levels"""
        recent_prices = prices.tail(20)
        support = recent_prices.min()
        resistance = recent_prices.max()
        current_price = prices.iloc[-1]

        distance_to_support = (current_price - support) / current_price if current_price > 0 else 0
        distance_to_resistance = (resistance - current_price) / current_price if current_price > 0 else 0

        return {
            'support_level': support, 
            'resistance_level': resistance,
            'distance_to_support': distance_to_support,
            'distance_to_resistance': distance_to_resistance
        }

    def _detect_patterns(self, prices: pd.Series) -> Dict:
        """Detect price patterns and trends"""
        if len(prices) < 10:
            return {
                'trend_pattern': 'Unknown',
                'consecutive_gains': 0,
                'consecutive_losses': 0
            }
            
        recent = prices.tail(10)
        if len(recent) >= 5:
            early = recent.head(5).mean()
            late = recent.tail(5).mean()
            
            if late > early * 1.02:
                trend = "Uptrend"
            elif late < early * 0.98:
                trend = "Downtrend"
            else:
                trend = "Sideways"
        else:
            trend = "Insufficient Data"

        return {
            'trend_pattern': trend,
            'consecutive_gains': self._count_consecutive_moves(prices, positive=True),
            'consecutive_losses': self._count_consecutive_moves(prices, positive=False)
        }

    def _count_consecutive_moves(self, prices: pd.Series, positive: bool = True) -> int:
        """Count consecutive gains or losses"""
        if len(prices) < 2:
            return 0
            
        changes = prices.diff().dropna()
        count = 0
        
        for change in reversed(changes.values):
            if (positive and change > 0) or (not positive and change < 0):
                count += 1
            else:
                break
                
        return count

    def _calculate_momentum_indicators(self, prices: pd.Series) -> Dict:
        """Calculate momentum indicators"""
        if len(prices) < 12:
            return {'roc_12d': 0.0, 'momentum_10d': 0.0, 'price_velocity': 0.0}
            
        roc_12 = ((prices.iloc[-1] / prices.iloc[-12]) - 1) * 100
        momentum = prices.iloc[-1] - prices.iloc[-10] if len(prices) >= 10 else 0
        velocity = prices.diff().tail(5).mean()
        
        return {
            'roc_12d': roc_12, 
            'momentum_10d': momentum,
            'price_velocity': velocity
        }

    def _detect_market_regime(self, prices: pd.Series) -> str:
        """Detect current market regime"""
        if len(prices) < 30:
            return MarketRegime.SIDEWAYS.value
            
        sma_short = prices.rolling(10).mean()
        sma_long = prices.rolling(30).mean()
        
        trend_strength = ((sma_short.iloc[-1] - sma_long.iloc[-1]) / 
                         sma_long.iloc[-1]) if sma_long.iloc[-1] != 0 else 0
        volatility = prices.pct_change().tail(20).std()

        if volatility > 0.05: 
            return MarketRegime.VOLATILE.value
        if trend_strength > 0.05: 
            return MarketRegime.BULL.value
        if trend_strength < -0.05: 
            return MarketRegime.BEAR.value
            
        return MarketRegime.SIDEWAYS.value

    def _calculate_technical_signal(self, indicators: Dict) -> int:
        """Calculate overall technical signal strength"""
        signals = []
        
        # RSI signals
        rsi = indicators.get('rsi', 50)
        if rsi < 30: 
            signals.append(30)
        elif rsi > 70: 
            signals.append(-30)
            
        # MACD signals
        if indicators.get('macd_crossover', False): 
            signals.append(25)
        elif indicators.get('macd_crossunder', False): 
            signals.append(-25)
            
        # Moving average signals
        sma_7 = indicators.get('sma_7', 0)
        sma_21 = indicators.get('sma_21', 0)
        if sma_7 and sma_21:
            if sma_7 > sma_21:
                signals.append(15)
            else:
                signals.append(-15)
                
        # Bollinger Band signals
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position < 0.2: 
            signals.append(20)
        elif bb_position > 0.8: 
            signals.append(-20)
            
        return int(np.clip(sum(signals), -100, 100))