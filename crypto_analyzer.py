"""
CryptoAnalyzer: Performs analysis on cryptocurrencies using live API data
"""
import logging
from typing import Dict
from datetime import datetime

from data_fetcher import DataFetcher
from technical_analyzer import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer  # Use SentimentAnalyzer instead of NewsAnalyzer
from prediction_engine import HybridPredictor, PredictionAdjuster
from analysis_engine import DecisionEngine
from config import Config

logger = logging.getLogger(__name__)


class ResultFormatter:
    def format_analysis(self, result: Dict) -> Dict:
        try:
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error(f"Failed to format analysis: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


class CryptoAnalyzer:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()  # Corrected
        self.prediction_engine = None
        self.adjuster = PredictionAdjuster()
        self.decision_engine = DecisionEngine(risk_tolerance=0.6, use_gpt=True)

    def analyze(self, symbol: str = None, user_input: str = "") -> Dict:
        # Ensure symbol is always defined
        symbol = (symbol or user_input or "UNKNOWN").strip().upper()

        try:
            # Initialize predictor
            self.prediction_engine = HybridPredictor(symbol=symbol)

            # Fetch market data
            market_data = self.data_fetcher.fetch_crypto_data(symbol)
            market_info = market_data.get("market_data", {})
            market_info.setdefault("current_price", 0.0)
            market_info.setdefault("market_cap", 0)
            market_info.setdefault("price_change_24h", 0.0)
            market_info.setdefault("market_cap_rank", 0)

            if not market_data:
                return {
                    "status": "error",
                    "symbol": symbol,
                    "message": f"Failed to fetch live data for {symbol}"
                }

            # Fetch historical data
            historical_df = self.data_fetcher.fetch_historical_data(symbol, days=30)
            if historical_df.empty:
                return {
                    "status": "error",
                    "symbol": symbol,
                    "message": f"Failed to fetch historical data for {symbol}"
                }

            # Technical analysis
            try:
                technical_result = self.technical_analyzer.analyze(historical_df)
            except Exception as e:
                logger.warning(f"Technical analysis failed: {e}")
                technical_result = {
                    "confidence": 50,
                    "support_level": 0,
                    "resistance_level": 0,
                    "volatility_30d": 0,
                    "trend": "Unknown",
                    "technical_signal": "Hold",
                    "rsi": 50,
                    "rsi_signal": "Neutral",
                    "macd_signal": "Neutral",
                    "market_regime": "sideways"
                }

            # Sentiment analysis
            try:
                sentiment_result = self.sentiment_analyzer.analyze(symbol)
            except AttributeError:
                logger.warning(f"Sentiment analysis method missing; using fallback for {symbol}")
                sentiment_result = {"score": 0, "label": "Neutral", "headlines": [], "headline_count": 0}
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                sentiment_result = {"score": 0, "label": "Neutral", "headlines": [], "headline_count": 0}

            # Hybrid prediction
            forecast = None
            try:
                forecast = self.prediction_engine.predict(historical_df)
                if forecast is None:
                    self.prediction_engine.train_or_load(historical_df)
                    forecast = self.prediction_engine.predict(historical_df)
                forecast = self.adjuster.adjust(forecast, market_info)
            except Exception as e:
                logger.warning(f"Forecast generation failed: {e}")
                forecast = []

            # Decision making
            try:
                confidence_scores = {"overall_confidence": technical_result.get("confidence", 50)}
                recommendation = self.decision_engine.make_decision(
                    technical=technical_result,
                    sentiment=sentiment_result,
                    confidence=confidence_scores,
                    market_data=market_info
                )
            except Exception as e:
                logger.warning(f"Decision engine failed: {e}")
                recommendation = {"action": "Hold", "confidence": 50, "confidence_level": "Medium",
                                  "risk_level": "Medium", "reasoning": "", "stop_loss": None,
                                  "take_profit": None, "position_size": 0}

            # Build result safely
            result = {
                "symbol": symbol,
                "current_price": market_info.get("current_price", 0.0),
                "market_cap": market_info.get("market_cap", 0),
                "price_change_24h": market_info.get("price_change_24h", 0.0),
                "market_cap_rank": market_info.get("market_cap_rank", 0),
                "forecast": forecast or [],
                "technical": technical_result,
                "sentiment": sentiment_result,
                "recommendation": recommendation,
                "risk_management": {
                    "stop_loss": recommendation.get("stop_loss"),
                    "take_profit": recommendation.get("take_profit"),
                    "position_size": recommendation.get("position_size"),
                    "support_level": technical_result.get("support_level"),
                    "resistance_level": technical_result.get("resistance_level"),
                    "volatility_30d": technical_result.get("volatility_30d"),
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return result

        except Exception as e:
            safe_symbol = symbol or "UNKNOWN"
            logger.error(f"Analysis failed for {safe_symbol}: {e}", exc_info=True)
            return {"status": "error", "symbol": safe_symbol, "message": str(e)}
