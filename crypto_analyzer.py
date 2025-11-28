import logging
from typing import Dict
from datetime import datetime

from data_fetcher import DataFetcher
from technical_analyzer import TechnicalAnalyzer
from sentiment_analyzer import NewsAnalyzer
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
        self.sentiment_analyzer = NewsAnalyzer()
        self.prediction_engine = None
        self.adjuster = PredictionAdjuster()
        self.decision_engine = DecisionEngine(risk_tolerance=0.6, use_gpt=True)

    def analyze(self, symbol: str = None, user_input: str = "") -> Dict:
        symbol = (symbol or user_input or "UNKNOWN").strip().upper()

        try:
            # Initialize predictor
            self.prediction_engine = HybridPredictor(symbol=symbol)

            # Fetch live market data
            market_data = self.data_fetcher.fetch_crypto_data(symbol)
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
            technical_result = self.technical_analyzer.analyze(historical_df)

            # Sentiment analysis (fallback if missing)
            try:
                sentiment_result = self.sentiment_analyzer.analyze(symbol)
            except Exception:
                logger.warning(f"Sentiment analysis method missing; using fallback for {symbol}")
                sentiment_result = {
                    "score": 0.0,
                    "label": "Neutral",
                    "headline_count": 0,
                    "headlines": []
                }

            # Hybrid forecast
            forecast = None
            try:
                forecast = self.prediction_engine.predict(historical_df)
                if forecast is None:
                    self.prediction_engine.train_or_load(historical_df)
                    forecast = self.prediction_engine.predict(historical_df)
                forecast = self.adjuster.adjust(forecast, market_data)
            except Exception as e:
                logger.warning(f"Forecast generation failed for {symbol}: {e}")
                forecast = []

            # Decision/recommendation
            try:
                confidence_scores = {"overall_confidence": technical_result.get("confidence", 50)}
                recommendation = self.decision_engine.make_decision(
                    technical=technical_result,
                    sentiment=sentiment_result,
                    confidence=confidence_scores,
                    market_data=market_data
                )
            except Exception as e:
                logger.warning(f"Recommendation generation failed: {e}")
                recommendation = {
                    "action": "Hold",
                    "confidence": 50.0,
                    "confidence_level": "Medium",
                    "risk_level": "Medium",
                    "reasoning": "Fallback recommendation due to analysis errors.",
                    "stop_loss": None,
                    "take_profit": None,
                    "position_size": 0
                }

            # Market metrics safely
            market_info = market_data.get("market_data", {})
            result = {
                "symbol": symbol,
                "current_price": market_info.get("current_price", 0.0),
                "market_cap": market_info.get("market_cap", 0),
                "price_change_24h": market_info.get("price_change_24h", 0.0),
                "market_cap_rank": market_data.get("market_cap_rank", 0),
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
                    "volatility_30d": technical_result.get("volatility_30d", 0.0),
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            return result

        except Exception as e:
            safe_symbol = symbol or "UNKNOWN"
            logger.error(f"Analysis failed for {safe_symbol}: {e}", exc_info=True)
            return {
                "status": "error",
                "symbol": safe_symbol,
                "message": str(e)
            }
