"""
CryptoAnalyzer: Performs analysis on cryptocurrencies using live API data
"""
import logging
from typing import Dict
from datetime import datetime

from data_fetcher import DataFetcher
from technical_analyzer import TechnicalAnalyzer
from sentiment_analyzer import SentimentAnalyzer, NewsAnalyzer
from prediction_engine import HybridPredictor, PredictionAdjuster
from analysis_engine import DecisionEngine
from config import Config

logger = logging.getLogger(__name__)


class ResultFormatter:
    def format_analysis(self, result: Dict) -> Dict:
        try:
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error(f"Failed to format analysis: {e}")
            return {"status": "error", "message": str(e)}


class CryptoAnalyzer:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = NewsAnalyzer()

        # Removed old PricePredictor()
        self.prediction_engine = None

        self.adjuster = PredictionAdjuster()
        self.decision_engine = DecisionEngine(risk_tolerance=0.6, use_gpt=True)

    def analyze(self, symbol: str, user_input: str) -> Dict:
        try:
            # 🔥 Initialize Hybrid XGBoost + LSTM predictor with symbol
            self.prediction_engine = HybridPredictor(symbol=symbol)

            # Live market data
            market_data = self.data_fetcher.fetch_crypto_data(symbol)
            if not market_data:
                return {
                    "status": "error",
                    "message": f"Failed to fetch live data for {symbol}"
                }

            # Historical data
            historical_df = self.data_fetcher.fetch_historical_data(symbol, days=30)
            if historical_df.empty:
                return {
                    "status": "error",
                    "message": f"Failed to fetch historical data for {symbol}"
                }

            # Technical analysis
            technical_result = self.technical_analyzer.analyze(historical_df)

            # Sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze(symbol)

            # Hybrid prediction (LSTM + XGBoost)
            forecast = self.prediction_engine.predict(historical_df)

            # Adjust forecast based on real-time volatility + market changes
            forecast = self.adjuster.adjust(forecast, market_data)

            # Decision engine confidence metrics
            confidence_scores = {
                "overall_confidence": technical_result.get("confidence", 50)
            }

            # Final buy/sell/hold decision
            recommendation = self.decision_engine.make_decision(
                technical=technical_result,
                sentiment=sentiment_result,
                confidence=confidence_scores,
                market_data=market_data
            )

            # Final formatted result
            result = {
                "symbol": symbol.upper(),
                "current_price": market_data["market_data"].get("current_price"),
                "market_cap": market_data["market_data"].get("market_cap"),
                "price_change_24h": market_data["market_data"].get("price_change_24h", 0),
                "market_cap_rank": market_data.get("market_cap_rank"),
                "forecast": forecast,
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
            logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
