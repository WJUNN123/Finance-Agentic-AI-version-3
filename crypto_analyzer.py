"""
Main cryptocurrency analysis engine
"""
import logging
from datetime import datetime
from typing import Dict, Optional

from data_fetcher import DataFetcher
from sentiment_analyzer import SentimentAnalyzer, NewsAnalyzer
from technical_analyzer import TechnicalAnalyzer
from analysis_engine import DecisionEngine
from prediction_engine import PricePredictor, PredictionAdjuster
from config import Config

logger = logging.getLogger(__name__)

class CryptoAnalyzer:
    """Main cryptocurrency analysis orchestrator"""
    
    def __init__(self, risk_tolerance: float = 0.6, csv_file_path: Optional[str] = None):
        self.data_fetcher = DataFetcher(csv_file_path=csv_file_path)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.confidence_calculator = ConfidenceCalculator()
        self.decision_engine = DecisionEngine(risk_tolerance=risk_tolerance)
        self.prediction_adjuster = PredictionAdjuster()
        self.predictors = {}

    def get_predictor(self, symbol: str) -> PricePredictor:
        """Get or create a price predictor for a symbol"""
        if symbol not in self.predictors:
            self.predictors[symbol] = PricePredictor(symbol)
        return self.predictors[symbol]

    def analyze(self, symbol: str, query: str = "") -> Dict:
        """Perform comprehensive cryptocurrency analysis"""
        try:
            logger.info(f"Starting analysis for {symbol}")
            
            # Fetch market data
            market_data_raw = self.data_fetcher.fetch_crypto_data(symbol)
            if not market_data_raw or 'market_data' not in market_data_raw:
                return {'error': f"Failed to fetch market data for {symbol}"}

            market_data = market_data_raw['market_data']

            # Fetch historical data
            historical_df = self.data_fetcher.fetch_historical_data(
                symbol, days=Config.HISTORICAL_DAYS
            )
            if historical_df.empty:
                return {'error': f"Failed to fetch historical data for {symbol}"}

            # Perform sentiment analysis
            sentiment_data = self.news_analyzer.fetch_and_analyze_news(symbol)
            if 'error' in sentiment_data:
                # Fallback to default headlines
                default_headlines = [
                    f"{symbol.upper()} market analysis shows mixed signals",
                    "Cryptocurrency market volatility continues",
                    "Digital asset investors remain cautious"
                ]
                sentiment_data = self.sentiment_analyzer.analyze_headlines(default_headlines)

            # Perform technical analysis
            technical_data = self.technical_analyzer.analyze(historical_df)

            # Calculate confidence scores
            confidence_data = self.confidence_calculator.calculate(
                technical_data, sentiment_data, market_data
            )

            # Make investment decision
            decision_data = self.decision_engine.make_decision(
                technical_data, sentiment_data, confidence_data, market_data_raw
            )

            # Generate price predictions
            raw_forecast = None
            adjusted_forecast = None
            
            try:
                predictor = self.get_predictor(symbol)
                predictor.train_or_load(historical_df)
                raw_forecast = predictor.predict_future(historical_df, Config.FORECAST_DAYS)
                
                if raw_forecast:
                    current_price = market_data.get('current_price', {}).get('usd', 
                        historical_df['price'].iloc[-1] if not historical_df.empty else 0
                    )
                    adjusted_forecast = self.prediction_adjuster.adjust(
                        raw_forecast, decision_data, confidence_data, current_price
                    )
                    
            except Exception as e:
                logger.warning(f"Price prediction failed: {e}")
                raw_forecast = []
                adjusted_forecast = []

            # Compile results
            result = {
                "symbol": symbol.upper(),
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "market_data": {
                    "current_price": market_data.get('current_price', {}).get('usd'),
                    "market_cap": market_data.get('market_cap', {}).get('usd'),
                    "market_cap_rank": market_data_raw.get('market_cap_rank'),
                    "price_change_24h": market_data.get('price_change_percentage_24h'),
                    "volume_24h": market_data.get('total_volume', {}).get('usd')
                },
                "sentiment_analysis": sentiment_data,
                "technical_analysis": technical_data,
                "confidence_assessment": confidence_data,
                "investment_decision": decision_data,
                "raw_forecast": raw_forecast or [],
                "adjusted_forecast": adjusted_forecast or []
            }

            logger.info(f"Analysis completed for {symbol}")
            return result

        except Exception as e:
            logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
            return {"error": str(e), "symbol": symbol}

class ResultFormatter:
    """Format analysis results for display"""
    
    def format_analysis(self, result: Dict) -> Dict:
        """Format analysis result for UI display"""
        if "error" in result:
            return {
                "status": "error",
                "message": result['error'],
                "data": None
            }

        symbol = result.get('symbol', 'N/A')
        market = result.get('market_data', {})
        decision = result.get('investment_decision', {})
        confidence = result.get('confidence_assessment', {})
        technical = result.get('technical_analysis', {})
        sentiment = result.get('sentiment_analysis', {})
        forecast = result.get('adjusted_forecast', [])        

        # Generate a more investor-friendly reasoning
        reasoning = self.generate_readable_summary(technical, sentiment, confidence, decision)

        # Format the main metrics
        formatted_result = {
            "status": "success",
            "data": {
                "symbol": symbol,
                "current_price": market.get('current_price', 0),
                "price_change_24h": market.get('price_change_24h', 0),
                "market_cap": market.get('market_cap', 0),
                "market_cap_rank": market.get('market_cap_rank', 'N/A'),
                
                "recommendation": {
                    "action": decision.get('action', 'Hold'),
                    "confidence": confidence.get('overall_confidence', 50),
                    "confidence_level": confidence.get('confidence_level', 'Medium'),
                    "risk_level": decision.get('risk_level', 'Medium'),
                    "position_size": decision.get('position_size', 0),
                    "reasoning": reasoning
                },
                
                # ... rest of your formatting unchanged ...
            }
        }

        return formatted_result

    def generate_readable_summary(self, technical, sentiment, confidence, decision) -> str:
        """Generate a human-friendly analysis summary for investors"""
        # Technical trend
        trend = technical.get('market_regime', 'sideways').lower()
        tech_signal = technical.get('technical_signal', 0)
        if tech_signal > 20:
            tech_summary = "strong bullish signals"
        elif tech_signal > 5:
            tech_summary = "mild bullish trend"
        elif tech_signal < -20:
            tech_summary = "strong bearish signals"
        elif tech_signal < -5:
            tech_summary = "moderate bearish trend"
        else:
            tech_summary = "neutral technical signals"

        # Sentiment
        sentiment_score = sentiment.get('sentiment_score', 0)
        if sentiment_score > 0.1:
            sentiment_summary = "positive market sentiment"
        elif sentiment_score < -0.1:
            sentiment_summary = "negative market sentiment"
        else:
            sentiment_summary = "neutral market sentiment"

        # Confidence
        conf_level = confidence.get('confidence_level', 'Medium')
        conf_score = confidence.get('overall_confidence', 50)

        # Decision action
        action = decision.get('action', 'Hold').lower()

        summary = (
            f"Market analysis indicates {tech_summary}, "
            f"with {sentiment_summary}. "
            f"The suggested action is **{action.capitalize()}**, "
            f"with an overall confidence of {conf_score:.1f}% ({conf_level}). "
            f"Investors should manage risk accordingly and consider position sizing based on their risk tolerance."
        )

        return summary


    def get_action_color(self, action: str) -> str:
        """Get color code for action"""
        color_map = {
            "Strong Buy": "green",
            "Buy": "lightgreen", 
            "Hold": "gray",
            "Sell": "orange",
            "Strong Sell": "red",
            "Wait": "blue"
        }
        return color_map.get(action, "gray")

    def get_risk_color(self, risk_level: str) -> str:
        """Get color code for risk level"""
        color_map = {
            "Low": "green",
            "Medium": "yellow",
            "High": "orange", 
            "Extreme": "red"
        }
        return color_map.get(risk_level, "gray")
