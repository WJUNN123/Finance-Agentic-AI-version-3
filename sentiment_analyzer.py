"""
Sentiment analysis for cryptocurrency news and social media
"""
import feedparser
import numpy as np
import logging
from typing import Dict, List
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from config import RSS_FEEDS

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        # Don't load the model yet
        self.sentiment = None
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        
        # Keyword-based fallback
        self.crypto_keywords = {
            'positive': ['bullish', 'moon', 'surge', 'rally', 'breakout', 'adoption', 'partnership', 'growth', 'pump', 'accumulate', 'buy', 'long'],
            'negative': ['crash', 'dump', 'bearish', 'decline', 'regulation', 'ban', 'hack', 'fall', 'drop', 'sell', 'short', 'liquidation']
        }

    def _lazy_load_model(self):
        """Load RoBERTa model only when needed"""
        if self.sentiment is None:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import logging
            logger = logging.getLogger(__name__)
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                logger.info("Twitter-RoBERTa model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Twitter-RoBERTa model: {e}")
                self.sentiment = None

    def analyze_headlines(self, headlines: List[str]) -> Dict:
        # Lazy load the model here
        self._lazy_load_model()

        if not headlines:
            return {'sentiment_score': 0, 'confidence': 0, 'analysis': [], 'headline_count': 0}

        # Use the model if available, fallback to keyword scoring
        sentiment_results = []
        try:
            if self.sentiment:
                sentiment_results = self.sentiment(headlines[:10])
            else:
                for headline in headlines[:10]:
                    score = self._calculate_keyword_sentiment(headline)
                    if score > 0.1:
                        sentiment_results.append({'label': 'positive', 'score': abs(score)})
                    elif score < -0.1:
                        sentiment_results.append({'label': 'negative', 'score': abs(score)})
                    else:
                        sentiment_results.append({'label': 'neutral', 'score': 0.5})
        except Exception as e:
            sentiment_results = []
        
        # Rest of your scoring logic...
        total_score = 0
        confidence_scores = []
        detailed_analysis = []

        for i, headline in enumerate(headlines[:10]):
            base_sentiment = sentiment_results[i] if i < len(sentiment_results) else {'label': 'neutral', 'score': 0.5}
            keyword_boost = self._calculate_keyword_sentiment(headline)

            label = base_sentiment['label'].lower()
            score = base_sentiment['score'] if label == 'positive' else -base_sentiment['score'] if label == 'negative' else 0
            final_score = max(min(score + keyword_boost * 0.3, 1), -1)
            total_score += final_score
            confidence_scores.append(base_sentiment['score'])

            detailed_analysis.append({
                'headline': headline,
                'sentiment': label,
                'score': final_score,
                'confidence': base_sentiment['score']
            })

        avg_sentiment = total_score / len(headlines) if headlines else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

        return {
            'sentiment_score': avg_sentiment,
            'confidence': avg_confidence,
            'analysis': detailed_analysis,
            'headline_count': len(headlines)
        }

    def _calculate_keyword_sentiment(self, text: str) -> float:
        text_lower = text.lower()
        pos_count = sum(1 for word in self.crypto_keywords['positive'] if word in text_lower)
        neg_count = sum(1 for word in self.crypto_keywords['negative'] if word in text_lower)
        if pos_count + neg_count == 0:
            return 0
        return (pos_count - neg_count) / (pos_count + neg_count)

class NewsAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()

    def fetch_and_analyze_news(self, symbol: str = "ethereum") -> Dict:
        """Fetch and analyze crypto news"""
        headlines = self._fetch_headlines(symbol)

        if not headlines:
            return {
                'error': 'No news found', 
                'sentiment_score': 0,
                'headlines': [],
                'total_headlines': 0
            }

        sentiment_data = self.sentiment_analyzer.analyze_headlines(headlines)

        sentiment_data['headlines'] = headlines[:10]
        sentiment_data['total_headlines'] = len(headlines)
        sentiment_data['sources'] = len(RSS_FEEDS)

        return sentiment_data

    def _fetch_headlines(self, symbol: str) -> List[str]:
        """Fetch headlines from multiple RSS feeds"""
        headlines = []
        symbol_keywords = [
            symbol.lower(), 
            'ethereum', 'eth', 'bitcoin', 'btc', 'crypto', 'cryptocurrency',
            'solana', 'sol', 'binance', 'bnb', 'cardano', 'ada'
        ]

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    title = entry.title.lower()
                    if any(keyword in title for keyword in symbol_keywords):
                        headlines.append(entry.title)
                        if len(headlines) >= 20:
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to fetch from {feed_url}: {e}")
                continue

        if not headlines:
            for feed_url in RSS_FEEDS[:2]:
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:3]:
                        headlines.append(entry.title)
                except:
                    continue

        return headlines
