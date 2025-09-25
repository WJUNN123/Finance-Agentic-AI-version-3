"""
Sentiment analysis for cryptocurrency news and social media
"""
import feedparser
import numpy as np
import logging
from typing import Dict, List
from transformers import pipeline
from config import RSS_FEEDS

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        try:
            self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            logger.warning(f"Failed to load FinBERT model: {e}")
            self.finbert = None
            
        self.crypto_keywords = {
            'positive': [
                'bullish', 'moon', 'surge', 'rally', 'breakout', 'adoption', 
                'partnership', 'growth', 'pump', 'accumulate', 'buy', 'long'
            ],
            'negative': [
                'crash', 'dump', 'bearish', 'decline', 'regulation', 'ban', 
                'hack', 'fall', 'drop', 'sell', 'short', 'liquidation'
            ]
        }

    def analyze_headlines(self, headlines: List[str]) -> Dict:
        """Enhanced sentiment analysis with keyword weighting"""
        if not headlines:
            return {
                'sentiment_score': 0, 
                'confidence': 0, 
                'analysis': [],
                'headline_count': 0
            }

        try:
            # Use FinBERT if available
            if self.finbert:
                sentiment_results = self.finbert(headlines[:10])
            else:
                # Fallback to simple keyword analysis
                sentiment_results = []
                for headline in headlines[:10]:
                    score = self._calculate_keyword_sentiment(headline)
                    if score > 0.1:
                        sentiment_results.append({'label': 'POSITIVE', 'score': abs(score)})
                    elif score < -0.1:
                        sentiment_results.append({'label': 'NEGATIVE', 'score': abs(score)})
                    else:
                        sentiment_results.append({'label': 'NEUTRAL', 'score': 0.5})
                        
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            sentiment_results = []

        total_score = 0
        confidence_scores = []
        detailed_analysis = []

        for i, headline in enumerate(headlines[:10]):
            if i < len(sentiment_results):
                base_sentiment = sentiment_results[i]
            else:
                base_sentiment = {'label': 'NEUTRAL', 'score': 0.5}
                
            keyword_boost = self._calculate_keyword_sentiment(headline)

            # Convert sentiment to score
            label = base_sentiment['label'].upper()
            if label == 'POSITIVE':
                score = base_sentiment['score']
            elif label == 'NEGATIVE':
                score = -base_sentiment['score']
            else:
                score = 0

            # Apply keyword boost
            final_score = np.clip(score + keyword_boost * 0.3, -1, 1)
            total_score += final_score
            confidence_scores.append(base_sentiment['score'])

            detailed_analysis.append({
                'headline': headline,
                'sentiment': label.lower(),
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
        """Calculate sentiment boost based on crypto-specific keywords"""
        text_lower = text.lower()
        pos_count = sum(1 for word in self.crypto_keywords['positive'] 
                       if word in text_lower)
        neg_count = sum(1 for word in self.crypto_keywords['negative'] 
                       if word in text_lower)

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

        # Analyze sentiment
        sentiment_data = self.sentiment_analyzer.analyze_headlines(headlines)

        # Add news metadata
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
                    # Check if headline is relevant
                    if any(keyword in title for keyword in symbol_keywords):
                        headlines.append(entry.title)
                        if len(headlines) >= 20:
                            break
                            
            except Exception as e:
                logger.warning(f"Failed to fetch from {feed_url}: {e}")
                continue

        # If no relevant headlines found, use general crypto headlines
        if not headlines:
            for feed_url in RSS_FEEDS[:2]:  # Try top 2 feeds for general news
                try:
                    feed = feedparser.parse(feed_url)
                    for entry in feed.entries[:3]:
                        headlines.append(entry.title)
                        
                except Exception as e:
                    continue

        return headlines