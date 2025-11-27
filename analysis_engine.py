import logging
from typing import Dict, Optional
import numpy as np
from config import Config, Action, RiskLevel
from gpt_engine import GPTInsightGenerator

logger = logging.getLogger(__name__)

class ConfidenceCalculator:
    """Calculate confidence scores based on multiple factors"""
    
    def __init__(self):
        self.weights = {
            'technical': 0.35, 
            'sentiment': 0.25, 
            'volume': 0.15,
            'market_structure': 0.15, 
            'risk_metrics': 0.10
        }

    def calculate(self, technical_data: Dict, sentiment_data: Dict, market_data: Dict) -> Dict:
        scores = {
            'technical': self._technical_confidence(technical_data),
            'sentiment': self._sentiment_confidence(sentiment_data),
            'volume': self._volume_confidence(market_data),
            'market_structure': self._market_structure_confidence(technical_data),
            'risk_metrics': self._risk_confidence(technical_data)
        }
        overall_confidence = sum(scores[key] * self.weights[key] for key in scores)
        return {
            'overall_confidence': round(overall_confidence, 1),
            'component_scores': scores,
            'confidence_level': self._confidence_level(overall_confidence)
        }

    # ----- Internal methods for ConfidenceCalculator -----
    def _technical_confidence(self, technical: Dict) -> float:
        if not technical: return 50.0
        signal_strength = abs(technical.get('technical_signal', 0))
        rsi_clarity = self._rsi_clarity(technical.get('rsi', 50))
        trend_consistency = self._trend_consistency(technical)
        weighted_score = signal_strength*0.4 + rsi_clarity*0.35 + trend_consistency*0.25
        return round(np.clip(weighted_score, 0, 100), 2)

    def _rsi_clarity(self, rsi: float) -> float:
        return min(abs(rsi-50)*2, 100)

    def _trend_consistency(self, technical: Dict) -> float:
        signals=[]
        macd = technical.get('macd', 0)
        macd_signal = technical.get('macd_signal', 0)
        signals.append(1 if macd>macd_signal else -1)
        sma_7 = technical.get('sma_7', 0)
        sma_21 = technical.get('sma_21', 0)
        if sma_7 and sma_21:
            signals.append(1 if sma_7>sma_21 else -1)
        if len(signals)>1:
            return (abs(sum(signals))/len(signals))*100
        return 50.0

    def _sentiment_confidence(self, sentiment: Dict) -> float:
        if not sentiment: return 50.0
        base_conf = sentiment.get('confidence', 0.5)*100
        strength_bonus = min(abs(sentiment.get('sentiment_score',0))*50,30)
        return min(base_conf+strength_bonus,100)

    def _volume_confidence(self, market_data: Dict) -> float:
        return 60.0

    def _market_structure_confidence(self, technical: Dict) -> float:
        regime = technical.get('market_regime','sideways')
        return {'bull':80,'bear':80,'volatile':35,'sideways':50}.get(regime,50)

    def _risk_confidence(self, technical: Dict) -> float:
        volatility_pct = technical.get('volatility_30d',0.5)*100
        if volatility_pct>100: return 30.0
        if volatility_pct>70: return 50.0
        if volatility_pct<40: return 80.0
        return 65.0

    def _confidence_level(self, score: float) -> str:
        if score>=80: return "Very High"
        if score>=65: return "High"
        if score>=50: return "Medium"
        if score>=35: return "Low"
        return "Very Low"

class DecisionEngine:
    """Make investment decisions based on analysis results, with optional GPT reasoning"""
    
    def __init__(self, risk_tolerance: float = 0.6, use_gpt: bool = True):
        self.risk_tolerance = risk_tolerance
        self.use_gpt = use_gpt
        if use_gpt:
            self.gpt_engine = GPTInsightGenerator()
    
    def make_decision(self, technical: Dict, sentiment: Dict, 
                      confidence: Dict, market_data: Dict) -> Dict:
        """Make investment decision based on all analysis"""
        
        technical_signal = technical.get('technical_signal', 0)
        sentiment_score = sentiment.get('sentiment_score', 0)
        overall_confidence = confidence.get('overall_confidence', 50)
        
        # Combine signals
        combined_signal = (technical_signal * 0.6 + sentiment_score * 40 * 0.4)
        
        # Apply market regime multiplier
        regime_multiplier = {
            'bull': 1.2,
            'bear': 1.2,
            'volatile': 0.7,
            'sideways': 1.0
        }.get(technical.get('market_regime', 'sideways'), 1.0)
        
        adjusted_signal = combined_signal * regime_multiplier
        
        # Determine action
        action = self._signal_to_action(adjusted_signal, overall_confidence)
        
        # Calculate position size and risk level
        position_size = self._calculate_position_size(adjusted_signal, overall_confidence, technical, action)
        risk_level = self._assess_risk_level(technical, sentiment, confidence, action)
        
        # Stop loss / take profit
        stop_loss, take_profit = None, None
        if action not in [Action.HOLD, Action.WAIT]:
            stop_loss = self._calculate_stop_loss(market_data, technical)
            take_profit = self._calculate_take_profit(market_data, technical)
        
        # Generate reasoning (GPT or fallback)
        reasoning = self._generate_reasoning(technical, sentiment, confidence, action, adjusted_signal, market_data)
        
        return {
            'action': action.value,
            'signal_strength': round(adjusted_signal, 1),
            'position_size': position_size,
            'risk_level': risk_level.value,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': reasoning
        }

    # ------------------------ Internal Methods ------------------------ #
    
    def _generate_reasoning(self, technical, sentiment, confidence, action, signal, market_data) -> str:
        """Generate reasoning using GPT or fallback template"""
        if self.use_gpt:
            try:
                # GPT expects the structured data
                return self.gpt_engine.generate(
                    technical=technical,
                    sentiment=sentiment,
                    confidence=confidence,
                    market_data=market_data,
                    action=action.value,
                    signal=signal
                )
            except Exception as e:
                logger.warning(f"GPT reasoning failed: {e}")
        
        # Fallback template reasoning
        regime = technical.get('market_regime', 'unknown')
        tech_signal = technical.get('technical_signal', 0)
        sent_score = sentiment.get('sentiment_score', 0)
        conf_score = confidence.get('overall_confidence', 0)
        conf_level = confidence.get('confidence_level', 'Unknown')

        reason = (
            f"The current market regime is {regime}. "
            f"The technical signal score is {tech_signal}, "
            f"while sentiment analysis shows a score of {sent_score:.2f}. "
            f"Combined, this gives an adjusted signal of {signal:.1f}. "
            f"Overall confidence in this analysis is {conf_score}% ({conf_level}). "
        )

        if action == Action.WAIT:
            reason += "Confidence is below the threshold, suggesting it's best to wait for a clearer market signal."
        elif action == Action.HOLD:
            reason += "The signal is not strong enough to suggest a new buy or sell action at this time."
        else:
            reason += f"This leads to a '{action.value}' recommendation based on the current risk assessment."

        return reason

    def _signal_to_action(self, signal: float, confidence: float) -> Action:
        if confidence < Config.CONFIDENCE_THRESHOLD - 20:
            return Action.WAIT
        if signal > 80: return Action.STRONG_BUY
        if signal > 40: return Action.BUY
        if signal < -80: return Action.STRONG_SELL
        if signal < -40: return Action.SELL
        return Action.HOLD

    def _calculate_position_size(self, signal, confidence, technical, action) -> float:
        if action in [Action.HOLD, Action.WAIT]: return 0.0
        base_size = abs(signal) / 100
        confidence_factor = confidence / 100
        volatility = technical.get('volatility_30d', 0.5)
        risk_factor = (1 - min(volatility, 1.0)) * self.risk_tolerance
        position_size = base_size * confidence_factor * risk_factor * 100
        return round(min(position_size, 25.0), 1)

    def _assess_risk_level(self, technical, sentiment, confidence, action) -> RiskLevel:
        risk_score = 0
        volatility = technical.get('volatility_30d', 0.5)
        if volatility > 0.8: risk_score += 40
        elif volatility > 0.5: risk_score += 20
        conf_score = confidence.get('overall_confidence', 50)
        if conf_score < 40: risk_score += 30
        elif conf_score < 60: risk_score += 15
        if abs(sentiment.get('sentiment_score', 0)) > 0.4: risk_score += 10
        if action not in [Action.HOLD, Action.WAIT]: risk_score += 10
        if risk_score > 75: return RiskLevel.EXTREME
        if risk_score > 50: return RiskLevel.HIGH
        if risk_score > 25: return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _calculate_stop_loss(self, market_data, technical) -> Optional[float]:
        current_price = market_data.get('market_data', {}).get('current_price', {}).get('usd')
        if not current_price: return None
        atr = technical.get('avg_true_range', current_price * 0.05)
        support = technical.get('support_level', current_price * 0.95)
        atr_stop = current_price - (2 * atr)
        support_stop = support * 0.98
        return round(max(atr_stop, support_stop), 2)

    def _calculate_take_profit(self, market_data, technical) -> Optional[float]:
        current_price = market_data.get('market_data', {}).get('current_price', {}).get('usd')
        if not current_price: return None
        resistance = technical.get('resistance_level', current_price * 1.1)
        return round(min(resistance * 0.98, current_price * 1.08), 2)
