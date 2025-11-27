import logging
from typing import Dict, Optional
from config import Config, Action, RiskLevel
from gpt_engine import GPTInsightGenerator 

logger = logging.getLogger(__name__)

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
