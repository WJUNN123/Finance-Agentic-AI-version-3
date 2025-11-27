# gpt_engine.py
import os
import logging
from typing import Dict, Any, List, Optional

# OpenAI-compatible Gemini client
from openai import OpenAI
from openai.types import ChatCompletionRequestMessage

logger = logging.getLogger(__name__)

class GPTInsightGenerator:
    """Generate investment insights using Gemini Flash 2.0."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment.")
        
        # Initialize OpenAI-compatible Gemini client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = model

    def generate(
        self,
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        confidence: Dict[str, Any],
        market_data: Dict[str, Any],
        forecast: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a natural-language insight and recommendation
        from structured crypto analysis input.
        """
        prompt = self._build_prompt(technical, sentiment, confidence, market_data, forecast)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=0.3,
                max_tokens=256
            )
            message = response.choices[0].message
            content = getattr(message, "content", None)
            return content or "<No response from GPT>"
        except Exception as e:
            logger.error(f"Error in GPTInsightGenerator.generate: {e}", exc_info=True)
            return f"GPT generation error: {e}"

    def _build_prompt(
        self,
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        confidence: Dict[str, Any],
        market_data: Dict[str, Any],
        forecast: Optional[List[Dict[str, Any]]]
    ) -> List[ChatCompletionRequestMessage]:
        """
        Build system + user messages for Gemini chat completion.
        """
        system_msg = {
            "role": "system",
            "content": (
                "You are a friendly, professional cryptocurrency investment analyst. "
                "Provide clear, concise insights and a recommendation (Buy / Hold / Sell / Wait) "
                "based on the data provided. Avoid mentioning internal code or implementation details."
            )
        }

        user_content = (
            "Here are the analysis inputs:\n"
            f"- Technical analysis: {technical}\n"
            f"- Sentiment analysis: {sentiment}\n"
            f"- Confidence metrics: {confidence}\n"
            f"- Market data: {market_data}\n"
        )
        if forecast is not None:
            user_content += f"- Forecast (next days): {forecast}\n"

        user_content += (
            "\nProvide a short paragraph: state your view of the market condition, "
            "your recommendation, and a brief reasoning in simple language."
        )

        user_msg = {"role": "user", "content": user_content}

        return [system_msg, user_msg]
