"""
Streamlit UI for Crypto Investment Analysis (Sidebar + Confidence Breakdown Removed)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta
from typing import Dict, List
import os

from crypto_analyzer import CryptoAnalyzer, ResultFormatter
from config import Config, SUPPORTED_CRYPTOS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Setup
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide",            # Make full-width since sidebar removed
    initial_sidebar_state="collapsed"
)

# Custom CSS (unchanged)
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
    }
    
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    .risk-extreme { background-color: #721c24; color: white; }
    
    .action-strong-buy { background-color: #28a745; color: white; }
    .action-buy { background-color: #17a2b8; color: white; }
    .action-hold { background-color: #6c757d; color: white; }
    .action-sell { background-color: #fd7e14; color: white; }
    .action-strong-sell { background-color: #dc3545; color: white; }
    .action-wait { background-color: #007bff; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_analyzer(csv_path=None):
    return CryptoAnalyzer(csv_file_path=csv_path)

# Removed: create_confidence_chart()


# Display Recommendation Card (unchanged)
def display_recommendation_card(recommendation: Dict):
    action = recommendation.get('action', 'Hold')
    confidence = recommendation.get('confidence', 50)
    risk_level = recommendation.get('risk_level', 'Medium')
    
    action_class = f"action-{action.lower().replace(' ', '-')}"
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(f"""
    <div class="recommendation-card">
        <h2 style="margin-top: 0;">📊 Investment Recommendation</h2>
        <div style="display: flex; justify-content: space-between; margin: 20px 0;">
            <div class="{action_class}" style="padding: 10px 20px; border-radius: 25px; text-align: center; flex: 1; margin: 0 5px;">
                <h3 style="margin: 0;">{action}</h3>
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 25px; text-align: center; flex: 1; margin: 0 5px;">
                <h3 style="margin: 0;">Confidence: {confidence:.1f}%</h3>
            </div>
            <div class="{risk_class}" style="padding: 10px 20px; border-radius: 25px; text-align: center; flex: 1; margin: 0 5px;">
                <h3 style="margin: 0;">Risk: {risk_level}</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():

    # HEADER
    st.title("🚀 Crypto Investment Analyzer")
    st.markdown("*Ask about BTC, ETH, SOL, etc. Educational only — not financial advice.*")

    # MAIN INPUT (no sidebar now)
    st.subheader("Your message")
    user_input = st.text_input(
        "Enter your query:",
        value=st.session_state.get('user_input', ''),
        placeholder="Example: 'ETH 7-day forecast' or 'Should I buy BTC?'"
    )

    if st.button("Analyze", type="primary") or user_input:
        if not user_input:
            st.warning("Please enter a cryptocurrency query.")
            return

        # SYMBOL DETECTION
        input_lower = user_input.lower()
        symbol = None

        for key, value in SUPPORTED_CRYPTOS.items():
            if key in input_lower:
                symbol = value
                break

        if not symbol:
            st.error("Cryptocurrency symbol not found in your query.")
            return

        # Run analysis
        with st.spinner(f"Analyzing {symbol.upper()}..."):
            try:
                analyzer = load_analyzer(None)
                formatter = ResultFormatter()
                result = analyzer.analyze(symbol, user_input)
                formatted = formatter.format_analysis(result)

                if formatted["status"] == "error":
                    st.error(formatted["message"])
                    return

                data = formatted["data"]

                # METRICS ROW
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        f"{data['symbol']} Price",
                        f"${data['current_price']:,.2f}",
                        f"{data['price_change_24h']:+.2f}%"
                    )

                with col2:
                    mc = data['market_cap']
                    mc_str = f"${mc/1e9:.1f}B" if mc > 1e9 else f"${mc/1e6:.1f}M"
                    st.metric("Market Cap", mc_str)

                with col3:
                    vol = data['risk_management']['volatility_30d']
                    st.metric("30D Volatility", f"{vol:.1f}%")

                with col4:
                    st.metric("Overall Confidence", f"{data['recommendation']['confidence']:.1f}%")

                # Recommendation Card
                display_recommendation_card(data["recommendation"])

                # TABS — Confidence Breakdown Removed
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Analysis Summary",
                    "🔮 7-Day Forecast",
                    "⚡ Technical",
                    "📰 Sentiment"
                ])

                # TAB 1 — Summary (Confidence Breakdown Removed)
                with tab1:
                    st.subheader("💡 Analysis Summary")
                    st.write(data['recommendation']['reasoning'])

                    colA, colB = st.columns(2)

                    with colA:
                        st.subheader("🛡️ Risk Management")
                        rm = data['risk_management']
                        if rm["stop_loss"]:
                            st.write(f"🔻 Stop Loss: ${rm['stop_loss']:,.2f}")
                        if rm["take_profit"]:
                            st.write(f"🎯 Take Profit: ${rm['take_profit']:,.2f}")
                        if rm["support_level"]:
                            st.write(f"📊 Support Level: ${rm['support_level']:,.2f}")
                        if rm["resistance_level"]:
                            st.write(f"📈 Resistance Level: ${rm['resistance_level']:,.2f}")


                # TAB 2 — Forecast (unchanged)
                with tab2:
                    st.subheader("🔮 7-Day Price Forecast")

                    hist = analyzer.data_fetcher.fetch_historical_data(symbol, days=30)
                    fig = make_subplots(rows=1, cols=1)
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['price'], name="Price"))
                    st.plotly_chart(fig, use_container_width=True)

                # TAB 3 — Technical (unchanged)
                with tab3:
                    tech = data["technical"]
                    st.metric("RSI", f"{tech['rsi']:.1f}")
                    st.metric("MACD", tech["macd_signal"])

                # TAB 4 — Sentiment (unchanged)
                with tab4:
                    sentiment = data["sentiment"]
                    st.metric("Sentiment Score", f"{sentiment['score']:.3f}")

                st.markdown("---")
                st.markdown(f"*Analysis completed at {data['timestamp']}*")

            except Exception as e:
                st.error(str(e))
                logger.error(e)


if __name__ == "__main__":
    main()
