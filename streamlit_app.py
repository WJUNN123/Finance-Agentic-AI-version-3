""" Streamlit UI for Crypto Investment Analysis """
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
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
.stAlert > div { padding-top: 10px; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(ttl=300)
def load_analyzer(csv_path=None):
    from crypto_analyzer import CryptoAnalyzer
    return CryptoAnalyzer()


def create_price_chart(historical_data: pd.DataFrame, forecast_data: List[Dict], symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Price Chart with Forecast', 'Volume'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )

    if not historical_data.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_data.index[-30:],
                y=historical_data['price'].iloc[-30:],
                mode='lines',
                name='Historical Price',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=historical_data.index[-30:],
                y=historical_data['volume'].iloc[-30:],
                name='Volume',
                opacity=0.7,
                marker_color='lightblue'
            ),
            row=2, col=1
        )

    if forecast_data:
        forecast_dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in forecast_data]
        forecast_prices = [item['predicted_price'] for item in forecast_data]

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_prices,
                mode='lines+markers',
                name='7-Day Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

    fig.update_layout(
        title=f'{symbol.upper()} Price Analysis',
        xaxis_title='Date',
        height=600,
        showlegend=True,
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_forecast_table(forecast_data: List[Dict], current_price: float) -> pd.DataFrame:
    if not forecast_data:
        return pd.DataFrame()

    df_forecast = pd.DataFrame(forecast_data)
    df_forecast['Change (%)'] = df_forecast['predicted_price'].pct_change() * 100
    df_forecast['Change ($)'] = df_forecast['predicted_price'].diff()
    df_forecast['Total Change (%)'] = (df_forecast['predicted_price'] - current_price) / current_price * 100

    df_forecast['predicted_price'] = df_forecast['predicted_price'].apply(lambda x: f"${x:.2f}")
    df_forecast['Change (%)'] = df_forecast['Change (%)'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A")
    df_forecast['Change ($)'] = df_forecast['Change ($)'].apply(lambda x: f"${x:+.2f}" if not pd.isna(x) else "N/A")
    df_forecast['Total Change (%)'] = df_forecast['Total Change (%)'].apply(lambda x: f"{x:+.2f}%")

    df_forecast.rename(columns={'date': 'Date', 'predicted_price': 'Predicted Price'}, inplace=True)
    return df_forecast[['Date', 'Predicted Price', 'Change (%)', 'Change ($)', 'Total Change (%)']]


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
    st.title("🚀 Crypto Investment Analyzer")
    st.markdown("*Ask about BTC, ETH, SOL, etc. This app renders a single, clean Summary dashboard. Educational only — not financial advice.*")

    # Removed sidebar ----- (per your requirement)

    # MAIN INPUT
    st.subheader("Your message")
    user_input = st.text_input(
        "Enter your query:",
        value=st.session_state.get('user_input', ''),
        placeholder="E.g. 'ETH 7-day forecast' or 'Should I buy BTC?'",
        key="main_input"
    )

    if st.button("Analyze", type="primary") or user_input:
        if not user_input:
            st.warning("Please enter a cryptocurrency query.")
            return

        input_lower = user_input.lower()
        symbol = None
        for key, value in SUPPORTED_CRYPTOS.items():
            if key in input_lower:
                symbol = value
                break

        if not symbol:
            st.error(f"Cryptocurrency not found: '{user_input}'.")
            return

        with st.spinner(f"Analyzing {symbol.upper()}..."):
            try:
                analyzer = load_analyzer()
                formatter = ResultFormatter()

                result = analyzer.analyze(symbol, user_input)
                formatted_result = formatter.format_analysis(result)

                if formatted_result['status'] == 'error':
                    st.error(f"Analysis failed: {formatted_result['message']}")
                    return

                data = formatted_result.get('data', {})

                # Safe retrievals
                symbol = data.get('symbol', 'UNKNOWN')
                current_price = data.get('current_price', 0.0)
                price_change = data.get('price_change_24h', 0.0)
                recommendation = data.get('recommendation', {})
                confidence = recommendation.get('confidence', 50.0)
                confidence_level = recommendation.get('confidence_level', 'Medium')
                risk_management = data.get('risk_management', {})
                vol = risk_management.get('volatility_30d', 0.0)
                market_cap = data.get('market_cap', 0)
                market_rank = data.get('market_cap_rank', 0)

                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"{symbol} Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
                with col2:
                    if market_cap > 1e9:
                        cap = f"${market_cap/1e9:.1f}B"
                    elif market_cap > 1e6:
                        cap = f"${market_cap/1e6:.1f}M"
                    else:
                        cap = f"${market_cap:,.0f}"
                    st.metric("Market Cap", cap, f"Rank #{market_rank}")
                with col3:
                    st.metric("30D Volatility", f"{vol:.1f}%", "High" if vol > 70 else "Medium" if vol > 40 else "Low")
                with col4:
                    st.metric("Overall Confidence", f"{confidence:.1f}%", confidence_level)

                display_recommendation_card(recommendation)

                # ---- TABS ----
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Analysis Summary",
                    "🔮 7-Day Forecast",
                    "⚡ Technical",
                    "📰 Sentiment"
                ])

                # TAB 1 — Analysis Summary
                with tab1:
                    st.subheader("💡 Analysis Summary")
                    st.write(recommendation.get('reasoning', 'No reasoning available.'))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🛡️ Risk Management")
                        r = risk_management
                        if r.get('stop_loss'): st.write(f"🔻 **Stop Loss**: ${r['stop_loss']:,.2f}")
                        if r.get('take_profit'): st.write(f"🎯 **Take Profit**: ${r['take_profit']:,.2f}")
                        if r.get('support_level'): st.write(f"📊 **Support Level**: ${r['support_level']:,.2f}")
                        if r.get('resistance_level'): st.write(f"📈 **Resistance Level**: ${r['resistance_level']:,.2f}")
                        if recommendation.get('position_size', 0) > 0:
                            st.write(f"💰 **Suggested Position Size**: {recommendation.get('position_size')}%")

                # TAB 2 — Forecast
                with tab2:
                    st.subheader("🔮 7-Day Price Forecast")
                    if data.get('forecast'):
                        hist = analyzer.data_fetcher.fetch_historical_data(symbol, days=30)
                        chart = create_price_chart(hist, data['forecast'], symbol)
                        st.plotly_chart(chart, use_container_width=True)

                        st.subheader("📅 Detailed Forecast")
                        df = create_forecast_table(data['forecast'], current_price)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Forecast data not available.")

                # TAB 3 — Technical Analysis
                with tab3:
                    st.subheader("⚡ Technical Analysis")
                    tech = data.get('technical', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI", f"{tech.get('rsi',0):.1f}", tech.get('rsi_signal','N/A'))
                    with col2:
                        st.metric("MACD Signal", tech.get('macd_signal','N/A'))
                    with col3:
                        st.metric("Market Regime", tech.get('market_regime','N/A').title())
                    st.write("**Technical Summary:**")
                    st.write(f"- **Trend**: {tech.get('trend','N/A')}")
                    st.write(f"- **Technical Signal**: {tech.get('technical_signal','N/A')}")
                    hist_data = analyzer.data_fetcher.fetch_historical_data(symbol, days=30)
                    if not hist_data.empty:
                        fig_tech = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=['Price & MA', 'RSI', 'Volume'],
                            vertical_spacing=0.1,
                            row_heights=[0.5, 0.25, 0.25]
                        )
                        fig_tech.add_trace(go.Scatter(x=hist_data.index, y=hist_data['price'], name='Price'), row=1, col=1)
                        st.plotly_chart(fig_tech, use_container_width=True)

                # TAB 4 — Sentiment
                with tab4:
                    st.subheader("📰 Market Sentiment")
                    sentiment = data.get('sentiment', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment Score", f"{sentiment.get('score',0):.3f}", sentiment.get('label','N/A'))
                    with col2:
                        st.metric("News Headlines", sentiment.get('headline_count',0))
                    headlines = sentiment.get('headlines',[])
                    if headlines:
                        st.subheader("📰 Recent Headlines")
                        for i, h in enumerate(headlines, 1):
                            st.write(f"{i}. {h}")
                    else:
                        st.info("No recent headlines available.")

                st.markdown("---")
                st.markdown(f"*Analysis completed at {data.get('timestamp','N/A')}*")
                st.markdown("*Educational purpose only — not financial advice.*")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
