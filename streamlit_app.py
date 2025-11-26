"""
Streamlit UI for Crypto Investment Analysis
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

# These imports will fail if the other files are not in the same directory.
# Make sure crypto_analyzer.py and config.py are present.
from crypto_analyzer import CryptoAnalyzer, ResultFormatter
from config import Config, SUPPORTED_CRYPTOS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="collapsed" # Collapsed by default
)

# Custom CSS
st.markdown("""
<style>
    /* Hide the default Streamlit sidebar button since we aren't using it */
    [data-testid="stSidebarNav"] {display: none;}
    section[data-testid="stSidebar"] {display: none;}

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
    
    .stAlert > div {
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_analyzer(csv_path=None):
    """Load the crypto analyzer with caching"""
    return CryptoAnalyzer(csv_file_path=csv_path)

def create_price_chart(historical_data: pd.DataFrame, forecast_data: List[Dict], 
                        symbol: str) -> go.Figure:
    """Create an interactive price chart with forecast"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Price Chart with Forecast', 'Volume'],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Historical prices
    if not historical_data.empty:
        fig.add_trace(
            go.Scatter(
                x=historical_data.index[-30:],  # Last 30 days
                y=historical_data['price'].iloc[-30:],
                mode='lines',
                name='Historical Price',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Volume
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
    
    # Forecast
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
    """Create a formatted forecast table"""
    if not forecast_data:
        return pd.DataFrame()
        
    df_forecast = pd.DataFrame(forecast_data)
    df_forecast['Change (%)'] = df_forecast['predicted_price'].pct_change() * 100
    df_forecast['Change ($)'] = df_forecast['predicted_price'].diff()
    df_forecast['Total Change (%)'] = ((df_forecast['predicted_price'] - current_price) / current_price * 100)
    
    # Format columns
    df_forecast['predicted_price'] = df_forecast['predicted_price'].apply(lambda x: f"${x:.2f}")
    df_forecast['Change (%)'] = df_forecast['Change (%)'].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "N/A")
    df_forecast['Change ($)'] = df_forecast['Change ($)'].apply(lambda x: f"${x:+.2f}" if not pd.isna(x) else "N/A")
    df_forecast['Total Change (%)'] = df_forecast['Total Change (%)'].apply(lambda x: f"{x:+.2f}%")
    
    # Rename columns
    df_forecast.rename(columns={
        'date': 'Date',
        'predicted_price': 'Predicted Price'
    }, inplace=True)
    
    return df_forecast[['Date', 'Predicted Price', 'Change (%)', 'Change ($)', 'Total Change (%)']]

def display_recommendation_card(recommendation: Dict):
    """Display the main recommendation card"""
    action = recommendation.get('action', 'Hold')
    confidence = recommendation.get('confidence', 50)
    risk_level = recommendation.get('risk_level', 'Medium')
    
    # Determine colors
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
    """Main Streamlit application"""
    
    # Header
    st.title("🚀 Crypto Investment Analyzer")
    st.markdown("*Ask about BTC, ETH, SOL, etc. This app renders a single, clean Summary dashboard. Educational only — not financial advice.*")
    
    # --- MOVED CONFIGURATION TO EXPANDER ---
    with st.expander("⚙️ Settings & Shortcuts (Data Upload, Quick Coins)", expanded=False):
        col_set1, col_set2 = st.columns(2)
        
        with col_set1:
            st.subheader("Upload Data")
            uploaded_file = st.file_uploader(
                "Upload Historical Data (CSV)", 
                type=['csv'],
                help="Upload your historical crypto data CSV file"
            )

        with col_set2:
            st.subheader("Quick Coins")
            quick_coins = ['Bitcoin', 'Ethereum', 'Solana', 'BNB', 'XRP', 'Cardano', 'Dogecoin']
            # Display buttons in a flex-wrap style using columns
            qc_cols = st.columns(4)
            for i, coin in enumerate(quick_coins):
                if qc_cols[i % 4].button(coin, key=f"quick_{coin}"):
                    st.session_state['selected_coin'] = coin.lower()
                    st.session_state['user_input'] = f"Should I buy {coin}?" # Auto-fill prompt

    # Main input
    st.subheader("Your message")
    user_input = st.text_input(
        "Enter your query:",
        value=st.session_state.get('user_input', ''),
        placeholder="E.g. 'ETH 7-day forecast' or 'Should I buy BTC?'",
        key="main_input"
    )
    
    # Process input
    if st.button("Analyze", type="primary") or user_input:
        if not user_input:
            st.warning("Please enter a cryptocurrency query.")
            return
            
        # Extract symbol from input
        input_lower = user_input.lower()
        symbol = None
        
        for key, value in SUPPORTED_CRYPTOS.items():
            if key in input_lower:
                symbol = value
                break
                
        if not symbol:
            st.error(f"Cryptocurrency not found in query: '{user_input}'. Please include a supported symbol like BTC, ETH, SOL, etc.")
            return
        
        # Show loading
        with st.spinner(f"Analyzing {symbol.upper()}..."):
            try:
                # Load analyzer
                csv_path = None
                if uploaded_file:
                    csv_path = f"temp_{uploaded_file.name}"
                    with open(csv_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                
                analyzer = load_analyzer(csv_path)
                formatter = ResultFormatter()
                
                # Perform analysis
                result = analyzer.analyze(symbol, user_input)
                formatted_result = formatter.format_analysis(result)
                
                # Clean up temp file
                if csv_path and os.path.exists(csv_path):
                    os.remove(csv_path)
                
                if formatted_result['status'] == 'error':
                    st.error(f"Analysis failed: {formatted_result['message']}")
                    return
                
                data = formatted_result['data']
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        f"{data['symbol']} Price",
                        f"${data['current_price']:,.2f}",
                        f"{data['price_change_24h']:+.2f}%"
                    )
                
                with col2:
                    market_cap = data['market_cap']
                    if market_cap > 1e9:
                        market_cap_str = f"${market_cap/1e9:.1f}B"
                    elif market_cap > 1e6:
                        market_cap_str = f"${market_cap/1e6:.1f}M"
                    else:
                        market_cap_str = f"${market_cap:,.0f}"
                        
                    st.metric(
                        "Market Cap",
                        market_cap_str,
                        f"Rank #{data['market_cap_rank']}" if data['market_cap_rank'] != 'N/A' else None
                    )
                
                with col3:
                    volatility = data['risk_management']['volatility_30d']
                    st.metric(
                        "30D Volatility", 
                        f"{volatility:.1f}%",
                        "High" if volatility > 70 else "Medium" if volatility > 40 else "Low"
                    )
                
                with col4:
                    st.metric(
                        "Overall Confidence",
                        f"{data['recommendation']['confidence']:.1f}%",
                        data['recommendation']['confidence_level']
                    )
                
                # Recommendation Card
                display_recommendation_card(data['recommendation'])
                
                # Main content tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis Summary", "🔮 7-Day Forecast", "⚡ Technical", "📰 Sentiment"])
                
                with tab1:
                    st.subheader("💡 Analysis Summary")
                    st.write(data['recommendation']['reasoning'])
                    
                    st.markdown("---")
                    
                    # --- REMOVED CONFIDENCE BREAKDOWN / RESTRUCTURED RISK ---
                    st.subheader("🛡️ Risk Management")
                    risk_data = data['risk_management']
                    
                    # Create 3 columns for risk metrics
                    r_col1, r_col2, r_col3 = st.columns(3)
                    
                    with r_col1:
                        if risk_data['stop_loss']:
                            st.metric("🔻 Stop Loss", f"${risk_data['stop_loss']:,.2f}")
                        if risk_data['support_level']:
                            st.metric("📊 Support Level", f"${risk_data['support_level']:,.2f}")

                    with r_col2:
                        if risk_data['take_profit']:
                            st.metric("🎯 Take Profit", f"${risk_data['take_profit']:,.2f}")
                        if risk_data['resistance_level']:
                            st.metric("📈 Resistance Level", f"${risk_data['resistance_level']:,.2f}")
                            
                    with r_col3:
                        pos_size = data['recommendation']['position_size']
                        if pos_size > 0:
                            st.metric("💰 Sug. Position Size", f"{pos_size}%")
                        else:
                            st.metric("💰 Sug. Position Size", "0%")

                with tab2:
                    st.subheader("🔮 7-Day Price Forecast")
                    
                    if data['forecast']:
                        historical_df = analyzer.data_fetcher.fetch_historical_data(symbol, days=30)
                        price_chart = create_price_chart(historical_df, data['forecast'], data['symbol'])
                        st.plotly_chart(price_chart, use_container_width=True)
                        
                        st.subheader("📅 Detailed Forecast")
                        forecast_df = create_forecast_table(data['forecast'], data['current_price'])
                        st.dataframe(forecast_df, use_container_width=True)
                    else:
                        st.warning("Forecast data not available. This could be due to insufficient historical data.")

                with tab3:
                    st.subheader("⚡ Technical Analysis")
                    tech = data['technical']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI", f"{tech['rsi']:.1f}", tech['rsi_signal'])
                    with col2:
                        st.metric("MACD Signal", tech['macd_signal'], None)
                    with col3:
                        st.metric("Market Regime", tech['market_regime'].title(), None)
                    
                    st.write("**Technical Summary:**")
                    st.write(f"- **Trend**: {tech['trend']}")
                    st.write(f"- **Technical Signal**: {tech['technical_signal']} ({'Bullish' if tech['technical_signal'] > 20 else 'Bearish' if tech['technical_signal'] < -20 else 'Neutral'})")
                    
                    hist_data_tech = analyzer.data_fetcher.fetch_historical_data(symbol, days=30)
                    if not hist_data_tech.empty:
                        fig_tech = make_subplots(rows=3, cols=1, subplot_titles=['Price & Moving Averages', 'RSI', 'Volume'], vertical_spacing=0.1, row_heights=[0.5, 0.25, 0.25])
                        
                        # Price and moving averages
                        fig_tech.add_trace(go.Scatter(x=hist_data_tech.index, y=hist_data_tech['price'], name='Price', line=dict(color='blue')), row=1, col=1)
                        if len(hist_data_tech) >= 7:
                            sma_7 = hist_data_tech['price'].rolling(7).mean()
                            fig_tech.add_trace(go.Scatter(x=hist_data_tech.index, y=sma_7, name='SMA 7', line=dict(color='orange', dash='dash')), row=1, col=1)
                        if len(hist_data_tech) >= 21:
                            sma_21 = hist_data_tech['price'].rolling(21).mean()
                            fig_tech.add_trace(go.Scatter(x=hist_data_tech.index, y=sma_21, name='SMA 21', line=dict(color='red', dash='dot')), row=1, col=1)
                            
                        # RSI
                        if len(hist_data_tech) >= 14:
                            delta = hist_data_tech['price'].diff()
                            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                            rs = gain / (loss + 1e-8)
                            rsi = 100 - (100 / (1 + rs))
                            fig_tech.add_trace(go.Scatter(x=hist_data_tech.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
                            fig_tech.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=2, col=1)
                            fig_tech.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=2, col=1)
                        
                        # Volume
                        fig_tech.add_trace(go.Bar(x=hist_data_tech.index, y=hist_data_tech['volume'], name='Volume', marker_color='lightblue'), row=3, col=1)
                        
                        fig_tech.update_layout(height=600, showlegend=True, title="Technical Indicators")
                        fig_tech.update_yaxes(title_text="Price (USD)", row=1, col=1)
                        fig_tech.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
                        fig_tech.update_yaxes(title_text="Volume", row=3, col=1)
                        
                        st.plotly_chart(fig_tech, use_container_width=True)

                with tab4:
                    st.subheader("📰 Market Sentiment")
                    sentiment = data['sentiment']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment Score", f"{sentiment['score']:.3f}", sentiment['label'])
                    with col2:
                        st.metric("News Headlines", sentiment['headline_count'], None)
                    
                    if sentiment['headlines']:
                        st.subheader("📰 Recent Headlines")
                        for i, headline in enumerate(sentiment['headlines'], 1):
                            st.write(f"{i}. {headline}")
                    else:
                        st.info("No recent headlines available for sentiment analysis.")
                
                # Footer
                st.markdown("---")
                st.markdown(f"*Analysis completed at {data['timestamp']}*")
                st.markdown("*This is educational content only — not financial advice. Always do your own research.*")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                logger.error(f"Analysis error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
