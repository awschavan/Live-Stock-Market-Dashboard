"""
📊 Live Stock Market Dashboard 
Filename: app.py
"""
from datetime import datetime
import argparse
import os
import requests
import io

# ---------------------- Core Libraries ----------------------
try:
    import pandas as pd
    import numpy as np
except Exception:
    raise RuntimeError("Install required packages: pip install pandas numpy")

# Optional Libraries
try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import plotly.graph_objs as go
    import plotly.express as px
except Exception:
    go = None
    px = None

# Detect Streamlit
STREAMLIT_AVAILABLE = True
try:
    import streamlit as st
except Exception:
    STREAMLIT_AVAILABLE = False

# ---------------------- Technical Indicators ----------------------
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = ema(series, span_short)
    ema_long = ema(series, span_long)
    macd_line = ema_short - ema_long
    signal = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

# ---------------------- Data Fetch ----------------------
if STREAMLIT_AVAILABLE:
    @st.cache_data(ttl=300)
    def fetch_data(ticker, period="1mo", interval="1d"):
        if yf is None:
            st.error("Install yfinance: pip install yfinance")
            return None
        try:
            df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False, auto_adjust=False)
            if df is None or df.empty:
                return None
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            return df.dropna()
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
            return None
else:
    def fetch_data(ticker, period="1mo", interval="1d"):
        if yf is None:
            print("Install yfinance to fetch live data: pip install yfinance")
            return None
        try:
            df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False, auto_adjust=False)
            if df is None or df.empty:
                return None
            if "Close" not in df.columns and "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            return df.dropna()
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

# ---------------------- Company List ----------------------
POPULAR_COMPANIES = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS"
}

# ---------------------- Streamlit UI ----------------------
def run_streamlit_ui():
    st.set_page_config(page_title="Live Stock Dashboard", layout="wide")
    st.title("📊 Live Stock Market Dashboard — Students · Traders · Analysts")
    st.caption("Analyze live stock data, visualize trends, and explore insights interactively.")

    # Sidebar
    with st.sidebar:
        st.header("Settings ⚙️")
        selected = st.multiselect("Select companies", list(POPULAR_COMPANIES.keys()), ["Apple (AAPL)", "Tesla (TSLA)"])
        tickers = [POPULAR_COMPANIES[name] for name in selected]

        extra = st.text_input("Add more tickers (comma separated)", "")
        if extra:
            tickers += [t.strip().upper() for t in extra.split(",") if t.strip()]

        period = st.selectbox("Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        interval = st.selectbox("Data Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], index=5)
        chart_type = st.radio("Chart Type", ["Line Chart", "Candlestick Chart"], index=0)

        st.markdown("---")
        st.subheader("Indicators 🎯")
        indicators = st.multiselect("Select Indicators", ["SMA", "EMA", "RSI", "MACD"], default=["SMA", "RSI"])
        sma_window = st.number_input("SMA Window", 2, 200, 20)
        ema_span = st.number_input("EMA Span", 2, 200, 20)
        rsi_window = st.number_input("RSI Window", 5, 50, 14)

    if not tickers:
        st.warning("Please select at least one ticker.")
        return

    tab1, tab2 = st.tabs(["📈 Dashboard", "📊 Research & Insights"])

    # ========================= TAB 1 ============================
    with tab1:
        st.subheader(f"📈 Comparing: {', '.join(tickers)}")
        combined = pd.DataFrame()
        for t in tickers:
            d = fetch_data(t, period, interval)
            if d is not None and not d.empty:
                combined[t] = d['Close']

        if combined.empty:
            st.error("No valid data received. Try changing period or tickers.")
            return

        if chart_type == "Line Chart":
            norm = combined.div(combined.iloc[0]).mul(100)
            fig_main = go.Figure()
            for col in norm.columns:
                fig_main.add_trace(go.Scatter(x=norm.index, y=norm[col], mode='lines', name=col))
            fig_main.update_layout(title="Normalized Close Prices (Base 100)", xaxis_title="Date", yaxis_title="Price Index")
        else:
            primary = tickers[0]
            d_primary = fetch_data(primary, period, interval)
            if d_primary is not None and not d_primary.empty:
                fig_main = go.Figure(data=[go.Candlestick(
                    x=d_primary.index, open=d_primary['Open'], high=d_primary['High'],
                    low=d_primary['Low'], close=d_primary['Close'], name=primary
                )])
                fig_main.update_layout(title=f"{primary} Candlestick Chart", xaxis_rangeslider_visible=False)
            else:
                st.warning("Fallback to line chart due to missing OHLC data.")
                norm = combined.div(combined.iloc[0]).mul(100)
                fig_main = go.Figure()
                for col in norm.columns:
                    fig_main.add_trace(go.Scatter(x=norm.index, y=norm[col], mode='lines', name=col))
        st.plotly_chart(fig_main, use_container_width=True)

        st.markdown("---")
        primary = tickers[0]
        df = fetch_data(primary, period, interval)
        if df is None or df.empty:
            st.warning(f"No detailed data for {primary}.")
            return

        st.subheader(f"Detailed Analysis: {primary}")
        fig_detail = go.Figure()
        if chart_type == "Candlestick Chart":
            fig_detail.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=primary))
        else:
            fig_detail.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

        if "SMA" in indicators:
            df[f"SMA_{sma_window}"] = sma(df['Close'], sma_window)
            fig_detail.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{sma_window}"], name=f'SMA {sma_window}'))
        if "EMA" in indicators:
            df[f"EMA_{ema_span}"] = ema(df['Close'], ema_span)
            fig_detail.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{ema_span}"], name=f'EMA {ema_span}'))

        fig_detail.update_layout(title=f"{primary} Price with Indicators", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_detail, use_container_width=True)

        with st.expander("📊 Technical Indicator Charts"):
            if "RSI" in indicators:
                df[f"RSI_{rsi_window}"] = rsi(df['Close'], rsi_window)
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df[f"RSI_{rsi_window}"], name='RSI'))
                fig_rsi.add_hline(y=70, line_dash='dash', annotation_text='Overbought')
                fig_rsi.add_hline(y=30, line_dash='dash', annotation_text='Oversold')
                st.plotly_chart(fig_rsi, use_container_width=True)

            if "MACD" in indicators:
                macd_line, signal, hist = macd(df['Close'])
                df["MACD_Line"], df["Signal"], df["Histogram"] = macd_line, signal, hist
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=macd_line, name='MACD'))
                fig_macd.add_trace(go.Scatter(x=df.index, y=signal, name='Signal'))
                fig_macd.add_trace(go.Bar(x=df.index, y=hist, name='Histogram'))
                st.plotly_chart(fig_macd, use_container_width=True)

        st.download_button(
            label='📥 Download CSV (with Indicators)',
            data=df.to_csv().encode('utf-8'),
            file_name=f"{primary}_data.csv",
            mime='text/csv'
        )

        st.caption("Built with ❤️ for students, traders, and analysts to explore live market data.")

    # ========================= TAB 2 ============================
    with tab2:
        st.subheader("📊 Research & Insights")
        primary = tickers[0]
        ticker_obj = yf.Ticker(primary)
        info = ticker_obj.info

        # --- Stock Summary ---
        st.markdown("### 📘 Stock Overview")
        if info:
            summary_data = {
                "Current Price": info.get("currentPrice", "N/A"),
                "Previous Close": info.get("previousClose", "N/A"),
                "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "Volume": info.get("volume", "N/A"),
                "Avg Volume": info.get("averageVolume", "N/A"),
                "P/E Ratio": info.get("trailingPE", "N/A"),
                "Dividend Yield": info.get("dividendYield", "N/A")
            }
            st.table(pd.DataFrame.from_dict(summary_data, orient='index', columns=['Value']))
        else:
            st.info("No detailed information available for this ticker.")

        # --- Correlation Matrix ---
        combined = pd.DataFrame()
        if len(tickers) > 1:
            st.markdown("### 🔗 Correlation Matrix")
            for t in tickers:
                d = fetch_data(t, period, interval)
                if d is not None and not d.empty:
                    combined[t] = d['Close']
            if not combined.empty:
                corr_df = combined.corr()
                fig_corr = px.imshow(corr_df, text_auto=True, title="Stock Correlation Matrix", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_corr, use_container_width=True)

        # --- Sentiment Overview ---
        st.markdown("### 🗞️ Sentiment Analysis (Recent News)")

        try:
            from textblob import TextBlob
            news_data_list = [] # List to store dicts of headline, url, sentiment

            # Fetch from yfinance
            news_items = getattr(ticker_obj, "news", []) or []
            for n in news_items[:10]:
                title = n.get("title") or n.get("content") or n.get("link_text")
                url = n.get("link")
                if title and url and isinstance(title, str):
                    news_data_list.append({"Headline": title, "URL": url})

            # Fallback to NewsAPI if no headlines from yfinance
            if not news_data_list:
                st.info("⏳ No news from yfinance — fetching live headlines...")
                api_key = "38e86b0734244344ace52fea67f9a570"
                url_newsapi = f"https://newsapi.org/v2/everything?q={primary}&language=en&sortBy=publishedAt&pageSize=10&apiKey={api_key}"
                resp = requests.get(url_newsapi)
                data = resp.json()
                for art in data.get("articles", []):
                    h = art.get("title")
                    u = art.get("url")
                    if h and u:
                        news_data_list.append({"Headline": h, "URL": u})

            if not news_data_list:
                st.warning("No recent news found from either source.")
            else:
                # Calculate sentiments and add to news_data_list
                for item in news_data_list:
                    item["Sentiment"] = TextBlob(item["Headline"]).sentiment.polarity

                df_sent = pd.DataFrame(news_data_list)
                avg_sent = df_sent["Sentiment"].mean()
                st.metric("Average Sentiment", f"{avg_sent:.2f}")

                fig_sent = px.histogram(df_sent, x="Sentiment", nbins=10, title="Sentiment Distribution")
                st.plotly_chart(fig_sent, use_container_width=True)

                st.markdown("**Recent News Headlines:**")
                for index, row in df_sent.iterrows():
                    st.markdown(f"- [{row['Headline']}]({row['URL']}) (Sentiment: {row['Sentiment']:.2f})")

                # ---------------------- PDF EXPORT ----------------------
                from fpdf import FPDF
                import tempfile
                import matplotlib.pyplot as plt
                import seaborn as sns

                st.markdown("### 📄 Export Research Report")

                if st.button("Generate PDF Report"):
                    try:
                        pdf = FPDF()
                        pdf.set_auto_page_break(auto=True, margin=15)
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 16)
                        # Removed emoji from the title for PDF export compatibility
                        pdf.cell(0, 10, f"Live Stock Market Report: {primary}", ln=True, align="C")
                        pdf.ln(5)
                        pdf.set_font("Arial", size=12)
                        pdf.multi_cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        pdf.multi_cell(0, 8, "Developed by: Swapnil Chavan")
                        pdf.multi_cell(0, 8, "Source: Yahoo Finance & NewsAPI")
                        pdf.ln(10)

                        pdf.set_font("Arial", "B", 14)
                        # Removed emoji from section title for PDF export compatibility
                        pdf.cell(0, 10, "Stock Overview", ln=True)
                        pdf.set_font("Arial", size=11)
                        for k, v in summary_data.items():
                            pdf.cell(0, 8, f"{k}: {v}", ln=True)
                        pdf.ln(8)

                        # --- SMA vs EMA Chart for PDF ---
                        if "SMA" in indicators or "EMA" in indicators:
                            pdf.set_font("Arial", "B", 14)
                            pdf.cell(0, 10, "Price Trends (SMA vs EMA)", ln=True)
                            pdf.ln(5)

                            sma_ema_img = io.BytesIO()
                            plt.figure(figsize=(10, 6))
                            plt.plot(df.index, df['Close'], label='Close Price', color='blue')

                            if "SMA" in indicators:
                                df_with_sma = df.copy()
                                df_with_sma[f"SMA_{sma_window}"] = sma(df_with_sma['Close'], sma_window)
                                plt.plot(df_with_sma.index, df_with_sma[f"SMA_{sma_window}"], label=f'SMA {sma_window}', color='red', linestyle='--')
                            if "EMA" in indicators:
                                df_with_ema = df.copy()
                                df_with_ema[f"EMA_{ema_span}"] = ema(df_with_ema['Close'], ema_span)
                                plt.plot(df_with_ema.index, df_with_ema[f"EMA_{ema_span}"], label=f'EMA {ema_span}', color='green', linestyle='-.')

                            plt.title('Price Trends (Close, SMA, and EMA)')
                            plt.xlabel('Date')
                            plt.ylabel('Price')
                            plt.legend()
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(sma_ema_img, format='png', bbox_inches='tight')
                            plt.close() # Close the figure to free memory

                            sma_ema_img.seek(0)
                            with open("sma_ema_temp.png", "wb") as f:
                                f.write(sma_ema_img.getbuffer())
                            pdf.image("sma_ema_temp.png", x=10, w=180)
                            os.remove("sma_ema_temp.png")
                            pdf.ln(10)

                        # --- MACD Chart for PDF ---
                        if "MACD" in indicators:
                            pdf.set_font("Arial", "B", 14)
                            pdf.cell(0, 10, "MACD (Moving Average Convergence Divergence)", ln=True)
                            pdf.ln(5)

                            macd_line_pdf, signal_pdf, hist_pdf = macd(df['Close'])

                            # Create a DataFrame for MACD plotting and drop NaNs
                            macd_data_for_plot = pd.DataFrame({
                                'MACD Line': macd_line_pdf,
                                'Signal Line': signal_pdf,
                                'Histogram': hist_pdf
                            }, index=df.index).dropna()

                            macd_img = io.BytesIO()
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                            ax1.plot(macd_data_for_plot.index, macd_data_for_plot['MACD Line'], label='MACD Line', color='blue')
                            ax1.plot(macd_data_for_plot.index, macd_data_for_plot['Signal Line'], label='Signal Line', color='red', linestyle='--')
                            ax1.set_title('MACD Line and Signal Line')
                            ax1.set_ylabel('Value')
                            ax1.legend()
                            ax1.grid(True)

                            ax2.bar(macd_data_for_plot.index, macd_data_for_plot['Histogram'], label='Histogram', color='grey', alpha=0.7)
                            ax2.set_title('MACD Histogram')
                            ax2.set_xlabel('Date')
                            ax2.set_ylabel('Value')
                            ax2.legend()
                            ax2.grid(True)

                            plt.tight_layout()
                            plt.savefig(macd_img, format='png', bbox_inches='tight')
                            plt.close() # Close the figure to free memory

                            macd_img.seek(0)
                            with open("macd_temp.png", "wb") as f:
                                f.write(macd_img.getbuffer())
                            pdf.image("macd_temp.png", x=10, w=180)
                            os.remove("macd_temp.png")
                            pdf.ln(10)


                        # --- Correlation Matrix for PDF ---
                        if len(tickers) > 1 and not combined.empty:
                            pdf.set_font("Arial", "B", 14)
                            pdf.cell(0, 10, "Stock Correlation Matrix", ln=True)
                            pdf.ln(5)
                            corr_img = io.BytesIO()
                            plt.figure(figsize=(8, 6))
                            sns.heatmap(corr_df, annot=True, cmap="RdBu_r", fmt=".2f")
                            plt.title("Stock Correlation Matrix")
                            plt.tight_layout()
                            plt.savefig(corr_img, format='png', bbox_inches='tight')
                            plt.close() # Close the figure to free memory

                            corr_img.seek(0)
                            with open("corr_temp.png", "wb") as f:
                                f.write(corr_img.getbuffer())
                            pdf.image("corr_temp.png", x=10, w=180)
                            os.remove("corr_temp.png")
                            pdf.ln(10)

                        # --- Sentiment Summary for PDF ---
                        pdf.set_font("Arial", "B", 14)
                        pdf.cell(0, 10, "Sentiment Summary", ln=True)
                        pdf.set_font("Arial", size=11)
                        pdf.cell(0, 8, f"Average Sentiment: {avg_sent:.2f}", ln=True)
                        pdf.ln(5)

                        sent_img = io.BytesIO()
                        plt.figure(figsize=(8, 4))
                        sns.histplot(df_sent["Sentiment"], bins=10, kde=True)
                        plt.title("Sentiment Distribution")
                        plt.xlabel("Sentiment")
                        plt.ylabel("Frequency")
                        plt.tight_layout()
                        plt.savefig(sent_img, format='png', bbox_inches='tight')
                        plt.close() # Close the figure to free memory

                        sent_img.seek(0)
                        with open("sent_temp.png", "wb") as f:
                            f.write(sent_img.getbuffer())
                        pdf.image("sent_temp.png", x=10, w=180)
                        os.remove("sent_temp.png")
                        pdf.ln(10)

                        pdf.set_font("Arial", "I", 10)
                        pdf.multi_cell(0, 8, "This report was auto-generated using the Live Stock Market Dashboard developed by Swapnil Chavan.")

                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"{primary}_Research_Report.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"⚠️ PDF Generation Failed: {e}")
        except Exception as e:
            st.warning(f"⚠️ Unable to analyze sentiment: {e}")

# ---------------------- CLI Fallback ----------------------
def run_cli_mode(tickers, period, interval, out_dir='output', indicators=None):
    os.makedirs(out_dir, exist_ok=True)
    for t in tickers:
        print(f"Fetching {t}...")
        df = fetch_data(t, period, interval)
        if df is None or df.empty:
            continue
        csv_path = os.path.join(out_dir, f"{t.replace('.', '_')}.csv")
        df.to_csv(csv_path)
        print(f"Saved {csv_path}")
    print("CLI run complete.")

# ---------------------- Entry Point ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', nargs='+', default=['AAPL'])
    p.add_argument('--period', default='1mo')
    p.add_argument('--interval', default='1d')
    p.add_argument('--out', default='output')
    p.add_argument('--indicators', nargs='*', default=[])
    # Parse known arguments and ignore unknown ones (like -f from Jupyter)
    args, unknown = p.parse_known_args()
    return args

if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        try:
            run_streamlit_ui()
        except Exception as e:
            print(f"Streamlit UI failed: {e}\nFalling back to CLI mode.")
            args = parse_args()
            run_cli_mode(args.tickers, args.period, args.interval, out_dir=args.out, indicators=args.indicators)
    else:
        print("Streamlit not installed. Running CLI fallback.")
        args = parse_args()
        run_cli_mode(args.tickers, args.period, args.interval, out_dir=args.out, indicators=args.indicators)
