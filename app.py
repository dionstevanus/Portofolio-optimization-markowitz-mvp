import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import matplotlib.pyplot as plt
import io

from data_processor import DataProcessor
from markowitz_optimizer import MarkowitzOptimizer
from rl_optimizer import RLOptimizer
from visualizer import Visualizer

# Set page configuration
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📊",
    layout="wide"
)

# Application title and description
st.title("Portfolio Optimization: Markowitz Model & Reinforcement Learning")
st.markdown("""
This application uses the Markowitz Model and Reinforcement Learning to optimize investment portfolios.
It helps identify the optimal asset allocation to maximize returns while minimizing risk.
""")

# Additional information for IDX
with st.expander("About Indonesian (IDX) Stock Market"):
    st.markdown("""
    ### Jakarta Stock Exchange (IDX)
    The Indonesia Stock Exchange (IDX) is a stock exchange based in Jakarta, Indonesia. 
    It was previously known as the Jakarta Stock Exchange (JSX) before its name changed in 2007.
    
    ### Popular IDX Stocks Include:
    - **BBCA.JK**: Bank Central Asia Tbk (BCA)
    - **BBRI.JK**: Bank Rakyat Indonesia Tbk
    - **ASII.JK**: Astra International Tbk
    - **TLKM.JK**: Telekomunikasi Indonesia Tbk
    - **BMRI.JK**: Bank Mandiri Tbk
    - **UNVR.JK**: Unilever Indonesia Tbk
    - **HMSP.JK**: HM Sampoerna Tbk
    - **ICBP.JK**: Indofood CBP Sukses Makmur Tbk
    
    ### Financial Years:
    The financial year in Indonesia follows the calendar year, starting from January 1 to December 31.
    """)

# Sidebar for user inputs
st.sidebar.header("Portfolio Settings")

# Time period selection
start_date = st.sidebar.date_input(
    "Start Date",
    datetime.date(2018, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    datetime.date.today()
)

# Stock selection
st.sidebar.subheader("Select Stocks")

# Market selection
market = st.sidebar.selectbox(
    "Select Market",
    ["US Market", "Indonesia (IDX)"]
)

# Default tickers based on market
if market == "Indonesia (IDX)":
    default_tickers = "BBCA,BBRI,ASII,TLKM,BMRI"
    st.sidebar.info("For Indonesian stocks, no need to add .JK suffix - it will be added automatically. Popular IDX stocks: BBCA, BBRI, ASII, TLKM, BMRI, UNVR, HMSP, ICBP, PGAS, ANTM")
else:
    default_tickers = "AAPL,MSFT,GOOG,AMZN,META"

ticker_input = st.sidebar.text_area(
    "Enter Stock Symbols (comma-separated)",
    default_tickers
)

# Parse ticker symbols
if market == "Indonesia (IDX)":
    # Add .JK suffix if not already present
    tickers = []
    for ticker in ticker_input.split(','):
        ticker = ticker.strip()
        if ticker and not ticker.endswith('.JK'):
            tickers.append(f"{ticker}.JK")
        elif ticker:
            tickers.append(ticker)
else:
    tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]

# Show selected tickers
st.sidebar.caption(f"Selected tickers: {', '.join(tickers)}")

# Risk-free rate for portfolio optimization
risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
) / 100

# Monte Carlo simulation parameters
st.sidebar.subheader("Monte Carlo Settings")
num_portfolios = st.sidebar.slider(
    "Number of Simulations",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=1000
)

# Reinforcement Learning parameters
st.sidebar.subheader("Reinforcement Learning Settings")
training_episodes = st.sidebar.slider(
    "Training Episodes",
    min_value=50,
    max_value=500,
    value=100,
    step=50
)

# Main process button
process_button = st.sidebar.button("Optimize Portfolio")

# Initialize session state to store data and results
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
    st.session_state.stock_data = None
    st.session_state.returns = None
    st.session_state.markowitz_results = None
    st.session_state.rl_results = None

# Main workflow
if process_button:
    with st.spinner("Fetching data and processing..."):
        try:
            # Initialize data processor
            data_processor = DataProcessor(tickers, start_date, end_date)
            
            # Get data
            stock_data = data_processor.get_stock_data()
            returns = data_processor.calculate_returns()
            
            # Store in session state
            st.session_state.stock_data = stock_data
            st.session_state.returns = returns
            st.session_state.data_processed = True
            
            # Display stock data
            st.subheader("Stock Data Overview")
            st.dataframe(stock_data.tail())
            
            # Display returns statistics
            st.subheader("Returns Statistics")
            st.dataframe(returns.describe())
            
            # Markowitz Optimization
            markowitz = MarkowitzOptimizer(returns, risk_free_rate, num_portfolios)
            results = markowitz.optimize()
            st.session_state.markowitz_results = results
            
            # RL Optimization
            rl_optimizer = RLOptimizer(returns, training_episodes)
            rl_results = rl_optimizer.optimize()
            st.session_state.rl_results = rl_results
            
            # Visualizations
            visualizer = Visualizer()
            
            # Display optimized portfolios
            st.subheader("Portfolio Optimization Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Markowitz Optimal Portfolio")
                st.dataframe(results['optimal_weights'])
                
                markowitz_metrics = pd.DataFrame({
                    'Expected Annual Return': [results['optimal_return'] * 100],
                    'Expected Volatility': [results['optimal_volatility'] * 100],
                    'Sharpe Ratio': [results['optimal_sharpe']]
                })
                st.dataframe(markowitz_metrics)
            
            with col2:
                st.markdown("### RL Optimal Portfolio")
                st.dataframe(rl_results['optimal_weights'])
                
                rl_metrics = pd.DataFrame({
                    'Expected Annual Return': [rl_results['expected_return'] * 100],
                    'Expected Volatility': [rl_results['expected_volatility'] * 100],
                    'Sharpe Ratio': [rl_results['sharpe_ratio']]
                })
                st.dataframe(rl_metrics)
            
            # Efficient Frontier Visualization
            st.subheader("Efficient Frontier")
            
            # Add detailed explanation
            st.markdown("""
            ### Memahami Efficient Frontier
            **Efficient Frontier** menggambarkan kombinasi optimal dari portfolio yang menawarkan _return_ tertinggi untuk tingkat risiko tertentu, atau risiko terendah untuk tingkat _return_ tertentu.
            
            **Keterangan Grafik**:
            - **Setiap titik biru** mewakili satu portfolio dengan bobot aset yang berbeda.
            - **Warna titik** menunjukkan Sharpe Ratio - semakin terang warnanya, semakin tinggi Sharpe Ratio-nya.
            - **Bintang merah** menandakan portfolio optimal dengan Sharpe Ratio tertinggi.
            - **Sumbu X (Volatilitas)** menunjukkan risiko portfolio dalam persentase.
            - **Sumbu Y (Return)** menunjukkan keuntungan yang diharapkan dalam persentase.
            
            **Implikasi Investasi**:
            - Portfolio di sepanjang bagian atas kurva (mendekati bintang merah) memberikan keseimbangan terbaik antara risiko dan return.
            - Investor yang menghindari risiko (risk-averse) cenderung memilih portfolio di kiri bawah kurva.
            - Investor yang menyukai risiko (risk-seeking) cenderung memilih portfolio di kanan atas kurva.
            """)
            
            ef_fig = visualizer.plot_efficient_frontier(
                results['returns'], 
                results['volatilities'], 
                results['optimal_return'], 
                results['optimal_volatility'], 
                results['sharpe_ratios']
            )
            st.pyplot(ef_fig)
            
            # Portfolio Allocation Visualizations
            st.subheader("Portfolio Allocations")
            
            # Add explanation for portfolio allocations
            st.markdown("""
            ### Perbandingan Alokasi Portfolio
            Berikut adalah perbandingan alokasi aset optimal yang dihasilkan oleh dua pendekatan berbeda: Model Markowitz dan Reinforcement Learning (RL).
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Markowitz Allocation")
                st.markdown("""
                **Tentang Markowitz Allocation:**
                - Alokasi ini dihasilkan dengan **Modern Portfolio Theory** yang dikembangkan oleh Harry Markowitz.
                - Mengoptimalkan bobot aset dengan memaksimalkan **Sharpe Ratio** (return per unit risiko).
                - Alokasi ini mencari keseimbangan terbaik antara risiko (volatilitas) dan return.
                - Persentase pada grafik menunjukkan proporsi investasi untuk setiap aset dalam portfolio.
                - Model ini umumnya menilai aset berdasarkan mean-variance optimization.
                
                **Keterangan Visualisasi:**
                - **Pie Chart** menampilkan alokasi aset secara visual (maksimum 10 aset teratas).
                - **Tabel** di bawah pie chart menampilkan daftar lengkap semua aset dengan bobot masing-masing.
                - Aset dengan bobot lebih tinggi ditandai dengan warna sel yang lebih gelap.
                - Jika jumlah aset melebihi 10, aset dengan bobot terkecil akan dikelompokkan sebagai "Others" pada pie chart.
                """)
                alloc_fig = visualizer.plot_allocation(
                    results['optimal_weights'], 
                    tickers
                )
                st.pyplot(alloc_fig)
            
            with col2:
                st.markdown("### RL Allocation")
                st.markdown("""
                **Tentang RL Allocation:**
                - Alokasi ini dihasilkan dengan pendekatan **Reinforcement Learning**.
                - RL menggunakan algoritma pembelajaran mesin untuk mengoptimalkan alokasi aset.
                - Berbeda dengan Markowitz, RL berusaha memaksimalkan reward jangka panjang.
                - Perhitungan mencakup diversifikasi portfolio selain hanya Sharpe Ratio.
                - Persentase pada grafik menunjukkan proporsi investasi untuk setiap aset yang direkomendasikan.
                - Pendekatan ini dapat lebih adaptif terhadap perubahan kondisi pasar.
                
                **Keterangan Visualisasi:**
                - **Pie Chart** menampilkan alokasi aset secara visual (maksimum 10 aset teratas).
                - **Tabel** di bawah pie chart menampilkan daftar lengkap semua aset dengan bobot masing-masing.
                - Aset dengan bobot lebih tinggi ditandai dengan warna sel yang lebih gelap.
                - Jika jumlah aset melebihi 10, aset dengan bobot terkecil akan dikelompokkan sebagai "Others" pada pie chart.
                """)
                rl_alloc_fig = visualizer.plot_allocation(
                    rl_results['optimal_weights'], 
                    tickers
                )
                st.pyplot(rl_alloc_fig)
            
            # Performance Comparison
            st.subheader("Performance Comparison")
            
            # Add explanation for performance comparison
            st.markdown("""
            ### Perbandingan Kinerja Portfolio
            
            Grafik ini menunjukkan perbandingan kinerja tiga strategi investasi berbeda selama periode waktu yang dipilih:
            
            **Keterangan Grafik:**
            - **Markowitz Portfolio** (garis biru): Kinerja portfolio yang dialokasikan menggunakan model Markowitz dengan Sharpe Ratio optimal.
            - **RL Portfolio** (garis oranye): Kinerja portfolio yang dialokasikan menggunakan algoritma Reinforcement Learning.
            - **Equal-Weight Portfolio** (garis hijau putus-putus): Kinerja portfolio dengan alokasi sama rata pada semua aset (benchmark).
            
            **Catatan Penting:**
            - Sumbu Y menunjukkan nilai portfolio (dimulai dari $1).
            - Performa historis tidak menjamin hasil yang sama di masa depan.
            - Gradien kurva yang lebih curam menunjukkan tingkat pertumbuhan yang lebih tinggi.
            - Periode volatilitas tinggi terlihat sebagai fluktuasi tajam pada grafik.
            
            **Cara Interpretasi:**
            - Strategi dengan kurva tertinggi di akhir periode memberikan return total terbaik.
            - Strategi dengan kurva yang lebih halus (kurang berfluktuasi) umumnya memiliki risiko lebih rendah.
            - Bandingkan bagaimana setiap strategi bereaksi terhadap peristiwa pasar dalam periode yang dipilih.
            """)
            
            markowitz_cumulative = (1 + returns.dot(results['optimal_weights'])).cumprod()
            rl_cumulative = (1 + returns.dot(rl_results['optimal_weights'])).cumprod()
            equal_weights = np.array([1/len(tickers)] * len(tickers))
            equal_cumulative = (1 + returns.dot(equal_weights)).cumprod()
            
            comparison_fig = visualizer.plot_performance_comparison(
                returns.index, 
                markowitz_cumulative, 
                rl_cumulative, 
                equal_cumulative
            )
            st.pyplot(comparison_fig)
            
            # Download options
            st.subheader("Download Results")
            
            csv_buffer = io.BytesIO()
            
            # Create a comprehensive results dataframe
            download_data = pd.DataFrame({
                'Ticker': tickers,
                'Markowitz Weights': results['optimal_weights'],
                'RL Weights': rl_results['optimal_weights']
            })
            
            download_data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            
            st.download_button(
                label="Download Optimization Results as CSV",
                data=csv_buffer,
                file_name=f"portfolio_optimization_results_{datetime.date.today()}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Show instructions if no data has been processed
if not st.session_state.data_processed:
    st.info("""
    ### How to use this application:
    1. Select a date range for historical data
    2. Choose market (US or Indonesia IDX)
    3. Enter stock symbols:
       - For US: AAPL, MSFT, GOOG, etc.
       - For Indonesia: BBCA, BBRI, ASII, etc. (no need to add .JK suffix)
    4. Adjust the risk-free rate and simulation parameters
    5. Click "Optimize Portfolio" to run the analysis
    6. Review the results and visualizations
    7. Download the optimization results for your records
    """)
    
    if market == "Indonesia (IDX)":
        st.info("""
        ### Indonesia (IDX) Market Tips:
        - Stocks available from Jakarta Stock Exchange (IDX)
        - Popular stocks include: BBCA (Bank Central Asia), BBRI (Bank Rakyat Indonesia),
          ASII (Astra International), TLKM (Telekomunikasi Indonesia), etc.
        - The suffix .JK will be automatically added to your ticker symbols
        - For a more complete list, expand the "About Indonesian (IDX) Stock Market" section above
        """)
    else:
        st.info("""
        ### US Market Tips:
        - Popular stocks include: AAPL (Apple), MSFT (Microsoft), GOOG (Google),
          AMZN (Amazon), META (Meta/Facebook), TSLA (Tesla), etc.
        - For ETFs, try SPY (S&P 500), QQQ (Nasdaq), VTI (Total Market), etc.
        """)
