import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess

st.set_page_config(page_title="DLR Ampacity Forecaster", layout="wide", page_icon="⚡")
st.title("⚡ Dynamic Line Rating (DLR) Dashboard")
st.markdown("Monitor and forecast power line ampacity using Open-Meteo NWP and LightGBM.")

st.sidebar.header("⚙️ Configuration")
lat = st.sidebar.text_input("Latitude", "11.758")
lon = st.sidebar.text_input("Longitude", "108.758")
azimuth = st.sidebar.slider("Line Azimuth (deg)", 0, 180, 90)

if st.sidebar.button("🚀 Run Forecast Pipeline"):
    with st.spinner("Running forecasting pipeline... (This may take a minute)"):
        env = os.environ.copy()
        env['DLR_LAT'] = str(lat)
        env['DLR_LON'] = str(lon)
        env['DLR_LINE_AZIMUTH_DEG'] = str(azimuth)
        
        # Run the backend script
        result = subprocess.run(["python", "-m", "dlr_planb.run"], env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            st.sidebar.success("Pipeline completed successfully!")
        else:
            st.sidebar.error("Pipeline failed!")
            st.sidebar.text(result.stderr)

try:
    # Load the 24-hour lead forecast
    df = pd.read_csv("results/predictions_lead_1.csv", parse_dates=["Time"])
    
    st.header("📈 24-Hour Ahead Forecast (Lead 1)")
    
    static_rating = 404.9  # Baseline static rating
    avg_actual = df['Actual'].mean()
    avg_p10 = df['P10'].mean()
    gain = avg_p10 - static_rating
    
    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Actual Ampacity", f"{avg_actual:.1f} A")
    col2.metric("P10 Safe Forecast (Avg)", f"{avg_p10:.1f} A")
    col3.metric("Capacity Gain Over Static", f"+{gain:.1f} A", delta_color="normal")
    
    # Plotting
    st.subheader("Forecast vs Actuals (Last 7 Days)")
    plot_df = df.tail(24 * 7).set_index("Time")  # Last 168 hours
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(plot_df.index, plot_df['P10'], plot_df['P90'], color='#add8e6', alpha=0.4, label='P10-P90 Confidence Bound')
    ax.plot(plot_df.index, plot_df['P50'], color='#00008b', label='P50 Forecast', linestyle='--')
    ax.plot(plot_df.index, plot_df['Actual'], color='black', label='Actual Computed Ampacity', linewidth=1.5)
    ax.axhline(static_rating, color='red', linestyle=':', linewidth=2, label=f'Static Rating ({static_rating} A)')
    
    ax.set_ylabel("Ampacity (A)")
    ax.set_xlabel("Date/Time")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    st.pyplot(fig)
    
    st.subheader("📊 Raw Forecast Data")
    st.dataframe(df.sort_values('Time', ascending=False).head(100), use_container_width=True)
    
except FileNotFoundError:
    st.warning("No results found. Please click 'Run Forecast Pipeline' in the sidebar to generate data.")
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
