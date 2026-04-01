import streamlit as st
import pandas as pd
from anomaly_model import load_data, build_model, detect_anomalies, plot_anomalies
import plotly.graph_objects as go

st.title("🕵️ LSTM Anomaly Detection Dashboard")
st.markdown("Upload GOOG.csv or use sample data for stock anomaly detection.")

uploaded_file = st.file_uploader("Choose CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=[
                     'Date'], index_col='Date')[['Close']]
else:
    st.info("Using sample data from notebook.")
    df = load_data()  # Assumes GOOG.csv in dir

if st.button("Detect Anomalies"):
    with st.spinner("Training model..."):
        input_shape = (30, 1)  # From notebook
        model = build_model(input_shape)
        anomaly_df, close_prices = detect_anomalies(model, None, df)

    st.success(f"Found {anomaly_df['anomaly'].sum()} anomalies!")

    fig = plot_anomalies(anomaly_df)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomalies Table")
    st.dataframe(anomaly_df[anomaly_df['anomaly']][['Close', 'loss']])

    # Download results
    csv = anomaly_df.to_csv()
    st.download_button("Download Results", csv, "anomalies.csv")
