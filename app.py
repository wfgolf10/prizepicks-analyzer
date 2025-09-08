import streamlit as st
import pandas as pd
from prizepicks_predictor import run_pipeline

st.set_page_config(page_title="PrizePicks Predictor", layout="wide")

st.title("🏈 PrizePicks NFL Prop Predictor")
st.write("Upload your historical player stats CSV to generate predictions.")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Stat selection
default_stats = ["passing_yards", "rushing_yards", "receiving_yards"]
stats = st.multiselect(
    "Choose which stats to model:",
    default_stats,
    default=default_stats
)

if uploaded_file is not None:
    hist = pd.read_csv(uploaded_file)

    with st.spinner("Training models and fetching PrizePicks lines..."):
        predictions = run_pipeline(hist, stats)

    st.success(f"Generated {len(predictions)} predictions.")

    st.dataframe(predictions, use_container_width=True)

    # Download button
    csv = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions CSV",
        csv,
        "predictions.csv",
        "text/csv",
        key="download-csv"
    )
else:
    st.info("👆 Upload a CSV to get started.")
