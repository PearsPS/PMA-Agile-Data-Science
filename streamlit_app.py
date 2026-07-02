import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Response TAT Prediction Dashboard", layout="wide")

st.title("Engineer Response Turnaround Time Prediction Dashboard")
st.write("This dashboard supports service operations by predicting Response TAT in business days.")

# Load model
model = joblib.load("best_model.pkl")

st.header("Predictive Output")

st.info(
    "The trained Random Forest model is loaded successfully. "
    "Prediction input form will be connected based on the final model features."
)

st.header("Dashboard Visualisations")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Used", "Random Forest")

with col2:
    st.metric("R² Score", "0.9078")

with col3:
    st.metric("MAE", "0.3292")

st.success("Dashboard prototype is running successfully.")
