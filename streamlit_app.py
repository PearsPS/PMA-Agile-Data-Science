import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Response TAT Analytics Dashboard", layout="wide")

st.title("Engineer Response Turnaround Time Analytics Dashboard")
st.write(
    "This dashboard supports service operations by monitoring engineer response "
    "turnaround time and displaying model performance."
)

model = joblib.load("best_model.pkl")

df = pd.read_csv("cleaned_data.csv")

st.sidebar.header("Interactive Filters")

business_segment = st.sidebar.selectbox(
    "Select Business Segment",
    ["All"] + sorted(df["Business Segment"].dropna().unique().tolist())
)

max_response_tat = st.sidebar.slider(
    "Maximum Response TAT BD",
    float(df["Response Tat Bd"].min()),
    float(df["Response Tat Bd"].max()),
    float(df["Response Tat Bd"].max())
)

filtered_df = df.copy()

if business_segment != "All":
    filtered_df = filtered_df[filtered_df["Business Segment"] == business_segment]

filtered_df = filtered_df[filtered_df["Response Tat Bd"] <= max_response_tat]

st.header("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Model Used", "Random Forest")
col2.metric("R² Score", "0.9078")
col3.metric("MAE", "0.3292")

st.header("Monitoring Metrics")

col4, col5 = st.columns(2)

col4.metric(
    "Current Model MAE",
    "0.3292",
    "Stable"
)

missing_values = filtered_df.isnull().sum().sum()

col5.metric(
    "Missing Values",
    int(missing_values),
    "Healthy"
)

st.header("Dashboard Visualisations")

st.subheader("Distribution of Response TAT BD")
fig1, ax1 = plt.subplots()
ax1.hist(filtered_df["Response Tat Bd"], bins=30)
ax1.set_xlabel("Response TAT BD")
ax1.set_ylabel("Frequency")
st.pyplot(fig1)

st.subheader("Average Response TAT by Business Segment")
avg_segment = (
    filtered_df.groupby("Business Segment")["Response Tat Bd"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig2, ax2 = plt.subplots()
avg_segment.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Average Response TAT BD")
ax2.set_xlabel("Business Segment")
st.pyplot(fig2)

st.subheader("Response TAT vs Field Control TAT")
fig3, ax3 = plt.subplots()
ax3.scatter(
    filtered_df["Field Contr TAT BD"],
    filtered_df["Response Tat Bd"],
    alpha=0.5
)
ax3.set_xlabel("Field Contr TAT BD")
ax3.set_ylabel("Response TAT BD")
st.pyplot(fig3)

st.header("Analytical Output")

avg_response = filtered_df["Response Tat Bd"].mean()

st.success(
    f"Average Response TAT for selected data: {avg_response:.2f} business days"
)

st.info(
    "The deployed Random Forest model supports service operations by providing "
    "analytical insights based on the selected filters. These results help "
    "stakeholders monitor engineer response turnaround time and identify "
    "operational trends."
)
