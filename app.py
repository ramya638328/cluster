import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spending Score Analyzer",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Spending Score Analyzer")
st.write("Find Spending Score based on Annual Income")

# ---------------- LOAD PKL DATA ----------------
@st.cache_resource
def load_data():
    with open("Hierarchikal Clustering.pkl", "rb") as file:
        data = pickle.load(file)
    return data

data = load_data()

# ---------------- VALIDATE PKL CONTENT ----------------
if not isinstance(data, pd.DataFrame):
    st.error("❌ The uploaded .pkl file does not contain a dataset")
    st.stop()

st.success("Dataset loaded successfully")

# ---------------- USER INPUT ----------------
annual_income = st.number_input(
    "Enter Annual Income (k$)",
    min_value=0.0,
    max_value=200.0,
    step=1.0
)

# ---------------- FIND SPENDING SCORE ----------------
if st.button("Get Spending Score"):

    # Find closest income
    closest_row = data.iloc[
        (data["Annual Income (k$)"] - annual_income).abs().argsort()[:1]
    ]

    spending_score = closest_row["Spending Score (1-100)"].values[0]

    st.subheader("Result")
    st.metric(
        label="Estimated Spending Score",
        value=int(spending_score)
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Hierarchical Clustering | Streamlit App")

