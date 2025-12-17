import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spending Score Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Spending Score Prediction App")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("Hierarchikal Clustering.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.success("Model loaded successfully")

# ---------------- USER INPUT ----------------
annual_income = st.number_input(
    "Annual Income (k$)",
    min_value=0.0,
    max_value=200.0,
    step=1.0
)

# ---------------- PREDICTION ----------------
if st.button("Predict Spending Score"):
    input_data = np.array([[annual_income]])
    prediction = model.predict(input_data)

    st.metric("Predicted Spending Score", round(float(prediction[0]), 2))

    # ---------------- GRAPH ----------------
    fig, ax = plt.subplots()
    ax.scatter(annual_income, prediction)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Income vs Spending Score")

    st.pyplot(fig)

