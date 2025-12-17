import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spending Score Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Spending Score Prediction App")
st.write("Predict Spending Score based on Annual Income")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("Hierarchikal Clustering.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()
st.success("Model loaded successfully")

# ---------------- USER INPUT ----------------
annual_income = st.number_input(
    "Enter Annual Income (k$)",
    min_value=0.0,
    max_value=200.0,
    step=1.0
)

# ---------------- PREDICTION ----------------
if st.button("Predict Spending Score"):
    input_data = np.array([[annual_income]])
    prediction = model.predict(input_data)

    st.subheader("Prediction Result")
    st.metric(
        label="Predicted Spending Score",
        value=round(float(prediction[0]), 2)
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Hierarchical Clustering | Streamlit App")
