import streamlit as st
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# Load model
model = joblib.load("model.pkl")

st.title("💰 PragyanAI Pricing Intelligence")

st.write("Enter student details to predict conversion & revenue")

# Inputs
price = st.number_input("Final Price", min_value=10000, max_value=300000, value=90000)
discount = st.slider("Discount %", 0, 50, 20)
income = st.number_input("Family Income", min_value=100000, max_value=2000000, value=500000)
cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)

# Prediction
if st.button("Predict"):
    input_data = [[price, discount, income, cgpa]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ High chance of conversion")
    else:
        st.error("❌ Low chance of conversion")

    revenue = price if prediction == 1 else 0
    st.metric("Estimated Revenue", f"₹{revenue}")
