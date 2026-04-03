import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("💰 PragyanAI Pricing Intelligence")

# Inputs
price = st.number_input("Final Price", 10000, 300000, 90000)
discount = st.slider("Discount %", 0, 50, 20)
income = st.number_input("Family Income", 100000, 2000000, 500000)

tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
program = st.selectbox("Program Type", ["DS", "AI", "GenAI"])

# Convert inputs (same encoding as training)
tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
program_map = {"DS": 0, "AI": 1, "GenAI": 2}

tier_val = tier_map[tier]
program_val = program_map[program]

# Prediction
if st.button("Predict"):
    input_data = [[price, discount, income, tier_val, program_val]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ High Conversion Chance")
        st.metric("Revenue", f"₹{price}")
    else:
        st.error("❌ Low Conversion Chance")
        st.metric("Revenue", "₹0")
