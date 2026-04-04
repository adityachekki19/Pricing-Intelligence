import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="PragyanAI Pricing", page_icon="💰")
st.title("💰 PragyanAI Pricing")

# Load dataset
file_path = "dataset.csv" 
df = pd.read_csv(file_path)
st.success("✅ Dataset loaded")

# Encode
tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
program_map = {"DS": 0, "AI": 1, "GenAI": 2}

df["College_Tier"] = df["College_Tier"].map(tier_map)
df["Program_Type"] = df["Program_Type"].map(program_map)

# Features
feature_cols = ["Base_Price", "Discount_%", "Family_Income", "College_Tier", "Program_Type"]
X = df[feature_cols]
y = df["Converted"]

# Model
model_file = "model.pkl"
if os.path.exists(model_file):
    model = joblib.load(model_file)
    st.success("✅ Model loaded")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    st.success("✅ Model trained")

st.metric("Accuracy", f"{model.score(X, y):.0%}")

# Inputs
st.subheader("Enter Student Details")
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Final Price", 10000, 300000, 90000)
    income = st.number_input("Family Income", 100000, 2000000, 500000)
    program = st.selectbox("Program", ["DS", "AI", "GenAI"])

with col2:
    discount = st.slider("Discount %", 0, 50, 20)
    tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])

if st.button("🎯 Predict Conversion"):
    input_df = pd.DataFrame([[price, discount, income, 
                             tier_map[tier], program_map[program]]], 
                            columns=feature_cols)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.success(f"✅ HIGH CHANCE ({probability:.0%})")
        col2.metric("Revenue", f"₹{price:,}")
    else:
        col1.error(f"❌ LOW CHANCE ({probability:.0%})")
        col2.metric("Revenue", "₹0")

st.markdown("---")
st.caption("PragyanAI Pricing Engine")
