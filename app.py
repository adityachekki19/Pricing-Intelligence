import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="PragyanAI", page_icon="💰")
st.title("💰 PragyanAI Pricing")

# Load & clean dataset
@st.cache_data
def load_clean_data():
    df = pd.read_csv("dataset.csv")
    
    # Clean numeric columns
    numeric_cols = ['Base_Price', 'Discount_%', 'Family_Income', 'CGPA', 'Revenue']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Encode categories
    tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2, "Tier 4": 3}
    program_map = {"DS": 0, "AI": 1, "GenAI": 2}
    
    df["College_Tier"] = df["College_Tier"].astype(str).map(tier_map).fillna(1)
    df["Program_Type"] = df["Program_Type"].astype(str).map(program_map).fillna(1)
    
    # Drop any remaining NaN rows
    df = df.dropna(subset=['Converted'])
    
    return df

df = load_clean_data()
st.success(f"✅ {len(df):,} clean records")

# Prepare features (NumPy arrays - sklearn safe)
feature_cols = ["Base_Price", "Discount_%", "Family_Income", "College_Tier", "Program_Type"]
X = df[feature_cols].fillna(0).values  # ✅ NumPy + No NaN
y = df["Converted"].astype(int).values

# Train model
st.info("🎓 Training...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

st.success(f"✅ Ready | Test Accuracy: {model.score(X_test, y_test):.0%}")

# User input
st.subheader("🎯 Optimize Pricing")
col1, col2 = st.columns(2)

with col1:
    base_price = st.number_input("Base Price ₹", 50000, 200000, 100000)
    income = st.number_input("Income ₹", 100000, 2000000, 500000)
    tier = st.selectbox("Tier", ["Tier 1", "Tier 2", "Tier 3"])
    program = st.selectbox("Program", ["DS", "AI", "GenAI"])

with col2:
    discount = st.slider("Discount %", 0, 50, 20)

if st.button("🚀 PREDICT", type="primary"):
    # Clean input (NumPy array)
    input_data = np.array([[
        base_price,
        discount,
        income,
        {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2, "Tier 4": 3}[tier],
        {"DS": 0, "AI": 1, "GenAI": 2}[program]
    ]])
    
    prob = model.predict_proba(input_data)[0][1]
    final_price = base_price * (1 - discount/100)
    revenue = final_price * prob
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Conversion", f"{prob:.0%}")
    col2.metric("💰 Price", f"₹{final_price:,.0f}")
    col3.metric("💸 Revenue", f"₹{revenue:,.0f}")

st.markdown("---")
st.caption("PragyanAI | Production Ready")
