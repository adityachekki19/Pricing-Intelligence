import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(page_title="PragyanAI Pricing Intelligence", page_icon="💰")
st.title("💰 PragyanAI Pricing Intelligence")
st.markdown("**Optimal Price + 20% Discount = MAX REVENUE** 🔥")

# --- Load dataset safely ---
@st.cache_data
def load_data():
    file_path = "dataset.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except:
        st.error("❌ dataset.csv not found!")
        st.info("📥 Download: https://raw.githubusercontent.com/pragyanaischool/VTU_Internship_DataSets/refs/heads/main/student_PRICING_SCHOLARSHIP_Analysis_Project_12.csv")
        st.stop()

df = load_data()
st.success(f"✅ Dataset: {len(df):,} records")

# --- Encode categorical columns ---
tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2, "Tier 4": 3}
program_map = {"DS": 0, "AI": 1, "GenAI": 2}

# Safe mapping
df["College_Tier"] = df["College_Tier"].astype(str).map(tier_map).fillna(1).astype(int)
df["Program_Type"] = df["Program_Type"].astype(str).map(program_map).fillna(1).astype(int)

# --- Features & target ---
feature_cols = ["Base_Price", "Discount_%", "Family_Income", "College_Tier", "Program_Type"]
X = df[feature_cols]
y = df["Converted"]

# --- Load or train model ---
model_file = "model.pkl"
if os.path.exists(model_file):
    model = joblib.load(model_file)
    st.success("✅ Model loaded")
else:
    st.info("🎓 Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    st.success("✅ Model trained!")

# Model accuracy
accuracy = model.score(X, y)
st.metric("Model Accuracy", f"{accuracy:.1%}")

# --- User Input ---
st.header("🎯 Revenue Optimizer")
col1, col2 = st.columns(2)

with col1:
    base_price = st.number_input("Base Price (₹)", 30000, 300000, 100000, 5000)
    family_income = st.number_input("Family Income (₹)", 100000, 5000000, 500000)
    college_tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3", "Tier 4"])
    program_type = st.selectbox("Program", ["DS", "AI", "GenAI"])

with col2:
    discount_pct = st.slider("Discount %", 0, 50, 20, 5)
    final_price = base_price * (1 - discount_pct / 100)

# === OPTIMIZE BUTTON ===
if st.button("🚀 FIND OPTIMAL STRATEGY", type="primary"):
    # Test 0-50% discounts
    discounts = list(range(0, 51, 5))
    results = []
    
    for discount in discounts:
        input_data = [
            base_price,
            discount,
            family_income,
            tier_map.get(college_tier, 1),
            program_map.get(program_type, 1)
        ]
        
        input_df = pd.DataFrame([input_data], columns=feature_cols)
        prob = model.predict_proba(input_df)[0][1]
        test_price = base_price * (1 - discount/100)
        revenue = test_price * prob
        
        results.append({
            'Discount': f"{discount}%",
            'Price': test_price,
            'Conversion': prob,
            'Revenue': revenue
        })
    
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['Revenue'].idxmax()]
    
    # === RESULTS ===
    st.subheader("✅ **BEST STRATEGY**")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("🎯 Discount", best['Discount'])
    col2.metric("💰 Price", f"₹{best['Price']:,.0f}")
    col3.metric("📊 Conversion", f"{best['Conversion']:.0%}")
    col4.metric("💸 Revenue", f"₹{best['Revenue']:,.0f}")
    
    # Native Streamlit Chart
    st.subheader("📈 Revenue vs Discount")
    chart_data = results_df[['Discount', 'Revenue']].copy()
    chart_data['Discount'] = chart_data['Discount'].str.replace('%', '').astype(float)
    
    st.bar_chart(chart_data.set_index('Discount')['Revenue']/1000, height=400)
    
    # Sweet spot highlight
    st.info(f"**🔥 SWEET SPOT**: {best['Discount']} discount = MAX revenue")
    st.success(f"**📈 Uplift**: +{((best['Revenue']/final_price)*100-100):+.0f}% vs current")

# === PRICE ELASTICITY ===
st.header("📊 Price Elasticity")
st.markdown("""
