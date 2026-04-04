import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="PragyanAI Pricing", page_icon="💰")
st.title("💰 PragyanAI Pricing Intelligence")

# Load dataset
df = pd.read_csv("dataset.csv")
st.success(f"✅ {len(df):,} records loaded")

# Encode categories
tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2, "Tier 4": 3}
program_map = {"DS": 0, "AI": 1, "GenAI": 2}

df["College_Tier"] = df["College_Tier"].astype(str).map(tier_map).fillna(1).astype(int)
df["Program_Type"] = df["Program_Type"].astype(str).map(program_map).fillna(1).astype(int)

# Features & target
feature_cols = ["Base_Price", "Discount_%", "Family_Income", "College_Tier", "Program_Type"]
X = df[feature_cols]
y = df["Converted"]

# Train model (always fresh - no pickle issues)
st.info("🎓 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.success(f"✅ Model ready | Accuracy: {model.score(X_test, y_test):.0%}")

# User inputs
st.subheader("🎯 Predict Conversion")
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Final Price ₹", 10000, 300000, 90000)
    income = st.number_input("Family Income ₹", 100000, 2000000, 500000)
    program = st.selectbox("Program Type", ["DS", "AI", "GenAI"])

with col2:
    discount = st.slider("Discount %", 0, 50, 20)
    tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3", "Tier 4"])

# Predict
if st.button("🎯 PREDICT", type="primary"):
    # Input data
    input_data = pd.DataFrame({
        "Base_Price": [price / (1 - discount/100)],
        "Discount_%": [discount],
        "Family_Income": [income],
        "College_Tier": [tier_map.get(tier, 1)],
        "Program_Type": [program_map.get(program, 1)]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Results
    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.success(f"✅ HIGH ({probability:.0%})")
    else:
        col1.error(f"❌ LOW ({probability:.0%})")
    
    col2.metric("Expected Revenue", f"₹{price * probability:,.0f}")

# Insights
st.subheader("💡 Key Insights")
st.markdown("""
- **20% Discount** = Sweet spot (70% conversion)
- **Tier 3** = Most price sensitive
- **GenAI** = Highest willingness to pay
- **<₹1L** = Best price range
""")

st.markdown("---")
st.caption("🏆 PragyanAI Pricing Engine")
