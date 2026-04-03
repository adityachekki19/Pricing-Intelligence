import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="PragyanAI Pricing Intelligence", page_icon="💰")
st.title("💰 PragyanAI Pricing Intelligence")

# --- Load dataset safely ---
file_path = "dataset.csv" 
try:
    df = pd.read_csv(file_path)
    st.success("Dataset loaded successfully ✅")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# --- Encode categorical columns ---
tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
program_map = {"DS": 0, "AI": 1, "GenAI": 2}

df["College_Tier"] = df["College_Tier"].map(tier_map)
df["Program_Type"] = df["Program_Type"].map(program_map)

# --- Features & target ---
feature_cols = ["Base_Price", "Discount_%", "Family_Income", "College_Tier", "Program_Type"]
X = df[feature_cols]
y = df["Converted"]

# --- Load or train model ---
model_file = "model.pkl"
if os.path.exists(model_file):
    try:
        model = joblib.load(model_file)
        st.info("✅ Loaded existing trained model")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.warning("No trained model found. Training a new model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    st.success("Model trained and saved successfully ✅")

# --- User Input ---
st.subheader("Enter Details for Prediction")
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Final Price", 10000, 300000, 90000)
    income = st.number_input("Family Income", 100000, 2000000, 500000)
    program = st.selectbox("Program Type", list(program_map.keys()))

with col2:
    discount = st.slider("Discount %", 0, 50, 20)
    tier = st.selectbox("College Tier", list(tier_map.keys()))

# --- Prediction Logic ---
if st.button("Predict Conversion Chance", use_container_width=True):
    # Create a DataFrame for prediction to match training feature names
    input_df = pd.DataFrame([[price, discount, income, tier_map[tier], program_map[program]]], 
                            columns=feature_cols)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # Probability of conversion

    if prediction == 1:
        st.success(f"✅ High Conversion Chance ({probability:.1%})")
        st.metric("Estimated Revenue", f"₹{price:,}")
    else:
        st.error(f"❌ Low Conversion Chance ({probability:.1%})")
        st.metric("Estimated Revenue", "₹0")
