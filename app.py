import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


st.write("Current working directory:", os.getcwd())
st.write("Files here:", os.listdir())
st.write("Files in data folder:", os.listdir("data") if os.path.exists("data") else "No data folder")
st.title("💰 PragyanAI Pricing Intelligence")

# --- Load dataset safely ---
file_path = os.path.join("data", "dataset.csv")
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
X = df[["Base_Price", "Discount_%", "Family_Income", "College_Tier", "Program_Type"]]
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
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
    st.success("Model trained and saved successfully ✅")

# --- User Input ---
st.subheader("Enter Details for Prediction")
price = st.number_input("Final Price", 10000, 300000, 90000)
discount = st.slider("Discount %", 0, 50, 20)
income = st.number_input("Family Income", 100000, 2000000, 500000)
tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
program = st.selectbox("Program Type", ["DS", "AI", "GenAI"])

# --- Convert categorical input ---
tier_val = tier_map[tier]
program_val = program_map[program]

# --- Prediction ---
if st.button("Predict"):
    input_data = [[price, discount, income, tier_val, program_val]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ High Conversion Chance")
        st.metric("Estimated Revenue", f"₹{price}")
    else:
        st.error("❌ Low Conversion Chance")
        st.metric("Estimated Revenue", "₹0")
