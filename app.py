import streamlit as st
import pandas as pd
df = pd.read_csv("dataset.csv")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Features & Target
X = df[["Base_Price", "Discount_%"]]import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("dataset.csv")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Encode categorical columns
tier_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
program_map = {"DS": 0, "AI": 1, "GenAI": 2}

df["College_Tier"] = df["College_Tier"].map(tier_map)
df["Program_Type"] = df["Program_Type"].map(program_map)

# ✅ USE ALL FEATURES
X = df[["Base_Price", "Discount_%", "Family_Income", "College_Tier", "Program_Type"]]
y = df["Converted"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# UI
st.title("💰 PragyanAI Pricing Intelligence")

price = st.number_input("Final Price", 10000, 300000, 90000)
discount = st.slider("Discount %", 0, 50, 20)
income = st.number_input("Family Income", 100000, 2000000, 500000)

tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
program = st.selectbox("Program Type", ["DS", "AI", "GenAI"])

# Convert inputs
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
y = df["Converted"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

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
