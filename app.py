import streamlit as st
import pandas as pd
import pickle
import joblib
st.title("💰 PragyanAI Pricing Intelligence Engine")

# Load data
df = pd.read_csv("dataset.csv")


model = joblib.load("model.pkl")

# Sidebar inputs
st.sidebar.header("Student Profile")

income = st.sidebar.selectbox("Family Income", df['Family_Income'].unique())
program = st.sidebar.selectbox("Program", df['Program_Type'].unique())
discount = st.sidebar.slider("Discount %", 0, 50, 20)

# Filter sample
sample = df.iloc[0].copy()

sample['Family_Income'] = income
sample['Discount_%'] = discount

# Predict
prediction = model.predict([sample.drop(['Converted','Revenue'], errors='ignore')])[0]

st.subheader("Prediction")
st.write("Conversion Probability:", prediction)

# Visualization
st.subheader("Price vs Conversion")
st.line_chart(df.groupby('Final_Price')['Converted'].mean())
