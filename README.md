# Pricing-Intelligence
# 💰 PragyanAI Pricing Optimization

PragyanAI is a machine learning-based pricing optimization tool built using Streamlit. It predicts customer conversion probability and helps estimate optimal pricing and expected revenue.

---

## 🚀 Features

- Predicts conversion probability
- Calculates final price after discount
- Estimates expected revenue
- Uses Machine Learning models
- Interactive UI with Streamlit

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Joblib

---

## 📂 Project Structure

PragyanAI/
│── app.py
│── model.py
│── dataset.csv
│── model.pkl
│── README.md

---

## 📊 How It Works

1. Data Processing  
- Cleans numeric columns  
- Encodes categorical values  

2. Model Training  
- Random Forest (used in app)  
- Logistic Regression (saved model)  

3. Prediction  
- Takes user inputs  
- Predicts conversion probability  
- Calculates price and revenue  

---

## ▶️ How to Run

1. Clone repo  
git clone https://github.com/your-username/pragyanai.git  
cd pragyanai  

2. Install dependencies  
pip install -r requirements.txt  

3. Run app  
streamlit run app.py  

---

## 🧠 Train Model

python model.py  

This creates:
model.pkl

---

## ⚠️ Notes

- Keep dataset.csv in the project folder  
- Accuracy depends on data quality  

---

## 👨‍💻 Author

Aditya Chekki
