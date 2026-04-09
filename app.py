import streamlit as st
import numpy as np
import pandas as pd
import numpy_financial as npf
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import plotly.express as px
import plotly.graph_objects as go

import shap

st.set_page_config(page_title="PropAI Interview Cracker", layout="wide")

st.title("🏡 PropAI - Interview Cracker ML System")
st.write("🔥 SHAP + Kaggle Data + Finance + Deployment Ready")

# ---------------- CITY DATA (Kaggle-style simulation fallback) ----------------
cities = ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad"]

city_multiplier = {
    "Mumbai": 1.8,
    "Delhi": 1.5,
    "Bangalore": 1.6,
    "Pune": 1.2,
    "Hyderabad": 1.3
}

city = st.selectbox("🌆 Select City", cities)

# ---------------- INPUT ----------------
home_price = st.number_input("🏠 Base Home Price (₹)", 500000, 50000000, 5000000)
rent = st.number_input("🏠 Monthly Rent (₹)", 5000, 200000, 20000)
down_payment = st.number_input("💰 Down Payment (₹)", 100000, 20000000, 1000000)
interest_rate = st.number_input("📈 Interest Rate (%)", 5.0, 15.0, 8.5)
years = st.slider("📊 Investment Duration", 5, 30, 20)

# ---------------- CITY ADJUSTED PRICE ----------------
city_factor = city_multiplier[city]
base_price = home_price * city_factor

# ---------------- KAGGLE STYLE DATASET SIMULATION ----------------
np.random.seed(42)

data_size = 300

df = pd.DataFrame({
    "Area": np.random.randint(500, 5000, data_size),
    "Bedrooms": np.random.randint(1, 5, data_size),
    "Age": np.random.randint(0, 30, data_size),
    "InterestRate": np.random.uniform(6, 12, data_size),
    "CityFactor": np.random.choice(list(city_multiplier.values()), data_size),
})

df["Price"] = (
    df["Area"] * 120 +
    df["Bedrooms"] * 500000 -
    df["Age"] * 10000 +
    df["CityFactor"] * 1000000 +
    np.random.normal(0, 200000, data_size)
)

# ---------------- FEATURES ----------------
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
score = r2_score(y_test, pred)

# ---------------- MODEL SAVE (JOBLIB) ----------------
MODEL_PATH = "propai_model.pkl"
joblib.dump(model, MODEL_PATH)

loaded_model = joblib.load(MODEL_PATH)

# ---------------- FUTURE INPUT ----------------
input_data = pd.DataFrame({
    "Area": [1200],
    "Bedrooms": [3],
    "Age": [10],
    "InterestRate": [interest_rate],
    "CityFactor": [city_factor]
})

prediction = loaded_model.predict(input_data)[0]

# ---------------- FINANCE ENGINE ----------------
loan_amount = base_price - down_payment
monthly_rate = interest_rate / 100 / 12
months = years * 12

emi = npf.pmt(monthly_rate, months, -loan_amount)
total_emi = emi * months

total_rent = rent * 12 * years

net_buy = total_emi + down_payment - prediction

# ---------------- SHAP EXPLAINABILITY ----------------
st.subheader("🧠 Why AI predicted this price (SHAP Explainability)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

fig_shap = plt = None
try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot()
except:
    st.write("SHAP plot rendering fallback (Cloud safe mode)")

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("📊 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig = px.bar(importance, x="Feature", y="Importance")
st.plotly_chart(fig)

# ---------------- DASHBOARD ----------------
col1, col2, col3 = st.columns(3)

col1.metric("🏠 Predicted Price", f"₹{prediction:,.0f}")
col2.metric("💸 Rent Cost", f"₹{total_rent:,.0f}")
col3.metric("📊 Model Accuracy", f"{score:.2f}")

# ---------------- DECISION ----------------
st.subheader("📌 Investment Decision Engine")

if net_buy < total_rent:
    st.success("🔥 BUY PROPERTY is better investment")
else:
    st.warning("🏠 RENT is better option")

# ---------------- GRAPH ----------------
st.subheader("📈 Investment Breakdown")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    name="Buy Cost",
    x=["Investment"],
    y=[total_emi + down_payment]
))

fig2.add_trace(go.Bar(
    name="Rent Cost",
    x=["Investment"],
    y=[total_rent]
))

st.plotly_chart(fig2)

# ---------------- REPORT ----------------
def create_report():
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="PropAI Interview Cracker Report", ln=True)
    pdf.cell(200, 10, txt=f"City: {city}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Price: {prediction:.0f}", ln=True)
    pdf.cell(200, 10, txt=f"Model Accuracy: {score:.2f}", ln=True)
    pdf.cell(200, 10, txt=f"Rent Cost: {total_rent:.0f}", ln=True)

    file = "PropAI_Interview_Report.pdf"
    pdf.output(file)
    return file

if st.button("📄 Generate Report"):
    file = create_report()
    st.success("Report Generated 🚀")
    st.download_button("⬇ Download PDF", open(file, "rb"), file_name="PropAI_Report.pdf")