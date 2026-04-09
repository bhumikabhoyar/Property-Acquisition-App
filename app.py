import streamlit as st
import numpy_financial as npf
import pandas as pd
from fpdf import FPDF
import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import requests

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Property Acquisition Decision Model", layout="wide")
st.title("Property Acquisition Decision Model")
st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72, #2a5298);
}

/* Selectbox label */
[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: bold !important;
}

/* Selectbox dropdown box */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #ffffff !important;
    border-radius: 10px !important;
}

/* Selected value text */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    color: #1e3c72 !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- PREMIUM UI DESIGN ----------------
st.markdown("""
<style>

/* ===== Main Background ===== */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== Main Title ===== */
h1 {
    text-align: center;
    font-size: 42px !important;
    font-weight: 700;
    background: linear-gradient(90deg,#00f2fe,#4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===== Metric Cards ===== */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    transition: 0.3s;
}
[data-testid="metric-container"]:hover {
    transform: scale(1.05);
}

/* ===== Section Cards ===== */
.section-card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#141e30,#243b55);
}

/* ===== Map Buttons ===== */
.map-btn {
    display:inline-block;
    padding:14px 25px;
    border-radius:30px;
    font-weight:bold;
    text-decoration:none;
    color:white !important;
    transition:0.3s;
    box-shadow:0 6px 15px rgba(0,0,0,0.4);
}
.buy-btn {
    background: linear-gradient(45deg,#00c6ff,#0072ff);
}
.rent-btn {
    background: linear-gradient(45deg,#ff512f,#dd2476);
}
.map-btn:hover {
    transform: scale(1.1);
}

/* ===== Download Button ===== */
.stDownloadButton button {
    background: linear-gradient(45deg,#00c6ff,#0072ff);
    color:white;
    border-radius:25px;
    padding:10px 30px;
    font-weight:bold;
}
.stDownloadButton button:hover {
    background: linear-gradient(45deg,#ff512f,#dd2476);
}

</style>

<script>
document.addEventListener("DOMContentLoaded", function(){
    console.log("Premium Dashboard Loaded");
});
</script>
""", unsafe_allow_html=True)
# ---------------- LANGUAGE ----------------
language = st.sidebar.selectbox("Select Language / भाषा चुनें", ["English", "Hindi"])
texts = {
    "English": {
        "home_price": "Home Price",
        "monthly_emi": "Monthly EMI",
        "ml_value": "ML Predicted Value",
        "investment_growth": "Investment Growth",
        "emi_vs_rent": "EMI vs Rent",
        "area_intelligence": "Area Intelligence",
        "risk_score": "Risk Score (1 Low - 10 High)",
        "rental_yield": "Rental Yield",
        "inflation_adj": "Inflation Adjusted Return",
        "break_even": "Break-even Year",
        "sensitivity": "Sensitivity Analysis (+1% Interest)",
        "client_stay": "Client Stay Based Final Decision",
        "total_rent": "Total Rent If Renting",
        "net_position": "Net Position If Buying",
        "decision_buy": "BUY is financially better.",
        "decision_rent": "RENT is financially better.",
        "download_report": "Download Full AI Report",
        "client_stay_years": "Client Stay Years"
    },
    "Hindi": {
        "home_price": "घर की कीमत",
        "monthly_emi": "मासिक ईएमआई",
        "ml_value": "एमएल अनुमानित मूल्य",
        "investment_growth": "निवेश वृद्धि",
        "emi_vs_rent": "ईएमआई बनाम किराया",
        "area_intelligence": "क्षेत्र बुद्धिमत्ता",
        "risk_score": "जोखिम स्कोर (1 कम - 10 अधिक)",
        "rental_yield": "किराया लाभांश",
        "inflation_adj": "मुद्रास्फीति समायोजित रिटर्न",
        "break_even": "ब्रेक-ईवन वर्ष",
        "sensitivity": "संवेदनशीलता विश्लेषण (+1% ब्याज)",
        "client_stay": "ग्राहक ठहराव आधारित अंतिम निर्णय",
        "total_rent": "किराये का कुल भुगतान",
        "net_position": "खरीद की स्थिति",
        "decision_buy": "खरीदना वित्तीय रूप से बेहतर है।",
        "decision_rent": "किराया लेना वित्तीय रूप से बेहतर है।",
        "download_report": "पूर्ण एआई रिपोर्ट डाउनलोड करें",
        "client_stay_years": "ग्राहक ठहराव वर्ष"
    }
}
t = texts[language]

# ---------------- CITY → AREA DATA ----------------
city_areas = {
    "Mumbai": {
        "South Mumbai": {"price_per_sqft": 35000, "monthly_rent": 45000},
        "Andheri": {"price_per_sqft": 22000, "monthly_rent": 28000},
        "Borivali": {"price_per_sqft": 15000, "monthly_rent": 18000}
    },
    "Pune": {
        "Kothrud": {"price_per_sqft": 9000, "monthly_rent": 18000},
        "Viman Nagar": {"price_per_sqft": 10000, "monthly_rent": 20000},
        "Hinjewadi": {"price_per_sqft": 8000, "monthly_rent": 15000}
    },
    "Bangalore": {
        "Whitefield": {"price_per_sqft": 14000, "monthly_rent": 22000},
        "Koramangala": {"price_per_sqft": 12000, "monthly_rent": 21000},
        "Yelahanka": {"price_per_sqft": 10000, "monthly_rent": 18000}
    },
    "Delhi": {
        "South Delhi": {"price_per_sqft": 18000, "monthly_rent": 30000},
        "Dwarka": {"price_per_sqft": 14000, "monthly_rent": 22000},
        "Rohini": {"price_per_sqft": 12000, "monthly_rent": 20000}
    },
    "Nagpur": {
        "Manish Nagar": {"price_per_sqft": 5471, "monthly_rent": 16500},
        "Mihan": {"price_per_sqft": 5229, "monthly_rent": 15750},
        "Besa": {"price_per_sqft": 5038, "monthly_rent": 15250},
        "Beltarodi": {"price_per_sqft": 5324, "monthly_rent": 15250},
        "Manewada": {"price_per_sqft": 4948, "monthly_rent": 14500},
        "KT Nagar": {"price_per_sqft": 5402, "monthly_rent": 15750},
        "Pratap Nagar": {"price_per_sqft": 8235, "monthly_rent": 21000},
        "Zingabai Takli": {"price_per_sqft": 4231, "monthly_rent": 13500},
        "Dharampeth": {"price_per_sqft": 10500, "monthly_rent": 25000},
        "Laxminagar": {"price_per_sqft": 9850, "monthly_rent": 23750},
        "Wardha Road": {"price_per_sqft": 6850, "monthly_rent": 20250},
        "Trimurti Nagar": {"price_per_sqft": 7950, "monthly_rent": 21500},
        "Dighori": {"price_per_sqft": 4550, "monthly_rent": 13500},
        "Nandanvan": {"price_per_sqft": 4900, "monthly_rent": 14750},
        "Sadar": {"price_per_sqft": 9150, "monthly_rent": 24500},
        "Ramdaspeth": {"price_per_sqft": 10850, "monthly_rent": 28000}
    }
}

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.markdown("""
<div style="
    font-size:20px;
    font-weight:900;
    color:#00f2fe;
    padding:8px 0px;
">
🏠 BUY OPTION INPUTS
</div>
""", unsafe_allow_html=True)
# City & Area selection
city = st.sidebar.selectbox("Select City", list(city_areas.keys()))
available_areas = list(city_areas[city].keys())
area_type = st.sidebar.selectbox("Select Area", available_areas)

# Auto-set price & rent based on area
base_price_per_sqft = city_areas[city][area_type]["price_per_sqft"]
base_monthly_rent = city_areas[city][area_type]["monthly_rent"]

sqft = st.sidebar.number_input("Area (sq.ft)", value=1000)
down_payment_pct = st.sidebar.slider("Down Payment (%)", 10, 90, 20)
interest_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 15.0, 9.0)
loan_tenure_yrs = st.sidebar.slider("Loan Tenure (Years)", 5, 30, 20)
analysis_years = st.sidebar.slider("Analysis Duration (Years)", 5, 30, 20)
mnt_tax_pct = st.sidebar.slider("Maintenance + Tax (% yearly)", 0.0, 5.0, 1.5)
investment_return_pct = st.sidebar.slider("Investment Return (%)", 0.0, 15.0, 10.0)
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 0.0, 10.0, 6.0)
st.sidebar.markdown("""
<div style="
    font-weight:800;
    font-size:16px;
    color:#ffc107;
    margin-top:10px;
">
⚠ ENABLE MARKET CRASH SIMULATION (-20%)
</div>
""", unsafe_allow_html=True)

crash_mode = st.sidebar.checkbox("", key="crash_checkbox")
client_stay_years = st.sidebar.number_input(t["client_stay_years"], min_value=1, max_value=50, value=10)

# Area intelligence
years_developed = st.sidebar.slider("Years Since Area Developed", 0, 50, 10)
near_location = st.sidebar.selectbox("Location Type", ["Near Metro City", "Near Town", "Rural Area"])
transport = st.sidebar.selectbox("Transport Facility", ["Excellent (Metro/Highway)", "Good (Bus/Auto)", "Average", "Poor"])


# Area multipliers & risk
area_price_multiplier = {"Prime/Civil Area": 1.3, "Suburban Area": 1.0, "Developing Area": 0.8, "Village Area": 0.5}
area_rent_multiplier = {"Prime/Civil Area": 1.3, "Suburban Area": 1.0, "Developing Area": 0.8, "Village Area": 0.5}
transport_bonus = {"Excellent (Metro/Highway)": 1.5, "Good (Bus/Auto)": 0.8, "Average": 0.3, "Poor": -0.5}
location_bonus = {"Near Metro City": 1.5, "Near Town": 0.5, "Rural Area": -0.5}
risk_dict = {"Prime/Civil Area": 2, "Suburban Area": 5, "Developing Area": 7, "Village Area": 9}

# ---------------- BUY ANALYSIS ----------------


def analyze(custom_interest=None):
    # Area-adjusted price & rent
    price_per_sqft_area = base_price_per_sqft
    monthly_rent_area = base_monthly_rent

    appreciation = 5 + transport_bonus[transport] + location_bonus[near_location]  # Default 5% appreciation
    if years_developed < 5: appreciation += 1.5
    elif years_developed > 30: appreciation -= 1

    home_price = price_per_sqft_area * sqft
    monthly_rent = monthly_rent_area
    down_payment = home_price * down_payment_pct / 100
    loan_amount = home_price - down_payment
    rate = interest_rate if custom_interest is None else custom_interest
    loan_months = loan_tenure_yrs * 12
    rate_monthly = rate / 100 / 12
    emi = npf.pmt(rate_monthly, loan_months, -loan_amount)

    home_val = home_price
    total_emi_paid = 0
    break_even_year = None
    yearly_data = []

    for year in range(1, analysis_years + 1):
        if crash_mode and year == 2: home_val *= 0.8
        home_val *= (1 + appreciation / 100)
        monthly_rent *= (1 + 4 / 100)  # Default 4% rent increase
        annual_rent = monthly_rent * 12
        annual_emi = emi * 12
        total_emi_paid += annual_emi
        if break_even_year is None and home_val >= total_emi_paid: break_even_year = year
        yearly_data.append({"Year": year, "Home Value": home_val, "Annual Rent": annual_rent, "Annual EMI": annual_emi})

    df = pd.DataFrame(yearly_data)
    net_buy_gain = home_val - home_price
    inflation_adjusted_return = net_buy_gain - (home_price * inflation_rate / 100 * analysis_years)
    rental_yield = (monthly_rent_area * 12 / home_price) * 100
    risk_score = 5  # Default fixed or could be calculated

    return home_price, emi, home_val, net_buy_gain, inflation_adjusted_return, rental_yield, risk_score, break_even_year, df

(home_price, emi, future_value, net_buy_gain, inflation_adj, rental_yield, risk_score, break_even_year, df) = analyze()

# ---------------- ML MODEL ----------------
years = np.arange(1, 21)
growth = np.array([1.05 ** y for y in years])
prices = home_price * growth
model = RandomForestRegressor()
model.fit(years.reshape(-1, 1), prices)
predicted_future = model.predict([[analysis_years]])[0]

# ---------------- CLIENT STAY DECISION ----------------
def client_decision_engine(years_to_stay):
    home_price_local = base_price_per_sqft * sqft
    down_payment = home_price_local * down_payment_pct / 100
    loan_amount = home_price_local - down_payment
    rate_monthly = interest_rate / 100 / 12
    emi_value = npf.pmt(rate_monthly, loan_tenure_yrs * 12, -loan_amount)
    current_home_value = home_price_local
    current_rent = base_monthly_rent
    total_rent_paid = 0
    total_emi_paid = 0
    for _ in range(years_to_stay):
        current_home_value *= 1.05  # Default 5% appreciation
        current_rent *= 1.04  # Default 4% rent increase
        total_rent_paid += current_rent * 12
        total_emi_paid += emi_value * 12
    net_buy_value = current_home_value - home_price_local - total_emi_paid
    decision_text = "BUY" if net_buy_value > -total_rent_paid else "RENT"
    return total_rent_paid, net_buy_value, decision_text

client_total_rent, client_net_buy, client_decision = client_decision_engine(client_stay_years)

# ---------------- RENT OPTION ----------------
st.sidebar.markdown("""
<div style="
    font-size:20px;
    font-weight:900;
    color:#ff4b5c;
    padding:8px 0px;
">
🏢 RENT OPTION INPUTS
</div>
""", unsafe_allow_html=True)
rent_area = st.sidebar.number_input("Rent Area (sq.ft)", value=1000)
rent_bhk = st.sidebar.selectbox("BHK", ["1BHK", "2BHK", "3BHK", "4BHK+"], key="rent_bhk")
monthly_rent_input = st.sidebar.number_input("Monthly Rent (Rs)", value=base_monthly_rent, key="monthly_rent_input")
rent_duration_years = st.sidebar.slider("Analysis Duration (Years)", 1, 30, 10, key="rent_duration_years")


def analyze_rent(area, monthly_rent, duration_years, rent_increase_pct=5):
    rent_growth = monthly_rent
    total_rent_paid = 0
    yearly_rent_data = []
    for year in range(1, duration_years + 1):
        yearly_rent = rent_growth * 12
        total_rent_paid += yearly_rent
        yearly_rent_data.append({"Year": year, "Annual Rent": yearly_rent})
        rent_growth *= (1 + rent_increase_pct / 100)
    df_rent = pd.DataFrame(yearly_rent_data)
    return total_rent_paid, df_rent

total_rent_paid, df_rent = analyze_rent(rent_area, monthly_rent_input, rent_duration_years)

# ---------------- DISPLAY ----------------
# ---------------- DISPLAY ----------------

# ===== TOP METRICS =====
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
    transition: 0.3s;
}
.metric-card:hover {
    transform: scale(1.05);
}
.metric-title {
    font-size: 18px;
    opacity: 0.9;
}
.metric-value {
    font-size: 28px;
    font-weight: 900;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{t["home_price"]}</div>
        <div class="metric-value">Rs {home_price:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
   st.markdown(f"""
    <div class="metric-card">
            <div class="metric-title">{t["monthly_emi"]}</div>
            <div class="metric-value">Rs {emi:,.0f}</div>
    </div>
    """,unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{t["ml_value"]}</div>
        <div class="metric-value">Rs {predicted_future:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# ===== INVESTMENT GROWTH SECTION =====
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("📈 Investment Growth Analysis")

col_g1, col_g2 = st.columns([2,1])

with col_g1:
    st.line_chart(df.set_index("Year")["Home Value"], height=350)



st.markdown('</div>', unsafe_allow_html=True)

# ===== EMI vs RENT =====
# ===== EMI vs RENT (Improved Visualization) =====
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Total Cost Comparison: EMI vs Rent")

# Calculate cumulative values
df["Total EMI Paid"] = df["Annual EMI"].cumsum()
df["Total Rent Paid"] = df["Annual Rent"].cumsum()

comparison_chart = df.set_index("Year")[["Total EMI Paid", "Total Rent Paid"]]

st.line_chart(comparison_chart, height=350)

st.caption("This chart shows how much total money you pay over time in EMI vs Rent.")

st.markdown('</div>', unsafe_allow_html=True)


# ===== CLIENT STAY =====
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader(t["client_stay"])
st.write(f"{t['client_stay_years']}: {client_stay_years}")
st.write(f"{t['total_rent']}: Rs {round(client_total_rent,2):,.0f}")
st.write(f"{t['net_position']}: Rs {round(client_net_buy,2):,.0f}")

if client_decision == "BUY":
    st.success(f"For {client_stay_years} years -> {t['decision_buy']}")
else:
    st.warning(f"For {client_stay_years} years -> {t['decision_rent']}")

st.markdown('</div>', unsafe_allow_html=True)
# ================= AREA INTELLIGENCE & RISK ANALYSIS =================

st.markdown("## 📊 Area Intelligence & Risk Analysis")

col_r1, col_r2 = st.columns([1, 1])

# ----------- RISK SCORE CARD -----------
with col_r1:

    # Risk Category
    if risk_score <= 3:
        color = "#00ff99"
        risk_text = "Low Risk Area"
        advice = "✅ Safe for Long-Term Investment"
    elif risk_score <= 6:
        color = "#ffc107"
        risk_text = "Moderate Risk Area"
        advice = "⚠️ Balanced Risk & Return"
    else:
        color = "#ff4b5c"
        risk_text = "High Risk Area"
        advice = "❌ High Volatility – Invest Carefully"

    st.markdown(f"""
        <div style='
            text-align:center;
            padding:18px;
            border-radius:15px;
            border:2px solid {color};
            box-shadow:0 5px 20px rgba(0,0,0,0.4);
            max-width:280px;
            margin:auto;
            background:rgba(255,255,255,0.05);
        '>
            <h2 style='color:{color}; font-size:28px; margin:0;'>
                {risk_score}/10
            </h2>
            <p style='font-size:15px; font-weight:600; margin:5px 0;'>
                {risk_text}
            </p>
            <p style='font-size:13px; margin:0;'>
                {advice}
            </p>
        </div>
    """, unsafe_allow_html=True)


# ----------- AREA METRICS TABLE -----------
with col_r2:

    if risk_score <= 3:
        safety_grade = "A (Very Safe)"
    elif risk_score <= 6:
        safety_grade = "B (Moderate)"
    else:
        safety_grade = "C (Risky)"

    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.05);
        padding:18px;
        border-radius:15px;
        box-shadow:0 4px 15px rgba(0,0,0,0.3);
    ">
        <table style="width:100%; border-collapse:collapse; font-size:15px;">
            <tr>
                <td style="font-weight:800; padding:8px;">Rental Yield</td>
                <td style="font-weight:700; text-align:right;">{round(rental_yield,2)}%</td>
            </tr>
            <tr>
                <td style="font-weight:800; padding:8px;">Inflation Adjusted Return</td>
                <td style="font-weight:700; text-align:right;">₹{round(inflation_adj,0):,.0f}</td>
            </tr>
            <tr>
                <td style="font-weight:800; padding:8px;">Break Even Year</td>
                <td style="font-weight:700; text-align:right;">
                    {break_even_year if break_even_year else "Not Reached"}
                </td>
            </tr>
            <tr>
                <td style="font-weight:800; padding:8px;">Safety Grade</td>
                <td style="font-weight:700; text-align:right;">{safety_grade}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ----------- INVESTMENT DECISION BANNER -----------

st.markdown("### 📌 AI Investment Recommendation")

if net_buy_gain > 0 and risk_score <= 6:
    rec_color = "#00ff99"
    recommendation = "🏆 BUY – Strong Investment Potential"
elif net_buy_gain > 0 and risk_score > 6:
    rec_color = "#ffc107"
    recommendation = "⚖️ Buy with Caution – High Risk Area"
else:
    rec_color = "#ff4b5c"
    recommendation = "🚫 Not Recommended for Investment"

st.markdown(f"""
    <div style='
        text-align:center;
        padding:15px;
        border-radius:12px;
        background:{rec_color};
        color:black;
        font-weight:bold;
        font-size:16px;
    '>
        {recommendation}
    </div>
""", unsafe_allow_html=True)
# Risk Card Design
 
st.write(f"**{t['rental_yield']}**: {round(rental_yield,2)}%")
st.write(f"**{t['inflation_adj']}**: Rs {round(inflation_adj,2):,.0f}")

if break_even_year:
    st.success(f"{t['break_even']}: {break_even_year}")
else:
    st.warning(t["break_even"] + " not reached")

st.markdown('</div>', unsafe_allow_html=True)

# ===== SENSITIVITY =====
_, emi2, *_ = analyze(custom_interest=interest_rate + 1)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader(t["sensitivity"])
st.write("EMI if Interest increases by 1%:", round(emi2, 2))
st.markdown('</div>', unsafe_allow_html=True)


# ===== RENT OPTION =====
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Rent Option Analysis")
st.write(f"Rent Area: {rent_area} sq.ft | BHK: {rent_bhk}")
st.write(f"Monthly Rent: Rs {monthly_rent_input:,.0f}")
st.write(f"Total Rent over {rent_duration_years} years: Rs {total_rent_paid:,.0f}")
st.bar_chart(df_rent.set_index("Year")["Annual Rent"], height=350)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- GOOGLE MAP PROPERTY VIEW (NEW FEATURE ADDED) ----------------
# ---------------- GOOGLE MAP PROPERTY VIEW ----------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("Explore Properties on Google Maps")

buy_map_url = f"https://www.google.com/maps/search/Property+for+sale+in+{area_type}+{city}"
rent_map_url = f"https://www.google.com/maps/search/Property+for+rent+in+{area_type}+{city}"

colA, colB = st.columns(2)

with colA:
    st.markdown(f"""
    <div style="text-align:center">
        <h3>🏠 Buy in {area_type}, {city}</h3>
        <a href="{buy_map_url}" target="_blank" class="map-btn buy-btn">
            View Buy Properties
        </a>
    </div>
    """, unsafe_allow_html=True)

with colB:
    st.markdown(f"""
    <div style="text-align:center">
        <h3>🏢 Rent in {area_type}, {city}</h3>
        <a href="{rent_map_url}" target="_blank" class="map-btn rent-btn">
            View Rental Properties
        </a>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
# ---------------- PDF REPORT ----------------
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8,
        f"AI Smart Real Estate Investment Report\n\n"
        f"Date: {datetime.date.today()}\n"
        f"City: {city}\n"
        f"Area: {area_type}\n"
        f"{t['home_price']}: Rs {home_price:,.0f}\n"
        f"{t['monthly_emi']}: Rs {emi:,.0f}\n"
        f"Future Value: Rs {future_value:,.0f}\n"
        f"{t['ml_value']}: Rs {predicted_future:,.0f}\n"
        f"Net Gain: Rs {net_buy_gain:,.0f}\n"
        f"{t['inflation_adj']}: Rs {inflation_adj:,.0f}\n"
        f"{t['rental_yield']}: {rental_yield:.2f}%\n"
        f"{t['risk_score']}: {risk_score}\n"
        f"{t['break_even']}: {break_even_year}\n\n"
        f"{t['client_stay']} ({client_stay_years} Years)\n"
        f"{t['total_rent']}: Rs {client_total_rent:,.0f}\n"
        f"{t['net_position']}: Rs {client_net_buy:,.0f}\n"
        f"Decision: {client_decision}\n\n"
        f"--- RENT OPTION ---\n"
        f"Rent Area: {rent_area} sq.ft | BHK: {rent_bhk}\n"
        f"Monthly Rent: Rs {monthly_rent_input:,.0f}\n"
        f"Total Rent ({rent_duration_years} yrs): Rs {total_rent_paid:,.0f}\n"
    )
    return bytes(pdf.output())

st.download_button(t["download_report"], generate_pdf(), file_name="AI_Property_Report.pdf")
# ---------------- FUTURE SCOPE (Only Add This Block) ----------------
st.subheader("Future Scope of Buying Property")

# Future Net Profit Projection (Extended 10 Years Beyond Analysis)
future_projection_years = 10
extended_home_value = future_value

for _ in range(future_projection_years):
    extended_home_value *= 1.05  # Assuming 5% steady appreciation

future_net_profit = extended_home_value - home_price

st.write(f"Projected Property Value after {analysis_years + future_projection_years} years: Rs {extended_home_value:,.0f}")
st.write(f"Estimated Future Net Profit: Rs {future_net_profit:,.0f}")

if future_net_profit > 0:
    st.success("Long-term future scope shows strong capital appreciation potential.")
else:
    st.warning("Future growth appears limited based on current assumptions.")