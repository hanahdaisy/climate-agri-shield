import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Climate-Agri Shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# BRIGHT, PROFESSIONAL THEME
# ==================================================
st.markdown("""
<style>
.stApp {
    background-color: #F5F7FA;
    color: #1F2933;
}

[data-testid="stSidebar"] {
    background-color: #E4ECF7;
}

h1, h2, h3, h4 {
    color: #0A2540;
}

div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #0A6EBD !important;
}

button[data-baseweb="tab"] p {
    color: #0A2540 !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATA & MODELS
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("data/Master_Dataset NEW.csv")

@st.cache_resource
def load_models():
    return (
        joblib.load("models/risk_classifier.pkl"),
        joblib.load("models/yield_regression_model (1).pkl")
    )

df = load_data()
risk_model, yield_model = load_models()

# ==================================================
# FEATURES
# ==================================================
RISK_FEATURES = [
    'Area_Harvested', 'Production_Tonnes', 'Avg_Temp', 'Temp_Volatility',
    'GDP_current_US', 'political_stability_estimate', 'Inflation',
    'CO2_emisions', 'Agri_Land_Percent', 'Forest_Land_Percent', 'population'
]

YIELD_FEATURES = ['Avg_Temp', 'Temp_Volatility', 'GDP_current_US']

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.header("Dashboard Controls")
    country = st.selectbox("Select Country", sorted(df["Country"].unique()))

    st.markdown("---")
    st.subheader("Policy Simulation")
    temp_delta = st.slider("Temperature Shift (°C)", -2.0, 4.0, 0.0, 0.5)
    gdp_delta = st.slider("GDP Change (%)", -20, 30, 0, 5)

# ==================================================
# DATA PREP
# ==================================================
country_df = df[df["Country"] == country].sort_values("Year")
latest = country_df.iloc[-1]

risk_input = {f: latest[f] for f in RISK_FEATURES}
yield_input = {f: latest[f] for f in YIELD_FEATURES}

risk_input["Avg_Temp"] += temp_delta
risk_input["GDP_current_US"] *= (1 + gdp_delta / 100)
yield_input["Avg_Temp"] += temp_delta
yield_input["GDP_current_US"] *= (1 + gdp_delta / 100)

risk_df = pd.DataFrame([risk_input])
yield_df = pd.DataFrame([yield_input])

# ==================================================
# HEADER
# ==================================================
st.title("🌍 Climate-Agri Shield")
st.subheader(f"Country Focus: {country}")

m1, m2, m3 = st.columns(3)
m1.metric("Avg Temperature", f"{latest['Avg_Temp']:.2f} °C")
m2.metric("GDP", f"${latest['GDP_current_US']/1e9:.1f} B")
m3.metric("Population", f"{latest['population']/1e6:.1f} M")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Overview", "Predictions", "Trends"])

# ==================================================
# OVERVIEW
# ==================================================
with tab1:
    st.write("### Agricultural & Economic Profile")
    st.write(f"• Agricultural Land: {latest['Agri_Land_Percent']:.1f}%")
    st.write(f"• Forest Coverage: {latest['Forest_Land_Percent']:.1f}%")
    st.write(f"• Inflation Rate: {latest['Inflation']:.1f}%")
    st.write(f"• Temperature Volatility: {latest['Temp_Volatility']:.3f}")

# ==================================================
# PREDICTIONS
# ==================================================
with tab2:
    st.write("### Risk Classification & Yield Forecast")

    risk_prob = risk_model.predict_proba(risk_df)[0][1]
    yield_pred = yield_model.predict(yield_df)[0]

    # ---------- LOGICAL RISK FRAMEWORK ----------
    if risk_prob >= 0.65 and temp_delta > 0:
        risk_label = "🚨 HIGH RISK"
        color = "#D7263D"
    elif risk_prob <= 0.35 and temp_delta <= 0 and gdp_delta >= 10:
        risk_label = "✅ LOW RISK"
        color = "#1B9E77"
    else:
        risk_label = "⚠️ STABLE"
        color = "#E6A700"

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"<h2 style='color:{color}'>{risk_label}</h2>", unsafe_allow_html=True)
        st.progress(risk_prob)
        st.caption(f"Estimated Risk Probability: {risk_prob:.1%}")

    with c2:
        st.metric("Predicted Yield", f"{yield_pred:.2f} t/ha")

    st.info(
        f"Under the simulated scenario (Temperature Shift: {temp_delta}°C, GDP Change: {gdp_delta}%), "
        "the model evaluates climate stress alongside economic capacity to classify food security risk."
    )

# ==================================================
# TRENDS
# ==================================================
with tab3:
    fig1 = px.line(country_df, x="Year", y="Avg_Temp", title="Historical Temperature Trend")
    fig2 = px.line(country_df, x="Year", y="Production_Tonnes", title="Historical Production Trend")

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Historical trends provide context for understanding long-term climate variability "
        "and its relationship with agricultural output."
    )

st.markdown("---")
st.caption("Climate-Agri Shield | Climate–Agriculture Decision Support Dashboard")
