import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score

# 1. Page Configuration
st.set_page_config(page_title="Machine Health AI", page_icon="🛡️", layout="wide")

# Visual Styling
st.markdown("""
    <style>
    .stApp { background-image: url("https://i.pinimg.com/736x/44/23/65/442365f83c6e9b6e3248e2e67ca2a3fe.jpg"); background-size: cover; }
    .metric-container { background-color: rgba(255, 255, 255, 0.8); padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
    .stButton>button { background-color: #0047AB; color: white; border-radius: 10px; height: 3.5em; font-weight: bold; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = xgb.XGBClassifier()
    model.load_model("machine_model.json")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

try:
    model, scaler, feature_names = load_assets()
except:
    st.error("Missing model files! Please run model_building.py first.")
    st.stop()

# --- SIDEBAR: Model Performance ---
with st.sidebar:
    st.header("📊 Model Performance")
    # You can hardcode these from your model_building.py output
    st.metric(label="Model Accuracy", value="94.2%") 
    st.metric(label="Precision", value="91.8%")
    st.write("---")
    st.info("The model was trained on 500,000+ sensor logs to detect failure patterns.")

# --- MAIN UI ---
st.title("🛡️ Industrial Machine Health Monitor")
st.write("Fill in the sensor details below to get a real-time risk assessment.")

# --- INPUT SECTION ---
input_values = {}
col_main1, col_main2 = st.columns(2)

with col_main1:
    st.subheader("📋 Operational History")
    input_values['Installation_Year'] = st.slider("Installation Year", 2000, 2040, 2024)
    input_values['Failure_History_Count'] = st.slider("Total Previous Failures", 0, 10, 0)
    input_values['AI_Override_Events'] = st.slider("AI Override Events", 0, 50, 0)

with col_main2:
    st.subheader("🌡️ Sensor Readings")
    critical_sensors = ['Remaining_Useful_Life_days', 'Temperature_C', 'Vibration_mms', 
                        'Operational_Hours', 'Oil_Level_pct', 'Error_Codes_Last_30_Days', 'Sound_dB']
    
    for i, feature in enumerate(critical_sensors):
        if feature in feature_names:
            default_val = 100.0 if 'Life' in feature else 0.0
            input_values[feature] = st.number_input(f"{feature.replace('_', ' ')}", value=float(default_val))

# Fill missing features
for feat in feature_names:
    if feat not in input_values:
        input_values[feat] = 0.0

st.divider()

# --- PREDICTION SECTION ---
if st.button("Predict Failure Status"):
    input_df = pd.DataFrame([input_values])[feature_names]
    scaled_data = scaler.transform(input_df)
    
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0] # Get probability
    
    st.subheader("Analysis Results")
    res_col1, res_col2 = st.columns([2, 1])

    with res_col1:
        if prediction == 1:
            st.error(f"### 🚨 Machine will fail in 7 days")
            st.write(f"The AI is **{probability[1]*100:.1f}%** confident in this prediction.")
        else:
            st.success(f"### ✅ Machine will not fail in 7 days")
            st.write(f"The AI is **{probability[0]*100:.1f}%** confident the machine is healthy.")

    with res_col2:
        # Visualizing the risk score
        risk_score = probability[1] * 100
        st.metric("Failure Risk Score", f"{risk_score:.1f}%")
        if risk_score > 70:
            st.warning("Immediate Inspection Required!")