import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from deep_translator import GoogleTranslator
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ----------------------------
# Gemini API Configuration
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# Load XGBoost Model
# ----------------------------
model = pickle.load(open("xgb_model.pkl", "rb"))

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🤱",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #f0f4f8;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(14,116,144,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(6,78,59,0.06) 0%, transparent 50%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1100px;
}

.header-banner {
    background: linear-gradient(135deg, #0c4a6e 0%, #0e7490 60%, #0891b2 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(14,116,144,0.30);
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.header-banner::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 320px; height: 320px;
    background: rgba(255,255,255,0.03);
    border-radius: 50%;
}
.header-title {
    font-family: 'Sora', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.header-subtitle {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.75);
    font-weight: 400;
    margin: 0;
}
.header-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}

.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid rgba(0,0,0,0.05);
}
.card-title {
    font-family: 'Sora', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #f1f5f9;
}

.stSelectbox > div > div {
    background: white;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif;
    color: #1e293b !important;
}
.stSelectbox > div > div:hover {
    border-color: #0e7490 !important;
}
/* Fix selected value text color */
.stSelectbox span, .stSelectbox div[data-baseweb="select"] * {
    color: #1e293b !important;
}
[data-baseweb="select"] > div {
    color: #1e293b !important;
    background: white !important;
}
[data-baseweb="select"] input {
    color: #1e293b !important;
}
/* ── Dropdown options – desktop + mobile ── */
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li,
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] ul,
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
    background-color: #1e293b !important;
    color: #ffffff !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"]    [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"],
[data-baseweb="menu"]    [aria-selected="true"] {
    background-color: #0e7490 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
/* Mobile listbox fallback */
ul[role="listbox"],
ul[role="listbox"] li,
ul[role="listbox"] [role="option"],
div[role="listbox"],
div[role="listbox"] [role="option"] {
    background-color: #1e293b !important;
    color: #ffffff !important;
}
li[role="option"]:hover,
div[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background-color: #0e7490 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
/* Override Streamlit default light blue hover */
[data-baseweb="list-item"]:hover,
[data-baseweb="list-item"]:focus,
[data-baseweb="menu-item"]:hover,
[data-baseweb="menu-item"]:focus {
    background-color: #0e7490 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
label {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

.stNumberInput > div > div > input {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #1e293b !important;
    padding: 0.55rem 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stNumberInput > div > div > input:focus {
    border-color: #0e7490 !important;
    box-shadow: 0 0 0 3px rgba(14,116,144,0.12) !important;
    background: white !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0c4a6e, #0e7490) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 14px rgba(14,116,144,0.35) !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(14,116,144,0.45) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

.risk-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.risk-low    { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1.5px solid #86efac; }
.risk-medium { background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 1.5px solid #fcd34d; }
.risk-high   { background: linear-gradient(135deg, #fff1f2, #ffe4e6); border: 1.5px solid #fca5a5; }
.risk-icon { font-size: 3rem; line-height: 1; }
.risk-label {
    font-family: 'Sora', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.risk-low .risk-label    { color: #15803d; }
.risk-medium .risk-label { color: #b45309; }
.risk-high .risk-label   { color: #b91c1c; }
.risk-sub { font-size: 0.85rem; color: #64748b; margin-top: 0.2rem; }

.explain-box {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border: 1.5px solid #7dd3fc;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.5rem;
}
.explain-box p { font-size: 0.95rem; line-height: 1.7; color: #1e3a5f; margin: 0; }

.translate-box {
    background: linear-gradient(135deg, #faf5ff, #f3e8ff);
    border: 1.5px solid #c4b5fd;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 0.5rem;
}
.translate-box p { font-size: 1rem; line-height: 1.8; color: #3b0764; margin: 0; }

.vitals-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 0.5rem;
}
.vital-chip {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.vital-chip-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #94a3b8;
    display: block;
    margin-bottom: 0.3rem;
}
.vital-chip-value {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="header-banner">
    <div class="header-badge">🤱 ASHA Worker Tool</div>
    <h1 class="header-title">Maternal Health Risk Predictor</h1>
    <p class="header-subtitle">AI-powered risk assessment · Multilingual support · Instant SHAP explanation</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Layout
# ----------------------------
left_col, right_col = st.columns([1, 1.1], gap="large")

with left_col:

    st.markdown('<div class="card"><div class="card-title">🌐 Language Preference</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Select Language",
        [
            "English",
            "Hindi",
            "Marathi",
            "Telugu",
            "Tamil",
            "Gujarati",
            "Odia",
            "Bhojpuri",
            "Bengali",
            "Kannada",
            "Malayalam",
            "Punjabi",
            "Urdu",
            "Assamese",
        ],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    lang_codes = {
        "English":    "en",
        "Hindi":      "hi",
        "Marathi":    "mr",
        "Telugu":     "te",
        "Tamil":      "ta",
        "Gujarati":   "gu",
        "Odia":       "or",
        "Bhojpuri":   "bho",
        "Bengali":    "bn",
        "Kannada":    "kn",
        "Malayalam":  "ml",
        "Punjabi":    "pa",
        "Urdu":       "ur",
        "Assamese":   "as",
    }

    st.markdown('<div class="card"><div class="card-title">📋 Patient Vitals</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age         = st.number_input("Age (years)",          min_value=10,  max_value=60,   value=25,   step=1,   help="Maternal age: 10–60 years")
        sys_bp      = st.number_input("Systolic BP (mmHg)",   min_value=70,  max_value=220,  value=120,  step=1,   help="Normal: 90–120 | High: >140 | Crisis: >180")
        blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=6.0,  step=0.1, format="%.1f", help="Normal fasting: 4.0–7.0 | High: >8.0 | Very high: >12.0")
    with col2:
        dia_bp      = st.number_input("Diastolic BP (mmHg)",  min_value=40,  max_value=140,  value=80,   step=1,   help="Normal: 60–80 | High: >90 | Crisis: >120")
        body_temp   = st.number_input("Body Temp (°F)",       min_value=94.0, max_value=107.0, value=98.6, step=0.1, format="%.1f", help="Normal: 97.0–99.0 | Fever: >100.4 | High fever: >103")
        heart_rate  = st.number_input("Heart Rate (bpm)",     min_value=30,  max_value=220,  value=75,   step=1,   help="Normal: 60–100 | High: >100 | Low: <60")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <div class="card-title">📊 Entered Values</div>
        <div class="vitals-grid">
            <div class="vital-chip">
                <span class="vital-chip-label">Age</span>
                <span class="vital-chip-value">{age} yrs</span>
            </div>
            <div class="vital-chip">
                <span class="vital-chip-label">BP</span>
                <span class="vital-chip-value">{sys_bp}/{dia_bp}</span>
            </div>
            <div class="vital-chip">
                <span class="vital-chip-label">Blood Sugar</span>
                <span class="vital-chip-value">{blood_sugar:.1f}</span>
            </div>
            <div class="vital-chip">
                <span class="vital-chip-label">Body Temp</span>
                <span class="vital-chip-value">{body_temp:.1f}°F</span>
            </div>
            <div class="vital-chip">
                <span class="vital-chip-label">Heart Rate</span>
                <span class="vital-chip-value">{heart_rate} bpm</span>
            </div>
            <div class="vital-chip">
                <span class="vital-chip-label">Language</span>
                <span class="vital-chip-value">{language[:3]}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("🔍  Analyse Risk", use_container_width=True)

# ----------------------------
# Results
# ----------------------------
with right_col:

    if not predict_btn:
        st.markdown("""
        <div style="
            background: white; border-radius: 20px; padding: 3rem 2rem;
            text-align: center; border: 2px dashed #e2e8f0; color: #94a3b8; margin-top: 0.5rem;
        ">
            <div style="font-size:3.5rem; margin-bottom:1rem;">🩺</div>
            <p style="font-family:'Sora',sans-serif; font-size:1rem; font-weight:600;
                      color:#cbd5e1; margin:0;">
                Fill in patient vitals<br>and click Analyse Risk
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        # ── Feature Engineering (core logic unchanged) ──
        bp_diff = sys_bp - dia_bp
        mean_bp = (sys_bp + 2 * dia_bp) / 3
        high_bs = 1 if blood_sugar > 8 else 0
        fever   = 1 if body_temp > 98.6 else 0

        features = pd.DataFrame({
            "Age":        [age],
            "SystolicBP": [sys_bp],
            "DiastolicBP":[dia_bp],
            "BS":         [blood_sugar],
            "BodyTemp":   [body_temp],
            "HeartRate":  [heart_rate],
            "BP_Diff":    [bp_diff],
            "Mean_BP":    [mean_bp],
            "HighBS":     [high_bs],
            "Fever":      [fever]
        })

        # ── ML Prediction (core logic unchanged) ──
        prediction = int(model.predict(features)[0])
        risk_map   = {1: "Low Risk", 2: "Medium Risk", 0: "High Risk"}
        risk_level = risk_map.get(prediction, "Unknown")

        # ── Risk Card ──
        if risk_level == "Low Risk":
            icon, css_class, sub = "✅", "risk-low", "Vitals are within acceptable range."
        elif risk_level == "Medium Risk":
            icon, css_class, sub = "⚠️", "risk-medium", "Some vitals need monitoring."
        else:
            icon, css_class, sub = "🚨", "risk-high", "Immediate medical attention advised."

        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div class="risk-icon">{icon}</div>
            <div>
                <p class="risk-label">{risk_level}</p>
                <p class="risk-sub">{sub}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP Explainability (core logic unchanged) ──
        st.markdown('<div class="card"><div class="card-title">🔬 Factors Influencing Prediction</div>', unsafe_allow_html=True)

        try:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
            shap_vals   = np.array(shap_values)

            if shap_vals.ndim == 3:
                shap_vals = shap_vals[0, :, prediction]
            elif shap_vals.ndim == 2:
                shap_vals = shap_vals[0]

            shap_vals     = shap_vals.flatten()
            feature_names = features.columns.tolist()

            shap_df = pd.DataFrame({
                "Feature":    feature_names,
                "SHAP Value": shap_vals[:len(feature_names)]
            })
            shap_df["abs"] = shap_df["SHAP Value"].abs()
            shap_df = shap_df.sort_values(by="abs", ascending=False).reset_index(drop=True)

            st.dataframe(
                shap_df[["Feature", "SHAP Value"]].style
                    .format({"SHAP Value": "{:+.4f}"})
                    .background_gradient(subset=["SHAP Value"], cmap="RdYlGn_r")
                    .set_properties(**{"font-family": "DM Sans, sans-serif", "font-size": "13px"}),
                use_container_width=True,
                hide_index=True,
                height=200
            )

            # Styled SHAP chart
            fig, ax = plt.subplots(figsize=(6, 3.5))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fafc')

            colors = ['#ef4444' if v > 0 else '#22c55e' for v in shap_df["SHAP Value"]]
            ax.barh(shap_df["Feature"], shap_df["SHAP Value"],
                    color=colors, edgecolor='white', linewidth=0.8, height=0.65)

            ax.axvline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
            ax.set_xlabel("Impact on Risk Prediction", fontsize=9, color='#64748b', labelpad=8)
            ax.set_title("Feature Impact (SHAP)", fontsize=10, color='#1e293b', fontweight='bold', pad=10)
            ax.invert_yaxis()
            ax.tick_params(axis='both', labelsize=8, colors='#475569')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('#e2e8f0')
            ax.grid(axis='x', alpha=0.3, color='#e2e8f0')

            pos_patch = mpatches.Patch(color='#ef4444', label='Increases Risk')
            neg_patch = mpatches.Patch(color='#22c55e', label='Decreases Risk')
            ax.legend(handles=[pos_patch, neg_patch], fontsize=7, loc='lower right', framealpha=0.8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            top_features = shap_df.head(3)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
            top_features = pd.DataFrame({"Feature": ["Age","SystolicBP","BS"], "SHAP Value": [0,0,0]})

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Dynamic Gemini Explanation ──

        # Build per-parameter severity strings with exact values + normal ranges
        def build_vital_details(age, sys_bp, dia_bp, blood_sugar, body_temp, heart_rate):
            details = []

            if sys_bp >= 180:
                details.append(f"Systolic BP {sys_bp} mmHg — CRISIS (normal <120), hypertensive emergency")
            elif sys_bp >= 140:
                details.append(f"Systolic BP {sys_bp} mmHg — HIGH (normal <120), stage-2 hypertension")
            elif sys_bp >= 130:
                details.append(f"Systolic BP {sys_bp} mmHg — ELEVATED (normal <120), stage-1 range")
            else:
                details.append(f"Systolic BP {sys_bp} mmHg — NORMAL (normal <120)")

            if dia_bp >= 120:
                details.append(f"Diastolic BP {dia_bp} mmHg — CRISIS (normal <80), organ-damage risk")
            elif dia_bp >= 90:
                details.append(f"Diastolic BP {dia_bp} mmHg — HIGH (normal <80)")
            elif dia_bp >= 80:
                details.append(f"Diastolic BP {dia_bp} mmHg — BORDERLINE (normal <80)")
            else:
                details.append(f"Diastolic BP {dia_bp} mmHg — NORMAL (normal <80)")

            if blood_sugar > 12:
                details.append(f"Blood sugar {blood_sugar} mmol/L — VERY HIGH (normal 4–8), severe hyperglycemia")
            elif blood_sugar > 8:
                details.append(f"Blood sugar {blood_sugar} mmol/L — HIGH (normal 4–8), elevated glucose")
            elif blood_sugar < 4:
                details.append(f"Blood sugar {blood_sugar} mmol/L — LOW (normal 4–8), hypoglycemia risk")
            else:
                details.append(f"Blood sugar {blood_sugar} mmol/L — NORMAL (normal 4–8)")

            if body_temp >= 103:
                details.append(f"Body temp {body_temp}°F — HIGH FEVER (normal 97–99°F), possible sepsis")
            elif body_temp >= 100.4:
                details.append(f"Body temp {body_temp}°F — FEVER (normal 97–99°F), signs of infection")
            elif body_temp >= 99.1:
                details.append(f"Body temp {body_temp}°F — LOW-GRADE FEVER (normal 97–99°F)")
            else:
                details.append(f"Body temp {body_temp}°F — NORMAL (normal 97–99°F)")

            if heart_rate > 130:
                details.append(f"Heart rate {heart_rate} bpm — SEVERELY HIGH (normal 60–100), severe tachycardia")
            elif heart_rate > 100:
                details.append(f"Heart rate {heart_rate} bpm — HIGH (normal 60–100), tachycardia")
            elif heart_rate < 50:
                details.append(f"Heart rate {heart_rate} bpm — LOW (normal 60–100), bradycardia")
            else:
                details.append(f"Heart rate {heart_rate} bpm — NORMAL (normal 60–100)")

            if age < 18:
                details.append(f"Age {age} — teenage pregnancy, higher risk")
            elif age > 35:
                details.append(f"Age {age} — advanced maternal age, elevated risk")
            else:
                details.append(f"Age {age} — normal childbearing age range")

            return details

        def get_abnormal_flags(sys_bp, dia_bp, blood_sugar, body_temp, heart_rate, age):
            flags = []
            if sys_bp >= 130:   flags.append(f"high systolic BP ({sys_bp} mmHg)")
            if dia_bp >= 80:    flags.append(f"elevated diastolic BP ({dia_bp} mmHg)")
            if blood_sugar > 8: flags.append(f"high blood sugar ({blood_sugar} mmol/L)")
            if blood_sugar < 4: flags.append(f"low blood sugar ({blood_sugar} mmol/L)")
            if body_temp >= 99.1: flags.append(f"elevated temperature ({body_temp}°F)")
            if heart_rate > 100 or heart_rate < 60: flags.append(f"abnormal heart rate ({heart_rate} bpm)")
            if age < 18 or age > 35: flags.append(f"age-related risk ({age} yrs)")
            return flags if flags else ["all vitals within normal range"]

        vital_details  = build_vital_details(age, sys_bp, dia_bp, blood_sugar, body_temp, heart_rate)
        abnormal_flags = get_abnormal_flags(sys_bp, dia_bp, blood_sugar, body_temp, heart_rate, age)

        detail_block  = "\n".join(f"  • {d}" for d in vital_details)
        abnormal_text = ", ".join(abnormal_flags)

        risk_tone = {
            "High Risk":   "Use an URGENT tone. Stress the danger clearly. Strongly recommend going to hospital immediately.",
            "Medium Risk": "Use a CONCERNED but calm tone. Highlight what needs attention. Advise check-up soon.",
            "Low Risk":    "Use a REASSURING tone. Acknowledge any mild values. Encourage healthy habits.",
        }.get(risk_level, "Provide a clear health explanation.")

        prompt = f"""
You are a maternal health assistant. A risk model predicted: {risk_level}.

Patient vitals with severity:
{detail_block}

Abnormal parameters: {abnormal_text}

Instruction: {risk_tone}

Write EXACTLY 4 sentences for a rural ASHA health worker. Rules:
- Sentence 1: State the risk level and mention the specific abnormal vitals with their values.
- Sentence 2: Explain in simple words WHY these values are concerning for this risk level.
- Sentence 3: Give specific actionable steps the health worker should take RIGHT NOW.
- Sentence 4: Give a follow-up or monitoring recommendation.
- Use simple, clear language — no medical jargon.
- NEVER use phrases like "some parameters appear risky" — reference actual numbers.
- Each patient gets a unique message based on their exact values.
"""

        st.markdown('<div class="card"><div class="card-title">🤖 AI Health Explanation</div>', unsafe_allow_html=True)

        with st.spinner("Generating explanation…"):
            try:
                response    = gemini_model.generate_content(prompt)
                explanation = response.text.strip()
            except Exception:
                # Fallback with actual values
                if risk_level == "High Risk":
                    explanation = (
                        f"This patient is HIGH RISK with BP {sys_bp}/{dia_bp} mmHg, "
                        f"blood sugar {blood_sugar} mmol/L, and heart rate {heart_rate} bpm — several values are dangerously outside normal. "
                        f"High blood pressure and elevated glucose together significantly increase the chance of serious complications during pregnancy. "
                        f"Take the patient to a hospital or emergency clinic immediately and do not wait. "
                        f"Check vitals every 30 minutes until medical help is reached."
                    )
                elif risk_level == "Medium Risk":
                    explanation = (
                        f"This patient is MEDIUM RISK — BP is {sys_bp}/{dia_bp} mmHg, "
                        f"blood sugar is {blood_sugar} mmol/L, temperature {body_temp}°F, and heart rate {heart_rate} bpm. "
                        f"Some of these readings are above the safe range and need attention before they worsen. "
                        f"Schedule a doctor visit within 1–2 days and advise the patient to rest and reduce salt intake. "
                        f"Monitor BP and blood sugar daily and record any changes."
                    )
                else:
                    explanation = (
                        f"This patient is LOW RISK — BP {sys_bp}/{dia_bp} mmHg, "
                        f"blood sugar {blood_sugar} mmol/L, temperature {body_temp}°F, and heart rate {heart_rate} bpm are all in a safe range. "
                        f"The vitals look healthy and there are no immediate danger signs at this time. "
                        f"Encourage the patient to maintain a balanced diet, stay hydrated, and attend routine antenatal check-ups. "
                        f"Recheck vitals at the next scheduled visit."
                    )

        st.markdown(f'<div class="explain-box"><p>{explanation}</p></div>', unsafe_allow_html=True)

        # ── Translation (core logic unchanged) ──
        if language != "English":
            with st.spinner(f"Translating to {language}…"):
                try:
                    translated = GoogleTranslator(
                        source="auto", target=lang_codes[language]
                    ).translate(explanation)
                except Exception:
                    translated = explanation

            st.markdown(f"""
            <div class="card-title" style="margin-top:1.2rem; padding-top:0;">
                🗣️ Explanation in {language}
            </div>
            <div class="translate-box"><p>{translated}</p></div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)