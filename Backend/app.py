import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator
import google.generativeai as genai
from dotenv import load_dotenv
import os

# ----------------------------
# Gemini API Configuration
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

st.set_page_config(page_title="Maternal Health Risk Predictor")

st.title("Maternal Health Risk Prediction System")
st.write("AI Assistant for ASHA Workers")

# ----------------------------
# Language Selection
# ----------------------------

language = st.selectbox(
    "Select Language",
    ["English", "Hindi", "Marathi", "Telugu", "Tamil"]
)

lang_codes = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Telugu": "te",
    "Tamil": "ta"
}

# ----------------------------
# Patient Inputs
# ----------------------------

st.subheader("Enter Patient Vitals")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 15, 50)
    sys_bp = st.number_input("Systolic BP", 80, 200)
    blood_sugar = st.number_input("Blood Sugar", 3.0, 20.0)

with col2:
    dia_bp = st.number_input("Diastolic BP", 50, 130)
    body_temp = st.number_input("Body Temperature", 95.0, 105.0)
    heart_rate = st.number_input("Heart Rate", 40, 180)

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict Risk"):

    # ----------------------------
    # Feature Engineering
    # ----------------------------

    bp_diff = sys_bp - dia_bp
    mean_bp = (sys_bp + 2 * dia_bp) / 3
    high_bs = 1 if blood_sugar > 8 else 0
    fever = 1 if body_temp > 98.6 else 0

    # ----------------------------
    # Final Feature DataFrame
    # ----------------------------

    features = pd.DataFrame({

        "Age":[age],
        "SystolicBP":[sys_bp],
        "DiastolicBP":[dia_bp],
        "BS":[blood_sugar],
        "BodyTemp":[body_temp],
        "HeartRate":[heart_rate],
        "BP_Diff":[bp_diff],
        "Mean_BP":[mean_bp],
        "HighBS":[high_bs],
        "Fever":[fever]

    })

    # ----------------------------
    # ML Prediction
    # ----------------------------

    prediction = int(model.predict(features)[0])

    risk_map = {
        1: "Low Risk",
        2: "Medium Risk",
        0: "High Risk"
    }

    risk_level = risk_map.get(prediction, "Unknown")

    st.subheader("Risk Level")

    if risk_level == "Low Risk":
        st.success("Low Risk")

    elif risk_level == "Medium Risk":
        st.warning("Medium Risk")

    else:
        st.error("High Risk")

    # ----------------------------
    # SHAP Explainability
    # ----------------------------

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(features)

        shap_vals = np.array(shap_values)

        # Handle different SHAP shapes
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[0, :, prediction]

        elif shap_vals.ndim == 2:
            shap_vals = shap_vals[0]

        shap_vals = shap_vals.flatten()

        feature_names = features.columns.tolist()

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_vals[:len(feature_names)]
        })

        shap_df["abs"] = shap_df["SHAP Value"].abs()

        shap_df = shap_df.sort_values(by="abs", ascending=False)

        st.subheader("Factors Influencing Prediction")

        st.dataframe(shap_df[["Feature","SHAP Value"]])

        # ----------------------------
        # SHAP Chart
        # ----------------------------

        fig, ax = plt.subplots()

        ax.barh(
            shap_df["Feature"],
            shap_df["SHAP Value"]
        )

        ax.set_xlabel("Impact on Risk")
        ax.set_title("Feature Impact on Prediction")

        ax.invert_yaxis()

        st.pyplot(fig)

        top_features = shap_df.head(3)

    except Exception as e:

        st.warning("SHAP explanation unavailable")

        top_features = pd.DataFrame({
            "Feature": ["Age","SystolicBP","BS"],
            "SHAP Value":[0,0,0]
        })

    # ----------------------------
    # Gemini Explanation
    # ----------------------------

    prompt = f"""
    A maternal health screening model predicted: {risk_level}.

    Important contributing factors were:

    {top_features.to_string(index=False)}

    Explain this result in very simple language for a rural health worker.

    Requirements:
    - Maximum 2 sentences explanation
    - 1 clear recommended action
    """

    try:

        response = gemini_model.generate_content(prompt)

        explanation = response.text

    except:

        explanation = "Patient vitals suggest this risk level. Please monitor the patient and consult a doctor if values worsen."

    st.subheader("AI Explanation (English)")
    st.write(explanation)

    # ----------------------------
    # Translation
    # ----------------------------

    if language != "English":

        translated = GoogleTranslator(
            source="auto",
            target=lang_codes[language]
        ).translate(explanation)

        st.subheader(f"Explanation in {language}")

        st.write(translated)