from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import google.generativeai as genai

# ---------------------------
# Load .env and Gemini key
# ---------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load ML model
# ---------------------------
model = pickle.load(open("xgb_model.pkl", "rb"))

# ---------------------------
# Input data model
# ---------------------------
class PatientData(BaseModel):
    age: int
    systolic_bp: float
    diastolic_bp: float
    blood_sugar: float
    body_temp: float
    heart_rate: int
    language: str  # for translation

# ---------------------------
# Helper: Build detailed severity analysis per parameter
# ---------------------------
def parameter_severity_text(data: PatientData):
    """
    Returns a list of detailed strings for each vital, including
    the exact value, normal range, and severity label.
    These are injected into the Gemini prompt to force dynamic output.
    """
    details = []

    # --- Systolic BP ---
    if data.systolic_bp >= 180:
        details.append(
            f"Systolic BP: {data.systolic_bp} mmHg — CRISIS LEVEL (normal: <120). "
            f"Hypertensive crisis, immediate emergency care needed."
        )
    elif data.systolic_bp >= 140:
        details.append(
            f"Systolic BP: {data.systolic_bp} mmHg — HIGH (normal: <120). "
            f"Stage 2 hypertension, urgent medical attention required."
        )
    elif data.systolic_bp >= 130:
        details.append(
            f"Systolic BP: {data.systolic_bp} mmHg — ELEVATED (normal: <120). "
            f"Stage 1 hypertension range, lifestyle changes needed."
        )
    else:
        details.append(
            f"Systolic BP: {data.systolic_bp} mmHg — NORMAL (normal: <120). "
            f"Within safe range."
        )

    # --- Diastolic BP ---
    if data.diastolic_bp >= 120:
        details.append(
            f"Diastolic BP: {data.diastolic_bp} mmHg — CRISIS LEVEL (normal: <80). "
            f"Dangerously high, risk of organ damage."
        )
    elif data.diastolic_bp >= 90:
        details.append(
            f"Diastolic BP: {data.diastolic_bp} mmHg — HIGH (normal: <80). "
            f"Stage 2 hypertension, monitor closely."
        )
    elif data.diastolic_bp >= 80:
        details.append(
            f"Diastolic BP: {data.diastolic_bp} mmHg — SLIGHTLY ELEVATED (normal: <80). "
            f"Borderline high, watch diet and stress."
        )
    else:
        details.append(
            f"Diastolic BP: {data.diastolic_bp} mmHg — NORMAL (normal: <80). "
            f"Within safe range."
        )

    # --- Blood Sugar ---
    if data.blood_sugar > 12:
        details.append(
            f"Blood Sugar: {data.blood_sugar} mmol/L — VERY HIGH (normal: 4–8). "
            f"Severe hyperglycemia, risk of gestational diabetes complications."
        )
    elif data.blood_sugar > 8:
        details.append(
            f"Blood Sugar: {data.blood_sugar} mmol/L — HIGH (normal: 4–8). "
            f"Elevated blood sugar, dietary control and monitoring needed."
        )
    elif data.blood_sugar < 4:
        details.append(
            f"Blood Sugar: {data.blood_sugar} mmol/L — LOW (normal: 4–8). "
            f"Hypoglycemia risk, immediate glucose intake may be needed."
        )
    else:
        details.append(
            f"Blood Sugar: {data.blood_sugar} mmol/L — NORMAL (normal: 4–8). "
            f"Within healthy range."
        )

    # --- Body Temperature ---
    if data.body_temp >= 103:
        details.append(
            f"Body Temperature: {data.body_temp}°F — VERY HIGH FEVER (normal: 97–99°F). "
            f"High risk of infection or sepsis, emergency evaluation needed."
        )
    elif data.body_temp >= 100.4:
        details.append(
            f"Body Temperature: {data.body_temp}°F — FEVER (normal: 97–99°F). "
            f"Signs of infection, antipyretics and medical review advised."
        )
    elif data.body_temp >= 99.1:
        details.append(
            f"Body Temperature: {data.body_temp}°F — LOW-GRADE FEVER (normal: 97–99°F). "
            f"Mild elevation, monitor for worsening symptoms."
        )
    elif data.body_temp < 97:
        details.append(
            f"Body Temperature: {data.body_temp}°F — LOW (normal: 97–99°F). "
            f"Hypothermia risk, keep warm and seek evaluation."
        )
    else:
        details.append(
            f"Body Temperature: {data.body_temp}°F — NORMAL (normal: 97–99°F). "
            f"Within healthy range."
        )

    # --- Heart Rate ---
    if data.heart_rate > 130:
        details.append(
            f"Heart Rate: {data.heart_rate} bpm — SEVERELY HIGH (normal: 60–100). "
            f"Severe tachycardia, cardiac evaluation urgently needed."
        )
    elif data.heart_rate > 100:
        details.append(
            f"Heart Rate: {data.heart_rate} bpm — HIGH (normal: 60–100). "
            f"Tachycardia, monitor for stress, dehydration, or fever cause."
        )
    elif data.heart_rate < 50:
        details.append(
            f"Heart Rate: {data.heart_rate} bpm — LOW (normal: 60–100). "
            f"Bradycardia, may need further cardiac assessment."
        )
    elif data.heart_rate < 60:
        details.append(
            f"Heart Rate: {data.heart_rate} bpm — SLIGHTLY LOW (normal: 60–100). "
            f"Mild bradycardia, monitor for symptoms like dizziness."
        )
    else:
        details.append(
            f"Heart Rate: {data.heart_rate} bpm — NORMAL (normal: 60–100). "
            f"Within healthy range."
        )

    # --- Age context ---
    if data.age < 18:
        details.append(f"Patient age: {data.age} — teenage pregnancy, higher risk category.")
    elif data.age > 35:
        details.append(f"Patient age: {data.age} — advanced maternal age, elevated risk factor.")
    else:
        details.append(f"Patient age: {data.age} — within typical childbearing age range.")

    return details

# ---------------------------
# Identify which parameters are abnormal (for focused prompt)
# ---------------------------
def get_abnormal_flags(data: PatientData):
    """Returns a list of abnormal parameter names to focus Gemini's explanation."""
    flags = []
    if data.systolic_bp >= 130:
        flags.append("high systolic blood pressure")
    if data.diastolic_bp >= 80:
        flags.append("elevated diastolic blood pressure")
    if data.blood_sugar > 8 or data.blood_sugar < 4:
        flags.append("abnormal blood sugar")
    if data.body_temp >= 99.1 or data.body_temp < 97:
        flags.append("abnormal body temperature")
    if data.heart_rate > 100 or data.heart_rate < 60:
        flags.append("abnormal heart rate")
    if data.age < 18 or data.age > 35:
        flags.append("age-related risk")
    return flags if flags else ["all vitals within normal range"]

# ---------------------------
# Gemini dynamic explanation
# ---------------------------
def generate_gemini_explanation(risk_level: str, details: list, abnormal_flags: list, data: PatientData, user_language: str):
    """
    Builds a highly specific, data-driven prompt using exact patient values
    so Gemini generates a unique explanation for each patient.
    """

    detail_block = "\n".join(f"  • {d}" for d in details)
    abnormal_block = ", ".join(abnormal_flags)

    # Risk-level-specific instruction
    risk_instructions = {
        "High Risk": (
            "This is a HIGH RISK patient — use urgent, serious tone. "
            "Emphasize the dangerous parameters clearly. "
            "Strongly recommend immediate hospital visit and close monitoring."
        ),
        "Medium Risk": (
            "This is a MEDIUM RISK patient — use a concerned but calm tone. "
            "Highlight which parameters need attention. "
            "Advise regular check-ups and specific lifestyle precautions."
        ),
        "Low Risk": (
            "This is a LOW RISK patient — use a reassuring, positive tone. "
            "Acknowledge any mildly elevated values if present. "
            "Encourage maintaining healthy habits and routine monitoring."
        ),
    }
    tone_instruction = risk_instructions.get(risk_level, "Provide a clear health explanation.")

    prompt = f"""
You are a maternal health assistant generating a personalized health explanation.

Patient vitals with severity analysis:
{detail_block}

Abnormal parameters: {abnormal_block}
Predicted risk level: {risk_level}

Instruction: {tone_instruction}

Rules:
- Write EXACTLY 4 sentences. No more, no less.
- Sentence 1: State the risk level and directly reference the SPECIFIC abnormal vitals with their values.
- Sentence 2: Explain WHY these specific values contribute to the {risk_level} classification.
- Sentence 3: Give SPECIFIC actionable steps tailored to the abnormal parameters listed above.
- Sentence 4: Give a monitoring/follow-up recommendation based on the severity.
- Do NOT use generic phrases like "some parameters appear risky."
- Do NOT repeat the same advice regardless of values — make it specific to THIS patient.
- Reference exact numbers (e.g., "your blood pressure of 160/95 mmHg").
"""

    try:
        response = gemini_model.generate_content(prompt)
        explanation = response.text.strip()
    except Exception as e:
        # Fallback: rule-based message if Gemini fails
        explanation = build_fallback_explanation(risk_level, abnormal_flags, data)

    # Translate if necessary
    if user_language != "English":
        lang_codes = {
            "Hindi":     "hi",
            "Marathi":   "mr",
            "Telugu":    "te",
            "Tamil":     "ta",
            "Gujarati":  "gu",
            "Odia":      "or",
            "Bhojpuri":  "bho",
            "Bengali":   "bn",
            "Kannada":   "kn",
            "Malayalam": "ml",
            "Punjabi":   "pa",
            "Urdu":      "ur",
            "Assamese":  "as",
        }
        target_lang = lang_codes.get(user_language, "en")
        try:
            explanation = GoogleTranslator(source="auto", target=target_lang).translate(explanation)
        except Exception:
            pass  # Return English if translation fails

    return explanation

# ---------------------------
# Fallback rule-based explanation (if Gemini API fails)
# ---------------------------
def build_fallback_explanation(risk_level: str, abnormal_flags: list, data: PatientData):
    """Generates a basic but parameter-specific message without Gemini."""
    abnormal_str = ", ".join(abnormal_flags)

    if risk_level == "High Risk":
        return (
            f"Your health assessment shows HIGH RISK due to {abnormal_str}. "
            f"Your blood pressure is {data.systolic_bp}/{data.diastolic_bp} mmHg, "
            f"blood sugar is {data.blood_sugar} mmol/L, and heart rate is {data.heart_rate} bpm — "
            f"some of these values are outside the safe range. "
            f"Please visit a hospital or emergency care facility immediately. "
            f"Do not delay seeking medical attention for these readings."
        )
    elif risk_level == "Medium Risk":
        return (
            f"Your health assessment shows MEDIUM RISK, with concerns in {abnormal_str}. "
            f"Key readings: BP {data.systolic_bp}/{data.diastolic_bp} mmHg, "
            f"blood sugar {data.blood_sugar} mmol/L, heart rate {data.heart_rate} bpm. "
            f"Schedule a doctor's appointment within the next 1–2 days. "
            f"Monitor your vitals daily and follow any prescribed medication plan."
        )
    else:
        return (
            f"Your health assessment shows LOW RISK. "
            f"Your readings — BP {data.systolic_bp}/{data.diastolic_bp} mmHg, "
            f"blood sugar {data.blood_sugar} mmol/L, heart rate {data.heart_rate} bpm — "
            f"are largely within normal ranges. "
            f"Continue maintaining a healthy diet, hydration, and regular prenatal check-ups. "
            f"Monitor for any changes and attend your scheduled antenatal visits."
        )

# ---------------------------
# Root route for testing
# ---------------------------
@app.get("/")
def home():
    return {"message": "API working"}

# ---------------------------
# Predict endpoint
# ---------------------------
@app.post("/predict")
def predict(data: PatientData):

    # Feature engineering
    bp_diff = data.systolic_bp - data.diastolic_bp
    mean_bp = (data.systolic_bp + 2 * data.diastolic_bp) / 3
    high_bs = 1 if data.blood_sugar > 8 else 0
    fever = 1 if data.body_temp > 98.6 else 0

    features = pd.DataFrame({
        "Age": [data.age],
        "SystolicBP": [data.systolic_bp],
        "DiastolicBP": [data.diastolic_bp],
        "BS": [data.blood_sugar],
        "BodyTemp": [data.body_temp],
        "HeartRate": [data.heart_rate],
        "BP_Diff": [bp_diff],
        "Mean_BP": [mean_bp],
        "HighBS": [high_bs],
        "Fever": [fever]
    })

    # Model prediction
    prediction_num = int(model.predict(features)[0])
    risk_map = {1: "Low Risk", 2: "Medium Risk", 0: "High Risk"}
    risk_level = risk_map.get(prediction_num, "Unknown")

    # Build detailed parameter analysis
    severity_details = parameter_severity_text(data)

    # Identify abnormal parameters
    abnormal_flags = get_abnormal_flags(data)

    # Generate dynamic Gemini explanation
    explanation = generate_gemini_explanation(
        risk_level, severity_details, abnormal_flags, data, data.language
    )

    return {
        "prediction": risk_level,
        "explanation": explanation
    }