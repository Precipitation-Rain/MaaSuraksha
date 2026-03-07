# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pickle
# import pandas as pd

# app = FastAPI()

# # Allow frontend to connect
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model
# model = pickle.load(open("xgb_model.pkl", "rb"))

# # Define input structure
# class PatientData(BaseModel):
#     age: int
#     systolic_bp: float
#     diastolic_bp: float
#     blood_sugar: float
#     body_temp: float
#     heart_rate: int


# @app.get("/")
# def home():
#     return {"message": "API working"}


# @app.post("/predict")
# def predict(data: PatientData):

#     age = data.age
#     sys_bp = data.systolic_bp
#     dia_bp = data.diastolic_bp
#     blood_sugar = data.blood_sugar
#     body_temp = data.body_temp
#     heart_rate = data.heart_rate

#     # Feature engineering
#     bp_diff = sys_bp - dia_bp
#     mean_bp = (sys_bp + 2 * dia_bp) / 3
#     high_bs = 1 if blood_sugar > 8 else 0
#     fever = 1 if body_temp > 98.6 else 0

#     features = pd.DataFrame({
#         "Age":[age],
#         "SystolicBP":[sys_bp],
#         "DiastolicBP":[dia_bp],
#         "BS":[blood_sugar],
#         "BodyTemp":[body_temp],
#         "HeartRate":[heart_rate],
#         "BP_Diff":[bp_diff],
#         "Mean_BP":[mean_bp],
#         "HighBS":[high_bs],
#         "Fever":[fever]
#     })

#     prediction = int(model.predict(features)[0])

#     risk_map = {
#         1: "Low Risk",
#         2: "Medium Risk",
#         0: "High Risk"
#     }

#     return {"prediction": risk_map[prediction]}


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
# Load XGBoost model
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
# Root route for testing
# ---------------------------
@app.get("/")
def home():
    return {"message": "API working"}

# ---------------------------
# Gemini explanation helper
# ---------------------------
def generate_gemini_explanation(risk_level: str, top_features_df, user_language: str):
    prompt = f"""
    A maternal health screening model predicted: {risk_level}.
    Important contributing factors were:
    {top_features_df.to_string(index=False)}

    Explain this result in very simple language for a rural health worker.
    Requirements:
    - Maximum 2 sentences explanation
    - 1 clear recommended action
    """

    try:
        response = gemini_model.generate_content(prompt)
        explanation = response.text
    except Exception:
        explanation = "Patient vitals suggest this risk level. Please monitor the patient and consult a doctor if values worsen."

    # Translate if needed
    if user_language != "English":
        lang_codes = {"Hindi": "hi", "Marathi": "mr", "Telugu": "te", "Tamil": "ta"}
        target_lang = lang_codes.get(user_language, "en")
        explanation = GoogleTranslator(source="auto", target=target_lang).translate(explanation)

    return explanation

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

    # Prepare top features for Gemini
    top_features = features.T.reset_index()
    top_features.columns = ["Feature", "Value"]

    # Generate Gemini explanation
    explanation = generate_gemini_explanation(risk_level, top_features, data.language)

    return {
        "prediction": risk_level,
        "explanation": explanation
    }