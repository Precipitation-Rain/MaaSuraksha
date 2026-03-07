from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = pickle.load(open("xgb_model.pkl", "rb"))

# Define input structure
class PatientData(BaseModel):
    age: int
    systolic_bp: float
    diastolic_bp: float
    blood_sugar: float
    body_temp: float
    heart_rate: int


@app.get("/")
def home():
    return {"message": "API working"}


@app.post("/predict")
def predict(data: PatientData):

    age = data.age
    sys_bp = data.systolic_bp
    dia_bp = data.diastolic_bp
    blood_sugar = data.blood_sugar
    body_temp = data.body_temp
    heart_rate = data.heart_rate

    # Feature engineering
    bp_diff = sys_bp - dia_bp
    mean_bp = (sys_bp + 2 * dia_bp) / 3
    high_bs = 1 if blood_sugar > 8 else 0
    fever = 1 if body_temp > 98.6 else 0

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

    prediction = int(model.predict(features)[0])

    risk_map = {
        1: "Low Risk",
        2: "Medium Risk",
        0: "High Risk"
    }

    return {"prediction": risk_map[prediction]}