from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
import os

app = FastAPI(title="Ledger API", description="Credit risk / loan default prediction API")

# CORS: allow the frontend (GitHub Pages, local file, etc.) to call this API.
# Tighten allow_origins to your real frontend domain once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "credit_risk_model.joblib")
model = joblib.load(MODEL_PATH)


class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=90, description="Applicant age")
    person_income: float = Field(..., gt=0, description="Annual income")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(..., ge=0, le=60, description="Years employed")
    loan_intent: Literal["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(..., ge=0, le=40, description="Interest rate (%)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan amount as a fraction of income")
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60, description="Credit history length (years)")


class PredictionResponse(BaseModel):
    default_probability: float
    risk_label: str
    prediction: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Ledger API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        row = pd.DataFrame([application.model_dump()])
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)

        if proba < 0.2:
            label = "Low Risk"
        elif proba < 0.5:
            label = "Moderate Risk"
        elif proba < 0.75:
            label = "High Risk"
        else:
            label = "Very High Risk"

        return PredictionResponse(
            default_probability=round(proba, 4),
            risk_label=label,
            prediction=pred,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
