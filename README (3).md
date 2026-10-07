# Ledger — Credit Risk Assessment Tool

Ledger is a full-stack credit risk / loan default prediction app. It takes an applicant's financial profile and returns a default probability, risk label, and prediction — the kind of signal a lender uses to price or decline a loan.

Built as a portfolio project to demonstrate end-to-end data science + software engineering: data cleaning, model training, a REST API, and a production-style frontend — not just a notebook.

## Live Demo
- Frontend: _add your GitHub Pages / deployed URL here_
- API: _add your Render URL here_

## Tech Stack
- **Model:** scikit-learn (Random Forest Classifier) inside a full preprocessing pipeline (imputation, scaling, one-hot encoding)
- **Backend:** FastAPI + Pydantic, serving predictions via a `/predict` endpoint
- **Frontend:** Vanilla HTML/CSS/JS (no framework) — calls the API directly via `fetch`
- **Dataset:** [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (~32,500 rows of applicant, loan, and credit bureau data)

## Model Performance
Evaluated on a held-out 20% test split:

| Metric | Score |
|---|---|
| Accuracy | 91% |
| ROC-AUC | 0.93 |
| Precision (Default class) | 0.83 |
| Recall (Default class) | 0.76 |

## Project Structure
```
ledger/
├── backend/
│   ├── main.py                  # FastAPI app and /predict endpoint
│   ├── train_model.py           # Data cleaning + model training script
│   ├── requirements.txt
│   └── credit_risk_model.joblib # Trained pipeline (preprocessing + model)
├── frontend/
│   └── index.html               # Single-page UI, calls the API
└── README.md
```

## Running Locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API will be live at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

**Frontend**
Just open `frontend/index.html` in a browser. It's a static file that calls the API at `http://localhost:8000` by default (see the `API_URL` constant near the top of the `<script>` tag — update this to your deployed API URL in production).

## Retraining the Model
```bash
cd backend
python train_model.py
```
This regenerates `credit_risk_model.joblib` from `credit_risk_dataset.csv`.

## API Reference

**POST `/predict`**

Request body:
```json
{
  "person_age": 25,
  "person_income": 45000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3,
  "loan_intent": "DEBTCONSOLIDATION",
  "loan_grade": "D",
  "loan_amnt": 20000,
  "loan_int_rate": 15.5,
  "loan_percent_income": 0.44,
  "cb_person_default_on_file": "Y",
  "cb_person_cred_hist_length": 4
}
```

Response:
```json
{
  "default_probability": 0.9894,
  "risk_label": "Very High Risk",
  "prediction": 1
}
```

## Disclaimer
This tool produces model estimates for demonstration purposes only. It is not financial advice and should not be used for actual lending decisions.

## Author
Built by [Garv Rajpoot](https://garvrajpoot.github.io)
