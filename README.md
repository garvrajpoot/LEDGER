# Ledger — Credit Risk Assessment Tool

Ledger is a full-stack credit risk / loan default prediction app. It takes an applicant's financial profile and returns a default probability, risk label, and prediction — the kind of signal a lender uses to price or decline a loan.

Built as a portfolio project to demonstrate end-to-end data science + software engineering: data cleaning, model training, a REST API, and a production-style frontend — not just a notebook.

## Live Demo
- Frontend: _add your GitHub Pages URL here_
- API: _add your Render URL here_ (interactive docs at `/docs`)

> The API runs on a free Render instance that sleeps when idle, so the first request after a quiet period can take 30–60 seconds to wake up.

## Tech Stack
- **Model:** scikit-learn (Random Forest Classifier) inside a full preprocessing pipeline (imputation, scaling, one-hot encoding)
- **Backend:** FastAPI + Pydantic, serving predictions via a `/predict` endpoint
- **Frontend:** Vanilla HTML/CSS/JS (no framework) — calls the API directly via `fetch`
- **Hosting:** Render (API) + GitHub Pages (frontend)
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
LEDGER/
├── backend/
│   ├── main.py                  # FastAPI app and /predict endpoint
│   ├── train_model.py           # Data cleaning + model training script
│   ├── requirements.txt
│   └── credit_risk_model.joblib # Trained pipeline (preprocessing + model)
├── docs/
│   └── index.html               # Single-page UI, served by GitHub Pages
└── README.md
```

## Running Locally

**Backend** (Python 3.11+)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API will be live at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

**Frontend**
Open `docs/index.html` in a browser. It calls the API at `http://localhost:8000` by default (see the `API_URL` constant in the `<script>` tag — it points at the deployed API in production).

## Deployment

- **API (Render):** Web Service with Root Directory `backend`, Build Command `pip install -r requirements.txt`, Start Command `uvicorn main:app --host 0.0.0.0 --port $PORT`, and environment variable `PYTHON_VERSION=3.12.3`.
- **Frontend (GitHub Pages):** Settings → Pages → Deploy from branch `main`, folder `/docs`.

The pinned versions in `requirements.txt` match the scikit-learn version the model was saved with — loading a pickled model under a different scikit-learn version can fail, so keep them in sync if you retrain.

## Retraining the Model
The dataset isn't included in this repo. Download `credit_risk_dataset.csv` from Kaggle, place it in `backend/`, then:
```bash
cd backend
python train_model.py
```
This regenerates `credit_risk_model.joblib`.

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

Other endpoints: `GET /` (status) and `GET /health` (health check).

## Disclaimer
This tool produces model estimates for demonstration purposes only. It is not financial advice and should not be used for actual lending decisions.

## Author
Built by [Garv Rajpoot](https://garvrajpoot.github.io)
