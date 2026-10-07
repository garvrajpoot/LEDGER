"""
Train a credit risk (loan default) classifier on the Kaggle Credit Risk Dataset
and save the fitted pipeline (preprocessing + model) as a single joblib file
so the FastAPI backend can load it directly for inference.
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("credit_risk_dataset.csv")

# Basic cleaning: a few known bad rows in this dataset (e.g. age 144, emp_length 123)
df = df[(df["person_age"] <= 90) & (df["person_emp_length"] <= 60)]

TARGET = "loan_status"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric_features = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
]
categorical_features = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Repaid", "Default"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

joblib.dump(model, "credit_risk_model.joblib")
print("Saved model to credit_risk_model.joblib")
