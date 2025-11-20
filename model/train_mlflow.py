import mlflow
import mlflow.sklearn
import joblib
import json
import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


os.environ["MLFLOW_TRACKING_USERNAME"] = "mzhammar"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "bebbb6c0c528440af049f0a4d6c07947e5b2908a"
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/mzhammar/Tugas-Action-SD-MLOps.mlflow"

mlflow.set_tracking_uri("https://dagshub.com/mzhammar/Tugas-Action-SD-MLOps.mlflow")
mlflow.set_experiment("Stroke Prediction")

# Konstanta
NUM = ["age", "avg_glucose_level", "bmi"]
CAT = ["gender","hypertension","heart_disease","ever_married",
       "work_type","Residence_type","smoking_status"]

# Load data
df = pd.read_csv("data/raw/healthcare-dataset-stroke-data.csv")
if "id" in df.columns:
    df = df.drop(columns=["id"])

X = df.drop("stroke", axis=1)
y = df["stroke"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Pipeline
num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])

clf = RandomForestClassifier(
    n_estimators=500, max_depth=8, random_state=42, class_weight="balanced"
)

pipe = ImbPipeline([
    ("preprocessor", pre),
    ("smote", SMOTE(random_state=42)),
    ("classifier", clf)
])

with mlflow.start_run():
    pipe.fit(X_tr, y_tr)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)

    mlflow.log_metric("roc_auc", round(auc, 4))
    mlflow.log_params({"n_estimators": 500, "max_depth": 8, "use_smote": True})

    # Simpan ke folder latest (API kamu tetap jalan)
    os.makedirs("model/artifacts/latest", exist_ok=True)
    joblib.dump(pipe, "model/artifacts/latest/stroke_pipeline.joblib")
    
    manifest = {
        "version": datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
        "auc": round(auc, 4),
        "model": "RandomForest+SMOTE"
    }
    with open("model/artifacts/latest/manifest.json", "w") as f:
        json.dump(manifest, f)

    mlflow.log_artifact("model/artifacts/latest/stroke_pipeline.joblib", artifact_path="model")
    mlflow.log_artifact("model/artifacts/latest/manifest.json", artifact_path="model")
 
    print(f"Training selesai! AUC = {auc:.4f}")
    print(f"Run: https://dagshub.com/mzhammar/stroke-prediction.mlflow/#/experiments/0/runs/{mlflow.active_run().info.run_id}")