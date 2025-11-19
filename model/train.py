# import os, json, shutil
# from datetime import datetime

# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import classification_report, roc_auc_score
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler

# NUM = ["age", "avg_glucose_level", "bmi"]
# CAT = ["gender","hypertension","heart_disease","ever_married",
#        "work_type","Residence_type","smoking_status"]

# def load_data(path="data/healthcare-dataset-stroke-data.csv"):
#     df = pd.read_csv(path)
#     for col in ["id"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])
#     return df

# def build_pipeline(use_smote=False):
#     # preprocess numerik & kategorikal
#     num_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])
#     cat_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("onehot", OneHotEncoder(handle_unknown="ignore"))
#     ])

#     pre = ColumnTransformer([
#         ("num", num_pipe, NUM),
#         ("cat", cat_pipe, CAT)
#     ])

#     clf = RandomForestClassifier(
#         n_estimators=300,
#         random_state=42,
#         class_weight=None 
#     )

#     if use_smote:
#         from imblearn.pipeline import Pipeline as ImbPipeline
#         from imblearn.over_sampling import SMOTE
#         pipe = ImbPipeline([
#             ("preprocessor", pre),
#             ("smote", SMOTE(random_state=42)),
#             ("classifier", clf)
#         ])
#     else:
#         pipe = Pipeline([
#             ("preprocessor", pre),
#             ("classifier", clf)
#         ])

#     return pipe

# def train(data_path="data/healthcare-dataset-stroke-data.csv", out_dir="model/artifacts"):
#     df = load_data(data_path)
#     X = df.drop(columns=["stroke"])
#     y = df["stroke"]

#     # stratified split penting untuk imbalance
#     X_tr, X_te, y_tr, y_te = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     use_smote = True
#     pipe = build_pipeline(use_smote=use_smote)
#     pipe.fit(X_tr, y_tr)

#     # evaluasi cepat
#     y_pred = pipe.predict(X_te)
#     y_proba = pipe.predict_proba(X_te)[:, 1]
#     print(classification_report(y_te, y_pred, digits=4))
#     print("ROC-AUC:", roc_auc_score(y_te, y_proba))

#     # simpan artefak versi
#     version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
#     vers_dir = os.path.join(out_dir, version)
#     os.makedirs(vers_dir, exist_ok=True)
#     joblib.dump(pipe, os.path.join(vers_dir, "stroke_pipeline.joblib"))

#     with open(os.path.join(vers_dir, "manifest.json"), "w") as f:
#         json.dump({"version": version, "created_at": datetime.utcnow().isoformat()+"Z"}, f)

#     # update "latest"
#     latest = os.path.join(out_dir, "latest")
#     if os.path.islink(latest) or os.path.exists(latest):
#         shutil.rmtree(latest)
#     shutil.copytree(vers_dir, latest)

# if __name__ == "__main__":
#     train()


# model/train_mlflow.py
import mlflow
import mlflow.sklearn
import joblib
import json
from datetime import datetime
from model.train import build_pipeline, load_data   # reuse semua fungsi lama kamu

mlflow.set_experiment("Stroke Prediction - Tugas Kuliah")

with mlflow.start_run():
    # Baca parameter dari params.yaml (otomatis ke-log)
    mlflow.log_params({
        "use_smote": True,
        "n_estimators": 500,
        "max_depth": 8,
        "model": "RandomForest"
    })

    # Training pake kode lama kamu (cuma dipanggil ulang)
    df = load_data("data/raw/healthcare-dataset-stroke-data.csv")
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_pipeline(use_smote=True)
    pipe.fit(X_tr, y_tr)

    # Evaluasi
    from sklearn.metrics import roc_auc_score
    y_proba = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba)

    mlflow.log_metric("roc_auc", round(auc, 4))

    # Simpan model ke folder latest (biar API kamu tetap jalan seperti biasa)
    version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    latest_dir = "model/artifacts/latest"
    import os, shutil
    os.makedirs(latest_dir, exist_ok=True)
    joblib.dump(pipe, f"{latest_dir}/stroke_pipeline.joblib")

    with open(f"{latest_dir}/manifest.json", "w") as f:
        json.dump({"version": version, "auc": round(auc, 4), "run_id": mlflow.active_run().info.run_id}, f)

    # Log model ke MLflow biar bisa dibandingin di UI
    mlflow.sklearn.log_model(pipe, "model")

    print(f"Training selesai! AUC = {auc:.4f} | Run ID: {mlflow.active_run().info.run_id}")
