# Stroke Prediction – Full MLOps Pipeline

**End-to-end MLOps project**: Data Versioning (DVC + DagsHub) → Experiment Tracking (MLflow) → CI/CD (GitHub Actions) → Inference API (FastAPI + Docker)

[![CI/CD](https://github.com/theammars/Tugas-Action-SD-MLOps/actions/workflows/train-and-register.yml/badge.svg)](https://github.com/theammars/Tugas-Action-SD-MLOps/actions/workflows/train-and-register.yml)
[![DagsHub](https://img.shields.io/badge/DagsHub-Repo-blue?logo=dvc)](https://dagshub.com/mzhammar/Tugas-Action-SD-MLOps)
[![MLflow](https://img.shields.io/badge/MLflow-Experiments-orange)](https://dagshub.com/mzhammar/Tugas-Action-SD-MLOps.mlflow)

## Features
- Dataset versioning dengan **DVC** + remote storage DagsHub (S3)
- Experiment tracking & model registry dengan **MLflow** (hosted di DagsHub)
- **CI/CD otomatis**: setiap push ke `main` → training ulang → model ter-update
- Inference service dengan **FastAPI + Docker** (Swagger UI siap pakai)
- Clean & production-ready project structure
- Unit test siap (pytest)

## Project Structure
stroke_prediction/
├── app/                  # FastAPI inference
│   ├── main.py
│   ├── schema.py
│   └── inferences.py
├── model/
│   ├── train_mlflow.py   # Training script + MLflow logging
│   └── artifacts/latest/ # Model terbaru otomatis
├── data/raw/
│   └── healthcare-dataset-stroke-data.csv.dvc
├── test/                 # Unit tests
├── .github/workflows/    # CI/CD GitHub Actions
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── params.yaml
└── requirements.txt

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/theammars/Tugas-Action-SD-MLOps.git
cd Tugas-Action-SD-MLOps
pip install -r requirements.txt
```

### 2. Pull Dataset From DVC Remote (DagsHub)
```bash
dvc pull
```

### 3. Re-Train Model 
```bash
dvc repro
```

### 4. Run API Inference with Docker
```bash
docker build -t stroke-api .
docker run -d -p 8080:8080 --name stroke-api stroke-api
```

Open API on: http://localhost:8080/docs

### Tech Stack  
- Python 3.11+
- DVC + DagsHub (remote storage & MLflow)
- MLflow (experiment tracking)
- scikit-learn + imbalanced-learn (SMOTE)
- FastAPI + Uvicorn
- Docker
- GitHub Actions (CI/CD)

### Dataset
Healthcare Dataset Stroke
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### Challenges & Lessons Learned
- Integration of DVC S3 with DagsHub → must synchronize repo name & endpoint
- New S3 bucket created after first dvc push → initial 404 error
- CI/CD failed due to old config (auth: basic) → must be unset
- Insight: 80% of MLOps is configuration & tool synchronization

“Bukan modelnya yang sulit, tapi bikin sistemnya hidup otomatis tanpa drama.” – Ammar, 2025