# Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

An end-to-end machine learning project for real-time credit card fraud detection using XGBoost, FastAPI, Streamlit, Docker, and SHAP explainability.

The project focuses on building a practical machine learning workflow where a trained fraud detection model is exposed through APIs, connected to an interactive dashboard, and packaged using Docker for easier deployment.

---

# Features

* Real-time fraud prediction API
* Interactive Streamlit dashboard
* XGBoost-based fraud classifier
* SHAP explainability integration
* Dockerized deployment
* FastAPI backend services
* Fraud risk scoring
* Swagger API documentation
* End-to-end ML workflow
* Modular project structure

---

# System Architecture

```text
User
   ↓
Streamlit Dashboard
   ↓
FastAPI Backend
   ↓
Feature Engineering Pipeline
   ↓
XGBoost Fraud Detection Model
   ↓
Prediction + SHAP Explainability
```

The application is fully containerized using Docker and orchestrated using Docker Compose.

---

# Tech Stack

| Category             | Technologies           |
| -------------------- | ---------------------- |
| Programming Language | Python                 |
| Machine Learning     | XGBoost, Scikit-learn  |
| Backend              | FastAPI                |
| Frontend             | Streamlit              |
| Explainability       | SHAP                   |
| Data Processing      | Pandas, NumPy          |
| Visualization        | Matplotlib, Seaborn    |
| API Validation       | Pydantic               |
| Deployment           | Docker, Docker Compose |
| Version Control      | Git, GitHub            |

---

# Project Structure

```text
credit-card-fraud-detection/
│
├── assets/
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_models.ipynb
│   └── 04_explainability.ipynb
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   └── dashboard/
│       └── app.py
│
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Model Development

The fraud detection pipeline was trained on an imbalanced credit card transaction dataset.

The workflow includes:

* data preprocessing
* feature engineering
* feature scaling
* model training
* imbalanced classification evaluation
* explainability analysis

XGBoost was selected because of:

* strong performance on tabular datasets
* efficient handling of imbalanced data
* fast inference speed
* suitability for real-time prediction systems

---

# Model Performance

| Model               | AUC-ROC | F1 Score |
| ------------------- | ------- | -------- |
| Logistic Regression | 0.97    | 0.86     |
| XGBoost             | 0.99    | 0.92     |
| LightGBM            | 0.98    | 0.90     |

The project prioritizes recall and fraud detection sensitivity to minimize missed fraudulent transactions.

---

# Explainability with SHAP

The system integrates SHAP explainability to improve prediction transparency and interpretability.

SHAP values help identify:

* which features contributed most to predictions
* why transactions are classified as fraudulent
* feature-level impact on model outputs

This improves trust and interpretability for fraud detection systems.

---

# Screenshots

## Dashboard Preview

![Dashboard](assets/dashboard.png)

## SHAP Explainability

![SHAP](assets/shap_analysis.png)

## Swagger API Documentation

![Swagger](assets/swagger_api.png)

---

# Installation

## Clone Repository

```bash
git clone https://github.com/SaicharanVeshala/credit-card-fraud-detection.git
```

## Navigate to Project Directory

```bash
cd credit-card-fraud-detection
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Locally

## Start FastAPI Backend

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 7860
```

FastAPI server:

```text
http://127.0.0.1:7860
```

Swagger documentation:

```text
http://127.0.0.1:7860/docs
```

## Start Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

Dashboard URL:

```text
http://127.0.0.1:8501
```

---

# Docker Deployment

## Start Application

```bash
docker compose up --build
```

## Stop Containers

```bash
docker compose down
```

---

# API Endpoints

| Method | Endpoint   | Description               |
| ------ | ---------- | ------------------------- |
| POST   | `/predict` | Fraud prediction          |
| GET    | `/health`  | API health check          |
| GET    | `/docs`    | Swagger API documentation |

---

# Sample API Request

```json
{
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536346,
  "Amount": 149.62,
  "Time": 0
}
```

# Sample API Response

```json
{
  "fraud_probability": 0.92,
  "verdict": "FRAUD",
  "confidence": "HIGH",
  "risk_score": 92
}
```

---

# Workflow

1. User submits transaction details through the Streamlit dashboard
2. FastAPI backend validates and processes the request
3. Feature engineering and preprocessing are applied
4. XGBoost model performs inference
5. Fraud probability and risk score are generated
6. SHAP explanations are returned
7. Results are visualized in the dashboard

---

# Current Status

* Dockerized local deployment
* FastAPI backend integration
* Streamlit dashboard integration
* SHAP explainability support
* GitHub project versioning

---

# Future Improvements

* Public cloud deployment
* CI/CD integration
* Model monitoring
* Drift detection
* Enhanced dashboard analytics
* Automated retraining workflows

---

# Author

**Sai Charan Veshala**

GitHub:
[https://github.com/SaicharanVeshala](https://github.com/SaicharanVeshala)

LinkedIn:
Add your LinkedIn profile link here.

---

# Conclusion

This project combines machine learning, backend development, explainable AI, and containerized deployment to build a real-time fraud detection system.

The system combines:

* XGBoost-based fraud prediction
* FastAPI backend services
* Streamlit dashboard interaction
* SHAP explainability
* Docker-based deployment workflow

The project emphasizes:

* real-time inference
* explainable AI
* modular architecture
* deployment-oriented design
* end-to-end ML application development
