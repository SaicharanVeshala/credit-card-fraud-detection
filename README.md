# Fraud Detection System

An end-to-end machine learning system for detecting fraudulent financial transactions in real time using LightGBM, FastAPI, Streamlit, SHAP explainability, and Docker.

The project is designed to simulate a production-style machine learning workflow where a trained model is served through APIs, integrated with an interactive dashboard, and deployed using containerized services.

---

# Problem Statement

Financial fraud is one of the biggest challenges in digital transactions and online payment systems. Fraudulent transactions lead to significant financial losses and reduce trust in online banking systems.

One major challenge in fraud detection is the highly imbalanced nature of financial datasets, where fraudulent transactions represent only a very small percentage of overall transactions.

This project aims to build a scalable machine learning system capable of:

* identifying fraudulent transactions in real time
* minimizing false negatives
* serving predictions through APIs
* supporting deployment-ready architecture
* improving interpretability using explainable AI

---

# System Architecture

The system follows a modular machine learning architecture designed for real-time fraud prediction.

```text id="jlwm137"
User
   ↓
Streamlit Dashboard
   ↓
FastAPI Backend
   ↓
Feature Engineering Pipeline
   ↓
LightGBM Fraud Detection Model
   ↓
Prediction + SHAP Explainability
```

The application is fully containerized using Docker and orchestrated using Docker Compose.

---

# Features

* Real-time fraud prediction
* FastAPI REST API backend
* Interactive Streamlit dashboard
* SHAP-based explainability
* Dockerized deployment
* Docker Compose orchestration
* Swagger API documentation
* Modular project structure
* Production-style ML deployment workflow

---

# Tech Stack

| Category             | Technologies                         |
| -------------------- | ------------------------------------ |
| Programming Language | Python                               |
| Machine Learning     | LightGBM, Scikit-learn               |
| Backend              | FastAPI                              |
| Frontend             | Streamlit                            |
| Explainability       | SHAP                                 |
| Data Processing      | Pandas, NumPy                        |
| Visualization        | Matplotlib, Seaborn                  |
| API Validation       | Pydantic                             |
| Deployment           | Docker, Docker Compose, Hugging Face |
| Version Control      | Git, GitHub                          |

---

# Project Structure

```text id="jlwm138"
fraud-detection-system/
│
├── data/
├── features/
├── models/
├── notebooks/
├── outputs/
├── screenshots/
├── src/
│   ├── api/
│   ├── dashboard/
│   └── models/
│
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

The project follows a modular architecture to improve scalability, maintainability, and deployment readiness.

---

# Model Development

The fraud detection model was trained on an imbalanced financial transaction dataset.

The workflow included:

* data preprocessing
* feature engineering
* feature scaling
* model training
* evaluation on imbalanced classification metrics

LightGBM was selected because of:

* strong performance on tabular datasets
* fast inference speed
* efficient handling of imbalanced data
* scalability for real-time prediction systems

---

# Evaluation Metrics

The fraud detection system was evaluated using metrics specifically designed for imbalanced classification problems.

## Metrics Used

* ROC-AUC Score
* F1-Score
* Precision-Recall Analysis
* Average Precision Score
* Confusion Matrix
* Classification Report

## Why These Metrics Matter

Fraud detection datasets are highly imbalanced, meaning fraudulent transactions represent only a very small percentage of overall transactions.

For this reason:

* ROC-AUC measures the model’s ability to distinguish between fraudulent and legitimate transactions
* F1-Score balances precision and recall
* Precision-Recall analysis is more informative than accuracy for imbalanced datasets
* Confusion Matrix helps analyze false positives and false negatives

The project prioritizes recall because missing fraudulent transactions can have severe financial impact.

---

# Explainability

The project integrates SHAP explainability to improve transparency and interpretability of predictions.

SHAP values help identify:

* which features contributed most to a prediction
* why a transaction was classified as fraudulent
* feature-level impact on model output

This improves trust and interpretability of the machine learning system.

---

# Installation

## Clone the Repository

```bash id="jlwm139"
git clone https://github.com/SaicharanVeshala/fraud-detection-system.git
```

## Navigate to Project Directory

```bash id="jlwm140"
cd fraud-detection-system
```

## Create Virtual Environment

```bash id="jlwm141"
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash id="jlwm142"
venv\Scripts\activate
```

### Linux / macOS

```bash id="jlwm143"
source venv/bin/activate
```

## Install Dependencies

```bash id="’wini144"
pip install -r requirements.txt
```

---

# Running the Application Locally

## Run FastAPI Backend

```bash id="’wini145"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 7860
```

FastAPI server will start at:

```text id="’wini146"
http://127.0.0.1:7860
```

## FastAPI Interactive Documentation

Swagger UI:

```text id="’wini147"
http://127.0.0.1:7860/docs
```

## Run Streamlit Frontend

```bash id="’wini148"
streamlit run src/dashboard/app.py
```

---

# Docker Deployment

The project is fully containerized using Docker and Docker Compose.

## Start the Entire Application

```bash id="’wini149"
docker compose up -d
```

This starts:

* FastAPI backend container
* Streamlit dashboard container

---

# Application URLs

## Streamlit Dashboard

```text id="’wini150"
http://localhost:8501
```

## FastAPI Swagger Documentation

```text id="’wini151"
http://localhost:7860/docs
```

---

# Stop Containers

```bash id="’wini152"
docker compose down
```

---

# API Endpoints

| Method | Endpoint | Description                    |
| ------ | -------- | ------------------------------ |
| POST   | /predict | Predict fraudulent transaction |
| GET    | /health  | API health check               |
| GET    | /docs    | Swagger API documentation      |

---

# Sample API Request

```json id="’wini153"
{
  "amount": 2500,
  "oldbalanceOrg": 5000,
  "newbalanceOrig": 2500,
  "oldbalanceDest": 1000,
  "newbalanceDest": 3500
}
```

# Sample API Response

```json id="’wini154"
{
  "prediction": "Fraudulent Transaction",
  "fraud_probability": 0.92
}
```

---

# Workflow

1. User submits transaction details through the Streamlit dashboard
2. FastAPI backend receives the request
3. Input data is validated using Pydantic schemas
4. Feature engineering and preprocessing are applied
5. The trained LightGBM model performs inference
6. Fraud probability and prediction result are returned
7. SHAP explainability is generated

---

# Deployment Architecture

The project uses Docker Compose to orchestrate multiple services together.

## FastAPI Container

* serves machine learning inference APIs
* loads trained model and preprocessing pipeline
* handles prediction requests

## Streamlit Container

* provides interactive user interface
* communicates with FastAPI backend
* visualizes predictions and explanations

---

# Screenshots

Project screenshots and deployment previews are available in the `screenshots/` directory.

---

# Current Deployment

* Dockerized local deployment
* Docker Compose orchestration
* Hugging Face deployment
* GitHub version control

---

# Future Improvements

Possible future enhancements include:

* AWS cloud deployment
* CI/CD pipeline integration
* Kafka-based real-time transaction streaming
* Drift monitoring
* Redis-based caching
* Authentication and security layers
* Automated retraining pipelines
* Advanced fraud analytics dashboard

---

# Conclusion

This project demonstrates the integration of machine learning, backend engineering, explainable AI, and containerized deployment for real-time fraud detection.

The system combines:

* LightGBM-based fraud prediction
* FastAPI backend services
* Streamlit frontend interaction
* SHAP explainability
* Docker-based deployment architecture

The project emphasizes:

* real-time inference
* modular architecture
* scalable backend integration
* deployment-oriented design
* production-style machine learning workflows
