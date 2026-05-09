# Real-Time Fraud Detection System using LightGBM and FastAPI

A production-oriented machine learning system for detecting fraudulent financial transactions in real time using LightGBM, FastAPI, and Streamlit.



## Problem Statement

Financial fraud is one of the biggest challenges in digital transactions and online payment systems. Fraudulent transactions cause significant financial losses and reduce customer trust in digital banking systems.

One major challenge in fraud detection is the highly imbalanced nature of financial datasets, where fraudulent transactions represent only a very small percentage of total transactions.

This project aims to build a scalable machine learning system capable of identifying fraudulent transactions in real time while minimizing false positives and maximizing recall.

---



## System Architecture

The system follows a modular machine learning architecture designed for real-time fraud prediction.

```text
User
   ↓
Streamlit Frontend
   ↓
FastAPI Backend
   ↓
Feature Engineering Pipeline
   ↓
LightGBM Fraud Detection Model
   ↓
Fraud Prediction Response




```

## Project Structure

```text
fraud-detection-system/
│
├── data/                  # Dataset files
├── features/              # Feature engineering scripts
├── models/                # Trained ML models
├── notebooks/             # Jupyter notebooks for experimentation
├── app/                   # FastAPI application
├── tests/                 # Unit tests
├── screenshots/           # Project screenshots
│
├── README.md
├── requirements.txt
├── Dockerfile
└── .gitignore
```

The project is organized using a modular architecture to improve scalability, maintainability, and deployment readiness.

---




### Workflow

1. User submits transaction details through the Streamlit interface.
2. FastAPI backend receives the request.
3. Input data is validated using Pydantic schemas.
4. Feature engineering and preprocessing are applied.
5. The trained LightGBM model performs inference.
6. Fraud probability and prediction result are returned to the user interface.

---

## Overview

This project is designed to identify potentially fraudulent transactions using machine learning techniques on highly imbalanced financial datasets. The system integrates a FastAPI backend for real-time inference and a Streamlit frontend for interactive user experience.

The project focuses not only on model accuracy but also on scalable backend integration, modular architecture, and real-world fraud detection challenges.

---

## Features

- Real-time fraud prediction
- FastAPI backend integration
- LightGBM-based machine learning model
- Streamlit interactive dashboard
- Feature engineering pipeline
- REST API endpoints
- Modular project architecture
- Fraud probability scoring
- Scalable deployment-ready design

---


## Tech Stack

| Category             |    Technologies        |
|----------------------|------------------------|
| Programming Language | Python                 |
| Machine Learning     | LightGBM, Scikit-learn |
| Backend              | FastAPI                |
| Frontend             | Streamlit              |
| Data Processing      | Pandas, NumPy          |
| Visualization        | Matplotlib, Seaborn    |
| API Validation       | Pydantic               |
| Deployment           | Hugging Face Spaces    |
| Version Control      | Git, GitHub            | 

---
