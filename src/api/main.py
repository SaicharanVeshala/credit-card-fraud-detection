import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

import shap
import joblib
import numpy as np
import pandas as pd
import fakeredis
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from xgboost import XGBClassifier
import uvicorn


# =========================
# PATHS
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "processed",
    "final_model.json"
)

SCALER_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "processed",
    "scaler.pkl"
)


# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = XGBClassifier()
model.load_model(MODEL_PATH)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

print("All loaded!")


# =========================
# CACHE
# =========================

cache = fakeredis.FakeRedis()
cache.flushall()

CACHE_TTL = 300


# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="Fraud Detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODEL
# =========================

class TransactionRequest(BaseModel):

    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

    Amount: float = Field(default=1.0, ge=0)
    Time: float = Field(default=0.0, ge=0)


# =========================
# PREPROCESS
# =========================

def preprocess(transaction):

    data = transaction.model_dump()

    data["Amount_log"] = np.log1p(data["Amount"])
    data["Hour"] = (data["Time"] // 3600) % 24

    data["Is_night"] = (
        1 if (
            data["Hour"] >= 22
            or data["Hour"] <= 5
        )
        else 0
    )

    df = pd.DataFrame([data])

    df = df.drop(
        ["Time", "Amount"],
        axis=1
    )

    scale_cols = [
        "Amount_log",
        "Hour",
        "Is_night"
    ]

    df[scale_cols] = scaler.transform(
        df[scale_cols]
    )

    return df


# =========================
# SHAP EXPLANATIONS
# =========================

def get_explanations(df):

    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = pd.DataFrame({
        "feature": df.columns,
        "impact": shap_values[0],
        "value": df.iloc[0].values
    })

    importance["abs"] = (
        importance["impact"].abs()
    )

    importance = importance.sort_values(
        "abs",
        ascending=False
    )

    top = importance.head(5)

    explanations = []

    for _, row in top.iterrows():

        explanations.append({
            "feature": row["feature"],
            "impact": float(row["impact"]),
            "value": float(row["value"]),
            "direction": (
                "increase"
                if row["impact"] > 0
                else "decrease"
            )
        })

    return explanations

# =========================
# ROOT
# =========================

@app.get("/")
def root():

    return {
        "message": "Fraud Detection API Running"
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================
# PREDICT
# =========================

@app.post("/predict")
def predict(transaction: TransactionRequest):

    try:

        start = time.time()

        df = preprocess(transaction)

        prob = float(
            model.predict_proba(df)[0][1]
        )

        result = {
            "fraud_probability": round(prob, 4),

            "verdict": (
                "FRAUD"
                if prob >= 0.5
                else "LEGIT"
            ),

            "confidence": (
                "HIGH"
                if abs(prob - 0.5) > 0.3
                else "LOW"
            ),

            "risk_score": int(prob * 100),

            "top_reasons": get_explanations(df),

            "response_time_ms": round(
                (time.time() - start) * 1000,
                2
            )
        }

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )
