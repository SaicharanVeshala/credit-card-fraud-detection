import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import joblib
import numpy as np
import pandas as pd
import fakeredis
import json
import time
import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# ── Load models once at startup ──────────────────────
MODEL_PATH     = os.path.join(os.path.dirname(__file__),
                              "..", "..", "data", "processed", "final_model.pkl")
EXPLAINER_PATH = os.path.join(os.path.dirname(__file__),
                              "..", "..", "data", "processed", "shap_explainer.pkl")
SCALER_PATH    = os.path.join(os.path.dirname(__file__),
                              "..", "..", "data", "processed", "scaler.pkl")

print("Loading model...")
model     = joblib.load(MODEL_PATH)
print("Loading explainer...")
explainer = joblib.load(EXPLAINER_PATH)
print("Loading scaler...")
scaler    = joblib.load(SCALER_PATH)
print("All loaded!")

# ── Fake Redis cache ─────────────────────────────────
cache = fakeredis.FakeRedis()
cache.flushall()
CACHE_TTL = 300  # seconds

# ── FastAPI app ──────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with SHAP explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ───────────────────────────────────
class TransactionRequest(BaseModel):
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)
    Amount: float = Field(default=1.0, ge=0)
    Time: float = Field(default=0.0, ge=0)


# ── Response schema ──────────────────────────────────
class PredictResponse(BaseModel):
    fraud_probability: float
    verdict: str
    confidence: str
    risk_score: int
    response_time_ms: float
    cached: bool
    top_reasons: list


# ── Helper functions ─────────────────────────────────
def preprocess(transaction: TransactionRequest) -> pd.DataFrame:
    data = transaction.model_dump()

    data["Amount_log"] = float(np.log1p(data["Amount"]))
    data["Hour"] = float((data["Time"] // 3600) % 24)
    data["Is_night"] = float(1 if (data["Hour"] >= 22 or data["Hour"] <= 5) else 0)

    df = pd.DataFrame([data])
    df = df.drop(["Time", "Amount"], axis=1)

    scale_cols = ["Amount_log", "Hour", "Is_night"]
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df


def get_explanations(df: pd.DataFrame, top_n: int = 5) -> list:
    shap_vals = explainer.shap_values(df)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    explanation = pd.DataFrame({
        "feature": df.columns,
        "value": df.values[0],
        "shap_value": shap_vals[0]
    })

    explanation = explanation.reindex(
        explanation["shap_value"].abs().sort_values(ascending=False).index
    )

    top_features = explanation.head(top_n)

    return [
        {
            "feature": row["feature"],
            "value": float(row["value"]),
            "impact": float(row["shap_value"]),
            "direction": "increase" if row["shap_value"] > 0 else "decrease"
        }
        for _, row in top_features.iterrows()
    ]


def make_cache_key(transaction: TransactionRequest) -> str:
    data = json.dumps(transaction.model_dump(), sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()


# ── API endpoints ────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "FRAUD DETECTION API RUNNING",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__,
        "cache_keys": cache.dbsize(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(transaction: TransactionRequest):
    start = time.time()
    cache_key = make_cache_key(transaction)
    cached_result = cache.get(cache_key)

    if cached_result:
        result = json.loads(cached_result)
        result["cached"] = True
        result["response_time_ms"] = round((time.time() - start) * 1000, 2)
        return result

    try:
        df = preprocess(transaction)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing error: {str(e)}")

    fraud_prob = float(model.predict_proba(df)[0][1])
    verdict = "FRAUD" if fraud_prob >= 0.5 else "LEGIT"
    confidence = "HIGH" if abs(fraud_prob - 0.5) > 0.3 else "LOW"
    risk_score = int(fraud_prob * 100)

    top_reasons = get_explanations(df)

    result = {
        "fraud_probability": round(fraud_prob, 4),
        "verdict": verdict,
        "confidence": confidence,
        "risk_score": risk_score,
        "cached": False,
        "top_reasons": top_reasons,
        "response_time_ms": round((time.time() - start) * 1000, 2)
    }

    cache.setex(cache_key, CACHE_TTL, json.dumps(result))
    return result


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionRequest]):
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 transactions allowed")

    results = []
    for txn in transactions:
        try:
            results.append(predict(txn))
        except Exception as e:
            results.append({"error": str(e)})

    return {"predictions": results, "count": len(results)}


@app.get("/stats")
def stats():
    return {
        "model_type": type(model).__name__,
        "features": getattr(model, "n_features_in_", "unknown"),
        "cache_entries": cache.dbsize(),
        "cache_ttl_secs": CACHE_TTL,
    }


# ── Entry point ──────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)