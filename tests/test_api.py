import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),".."))

from fastapi.testclient import TestClient
from src.api.main import app,cache

client=TestClient(app)
def setup_function():
    cache.flushall()
LEGIT_TRANSACTION={
        "V1": -1.35, "V2": -0.07, "V3": 2.53,  "V4": 1.37,
    "V5": -0.33, "V6":  0.46, "V7": 0.23,  "V8": 0.09,
    "V9":  0.36, "V10": 0.09, "V11": -0.55, "V12": -0.61,
    "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
    "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
    "V25": 0.12, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62, "Time": 406

}

FRAUD_TRANSACTION={
     "V1": -3.04, "V2": -3.15, "V3": 1.08,  "V4": 2.28,
    "V5":  4.53, "V6": -1.47, "V7": -3.63, "V8": 0.10,
    "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.89,
    "V13": -0.59, "V14": -5.92, "V15": 0.09, "V16": -2.76,
    "V17": -3.43, "V18": -0.98, "V19": -0.62, "V20": -1.42,
    "V21": -0.35, "V22": -0.93, "V23": 0.17, "V24": 0.70,
    "V25": -0.50, "V26": -0.44, "V27": 0.00, "V28": 0.02,
    "Amount": 9.99, "Time": 87120
}

def test_health():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="healthy"

def test_predict_returns_200():
    response=client.post("/predict",json=LEGIT_TRANSACTION)
    assert response.status_code==200

def test_predict_has_requires_fields():
    response=client.post( "/predict",json=LEGIT_TRANSACTION)
    data=response.json()
    assert "fraud_probability" in data
    assert "verdict" in data
    assert "confidence" in data
    assert "risk_score" in data
    assert "top_reasons" in data

def test_fraud_probability_range():
    response=client.post("/predict",json=LEGIT_TRANSACTION)
    prob=response.json()["fraud_probability"]
    assert 0.0<=prob<=1.0

def test_risk_score_range():
    response=client.post("/predict",json=LEGIT_TRANSACTION)
    prob=response.json()["fraud_probability"]
    assert 0.0<=prob<=1.0

def test_ris_score_range():
    response=client.post("/predict",json=LEGIT_TRANSACTION)
    score=response.json()["risk_score"]
    assert 0<=score <=100

def test_legit_transaction_verdict():
    response=client.post("/predict",json=LEGIT_TRANSACTION)
    verdict=response.json()["verdict"]
    assert verdict in ["FRAUD","LEGIT"]

def test_fraud_transaction_high_probability():
    response = client.post("/predict", json=FRAUD_TRANSACTION)
    prob     = response.json()["fraud_probability"]
    assert prob > 0.5, f"Expected fraud probability > 0.5, got {prob}"

def test_caching_works():
    r1=client.post("/predict",json=LEGIT_TRANSACTION)
    assert r1.json()["cached"]==False

    r2=client.post("/predict",json=LEGIT_TRANSACTION)
    assert r2.json()["cached"]==True

def test_top_reasons_not_empty():
    response=client.post("/predict", json=LEGIT_TRANSACTION)
    reasons=response.json()["top_reasons"]
    assert len(reasons)>0
    assert "feature" in reasons[0]
    assert "impact" in reasons[0]
    assert "direction" in reasons[0]

def test_batch_predict():
    response=client.post("/predict/batch",
                           json=[LEGIT_TRANSACTION, FRAUD_TRANSACTION])
    data=response.json()
    assert data["count"]==2
    assert len(data["predictions"])==2

def test_batch_limit():
    big_batch=[LEGIT_TRANSACTION]*101
    response=client.post("/predict/batch",json=big_batch)
    assert response.status_code==400