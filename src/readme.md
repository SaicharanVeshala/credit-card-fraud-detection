# Fraud Detection API

## Endpoints

| Method | Endpoint        | Description                          |
|--------|-----------------|--------------------------------------|
| GET    | /               | API info                             |
| GET    | /health         | Health check                         |
| GET    | /docs           | Swagger UI (auto-generated)          |
| POST   | /predict        | Score one transaction                |
| POST   | /predict/batch  | Score up to 100 transactions         |
| GET    | /stats          | Model and cache stats                |

## Example request

```json
POST /predict
{
  "V1": -1.35,
  "V14": -5.92,
  "Amount": 149.62,
  "Time": 406
}
```

## Example response

```json
{
  "fraud_probability": 0.9823,
  "verdict": "FRAUD",
  "confidence": "HIGH",
  "risk_score": 98,
  "cached": false,
  "response_time_ms": 23.4,
  "top_reasons": [
    {
      "feature": "V14",
      "value": -5.92,
      "impact": 0.8431,
      "direction": "toward fraud"
    }
  ]
}
```

## Run locally

```powershell
cd src\api
uvicorn main:app --reload --port 8000
```