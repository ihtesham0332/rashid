# Diabetes Risk Prediction API — Complete Documentation

> **A production-ready REST API** built with FastAPI that uses a pre-trained SVM model to predict diabetes risk severity (Low / Medium / High) from patient health metrics.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Installation & Setup](#3-installation--setup)
4. [Running the API](#4-running-the-api)
5. [API Endpoints Reference](#5-api-endpoints-reference)
6. [Testing the API — Real Data Examples](#6-testing-the-api--real-data-examples)
7. [Input Validation & Error Handling](#7-input-validation--error-handling)
8. [Model Details](#8-model-details)
9. [Frequently Asked Questions](#9-frequently-asked-questions)

---

## 1. Project Overview

| Item | Detail |
|------|--------|
| **Framework** | FastAPI (Python) |
| **ML Model** | SVM (`SVC`, RBF kernel, C=100) |
| **Scaler** | `StandardScaler` |
| **Target** | Diabetes risk: `Low`, `Medium`, `High` |
| **Training Data** | 1,898 patient records |
| **Accuracy** | ~96.58% (test set) |
| **Interactive Docs** | Swagger UI at `/docs`, ReDoc at `/redoc` |

### Risk Level Definitions

| Level | HbA1c Range | Clinical Meaning |
|-------|-------------|------------------|
| **Low** | < 5.7% | Normal — low diabetes risk |
| **Medium** | 5.7% – 6.4% | Pre-diabetic |
| **High** | ≥ 6.5% | Diabetic |

---

## 2. Project Structure

```
Model__Rasheed/
├── main.py                    # FastAPI application (the API server)
├── test_api.py                # Automated end-to-end test suite
├── diabetes_model.pkl         # Pre-trained SVM model (joblib format)
├── cleaned_diabetes_data.csv  # Reference training dataset (1,898 rows)
├── requirements.txt           # Python dependencies
├── README.md                  # Complete API & testing documentation
└── Untitled20.ipynb           # Original Colab training notebook
```

---

## 3. Installation & Setup

### Prerequisites

- **Python 3.9+** (tested on 3.11)
- **pip** package manager

### Step-by-step

```bash
# 1. Navigate to the project folder
cd Model__Rasheed

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies installed

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `scikit-learn` | ML model (SVM, StandardScaler) |
| `pandas` | Data handling |
| `numpy` | Numerical operations |
| `pydantic` | Request/response validation |
| `joblib` | Model loading (comes with scikit-learn) |

---

## 4. Running the API

### Start the server

```bash
python main.py
```

Output:

```
============================================================
  Diabetes Risk Prediction API
  Starting server at http://localhost:8000
  Swagger docs at   http://localhost:8000/docs
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Access points

| URL | Description |
|-----|-------------|
| `http://localhost:8000` | Root / welcome message |
| `http://localhost:8000/docs` | **Swagger UI** — interactive API testing |
| `http://localhost:8000/redoc` | ReDoc — read-only API docs |
| `http://localhost:8000/health` | Health check |

---

## 5. API Endpoints Reference

### 5.1 `GET /` — Welcome

Returns a welcome message and a list of all endpoints.

```bash
curl http://localhost:8000/
```

---

### 5.2 `GET /health` — Health Check

Verify the API is running and the model is loaded.

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "SVC",
  "kernel": "rbf",
  "C": 100.0,
  "features": ["Age", "Glucose", "weight", "height", "bmi", "systolic_bp", "diastolic_bp", "Hba1c", "gender_Male"],
  "target_classes": ["High", "Low", "Medium"]
}
```

---

### 5.3 `GET /model-info` — Model Metadata

Returns full details about the model, scaler means/scales, and feature information.

```bash
curl http://localhost:8000/model-info
```

---

### 5.4 `POST /predict` — Single Patient Prediction ⭐

**The main endpoint.** Submit one patient's health data and get a diabetes risk prediction.

#### Request Body Schema

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `Age` | int | 1–120 | Patient age in years |
| `gender` | string | `"Male"` or `"Female"` | Patient gender |
| `Glucose` | float | 0–700 | Fasting blood glucose (mg/dL) |
| `weight` | float | 1–300 | Body weight (kg) |
| `height` | float | 0.5–2.5 | Height (metres) |
| `bmi` | float | 1–80 | Body Mass Index (kg/m²) |
| `systolic_bp` | int | 50–300 | Systolic blood pressure (mmHg) |
| `diastolic_bp` | int | 30–200 | Diastolic blood pressure (mmHg) |
| `Hba1c` | float | 3.0–15.0 | Glycated haemoglobin (%) |

#### Example — curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 42,
    "gender": "Female",
    "Glucose": 105.84,
    "weight": 70.2,
    "height": 1.65,
    "bmi": 25.75,
    "systolic_bp": 110,
    "diastolic_bp": 73,
    "Hba1c": 5.1
  }'
```

#### Example — Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "Age": 42,
        "gender": "Female",
        "Glucose": 105.84,
        "weight": 70.2,
        "height": 1.65,
        "bmi": 25.75,
        "systolic_bp": 110,
        "diastolic_bp": 73,
        "Hba1c": 5.1
    }
)

print(response.json())
```

#### Example — PowerShell

```powershell
$body = @{
    Age = 42
    gender = "Female"
    Glucose = 105.84
    weight = 70.2
    height = 1.65
    bmi = 25.75
    systolic_bp = 110
    diastolic_bp = 73
    Hba1c = 5.1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method POST -Body $body -ContentType "application/json"
```

#### Response

```json
{
  "prediction": "Low",
  "risk_description": "Normal / low risk — HbA1c typically below 5.7%",
  "input_data": {
    "Age": 42,
    "gender": "Female",
    "Glucose": 105.84,
    "weight": 70.2,
    "height": 1.65,
    "bmi": 25.75,
    "systolic_bp": 110,
    "diastolic_bp": 73,
    "Hba1c": 5.1
  }
}
```

---

### 5.5 `POST /predict/batch` — Batch Predictions

Submit up to **500 patient records** at once.

#### Example — curl

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {
        "Age": 42, "gender": "Female", "Glucose": 105.84,
        "weight": 70.2, "height": 1.65, "bmi": 25.75,
        "systolic_bp": 110, "diastolic_bp": 73, "Hba1c": 5.1
      },
      {
        "Age": 35, "gender": "Female", "Glucose": 102.78,
        "weight": 42.5, "height": 1.47, "bmi": 19.58,
        "systolic_bp": 125, "diastolic_bp": 68, "Hba1c": 6.0
      },
      {
        "Age": 62, "gender": "Female", "Glucose": 123.30,
        "weight": 47.0, "height": 1.52, "bmi": 20.24,
        "systolic_bp": 127, "diastolic_bp": 74, "Hba1c": 7.2
      }
    ]
  }'
```

#### Response

```json
{
  "predictions": [
    {
      "prediction": "Low",
      "risk_description": "Normal / low risk — HbA1c typically below 5.7%",
      "input_data": { ... }
    },
    {
      "prediction": "Medium",
      "risk_description": "Pre-diabetic — HbA1c typically between 5.7% and 6.4%",
      "input_data": { ... }
    },
    {
      "prediction": "High",
      "risk_description": "Diabetic — HbA1c typically 6.5% or higher",
      "input_data": { ... }
    }
  ],
  "total": 3
}
```

---

### 5.6 `GET /data/stats` — Dataset Statistics

```bash
curl http://localhost:8000/data/stats
```

Returns total rows, columns, class distribution, and numeric summary statistics.

---

### 5.7 `GET /data/sample?n=5` — Sample Dataset Rows

```bash
curl "http://localhost:8000/data/sample?n=5"
```

Returns **n** random rows from the training dataset (max 50).

---

## 6. Testing the API — Real Data Examples

### 6.1 Automated One-Command Test Suite (`test_api.py`)

A full end-to-end test suite is included in [`test_api.py`](file:///c:/Users/khani/OneDrive/Desktop(1)/Model__Rasheed/test_api.py). You can test all endpoints, real data predictions, batch processing, and validation error handlers with a single command:

```bash
python test_api.py
```

Expected output:
```
Starting Diabetes Risk Prediction API Automated Test Suite...
============================================================
  1. Testing Root Endpoint GET /
  [PASS] Root endpoint returned status 200 with documentation links.
  2. Testing Health Endpoint GET /health
  [PASS] Health check verified: Status: healthy, Model Type: SVC
  3. Testing Model Info Endpoint GET /model-info
  [PASS] Model info returned 9 features and training size: 1898 records.
  4. Testing Single Prediction POST /predict with Real Patient Data
  [PASS] Low Risk Patient: Prediction: Low (Expected: Low)
  [PASS] Medium Risk Patient: Prediction: Medium (Expected: Medium)
  [PASS] High Risk Patient: Prediction: High (Expected: High)
  5. Testing Batch Prediction POST /predict/batch
  [PASS] Batch of 2 patients processed successfully. Total: 2.
  6. Testing Data Endpoints GET /data/stats & GET /data/sample
  [PASS] /data/stats total rows: 1898
  7. Testing Input Validation Error Handling (422 Unprocessable Entity)
  [PASS] Correctly rejected invalid Age (150) with HTTP 422.
  [PASS] Correctly rejected invalid gender ('Alien') with HTTP 422.
============================================================
  ALL TESTS PASSED SUCCESSFULLY! (7/7 suites passed)
============================================================
```

---

### 6.2 Manual Real Data Test Cases (curl / Postman / Swagger)

These examples use **actual rows** from the training dataset (`cleaned_diabetes_data.csv`) so you can verify the predictions match.

### Test Case 1: Low Risk Patient (Row 0)

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Age\": 42, \"gender\": \"Female\", \"Glucose\": 105.84, \"weight\": 70.2, \"height\": 1.65, \"bmi\": 25.75, \"systolic_bp\": 110, \"diastolic_bp\": 73, \"Hba1c\": 5.1}"
```

**Expected:** `"prediction": "Low"` ✅ (HbA1c = 5.1, label = Low)

---

### Test Case 2: Medium Risk Patient (Row 1)

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Age\": 35, \"gender\": \"Female\", \"Glucose\": 102.78, \"weight\": 42.5, \"height\": 1.47, \"bmi\": 19.58, \"systolic_bp\": 125, \"diastolic_bp\": 68, \"Hba1c\": 6.0}"
```

**Expected:** `"prediction": "Medium"` ✅ (HbA1c = 6.0, label = Medium)

---

### Test Case 3: High Risk Patient (Row 2)

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Age\": 62, \"gender\": \"Female\", \"Glucose\": 123.30, \"weight\": 47.0, \"height\": 1.52, \"bmi\": 20.24, \"systolic_bp\": 127, \"diastolic_bp\": 74, \"Hba1c\": 7.2}"
```

**Expected:** `"prediction": "High"` ✅ (HbA1c = 7.2, label = High)

---

### Test Case 4: Male Patient (Row 3)

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Age\": 73, \"gender\": \"Male\", \"Glucose\": 113.04, \"weight\": 57.4, \"height\": 1.63, \"bmi\": 21.72, \"systolic_bp\": 193, \"diastolic_bp\": 112, \"Hba1c\": 5.5}"
```

**Expected:** `"prediction": "Low"` ✅ (HbA1c = 5.5, label = Low)

---

### Test Case 5: Python Script — Test All Three Classes

```python
import requests

url = "http://localhost:8000/predict"

# Test cases from the real dataset
test_cases = [
    {
        "name": "Low Risk (Row 0)",
        "data": {"Age": 42, "gender": "Female", "Glucose": 105.84,
                 "weight": 70.2, "height": 1.65, "bmi": 25.75,
                 "systolic_bp": 110, "diastolic_bp": 73, "Hba1c": 5.1},
        "expected": "Low"
    },
    {
        "name": "Medium Risk (Row 1)",
        "data": {"Age": 35, "gender": "Female", "Glucose": 102.78,
                 "weight": 42.5, "height": 1.47, "bmi": 19.58,
                 "systolic_bp": 125, "diastolic_bp": 68, "Hba1c": 6.0},
        "expected": "Medium"
    },
    {
        "name": "High Risk (Row 2)",
        "data": {"Age": 62, "gender": "Female", "Glucose": 123.30,
                 "weight": 47.0, "height": 1.52, "bmi": 20.24,
                 "systolic_bp": 127, "diastolic_bp": 74, "Hba1c": 7.2},
        "expected": "High"
    },
    {
        "name": "Male Low Risk (Row 3)",
        "data": {"Age": 73, "gender": "Male", "Glucose": 113.04,
                 "weight": 57.4, "height": 1.63, "bmi": 21.72,
                 "systolic_bp": 193, "diastolic_bp": 112, "Hba1c": 5.5},
        "expected": "Low"
    },
]

print("=" * 50)
print("  DIABETES API - TEST SUITE")
print("=" * 50)

for tc in test_cases:
    r = requests.post(url, json=tc["data"])
    result = r.json()
    status = "PASS" if result["prediction"] == tc["expected"] else "FAIL"
    print(f"\n  [{status}] {tc['name']}")
    print(f"    Expected: {tc['expected']}")
    print(f"    Got:      {result['prediction']}")
    print(f"    Risk:     {result['risk_description']}")

print("\n" + "=" * 50)
```

---

## 7. Input Validation & Error Handling

The API validates all inputs automatically. If you send bad data, you'll get a clear `422` error.

### Example — Missing Field

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 42, "gender": "Female"}'
```

**Response (422):**

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "Glucose"],
      "msg": "Field required"
    },
    ...
  ]
}
```

### Example — Invalid Gender

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 42, "gender": "Unknown", "Glucose": 100, "weight": 70, "height": 1.65, "bmi": 25, "systolic_bp": 120, "diastolic_bp": 80, "Hba1c": 5.5}'
```

**Response (422):**

```json
{
  "detail": [
    {
      "loc": ["body", "gender"],
      "msg": "Value error, gender must be 'Male' or 'Female'"
    }
  ]
}
```

### Example — Value Out of Range

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 200, "gender": "Male", "Glucose": 100, "weight": 70, "height": 1.65, "bmi": 25, "systolic_bp": 120, "diastolic_bp": 80, "Hba1c": 5.5}'
```

**Response (422):**

```json
{
  "detail": [
    {
      "loc": ["body", "Age"],
      "msg": "Input should be less than or equal to 120"
    }
  ]
}
```

---

## 8. Model Details

### Architecture

```
Input (9 features)
    │
    ▼
┌──────────────────────┐
│   gender → gender_Male  (0 if Female, 1 if Male)
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│   StandardScaler     │  (z-score normalisation)
│   trained on 1,898   │
│   patient records    │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│   SVM Classifier     │
│   kernel = RBF       │
│   C = 100            │
└──────────────────────┘
    │
    ▼
Prediction: Low / Medium / High
```

### Features (in model order)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `Age` | int | Patient age (years) |
| 2 | `Glucose` | float | Fasting blood glucose (mg/dL) |
| 3 | `weight` | float | Body weight (kg) |
| 4 | `height` | float | Height (metres) |
| 5 | `bmi` | float | Body Mass Index |
| 6 | `systolic_bp` | int | Systolic BP (mmHg) |
| 7 | `diastolic_bp` | int | Diastolic BP (mmHg) |
| 8 | `Hba1c` | float | Glycated haemoglobin (%) |
| 9 | `gender_Male` | int | 1 if Male, 0 if Female (auto-encoded by API) |

### Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 96.58% |
| Macro F1 | 96.47% |
| 5-Fold CV F1 | 95.34% ± 1.51% |

---

## 9. Frequently Asked Questions

### Q: How do I test without curl?

Open `http://localhost:8000/docs` in your browser. The **Swagger UI** lets you test every endpoint interactively — just click "Try it out", fill in the JSON, and click "Execute".

### Q: Can I deploy this to a server?

Yes. Options include:

- **Docker**: Create a `Dockerfile`, copy the project, and run `uvicorn main:app`
- **Cloud Run / AWS Lambda**: Package as a container
- **Heroku / Railway / Render**: Push the repo and set start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Q: What happens if I send wrong data types?

FastAPI automatically returns a `422 Unprocessable Entity` response with a detailed error message explaining which field failed validation and why.

### Q: Can gender be "male" (lowercase)?

Yes. The API normalises gender input — `"male"`, `"Male"`, `"MALE"`, `" Male "` all work correctly.

### Q: Why does the API return the input data in the response?

For traceability. When processing batch predictions or integrating with other systems, echoing the input alongside the prediction makes it easy to match results to records.

---

**Built with FastAPI + scikit-learn | Model trained on 1,898 patient records**
