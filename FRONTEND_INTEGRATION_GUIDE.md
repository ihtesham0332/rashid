# Frontend Developer Integration Guide — Diabetes Risk Prediction API

Welcome! This guide contains everything you need to integrate and test the **Diabetes Risk Prediction API** (powered by a 98.60% accuracy PyTorch Deep Tabular ResNet) in web, mobile, or desktop applications.

---

## 1. Live API Environments & URLs

| Environment | Base URL | Interactive Swagger Docs | ReDoc Reference |
| :--- | :--- | :--- | :--- |
| **Production (Hugging Face)** | `https://ihtesham0345-kidney-stone.hf.space` | [`/docs`](https://ihtesham0345-kidney-stone.hf.space/docs) | [`/redoc`](https://ihtesham0345-kidney-stone.hf.space/redoc) |
| **Local Development** | `http://localhost:8000` | [`http://localhost:8000/docs`](http://localhost:8000/docs) | [`http://localhost:8000/redoc`](http://localhost:8000/redoc) |

> **CORS Enabled**: The API has Cross-Origin Resource Sharing (`CORS`) enabled for all origins (`*`), so you can make direct client-side `fetch` or `axios` calls from `localhost:3000`, Vercel, Netlify, or any domain.

---

## 2. Interactive Testing via Swagger UI

You can test every endpoint without writing a single line of code:

1. Open **[https://ihtesham0345-kidney-stone.hf.space/docs](https://ihtesham0345-kidney-stone.hf.space/docs)** in your browser.
2. Click on the **`POST /predict`** accordion.
3. Click the **"Try it out"** button on the top-right.
4. Modify the sample JSON payload (e.g. change `Glucose` or `Hba1c`).
5. Click the large blue **"Execute"** button.
6. Scroll down to see the real-time **Server Response (HTTP 200)**, confidence percentages, and class probabilities.

---

## 3. Core Endpoint: `POST /predict`

### Request URL
```http
POST https://ihtesham0345-kidney-stone.hf.space/predict
Content-Type: application/json
```

### Request Headers
```http
Content-Type: application/json
Accept: application/json
```

### Request Payload (JSON)
```json
{
  "Age": 45,
  "gender": "Male",
  "Glucose": 115.0,
  "weight": 76.0,
  "height": 1.72,
  "bmi": 25.7,
  "systolic_bp": 128,
  "diastolic_bp": 82,
  "Hba1c": 5.9
}
```

### Input Field Specifications & Valid Ranges

| Field | Type | Valid Range | Unit / Description | Auto-calculable? |
| :--- | :---: | :---: | :--- | :---: |
| `Age` | `integer` | `1` – `120` | Patient age in years | No |
| `gender` | `string` | `"Male"` or `"Female"` | Case-sensitive | No |
| `Glucose` | `number` | `30.0` – `400.0` | Fasting blood glucose in mg/dL | No |
| `weight` | `number` | `20.0` – `300.0` | Weight in kilograms | No |
| `height` | `number` | `0.5` – `2.5` | Height in meters (e.g. 1.75) | No |
| `bmi` | `number` | `10.0` – `80.0` | Body Mass Index ($kg/m^2$) | **Yes**: `weight / (height * height)` |
| `systolic_bp` | `number` | `60` – `260` | Systolic blood pressure (mmHg) | No |
| `diastolic_bp` | `number` | `40` – `160` | Diastolic blood pressure (mmHg, must be < systolic) | No |
| `Hba1c` | `number` | `3.0` – `20.0` | Glycated hemoglobin percentage (%) | No |

---

### Response Payload (HTTP 200 OK)
```json
{
  "prediction": "Medium",
  "risk_description": "Pre-diabetic — HbA1c typically between 5.7% and 6.4%",
  "confidence": 98.4,
  "probabilities": {
    "Low": 0.5,
    "Medium": 98.4,
    "High": 1.1
  },
  "input_data": {
    "Age": 45,
    "gender": "Male",
    "Glucose": 115.0,
    "weight": 76.0,
    "height": 1.72,
    "bmi": 25.7,
    "systolic_bp": 128.0,
    "diastolic_bp": 82.0,
    "Hba1c": 5.9
  }
}
```

### Risk Level UI Badge Guide

| Severity | Color Code | Clinical Meaning | Recommended UI Treatment |
| :--- | :---: | :--- | :--- |
| **`Low`** | `#10b981` (Green) | Normal risk (HbA1c < 5.7%) | Green badge with check icon |
| **`Medium`** | `#f59e0b` (Amber) | Pre-diabetic (HbA1c 5.7% – 6.4%) | Yellow/amber warning badge |
| **`High`** | `#ef4444` (Red) | Diabetic (HbA1c ≥ 6.5%) | Red alert badge with urgent flag |

---

## 4. Frontend Code Snippets

### A. Modern JavaScript / TypeScript (Fetch API)

```typescript
export interface PatientInput {
  Age: number;
  gender: 'Male' | 'Female';
  Glucose: number;
  weight: number;
  height: number;
  bmi: number;
  systolic_bp: number;
  diastolic_bp: number;
  Hba1c: number;
}

export interface PredictionResult {
  prediction: 'Low' | 'Medium' | 'High';
  confidence: number;
  probabilities: {
    Low: number;
    Medium: number;
    High: number;
  };
  risk_description: string;
}

const API_BASE_URL = 'https://ihtesham0345-kidney-stone.hf.space';

export async function predictDiabetesRisk(patient: PatientInput): Promise<PredictionResult> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(patient),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || `Prediction error (${response.status})`);
  }

  return response.json();
}
```

---

### B. React Hook Example (`useDiabetesPrediction.ts`)

```tsx
import { useState } from 'react';
import { PatientInput, PredictionResult, predictDiabetesRisk } from './api';

export function useDiabetesPrediction() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const executePrediction = async (patientData: PatientInput) => {
    setLoading(true);
    setError(null);
    try {
      const data = await predictDiabetesRisk(patientData);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Failed to calculate prediction');
    } finally {
      setLoading(false);
    }
  };

  return { executePrediction, result, loading, error };
}
```

---

### C. Automatic BMI Calculation Helper

You can automatically compute `bmi` when the user types `weight` or `height`:

```javascript
function calculateBMI(weightKg, heightMeters) {
  if (weightKg > 0 && heightMeters > 0) {
    return parseFloat((weightKg / (heightMeters * heightMeters)).toFixed(1));
  }
  return 0;
}
```

---

## 5. Other Useful Endpoints

### Health Check: `GET /health`
Verify that the PyTorch neural network and API are online:
```http
GET https://ihtesham0345-kidney-stone.hf.space/health
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "DiabetesTabularResNet",
  "accuracy": "98.60%",
  "macro_roc_auc": 0.9994
}
```

### Batch Prediction: `POST /predict/batch`
Predict up to 500 patient records in a single request:
```http
POST https://ihtesham0345-kidney-stone.hf.space/predict/batch
```
Accepts an array of patient objects `[ { ... }, { ... } ]`.

---

## 6. Error Handling (HTTP 422 & 500)

If invalid values are supplied (e.g. `Age: 200` or `diastolic_bp > systolic_bp`), the API responds with **HTTP 422 Unprocessable Entity**:

```json
{
  "detail": [
    {
      "loc": ["body", "Age"],
      "msg": "Input should be less than or equal to 120",
      "type": "less_than_equal"
    }
  ]
}
```

In your UI, check `if (!response.ok)` and display user-friendly error messages based on the `detail` array.
