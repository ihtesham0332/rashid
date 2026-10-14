"""
Diabetes Risk Prediction API
=============================
FastAPI service that serves a pre-trained SVM model for predicting diabetes
risk severity (Low / Medium / High) from patient health metrics.

Model details
-------------
* Algorithm  : SVM (SVC with RBF kernel, C=100)
* Scaler     : StandardScaler (applied before SVC)
* Features   : Age, Glucose, weight, height, bmi,
               systolic_bp, diastolic_bp, Hba1c, gender_Male
* Target     : diabetic severity  (Low | Medium | High)
* File       : diabetes_model.pkl  (joblib-serialised dict)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Suppress sklearn version mismatch warnings (1.6.1 → 1.6.0)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
DATA_PATH = BASE_DIR / "cleaned_diabetes_data.csv"

# ---------------------------------------------------------------------------
# Load model artefacts at startup
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at {MODEL_PATH}. "
        "Make sure 'diabetes_model.pkl' is in the project root."
    )

artefacts = joblib.load(MODEL_PATH)
svc_model = artefacts["model"]        # sklearn.svm.SVC
scaler = artefacts["scaler"]          # sklearn.preprocessing.StandardScaler
feature_names = artefacts["feature_names"]
# feature_names = ['Age','Glucose','weight','height','bmi',
#                  'systolic_bp','diastolic_bp','Hba1c','gender_Male']

# ---------------------------------------------------------------------------
# Load reference dataset (for /data endpoints)
# ---------------------------------------------------------------------------
if DATA_PATH.exists():
    reference_df = pd.read_csv(DATA_PATH)
else:
    reference_df = None

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description=(
        "A production-ready REST API that uses a pre-trained **SVM model** "
        "(RBF kernel, C=100) to predict the diabetes risk severity of a "
        "patient.\n\n"
        "### Risk Levels\n"
        "| Level | HbA1c Range | Meaning |\n"
        "|-------|-------------|----------|\n"
        "| **Low** | < 5.7 | Normal / low risk |\n"
        "| **Medium** | 5.7 – 6.4 | Pre-diabetic |\n"
        "| **High** | ≥ 6.5 | Diabetic |\n\n"
        "### Input Features\n"
        "`Age`, `gender` (Male / Female), `Glucose` (mg/dL), `weight` (kg), "
        "`height` (m), `bmi`, `systolic_bp` (mmHg), `diastolic_bp` (mmHg), "
        "`Hba1c` (%)\n\n"
        "### How it works\n"
        "1. The API converts `gender` → `gender_Male` (1 if Male, 0 if Female)\n"
        "2. Applies `StandardScaler` trained on the original data\n"
        "3. Feeds scaled features to the SVM classifier\n"
        "4. Returns the predicted risk level"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for local / dev usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Pydantic Schemas
# ===========================================================================

class PatientInput(BaseModel):
    """Input schema for a single patient prediction."""

    Age: int = Field(
        ..., ge=1, le=120,
        description="Patient age in years",
        examples=[45],
    )
    gender: str = Field(
        ...,
        description="Patient gender: 'Male' or 'Female'",
        examples=["Female"],
    )
    Glucose: float = Field(
        ..., ge=0, le=700,
        description="Fasting blood glucose level (mg/dL)",
        examples=[120.5],
    )
    weight: float = Field(
        ..., ge=1, le=300,
        description="Body weight in kilograms",
        examples=[70.2],
    )
    height: float = Field(
        ..., ge=0.5, le=2.5,
        description="Height in metres",
        examples=[1.65],
    )
    bmi: float = Field(
        ..., ge=1, le=80,
        description="Body Mass Index (kg/m²)",
        examples=[25.75],
    )
    systolic_bp: int = Field(
        ..., ge=50, le=300,
        description="Systolic blood pressure (mmHg)",
        examples=[120],
    )
    diastolic_bp: int = Field(
        ..., ge=30, le=200,
        description="Diastolic blood pressure (mmHg)",
        examples=[80],
    )
    Hba1c: float = Field(
        ..., ge=3.0, le=15.0,
        description="Glycated haemoglobin percentage (%)",
        examples=[5.5],
    )

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in ("male", "female"):
            raise ValueError("gender must be 'Male' or 'Female'")
        return v_lower.capitalize()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 42,
                    "gender": "Female",
                    "Glucose": 105.84,
                    "weight": 70.2,
                    "height": 1.65,
                    "bmi": 25.75,
                    "systolic_bp": 110,
                    "diastolic_bp": 73,
                    "Hba1c": 5.1,
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    """Output schema for a single prediction."""
    prediction: str = Field(
        ..., description="Predicted risk level: Low, Medium, or High"
    )
    risk_description: str = Field(
        ..., description="Human-readable explanation of the risk level"
    )
    input_data: dict = Field(
        ..., description="Echo of the input values used for prediction"
    )


class BatchInput(BaseModel):
    """Input schema for batch predictions."""
    patients: List[PatientInput] = Field(
        ..., min_length=1, max_length=500,
        description="List of patient records (max 500 per request)",
    )


class BatchPredictionResult(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionResult]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    kernel: str
    C: float
    features: List[str]
    target_classes: List[str]


class ModelInfoResponse(BaseModel):
    model_type: str
    kernel: str
    C: float
    scaler_type: str
    feature_names: List[str]
    input_features_for_user: List[str]
    target_classes: List[str]
    scaler_means: dict
    scaler_scales: dict
    training_data_rows: Optional[int] = None


class DataStatsResponse(BaseModel):
    total_rows: int
    columns: List[str]
    class_distribution: dict
    numeric_summary: dict


# ===========================================================================
# Helper functions
# ===========================================================================

RISK_DESCRIPTIONS = {
    "Low": "Normal / low risk — HbA1c typically below 5.7%",
    "Medium": "Pre-diabetic — HbA1c typically between 5.7% and 6.4%",
    "High": "Diabetic — HbA1c typically 6.5% or higher",
}

USER_FACING_FEATURES = [
    "Age", "gender", "Glucose", "weight", "height",
    "bmi", "systolic_bp", "diastolic_bp", "Hba1c",
]


def _patient_to_array(patient: PatientInput) -> np.ndarray:
    """Convert a PatientInput to a 1×9 NumPy array matching feature_names."""
    gender_male = 1 if patient.gender == "Male" else 0
    row = np.array([[
        patient.Age,
        patient.Glucose,
        patient.weight,
        patient.height,
        patient.bmi,
        patient.systolic_bp,
        patient.diastolic_bp,
        patient.Hba1c,
        gender_male,
    ]])
    return row


def predict_single(patient: PatientInput) -> str:
    """Return the predicted class for one patient."""
    try:
        X = _patient_to_array(patient)
        X_scaled = scaler.transform(X)
        prediction = svc_model.predict(X_scaled)
        return str(prediction[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {exc}",
        ) from exc


# ===========================================================================
# Endpoints
# ===========================================================================

# ---- General / Health -----------------------------------------------------

@app.get(
    "/",
    summary="Root – welcome message",
    tags=["General"],
)
def root():
    """Returns a welcome message with links to documentation."""
    return {
        "message": "Welcome to the Diabetes Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs (Swagger UI)": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "predict": "/predict  [POST]",
            "batch predict": "/predict/batch  [POST]",
            "model info": "/model-info",
            "data stats": "/data/stats",
            "data sample": "/data/sample?n=5",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["General"],
)
def health_check():
    """Verify the API is running and the model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=svc_model is not None,
        model_type=type(svc_model).__name__,
        kernel=svc_model.kernel,
        C=svc_model.C,
        features=feature_names,
        target_classes=list(svc_model.classes_),
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Detailed model metadata",
    tags=["General"],
)
def model_info():
    """Return full metadata about the loaded model, scaler, and features."""
    means = dict(zip(feature_names, scaler.mean_.tolist()))
    scales = dict(zip(feature_names, scaler.scale_.tolist()))

    return ModelInfoResponse(
        model_type=type(svc_model).__name__,
        kernel=svc_model.kernel,
        C=svc_model.C,
        scaler_type=type(scaler).__name__,
        feature_names=feature_names,
        input_features_for_user=USER_FACING_FEATURES,
        target_classes=list(svc_model.classes_),
        scaler_means=means,
        scaler_scales=scales,
        training_data_rows=len(reference_df) if reference_df is not None else None,
    )


# ---- Predictions ----------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Predict diabetes risk for a single patient",
    tags=["Predictions"],
)
def predict(patient: PatientInput):
    """
    Submit a single patient's health metrics and receive a diabetes-risk
    prediction: **Low**, **Medium**, or **High**.

    ### Example request body
    ```json
    {
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
    ```
    """
    prediction = predict_single(patient)
    return PredictionResult(
        prediction=prediction,
        risk_description=RISK_DESCRIPTIONS.get(prediction, "Unknown"),
        input_data=patient.model_dump(),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    summary="Batch-predict diabetes risk for multiple patients",
    tags=["Predictions"],
)
def predict_batch(batch: BatchInput):
    """
    Submit up to **500 patient records** at once and receive predictions
    for all of them in a single response.

    ### Example request body
    ```json
    {
      "patients": [
        {
          "Age": 42, "gender": "Female", "Glucose": 105.84,
          "weight": 70.2, "height": 1.65, "bmi": 25.75,
          "systolic_bp": 110, "diastolic_bp": 73, "Hba1c": 5.1
        },
        {
          "Age": 62, "gender": "Female", "Glucose": 123.30,
          "weight": 47.0, "height": 1.52, "bmi": 20.24,
          "systolic_bp": 127, "diastolic_bp": 74, "Hba1c": 7.2
        }
      ]
    }
    ```
    """
    results = []
    for patient in batch.patients:
        pred = predict_single(patient)
        results.append(
            PredictionResult(
                prediction=pred,
                risk_description=RISK_DESCRIPTIONS.get(pred, "Unknown"),
                input_data=patient.model_dump(),
            )
        )
    return BatchPredictionResult(predictions=results, total=len(results))


# ---- Data Exploration -----------------------------------------------------

@app.get(
    "/data/stats",
    response_model=DataStatsResponse,
    summary="Summary statistics of the training dataset",
    tags=["Data"],
)
def data_stats():
    """Return high-level statistics about the reference / training dataset."""
    if reference_df is None:
        raise HTTPException(status_code=404, detail="Reference dataset not available.")

    class_dist = reference_df["diabetic"].value_counts().to_dict()
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    summary = reference_df[numeric_cols].describe().round(2).to_dict()

    return DataStatsResponse(
        total_rows=len(reference_df),
        columns=reference_df.columns.tolist(),
        class_distribution={str(k): int(v) for k, v in class_dist.items()},
        numeric_summary=summary,
    )


@app.get(
    "/data/sample",
    summary="Get sample rows from the dataset",
    tags=["Data"],
)
def data_sample(
    n: int = Query(default=5, ge=1, le=50, description="Number of rows to return"),
):
    """Return **n** random sample rows from the training dataset."""
    if reference_df is None:
        raise HTTPException(status_code=404, detail="Reference dataset not available.")

    sample = reference_df.sample(n=min(n, len(reference_df)), random_state=42)
    return sample.to_dict(orient="records")


# ===========================================================================
# Run with:  python main.py
# ===========================================================================
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  Diabetes Risk Prediction API")
    print("  Starting server at http://localhost:8000")
    print("  Swagger docs at   http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
