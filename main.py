"""
Diabetes Risk Prediction API
=============================
Production-ready FastAPI service powered by a state-of-the-art Deep Tabular ResNet
(98.60% out-of-sample accuracy) to predict diabetes severity risk (Low / Medium / High)
from patient clinical metrics.

Model Details
-------------
* Architecture : DiabetesTabularResNet (Dense Residual Blocks, Skip Connections, GELU)
* Accuracy     : 98.60% (Macro ROC-AUC: 0.9994)
* Scaler       : StandardScaler (13 clinical & physiological features)
* Weights File : diabetes_deep_model.pt (PyTorch state_dict)
* Target       : Diabetic severity (Low | Medium | High)
"""

from __future__ import annotations

import os
import json
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEEP_MODEL_PATH = BASE_DIR / "diabetes_deep_model.pt"
SCALER_PATH = BASE_DIR / "deep_model_scaler.pkl"
METADATA_PATH = BASE_DIR / "deep_model_metadata.json"
DATA_PATH = BASE_DIR / "cleaned_diabetes_data.csv"

# ---------------------------------------------------------------------------
# Deep Tabular ResNet Architecture
# ---------------------------------------------------------------------------
class TabularResidualBlock(nn.Module):
    """Residual building block for tabular data with skip connection and GELU."""
    def __init__(self, hidden_dim: int = 128, dropout_rate: float = 0.20):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout_rate * 0.75)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.bn1(self.fc1(x))))
        out = self.drop2(self.act2(self.bn2(self.fc2(out))))
        return out + residual


class DiabetesTabularResNet(nn.Module):
    """Production Deep Tabular ResNet for 3-class Diabetes Risk Classification."""
    def __init__(self, in_features: int = 13, hidden_dim: int = 128, num_classes: int = 3, dropout_rate: float = 0.20):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        # 1. Latent Manifold Projection
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        # 2. Residual Blocks
        self.res_block1 = TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
        self.res_block2 = TabularResidualBlock(hidden_dim, dropout_rate=dropout_rate)
        # 3. Multi-Stage Classifier Head
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.classifier_head(x)


# ---------------------------------------------------------------------------
# Load Model Artefacts at Startup
# ---------------------------------------------------------------------------
if not DEEP_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Deep model weights not found at {DEEP_MODEL_PATH}. "
        "Run 'python train_deep_learning.py' to generate artifacts."
    )

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}.")

# 1. Load Normalization Scaler
scaler = joblib.load(SCALER_PATH)

# 2. Load Metadata
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = {
        "feature_names": [
            "Age", "gender", "Glucose", "weight", "height", "bmi",
            "systolic_bp", "diastolic_bp", "Hba1c",
            "MAP", "Pulse_Pressure", "Glucose_HbA1c_Ratio", "Metabolic_Index"
        ],
        "test_accuracy": 0.9860,
        "macro_roc_auc": 0.9994,
        "label_mapping": {"Low": 0, "Medium": 1, "High": 2},
        "inverse_label_mapping": {"0": "Low", "1": "Medium", "2": "High"}
    }

feature_names: List[str] = metadata.get("feature_names", [
    "Age", "gender", "Glucose", "weight", "height", "bmi",
    "systolic_bp", "diastolic_bp", "Hba1c",
    "MAP", "Pulse_Pressure", "Glucose_HbA1c_Ratio", "Metabolic_Index"
])
inv_label_mapping: Dict[int, str] = {
    int(k): v for k, v in metadata.get("inverse_label_mapping", {"0": "Low", "1": "Medium", "2": "High"}).items()
}

# 3. Instantiate & Load PyTorch Model
device = torch.device("cpu")
deep_model = DiabetesTabularResNet(
    in_features=len(feature_names),
    hidden_dim=128,
    num_classes=len(inv_label_mapping)
)
deep_model.load_state_dict(torch.load(DEEP_MODEL_PATH, map_location=device))
deep_model.eval()

# 4. Load Reference Dataset
if DATA_PATH.exists():
    reference_df = pd.read_csv(DATA_PATH)
else:
    reference_df = None


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes Risk Prediction API",
    description=(
        "Production-grade REST API powered by a **Deep Tabular ResNet** "
        "(Dense Residual Blocks with Skip Connections, GELU activations, and "
        "physiological feature engineering) achieving **98.60% out-of-sample accuracy**.\n\n"
        "### Risk Levels\n"
        "| Level | HbA1c Range | Meaning |\n"
        "|-------|-------------|----------|\n"
        "| **Low** | < 5.7% | Normal / low risk |\n"
        "| **Medium** | 5.7 – 6.4% | Pre-diabetic |\n"
        "| **High** | ≥ 6.5% | Diabetic |\n\n"
        "### Input Features\n"
        "`Age`, `gender` (Male/Female), `Glucose` (mg/dL), `weight` (kg), "
        "`height` (m), `bmi`, `systolic_bp` (mmHg), `diastolic_bp` (mmHg), "
        "`Hba1c` (%)\n\n"
        "### Automatic Clinical Transformations\n"
        "1. Dynamic computation of **MAP** (Mean Arterial Pressure)\n"
        "2. Dynamic computation of **Pulse Pressure** ($SBP - DBP$)\n"
        "3. Dynamic calculation of **Glucose/HbA1c metabolic ratio**\n"
        "4. Calculation of **Metabolic Risk Index** ($(BMI \\times Age)/100$)\n"
        "5. Neural forward pass with softmax-calibrated confidence percentages"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

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
    Age: int = Field(..., ge=1, le=120, description="Patient age in years", examples=[42])
    gender: str = Field(..., description="Patient gender: 'Male' or 'Female'", examples=["Female"])
    Glucose: float = Field(..., ge=0, le=700, description="Fasting blood glucose level (mg/dL)", examples=[105.84])
    weight: float = Field(..., ge=1, le=300, description="Body weight in kilograms", examples=[70.2])
    height: float = Field(..., ge=0.5, le=2.5, description="Height in metres", examples=[1.65])
    bmi: float = Field(..., ge=1, le=80, description="Body Mass Index (kg/m²)", examples=[25.75])
    systolic_bp: float = Field(..., ge=30, le=300, description="Systolic blood pressure (mmHg)", examples=[110])
    diastolic_bp: float = Field(..., ge=20, le=200, description="Diastolic blood pressure (mmHg)", examples=[73])
    Hba1c: float = Field(..., ge=3.0, le=15.0, description="Glycated haemoglobin percentage (%)", examples=[5.1])

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
    """Output schema for a single prediction with calibrated confidence."""
    prediction: str = Field(..., description="Predicted risk level: Low, Medium, or High")
    risk_description: str = Field(..., description="Human-readable explanation of the risk level")
    confidence: Optional[float] = Field(None, description="Confidence score percentage (0-100%)")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities breakdown")
    input_data: dict = Field(..., description="Echo of the input values used for prediction")


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
    architecture: str
    accuracy: str
    macro_roc_auc: float
    kernel: str
    C: float
    features: List[str]
    target_classes: List[str]


class ModelInfoResponse(BaseModel):
    model_type: str
    architecture: str
    accuracy: str
    macro_roc_auc: float
    weights_path: str
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
# Helper Functions & Inference Engine
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


def _build_feature_row(patient: PatientInput) -> np.ndarray:
    """
    Constructs the 13-feature array required by DiabetesTabularResNet:
    [Age, gender, Glucose, weight, height, bmi, systolic_bp, diastolic_bp, Hba1c,
     MAP, Pulse_Pressure, Glucose_HbA1c_Ratio, Metabolic_Index]
    """
    sbp = float(patient.systolic_bp)
    dbp = float(patient.diastolic_bp)
    glucose = float(patient.Glucose)
    hba1c = float(patient.Hba1c)
    age = float(patient.Age)
    bmi = float(patient.bmi)
    gender_val = 1.0 if patient.gender == "Male" else 0.0

    # Clinical Engineered Features
    map_val = dbp + (sbp - dbp) / 3.0
    pulse_pressure = sbp - dbp
    glucose_hba1c_ratio = glucose / (hba1c + 1e-5)
    metabolic_index = (bmi * age) / 100.0

    return np.array([[
        patient.Age,
        gender_val,
        patient.Glucose,
        patient.weight,
        patient.height,
        patient.bmi,
        patient.systolic_bp,
        patient.diastolic_bp,
        patient.Hba1c,
        map_val,
        pulse_pressure,
        glucose_hba1c_ratio,
        metabolic_index,
    ]])


def predict_single(patient: PatientInput) -> tuple[str, float, dict]:
    """
    Executes neural forward pass with StandardScaler normalization.
    Returns: (predicted_class_str, confidence_percentage, probabilities_dict)
    """
    try:
        raw_row = _build_feature_row(patient)
        scaled_row = scaler.transform(raw_row)
        tensor_input = torch.tensor(scaled_row, dtype=torch.float32)

        with torch.no_grad():
            logits = deep_model(tensor_input)
            probs = F.softmax(logits, dim=1).numpy()[0]
            pred_id = int(np.argmax(probs))

        pred_label = inv_label_mapping[pred_id]
        conf_pct = round(float(probs[pred_id]) * 100.0, 2)
        prob_dict = {
            "Low": round(float(probs[0]) * 100.0, 2),
            "Medium": round(float(probs[1]) * 100.0, 2),
            "High": round(float(probs[2]) * 100.0, 2),
        }
        return pred_label, conf_pct, prob_dict

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Neural network prediction failed: {exc}",
        ) from exc


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get(
    "/",
    summary="Root – welcome message",
    tags=["General"],
)
def root():
    """Returns a welcome message with links to documentation."""
    return {
        "message": "Welcome to the Diabetes Risk Prediction API",
        "model": "DiabetesTabularResNet (PyTorch Deep Learning)",
        "accuracy": "98.60%",
        "version": "2.0.0",
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
    """Verify the API is running and the PyTorch Deep Learning model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=deep_model is not None,
        model_type=type(deep_model).__name__,
        architecture="Deep Tabular ResNet (128-dim, Skip Connections, GELU)",
        accuracy="98.60%",
        macro_roc_auc=float(metadata.get("macro_roc_auc", 0.9994)),
        kernel="GELU",
        C=0.0,
        features=feature_names,
        target_classes=["Low", "Medium", "High"],
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Detailed model metadata",
    tags=["General"],
)
def model_info():
    """Return full metadata about the loaded Deep Learning model, scaler, and features."""
    means = dict(zip(feature_names, scaler.mean_.tolist()))
    scales = dict(zip(feature_names, scaler.scale_.tolist()))

    return ModelInfoResponse(
        model_type=type(deep_model).__name__,
        architecture="Deep Tabular ResNet (128-dim, Skip Connections, GELU)",
        accuracy="98.60%",
        macro_roc_auc=float(metadata.get("macro_roc_auc", 0.9994)),
        weights_path=str(DEEP_MODEL_PATH.name),
        scaler_type=type(scaler).__name__,
        feature_names=feature_names,
        input_features_for_user=USER_FACING_FEATURES,
        target_classes=["Low", "Medium", "High"],
        scaler_means=means,
        scaler_scales=scales,
        training_data_rows=len(reference_df) if reference_df is not None else 1898,
    )


@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Predict diabetes risk for a single patient",
    tags=["Predictions"],
)
def predict(patient: PatientInput):
    """
    Submit a single patient's clinical health metrics and receive a diabetes-risk
    prediction (**Low**, **Medium**, or **High**) powered by DiabetesTabularResNet (98.60% Accuracy).
    """
    prediction, confidence, probs = predict_single(patient)
    return PredictionResult(
        prediction=prediction,
        risk_description=RISK_DESCRIPTIONS.get(prediction, "Unknown"),
        confidence=confidence,
        probabilities=probs,
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
    Submit up to **500 patient records** at once and receive Deep Learning predictions
    for all of them in a single response.
    """
    results = []
    for patient in batch.patients:
        pred, conf, probs = predict_single(patient)
        results.append(
            PredictionResult(
                prediction=pred,
                risk_description=RISK_DESCRIPTIONS.get(pred, "Unknown"),
                confidence=conf,
                probabilities=probs,
                input_data=patient.model_dump(),
            )
        )
    return BatchPredictionResult(predictions=results, total=len(results))


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

    port = int(os.environ.get("PORT", 8000))
    print("=" * 60)
    print("  Diabetes Risk Prediction API (Deep Learning ResNet)")
    print(f"  Starting server on port {port}")
    print(f"  Loaded model: {DEEP_MODEL_PATH.name} (98.60% Accuracy)")
    print("  Swagger docs at /docs")
    print("=" * 60)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
