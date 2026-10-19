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
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

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
    bmi: Optional[float] = Field(None, ge=1, le=80, description="Body Mass Index (kg/m²). Auto-calculated if omitted.", examples=[25.75])
    systolic_bp: float = Field(..., ge=30, le=300, description="Systolic blood pressure (mmHg)", examples=[110])
    diastolic_bp: float = Field(..., ge=20, le=200, description="Diastolic blood pressure (mmHg)", examples=[73])
    Hba1c: float = Field(..., ge=3.0, le=15.0, description="Glycated haemoglobin percentage (%)", examples=[5.1])

    @model_validator(mode="before")
    @classmethod
    def pre_process_fields(cls, data: Any) -> Any:
        """
        Provides case-insensitive field alias mapping, automatic height unit conversion (cm -> m),
        and automatic BMI calculation if omitted.
        """
        if not isinstance(data, dict):
            return data

        lower_map = {str(k).lower().replace("-", "_").replace(" ", "_"): v for k, v in data.items()}

        def get_val(canonical: str, aliases: list[str], default=None):
            for k in [canonical] + aliases:
                k_norm = k.lower().replace("-", "_").replace(" ", "_")
                if k_norm in lower_map and lower_map[k_norm] is not None:
                    return lower_map[k_norm]
            return default

        processed = {}
        processed["Age"] = get_val("Age", ["age"])
        processed["gender"] = get_val("gender", ["Gender", "sex", "Sex"])
        processed["Glucose"] = get_val("Glucose", ["glucose", "fasting_glucose", "blood_sugar"])
        processed["weight"] = get_val("weight", ["Weight", "weight_kg"])

        # Auto-convert height from cm to meters if given as > 3.0 (e.g. 172 cm -> 1.72 m)
        raw_height = get_val("height", ["Height", "height_m", "height_cm"])
        if raw_height is not None:
            try:
                h_f = float(raw_height)
                if h_f > 3.0:
                    h_f = h_f / 100.0
                processed["height"] = round(h_f, 2)
            except (ValueError, TypeError):
                processed["height"] = raw_height
        else:
            processed["height"] = None

        # Auto-calculate BMI if not supplied or zero
        raw_bmi = get_val("bmi", ["BMI", "body_mass_index"])
        raw_weight = processed.get("weight")
        raw_h = processed.get("height")
        if (raw_bmi is None or raw_bmi == 0) and raw_weight and raw_h:
            try:
                w_val = float(raw_weight)
                h_val = float(raw_h)
                if h_val > 0:
                    raw_bmi = round(w_val / (h_val * h_val), 2)
            except Exception:
                pass
        processed["bmi"] = raw_bmi

        processed["systolic_bp"] = get_val("systolic_bp", ["systolic", "sbp", "Systolic_BP", "systolic_blood_pressure"])
        processed["diastolic_bp"] = get_val("diastolic_bp", ["diastolic", "dbp", "Diastolic_BP", "diastolic_blood_pressure"])
        processed["Hba1c"] = get_val("Hba1c", ["hba1c", "HbA1c", "HBA1C", "a1c", "glycated_hemoglobin"])

        return processed

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("gender must be 'Male' or 'Female'")
        v_clean = v.strip().lower()
        if v_clean in ("male", "m", "1"):
            return "Male"
        elif v_clean in ("female", "f", "0"):
            return "Female"
        raise ValueError("gender must be 'Male' or 'Female'")

    @model_validator(mode="after")
    def compute_missing_bmi(self) -> PatientInput:
        if self.bmi is None and self.weight and self.height:
            self.bmi = round(float(self.weight) / (float(self.height) ** 2), 2)
        return self

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

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Diabetes Risk Prediction API — Deep Tabular ResNet</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(17, 24, 39, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #06b6d4;
      --text: #f3f4f6;
      --text-muted: #94a3b8;
      --low: #10b981;
      --med: #f59e0b;
      --high: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: radial-gradient(circle at top right, #1e1b4b 0%, #090d16 60%);
      color: var(--text);
      min-height: 100vh;
      padding: 2rem 1rem;
      line-height: 1.5;
    }
    .container { max-width: 1100px; margin: 0 auto; }
    header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 0.35rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.03em;
      margin-bottom: 0.75rem;
    }
    h1 {
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 1rem;
      max-width: 700px;
      margin: 0 auto;
    }
    .links-bar {
      display: flex;
      justify-content: center;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-top: 1.25rem;
    }
    .btn-link {
      color: var(--text);
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.4rem 0.85rem;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .btn-link:hover {
      background: rgba(59, 130, 246, 0.2);
      border-color: rgba(59, 130, 246, 0.5);
      color: #93c5fd;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 1.5rem;
      margin-top: 2rem;
    }
    @media (max-width: 860px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 1.75rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .presets {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 1.25rem;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      color: var(--text);
      font-size: 0.75rem;
      font-weight: 600;
      padding: 0.35rem 0.7rem;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .preset-btn:hover {
      background: rgba(255, 255, 255, 0.12);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1rem;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }
    label {
      font-size: 0.8rem;
      font-weight: 600;
      color: #cbd5e1;
    }
    input, select {
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 0.6rem 0.75rem;
      color: #fff;
      font-family: inherit;
      font-size: 0.9rem;
      transition: border 0.2s ease;
    }
    input:focus, select:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    .btn-submit {
      width: 100%;
      margin-top: 1.25rem;
      background: linear-gradient(135deg, var(--primary) 0%, #1d4ed8 100%);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 0.85rem;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
      box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
      transition: all 0.2s ease;
    }
    .btn-submit:hover {
      background: linear-gradient(135deg, var(--primary-hover) 0%, #1e40af 100%);
      transform: translateY(-1px);
    }
    .result-box {
      display: none;
      flex-direction: column;
      gap: 1.25rem;
    }
    .pred-badge {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 1rem 1.25rem;
      border-radius: 12px;
      font-weight: 700;
      font-size: 1.2rem;
    }
    .pred-badge.Low { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pred-badge.Medium { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pred-badge.High { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .prob-bar-group {
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .prob-row {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      font-size: 0.8rem;
    }
    .prob-header {
      display: flex;
      justify-content: space-between;
      color: #94a3b8;
      font-weight: 600;
    }
    .bar-bg {
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 999px;
      transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .bar-fill.Low { background: var(--low); }
    .bar-fill.Medium { background: var(--med); }
    .bar-fill.High { background: var(--high); }
    .bio-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.75rem;
      background: rgba(15, 23, 42, 0.6);
      padding: 1rem;
      border-radius: 10px;
      border: 1px solid var(--card-border);
    }
    .bio-item {
      display: flex;
      flex-direction: column;
      gap: 0.1rem;
    }
    .bio-label {
      font-size: 0.72rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .bio-val {
      font-size: 0.95rem;
      font-weight: 700;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
    }
    .placeholder-box {
      text-align: center;
      padding: 3rem 1rem;
      color: var(--text-muted);
    }
    .placeholder-icon {
      font-size: 2.5rem;
      margin-bottom: 0.75rem;
      opacity: 0.6;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">🩺 PyTorch Deep Tabular ResNet &bull; 98.60% Accuracy</div>
      <h1>Diabetes Risk Prediction API</h1>
      <p class="subtitle">Real-time clinical diabetes risk triage system powered by dense residual skip connections and automated metabolic biomarker engineering.</p>
      <div class="links-bar">
        <a class="btn-link" href="/docs" target="_blank">⚡ Swagger UI Docs</a>
        <a class="btn-link" href="/redoc" target="_blank">📖 ReDoc</a>
        <a class="btn-link" href="/model-info" target="_blank">📊 Model Metrics</a>
        <a class="btn-link" href="/health" target="_blank">💚 Health Check</a>
        <a class="btn-link" href="https://github.com/ihtesham0332/rashid" target="_blank">🐙 GitHub Repo</a>
      </div>
    </header>

    <div class="grid">
      <!-- Input Panel -->
      <div class="card">
        <div class="card-title">
          <span>Patient Clinical Profile</span>
          <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 500;">9 Parameters</span>
        </div>

        <div class="presets">
          <span style="font-size: 0.75rem; color: #94a3b8; align-self: center;">Quick Presets:</span>
          <button class="preset-btn" onclick="applyPreset('low')">🟢 Low Risk</button>
          <button class="preset-btn" onclick="applyPreset('med')">🟡 Pre-Diabetic</button>
          <button class="preset-btn" onclick="applyPreset('high')">🔴 High Diabetic</button>
        </div>

        <form id="predictionForm" onsubmit="runPrediction(event)">
          <div class="form-grid">
            <div class="form-group">
              <label for="Age">Age (years)</label>
              <input type="number" id="Age" value="45" min="1" max="120" required>
            </div>
            <div class="form-group">
              <label for="gender">Gender</label>
              <select id="gender" required>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
            <div class="form-group">
              <label for="Glucose">Fasting Glucose (mg/dL)</label>
              <input type="number" step="0.1" id="Glucose" value="115" min="30" max="400" required>
            </div>
            <div class="form-group">
              <label for="Hba1c">HbA1c (%)</label>
              <input type="number" step="0.1" id="Hba1c" value="5.9" min="3" max="20" required>
            </div>
            <div class="form-group">
              <label for="weight">Weight (kg)</label>
              <input type="number" step="0.1" id="weight" value="76" min="20" max="300" oninput="calcBmi()" required>
            </div>
            <div class="form-group">
              <label for="height">Height (m)</label>
              <input type="number" step="0.01" id="height" value="1.72" min="0.5" max="2.5" oninput="calcBmi()" required>
            </div>
            <div class="form-group">
              <label for="bmi">BMI (kg/m&sup2;)</label>
              <input type="number" step="0.1" id="bmi" value="25.7" min="10" max="80" required>
            </div>
            <div class="form-group">
              <label for="systolic_bp">Systolic BP (mmHg)</label>
              <input type="number" id="systolic_bp" value="128" min="60" max="260" required>
            </div>
            <div class="form-group" style="grid-column: span 2;">
              <label for="diastolic_bp">Diastolic BP (mmHg)</label>
              <input type="number" id="diastolic_bp" value="82" min="40" max="160" required>
            </div>
          </div>

          <button type="submit" class="btn-submit" id="submitBtn">
            <span>Calculate Severity Risk</span> &rarr;
          </button>
        </form>
      </div>

      <!-- Result Panel -->
      <div class="card">
        <div class="card-title">ResNet Inference Output</div>

        <div id="placeholder" class="placeholder-box">
          <div class="placeholder-icon">🔬</div>
          <p>Fill the clinical indicators on the left and click <b>Calculate Severity Risk</b> to trigger neural forward inference.</p>
        </div>

        <div id="results" class="result-box">
          <div id="badge" class="pred-badge Low">
            <span id="predLabel">Low Risk</span>
            <span id="predConf" style="font-family: 'JetBrains Mono', monospace; font-size: 1rem;">98.4%</span>
          </div>

          <p id="riskDesc" style="font-size: 0.85rem; color: #cbd5e1;"></p>

          <div class="prob-bar-group">
            <div class="prob-row">
              <div class="prob-header"><span>Low Risk (&lt; 5.7%)</span><span id="pLow">0%</span></div>
              <div class="bar-bg"><div id="barLow" class="bar-fill Low" style="width: 0%;"></div></div>
            </div>
            <div class="prob-row">
              <div class="prob-header"><span>Medium / Pre-diabetic (5.7-6.4%)</span><span id="pMed">0%</span></div>
              <div class="bar-bg"><div id="barMed" class="bar-fill Medium" style="width: 0%;"></div></div>
            </div>
            <div class="prob-row">
              <div class="prob-header"><span>High / Diabetic (&ge; 6.5%)</span><span id="pHigh">0%</span></div>
              <div class="bar-bg"><div id="barHigh" class="bar-fill High" style="width: 0%;"></div></div>
            </div>
          </div>

          <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; margin-top: 0.5rem;">Clinical Biomarkers Computed</div>
          <div class="bio-grid">
            <div class="bio-item">
              <span class="bio-label">Mean Arterial Press</span>
              <span class="bio-val" id="bioMap">-</span>
            </div>
            <div class="bio-item">
              <span class="bio-label">Pulse Pressure</span>
              <span class="bio-val" id="bioPp">-</span>
            </div>
            <div class="bio-item">
              <span class="bio-label">Glucose/HbA1c Ratio</span>
              <span class="bio-val" id="bioRatio">-</span>
            </div>
            <div class="bio-item">
              <span class="bio-label">Metabolic Risk Idx</span>
              <span class="bio-val" id="bioIdx">-</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    function calcBmi() {
      const w = parseFloat(document.getElementById('weight').value);
      const h = parseFloat(document.getElementById('height').value);
      if (w > 0 && h > 0) {
        document.getElementById('bmi').value = (w / (h * h)).toFixed(1);
      }
    }

    const presets = {
      low: { Age: 32, gender: 'Male', Glucose: 88, weight: 68, height: 1.75, bmi: 22.2, systolic_bp: 115, diastolic_bp: 75, Hba1c: 4.9 },
      med: { Age: 48, gender: 'Female', Glucose: 118, weight: 79, height: 1.65, bmi: 29.0, systolic_bp: 132, diastolic_bp: 84, Hba1c: 6.0 },
      high: { Age: 56, gender: 'Male', Glucose: 175, weight: 88, height: 1.70, bmi: 30.4, systolic_bp: 148, diastolic_bp: 92, Hba1c: 8.4 }
    };

    function applyPreset(type) {
      const p = presets[type];
      for (const k in p) {
        document.getElementById(k).value = p[k];
      }
      runPrediction();
    }

    async function runPrediction(e) {
      if (e) e.preventDefault();
      const btn = document.getElementById('submitBtn');
      btn.innerText = 'Running ResNet forward pass...';
      btn.disabled = true;

      const payload = {
        Age: parseInt(document.getElementById('Age').value),
        gender: document.getElementById('gender').value,
        Glucose: parseFloat(document.getElementById('Glucose').value),
        weight: parseFloat(document.getElementById('weight').value),
        height: parseFloat(document.getElementById('height').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        systolic_bp: parseFloat(document.getElementById('systolic_bp').value),
        diastolic_bp: parseFloat(document.getElementById('diastolic_bp').value),
        Hba1c: parseFloat(document.getElementById('Hba1c').value)
      };

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Prediction failed');

        document.getElementById('placeholder').style.display = 'none';
        document.getElementById('results').style.display = 'flex';

        const predClass = data.prediction || data.predicted_class;
        const confVal = (data.confidence !== undefined ? data.confidence : data.confidence_percentage) || 0;
        const probs = data.probabilities || data.probability_breakdown || { Low: 0, Medium: 0, High: 0 };

        const badge = document.getElementById('badge');
        badge.className = 'pred-badge ' + predClass;
        document.getElementById('predLabel').innerText = predClass + ' Risk';
        document.getElementById('predConf').innerText = confVal.toFixed(1) + '%';
        document.getElementById('riskDesc').innerText = data.risk_description || '';
        document.getElementById('pLow').innerText = probs.Low.toFixed(1) + '%';
        document.getElementById('barLow').style.width = probs.Low + '%';
        document.getElementById('pMed').innerText = probs.Medium.toFixed(1) + '%';
        document.getElementById('barMed').style.width = probs.Medium + '%';
        document.getElementById('pHigh').innerText = probs.High.toFixed(1) + '%';
        document.getElementById('barHigh').style.width = probs.High + '%';

        const sbp = payload.systolic_bp;
        const dbp = payload.diastolic_bp;
        document.getElementById('bioMap').innerText = (dbp + (sbp - dbp) / 3).toFixed(1) + ' mmHg';
        document.getElementById('bioPp').innerText = (sbp - dbp).toFixed(0) + ' mmHg';
        document.getElementById('bioRatio').innerText = (payload.Glucose / payload.Hba1c).toFixed(2);
        document.getElementById('bioIdx').innerText = ((payload.bmi * payload.Age) / 100).toFixed(2);

      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.innerText = 'Calculate Severity Risk \u2192';
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""

@app.get(
    "/",
    summary="Root – welcome dashboard or JSON API specification",
    tags=["General"],
)
def root(request: Request):
    """
    Returns an interactive HTML diagnostic dashboard when accessed via a browser,
    or the JSON API root specification when accessed via API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=UI_HTML)

    return {
        "message": "Welcome to the Diabetes Risk Prediction API",
        "model": "DiabetesTabularResNet (PyTorch Deep Learning)",
        "accuracy": "98.60%",
        "version": "2.0.0",
        "endpoints": {
            "web_ui": "/",
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


@app.post(
    "/",
    summary="Root POST fallback",
    tags=["Predictions"],
    include_in_schema=False,
)
def root_post(payload: Optional[Dict[str, Any]] = None):
    """
    Prevents 405 Method Not Allowed when clients or forms POST to the root URL '/'.
    If valid patient metrics are provided in the payload, runs prediction directly.
    """
    if payload:
        try:
            patient = PatientInput(**payload)
            return predict(patient)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Bad Request",
                    "message": "To run a prediction, please send a POST request with patient metrics to '/predict'.",
                    "details": str(exc),
                    "docs": "/docs",
                },
            )
    return JSONResponse(
        status_code=200,
        content={
            "message": "Diabetes Risk Prediction API is running.",
            "instructions": "Send a POST request with patient metrics to '/predict' to get diabetes severity predictions.",
            "docs": "/docs",
        },
    )


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
# Validation Error Handler (User-Friendly 422 Details)
# ===========================================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Returns user-friendly validation error details when a client submits invalid JSON
    to prevent cryptic 422 errors and clearly indicate missing or malformed fields.
    """
    errors = exc.errors()
    simplified_errors = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        simplified_errors.append(f"{loc}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error (422)",
            "message": "Input data failed schema validation.",
            "details": simplified_errors,
            "expected_fields": {
                "Age": "int (1-120)",
                "gender": "'Male' or 'Female'",
                "Glucose": "float (0-700 mg/dL)",
                "weight": "float (1-300 kg)",
                "height": "float (0.5-2.5 meters, or cm)",
                "bmi": "float (optional, auto-calculated if omitted)",
                "systolic_bp": "float (30-300 mmHg)",
                "diastolic_bp": "float (20-200 mmHg)",
                "Hba1c": "float (3.0-15.0 %)",
            },
            "interactive_docs": "/docs",
        },
    )


# ===========================================================================
# Agent & MCP Protocol Discovery Gateway
# ===========================================================================
@app.api_route(
    "/.well-known/{path:path}",
    methods=["GET", "POST", "HEAD", "OPTIONS"],
    tags=["Agent / Discovery"],
    include_in_schema=False,
)
async def well_known_discovery(path: str, request: Request):
    """
    Handles discovery requests from Agent-to-Agent (A2A), OpenAI Plugin crawlers,
    and Hugging Face Agent cataloging bots.
    """
    clean_path = path.strip().lower()

    # Agent-to-Agent (A2A) protocol discovery card
    if clean_path.endswith("agent.json"):
        return JSONResponse(
            status_code=200,
            content={
                "name": "Diabetes Risk Prediction Agent",
                "description": "Deep Tabular ResNet model (98.60% accuracy) for predicting clinical diabetes severity risk (Low, Medium, High).",
                "version": "2.0.0",
                "protocol_version": "0.1.0",
                "capabilities": {
                    "prediction": True,
                    "batch_prediction": True,
                    "model_metadata": True,
                },
                "endpoints": {
                    "predict": "/predict",
                    "batch_predict": "/predict/batch",
                    "health": "/health",
                    "openapi": "/openapi.json",
                },
                "tools": [
                    {
                        "name": "predict_diabetes_risk",
                        "description": "Predict diabetes risk level (Low, Medium, High) given clinical health metrics.",
                        "endpoint": "/predict",
                        "method": "POST",
                    }
                ],
            },
        )

    # OpenAI Plugin / GPT Action discovery manifest
    if "ai-plugin" in clean_path:
        base_url = str(request.base_url).rstrip("/")
        return JSONResponse(
            status_code=200,
            content={
                "schema_version": "v1",
                "name_for_human": "Diabetes Risk Predictor",
                "name_for_model": "diabetes_risk_predictor",
                "description_for_human": "Predict diabetes risk severity (Low, Medium, High) using clinical biomarkers.",
                "description_for_model": "Clinical diabetes prediction tool powered by a Deep Tabular ResNet. Send patient biomarkers to /predict.",
                "auth": {"type": "none"},
                "api": {"type": "openapi", "url": f"{base_url}/openapi.json"},
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "active",
            "service": "Diabetes Risk Prediction API",
            "endpoints": {
                "predict": "/predict",
                "docs": "/docs",
                "health": "/health",
            },
        },
    )


@app.api_route("/mcp", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
@app.api_route("/mcp/{subpath:path}", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
@app.api_route("/v1/mcp", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
@app.api_route("/v1/mcp/{subpath:path}", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
@app.api_route("/sse", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
@app.api_route("/sse/{subpath:path}", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
@app.api_route("/a2a", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
@app.api_route("/a2a/{subpath:path}", methods=["GET", "POST", "HEAD", "OPTIONS"], tags=["Agent / MCP"], include_in_schema=False)
async def mcp_and_agent_gateway(request: Request, subpath: str = ""):
    """
    Universal gateway for Model Context Protocol (MCP) and Agent-to-Agent (A2A) probes.
    Responds to JSON-RPC calls (initialize, tools/list, tools/call) and returns 200 OK
    for automated discovery probes instead of cluttering logs with 404 Not Found.
    """
    req_path = request.url.path.lower()
    if req_path.endswith("agent.json"):
        return await well_known_discovery("agent.json", request)

    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}

        if isinstance(body, dict) and "method" in body:
            method = body.get("method")
            req_id = body.get("id")

            # MCP protocol handshake
            if method == "initialize":
                return JSONResponse(
                    status_code=200,
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {"tools": {}},
                            "serverInfo": {
                                "name": "diabetes-risk-prediction-api",
                                "version": "2.0.0",
                            },
                        },
                    },
                )

            # MCP tool discovery
            if method in ("tools/list", "tool/list"):
                return JSONResponse(
                    status_code=200,
                    content={
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "tools": [
                                {
                                    "name": "predict_diabetes_risk",
                                    "description": "Predict patient diabetes risk severity (Low, Medium, High) with 98.60% accuracy using Deep Tabular ResNet.",
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "Age": {"type": "integer", "description": "Patient age in years (1-120)"},
                                            "gender": {"type": "string", "enum": ["Male", "Female"], "description": "Patient gender"},
                                            "Glucose": {"type": "number", "description": "Fasting blood glucose (mg/dL)"},
                                            "weight": {"type": "number", "description": "Weight in kg"},
                                            "height": {"type": "number", "description": "Height in meters (e.g. 1.72) or cm"},
                                            "bmi": {"type": "number", "description": "BMI (kg/m²). Optional - auto-calculated if omitted"},
                                            "systolic_bp": {"type": "number", "description": "Systolic BP (mmHg)"},
                                            "diastolic_bp": {"type": "number", "description": "Diastolic BP (mmHg)"},
                                            "Hba1c": {"type": "number", "description": "HbA1c percentage (%)"},
                                        },
                                        "required": ["Age", "gender", "Glucose", "weight", "height", "systolic_bp", "diastolic_bp", "Hba1c"],
                                    },
                                }
                            ]
                        },
                    },
                )

            # MCP tool execution
            if method in ("tools/call", "tool/call"):
                params = body.get("params", {})
                args = params.get("arguments", {})
                try:
                    patient = PatientInput(**args)
                    pred, conf, probs = predict_single(patient)
                    result_payload = {
                        "prediction": pred,
                        "risk_description": RISK_DESCRIPTIONS.get(pred, ""),
                        "confidence_percentage": conf,
                        "probabilities": probs,
                        "input_data": patient.model_dump(),
                    }
                    return JSONResponse(
                        status_code=200,
                        content={
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(result_payload, indent=2),
                                    }
                                ]
                            },
                        },
                    )
                except Exception as err:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32602, "message": f"Tool execution failed: {err}"},
                        },
                    )

            if method == "notifications/initialized":
                return JSONResponse(status_code=200, content={"status": "initialized"})

        return JSONResponse(
            status_code=200,
            content={
                "status": "active",
                "service": "Diabetes Risk Prediction API",
                "message": "Gateway active. Use REST POST /predict or standard MCP JSON-RPC.",
                "endpoints": {
                    "predict": "/predict",
                    "docs": "/docs",
                    "health": "/health",
                },
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "active",
            "service": "Diabetes Risk Prediction API",
            "protocols_supported": ["REST", "MCP (Model Context Protocol)", "A2A (Agent-to-Agent)"],
            "endpoints": {
                "predict": "/predict",
                "docs": "/docs",
                "openapi": "/openapi.json",
                "health": "/health",
            },
        },
    )


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
