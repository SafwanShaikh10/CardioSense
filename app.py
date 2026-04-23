"""CardioSense FastAPI backend.

Endpoints
---------
GET  /health                  — liveness check
POST /predict                 — prediction only (fast)
POST /explain/shap            — prediction + SHAP explanation
POST /explain/lime            — prediction + LIME explanation
POST /explain/both            — prediction + SHAP + LIME (default for frontend)
GET  /features                — feature metadata (labels, ranges) for the UI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Import everything from the model module (must be in same directory)
from model import (
    CATEGORICAL_LABELS,
    FEATURE_DISPLAY_NAMES,
    ModelBundle,
    ensure_model_ready,
    explain_lime,
    explain_shap,
    predict_patient,
    predict_with_both_explanations,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CardioSense API",
    description="Explainable cardiovascular risk assessment powered by an ensemble ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parent / "Cardiovascular_Disease_Dataset.csv"
_bundle: Optional[ModelBundle] = None


@app.on_event("startup")
def load_model() -> None:
    global _bundle
    print("Loading / training model…")
    _bundle = ensure_model_ready(DATASET_PATH)
    print("Model ready.")


def get_bundle() -> ModelBundle:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    return _bundle


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PatientData(BaseModel):
    age: float = Field(..., ge=20, le=80, description="Age in years")
    gender: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male")
    chestpain: int = Field(..., ge=0, le=3, description="0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic")
    restingBP: float = Field(..., ge=80, le=220, description="Resting blood pressure (mmHg)")
    serumcholestrol: float = Field(..., ge=0, le=602, description="Serum cholesterol (mg/dl). 0 = missing, will be imputed.")
    fastingbloodsugar: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl: 0 = No, 1 = Yes")
    restingrelectro: int = Field(..., ge=0, le=2, description="0=Normal, 1=ST-T abnormality, 2=LVH")
    maxheartrate: float = Field(..., ge=60, le=210, description="Maximum heart rate achieved (bpm)")
    exerciseangia: int = Field(..., ge=0, le=1, description="Exercise-induced angina: 0 = No, 1 = Yes")
    oldpeak: float = Field(..., ge=0.0, le=6.5, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=3, description="0=Upsloping, 1=Flat, 2=Downsloping, 3=N/A")
    noofmajorvessels: int = Field(..., ge=0, le=3, description="Major vessels coloured by fluoroscopy (0-3)")

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()


class FeatureContribution(BaseModel):
    feature: str
    display_name: str
    value: str
    importance: float
    direction: str          # "increases" | "decreases"
    shap_value: Optional[float] = None
    lime_weight: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int         # 0 = No CVD, 1 = CVD
    probability: float      # CVD probability (0–1)
    risk_level: str         # "Low" | "Moderate" | "High"
    risk_percent: int


class ExplainResponse(PredictionResponse):
    method: str
    base_value: Optional[float] = None
    feature_importance: List[FeatureContribution]


class BothExplainResponse(PredictionResponse):
    shap: ExplainResponse
    lime: ExplainResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_level(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    if prob >= 0.40:
        return "Moderate"
    return "Low"


def _format_explain(raw: Dict[str, Any]) -> ExplainResponse:
    features = [
        FeatureContribution(
            feature=f["feature"],
            display_name=f["display_name"],
            value=f["value"],
            importance=f["importance"],
            direction=f["direction"],
            shap_value=f.get("shap_value"),
            lime_weight=f.get("lime_weight"),
        )
        for f in raw["feature_importance"]
    ]
    prob = raw["probability"]
    return ExplainResponse(
        prediction=raw["prediction"],
        probability=prob,
        risk_level=_risk_level(prob),
        risk_percent=round(prob * 100),
        method=raw["method"],
        base_value=raw.get("base_value"),
        feature_importance=features,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Utility"])
def health():
    return {"status": "ok", "model_loaded": _bundle is not None}


@app.get("/features", tags=["Utility"])
def features():
    """Return feature metadata — used by the frontend to build the question flow."""
    return {
        "feature_display_names": FEATURE_DISPLAY_NAMES,
        "categorical_labels": {
            k: {str(idx): label for idx, label in v.items()}
            for k, v in CATEGORICAL_LABELS.items()
        },
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(patient: PatientData):
    """Fast prediction — no explanation, minimal latency."""
    bundle = get_bundle()
    result = predict_patient(patient.to_dict(), bundle)
    prob = result["probability"]
    return PredictionResponse(
        prediction=result["prediction"],
        probability=prob,
        risk_level=_risk_level(prob),
        risk_percent=round(prob * 100),
    )


@app.post("/explain/shap", response_model=ExplainResponse, tags=["Inference"])
def explain_shap_endpoint(patient: PatientData):
    """Prediction with SHAP feature-attribution explanation."""
    bundle = get_bundle()
    try:
        raw = explain_shap(patient.to_dict(), bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP explanation failed: {exc}")
    return _format_explain(raw)


@app.post("/explain/lime", response_model=ExplainResponse, tags=["Inference"])
def explain_lime_endpoint(patient: PatientData):
    """Prediction with LIME local linear explanation."""
    bundle = get_bundle()
    try:
        raw = explain_lime(patient.to_dict(), bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LIME explanation failed: {exc}")
    return _format_explain(raw)


@app.post("/explain/both", response_model=BothExplainResponse, tags=["Inference"])
def explain_both_endpoint(patient: PatientData):
    """Prediction with both SHAP and LIME explanations.
    This is the primary endpoint used by the CardioSense frontend.
    """
    bundle = get_bundle()
    try:
        raw = predict_with_both_explanations(patient.to_dict(), bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}")

    prob = raw["probability"]
    return BothExplainResponse(
        prediction=raw["prediction"],
        probability=prob,
        risk_level=raw["risk_level"],
        risk_percent=round(prob * 100),
        shap=_format_explain(raw["shap"]),
        lime=_format_explain(raw["lime"]),
    )
