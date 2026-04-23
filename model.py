"""Training and inference engine for cardiovascular risk assessment.

Enhancements over v1:
  - LIME tabular explainer alongside SHAP
  - Stratified k-fold cross-validation with AUC / F1 reporting
  - Class-weight balancing for the 58 / 42 positive / negative skew
  - Outlier capping for restingBP and serumcholestrol
  - Human-readable feature display names used by the API layer
  - predict_with_both_explanations() convenience wrapper
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "Cardiovascular_Disease_Dataset.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "ensemble_model.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.joblib"

TARGET_COLUMN = "target"
ID_COLUMN = "patientid"

# ---------------------------------------------------------------------------
# Human-readable feature labels (used by frontend and explanation layers)
# ---------------------------------------------------------------------------

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "age": "Age",
    "gender": "Biological Sex",
    "chestpain": "Chest Pain Type",
    "restingBP": "Resting Blood Pressure",
    "serumcholestrol": "Serum Cholesterol",
    "fastingbloodsugar": "Fasting Blood Sugar > 120 mg/dl",
    "restingrelectro": "Resting ECG Result",
    "maxheartrate": "Max Heart Rate Achieved",
    "exerciseangia": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "ST Slope",
    "noofmajorvessels": "Major Vessels (Fluoroscopy)",
}

# Categorical value labels for readable LIME/SHAP explanations
CATEGORICAL_LABELS: Dict[str, Dict[int, str]] = {
    "gender": {0: "Female", 1: "Male"},
    "chestpain": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Asymptomatic",
    },
    "fastingbloodsugar": {0: "No (≤120 mg/dl)", 1: "Yes (>120 mg/dl)"},
    "restingrelectro": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy",
    },
    "exerciseangia": {0: "No", 1: "Yes"},
    "slope": {
        0: "Upsloping",
        1: "Flat",
        2: "Downsloping",
        3: "Not Applicable",
    },
}

# Features that are categorical (used for LIME)
CATEGORICAL_FEATURE_NAMES: List[str] = list(CATEGORICAL_LABELS.keys())


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    """Runtime bundle containing the trained model and preprocessing state."""

    model: VotingClassifier
    scaler: StandardScaler
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sanitize_float(val: Any) -> float:
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return 0.0
        return fval
    except Exception:
        return 0.0


def readable_value(feature: str, raw_value: Any) -> str:
    """Return a human-readable representation of a feature value."""
    labels = CATEGORICAL_LABELS.get(feature)
    if labels is not None:
        return labels.get(int(raw_value), str(raw_value))
    if feature == "restingBP":
        return f"{int(raw_value)} mmHg"
    if feature == "serumcholestrol":
        return f"{int(raw_value)} mg/dl"
    if feature == "maxheartrate":
        return f"{int(raw_value)} bpm"
    if feature == "age":
        return f"{int(raw_value)} yrs"
    if feature == "oldpeak":
        return f"{float(raw_value):.1f}"
    return str(raw_value)


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def feature_columns_from_dataframe(dataframe: pd.DataFrame) -> List[str]:
    return [
        col for col in dataframe.columns if col not in {TARGET_COLUMN, ID_COLUMN}
    ]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# Clip thresholds derived from domain knowledge and EDA
_BP_MAX = 200
_CHOL_MAX = 564  # 99th-percentile cap to reduce outlier leverage
_HR_MIN = 60


def _cap_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme physiological values that inflate feature variance."""
    data = data.copy()
    data["restingBP"] = data["restingBP"].clip(upper=_BP_MAX)
    data["serumcholestrol"] = data["serumcholestrol"].clip(upper=_CHOL_MAX)
    data["maxheartrate"] = data["maxheartrate"].clip(lower=_HR_MIN)
    return data


def preprocess_dataframe(dataframe: pd.DataFrame) -> Dict[str, Any]:
    """Repair invalid values, cap outliers, and produce scaled train-test splits."""

    data = _cap_outliers(dataframe.copy())
    feature_columns = feature_columns_from_dataframe(data)

    # Impute zero cholesterol with the median of valid (non-zero) readings
    chol_median = float(data.loc[data["serumcholestrol"] > 0, "serumcholestrol"].median())
    data.loc[data["serumcholestrol"] == 0, "serumcholestrol"] = chol_median

    x = data[feature_columns]
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return {
        "feature_columns": feature_columns,
        "chol_median": chol_median,
        "x_train": x_train,
        "x_test": x_test,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_ensemble_model() -> VotingClassifier:
    """Build a soft-voting ensemble with class-weight balancing.

    The dataset has a mild positive-class skew (58 % CVD).  Passing
    ``class_weight='balanced'`` to the base estimators that support it keeps
    recall high for the minority class while XGBoost is tuned via
    ``scale_pos_weight``.
    """
    pos_weight = 420 / 580  # approx. neg / pos ratio

    return VotingClassifier(
        estimators=[
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    scale_pos_weight=pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    use_label_encoder=False,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_split=4,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ],
        voting="soft",
    )


# ---------------------------------------------------------------------------
# Cross-validation evaluation (optional — call before or after training)
# ---------------------------------------------------------------------------


def cross_validate_model(dataframe: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
    """Run stratified k-fold CV and return mean AUC / F1 / classification report."""

    processed = preprocess_dataframe(dataframe)
    feature_columns = processed["feature_columns"]

    # Re-scale the full dataset for CV
    data = _cap_outliers(dataframe.copy())
    chol_median = processed["chol_median"]
    data.loc[data["serumcholestrol"] == 0, "serumcholestrol"] = chol_median

    x = data[feature_columns].values
    y = data[TARGET_COLUMN].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores, f1_scores = [], []
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), 1):
        x_tr, x_val = x[train_idx], x[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        sc = StandardScaler()
        x_tr_sc = sc.fit_transform(x_tr)
        x_val_sc = sc.transform(x_val)

        m = build_ensemble_model()
        m.fit(x_tr_sc, y_tr)

        y_prob = m.predict_proba(x_val_sc)[:, 1]
        y_pred = m.predict(x_val_sc)

        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)

        auc_scores.append(auc)
        f1_scores.append(f1)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        print(f"  Fold {fold}: AUC={auc:.4f}  F1={f1:.4f}")

    mean_auc = float(np.mean(auc_scores))
    mean_f1 = float(np.mean(f1_scores))
    print(f"\nCV Summary — Mean AUC: {mean_auc:.4f}  Mean F1: {mean_f1:.4f}")
    print(classification_report(all_y_true, all_y_pred, target_names=["No CVD", "CVD"]))

    return {
        "mean_auc": mean_auc,
        "mean_f1": mean_f1,
        "auc_per_fold": auc_scores,
        "f1_per_fold": f1_scores,
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def train_and_save_artifacts(dataset_path: Path = DATASET_PATH) -> ModelBundle:
    """Train the ensemble and persist model, scaler, and metadata."""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = load_dataset(dataset_path)
    processed = preprocess_dataframe(dataframe)

    model = build_ensemble_model()
    model.fit(processed["x_train_scaled"], processed["y_train"])

    # Evaluate on held-out test set
    y_pred = model.predict(processed["x_test_scaled"])
    y_prob = model.predict_proba(processed["x_test_scaled"])[:, 1]
    print("=== Hold-out test set ===")
    print(classification_report(processed["y_test"], y_pred, target_names=["No CVD", "CVD"]))
    print(f"ROC-AUC: {roc_auc_score(processed['y_test'], y_prob):.4f}")

    # Identify categorical feature indices for LIME
    feature_cols = processed["feature_columns"]
    categorical_indices = [
        i for i, f in enumerate(feature_cols) if f in CATEGORICAL_FEATURE_NAMES
    ]

    metadata = {
        "feature_columns": feature_cols,
        "feature_display_names": {f: FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_cols},
        "categorical_labels": CATEGORICAL_LABELS,
        "categorical_indices": categorical_indices,
        "chol_median": processed["chol_median"],
        # SHAP: keep 100-row background sample from training set
        "background_data": processed["x_train_scaled"][:100],
        # LIME: keep unscaled training data as reference distribution
        "lime_training_data": processed["x_train"].values,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(processed["scaler"], SCALER_PATH)
    joblib.dump(metadata, METADATA_PATH)

    return ModelBundle(model=model, scaler=processed["scaler"], metadata=metadata)


def load_artifacts() -> ModelBundle:
    return ModelBundle(
        model=joblib.load(MODEL_PATH),
        scaler=joblib.load(SCALER_PATH),
        metadata=joblib.load(METADATA_PATH),
    )


def ensure_model_ready(dataset_path: Path = DATASET_PATH) -> ModelBundle:
    if MODEL_PATH.exists() and SCALER_PATH.exists() and METADATA_PATH.exists():
        return load_artifacts()
    return train_and_save_artifacts(dataset_path)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def normalize_patient_data(
    patient_data: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply inference-time repairs to incoming patient data."""
    normalized = dict(patient_data)
    # Cap outliers to match training-time transforms
    if float(normalized.get("restingBP", 0)) > _BP_MAX:
        normalized["restingBP"] = _BP_MAX
    if float(normalized.get("serumcholestrol", 0)) == 0:
        normalized["serumcholestrol"] = metadata["chol_median"]
    if float(normalized.get("serumcholestrol", 0)) > _CHOL_MAX:
        normalized["serumcholestrol"] = _CHOL_MAX
    return normalized


def dataframe_from_patient(
    patient_data: Dict[str, Any],
    feature_columns: List[str],
) -> pd.DataFrame:
    payload = {f: patient_data[f] for f in feature_columns}
    return pd.DataFrame([payload], columns=feature_columns)


def predict_patient(
    patient_data: Dict[str, Any],
    bundle: ModelBundle,
) -> Dict[str, Any]:
    normalized = normalize_patient_data(patient_data, bundle.metadata)
    input_frame = dataframe_from_patient(normalized, bundle.metadata["feature_columns"])
    transformed = bundle.scaler.transform(input_frame)
    prediction = int(bundle.model.predict(transformed)[0])
    probability = sanitize_float(bundle.model.predict_proba(transformed)[0, 1])
    return {
        "normalized_patient": normalized,
        "input_frame": input_frame,
        "transformed": transformed,
        "prediction": prediction,
        "probability": probability,
    }


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------


def explain_shap(
    patient_data: Dict[str, Any],
    bundle: ModelBundle,
) -> Dict[str, Any]:
    """Generate SHAP values and ranked feature importance for one patient."""

    prediction_payload = predict_patient(patient_data, bundle)
    background = np.asarray(bundle.metadata["background_data"], dtype=float)
    explainer = shap.Explainer(bundle.model.predict_proba, background)
    shap_result = explainer(prediction_payload["transformed"])
    positive_class = shap_result[..., 1]
    shap_values = positive_class.values[0]
    feature_names = bundle.metadata["feature_columns"]
    display_names = bundle.metadata.get("feature_display_names", {})

    ranked_features = sorted(
        [
            {
                "feature": feature,
                "display_name": display_names.get(feature, feature),
                "value": readable_value(feature, patient_data.get(feature, 0)),
                "shap_value": sanitize_float(sv),
                "importance": abs(sanitize_float(sv)),
                "direction": "increases" if sanitize_float(sv) > 0 else "decreases",
            }
            for feature, sv in zip(feature_names, shap_values)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    return {
        "method": "SHAP",
        "prediction": prediction_payload["prediction"],
        "probability": prediction_payload["probability"],
        "base_value": sanitize_float(
            np.asarray(positive_class.base_values).reshape(-1)[0]
        ),
        "shap_values": [sanitize_float(sv) for sv in shap_values],
        "feature_importance": ranked_features,
    }


# ---------------------------------------------------------------------------
# LIME explanation
# ---------------------------------------------------------------------------


def explain_lime(
    patient_data: Dict[str, Any],
    bundle: ModelBundle,
    num_features: int = 8,
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """Generate LIME local explanation for one patient.

    LIME fits a locally-weighted linear model around the query point using
    ``num_samples`` perturbed neighbours from the training distribution.
    The signed coefficients indicate direction and magnitude of each feature's
    local contribution to the positive-class probability.
    """

    prediction_payload = predict_patient(patient_data, bundle)
    feature_names = bundle.metadata["feature_columns"]
    display_names = bundle.metadata.get("feature_display_names", {})
    cat_indices = bundle.metadata.get("categorical_indices", [])

    # LIME operates in the original (unscaled) feature space
    lime_training_data = np.asarray(bundle.metadata["lime_training_data"], dtype=float)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=lime_training_data,
        feature_names=feature_names,
        categorical_features=cat_indices,
        class_names=["No CVD", "CVD"],
        mode="classification",
        random_state=42,
    )

    # We need a predict function that accepts unscaled inputs
    def predict_fn(x_raw: np.ndarray) -> np.ndarray:
        x_scaled = bundle.scaler.transform(x_raw)
        return bundle.model.predict_proba(x_scaled)

    input_raw = prediction_payload["input_frame"].values[0]
    explanation = explainer.explain_instance(
        data_row=input_raw,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(1,),
    )

    lime_map: Dict[str, float] = {
        feat: weight
        for feat, weight in explanation.as_map()[1]  # class index 1 = CVD
    }

    # Build ranked list using feature index → name mapping
    ranked_features = sorted(
        [
            {
                "feature": feature_names[idx],
                "display_name": display_names.get(feature_names[idx], feature_names[idx]),
                "value": readable_value(feature_names[idx], patient_data.get(feature_names[idx], 0)),
                "lime_weight": sanitize_float(lime_map.get(idx, 0.0)),
                "importance": abs(sanitize_float(lime_map.get(idx, 0.0))),
                "direction": "increases" if sanitize_float(lime_map.get(idx, 0.0)) > 0 else "decreases",
            }
            for idx in lime_map
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    return {
        "method": "LIME",
        "prediction": prediction_payload["prediction"],
        "probability": prediction_payload["probability"],
        "feature_importance": ranked_features,
        "intercept": sanitize_float(explanation.intercept[1]),
        "local_pred": sanitize_float(explanation.local_pred[0]),
        "score": sanitize_float(explanation.score),  # R² of the local linear fit
    }


# ---------------------------------------------------------------------------
# Combined convenience wrapper
# ---------------------------------------------------------------------------


def predict_with_both_explanations(
    patient_data: Dict[str, Any],
    bundle: ModelBundle,
    lime_num_features: int = 8,
) -> Dict[str, Any]:
    """Return prediction, SHAP explanation, and LIME explanation in one call.

    The SHAP result is global-kernel based (model-agnostic).
    The LIME result is a locally-weighted linear approximation.
    Having both lets the API layer present either or cross-validate them.
    """
    shap_result = explain_shap(patient_data, bundle)
    lime_result = explain_lime(patient_data, bundle, num_features=lime_num_features)

    return {
        "prediction": shap_result["prediction"],
        "probability": shap_result["probability"],
        "risk_level": (
            "High" if shap_result["probability"] >= 0.65
            else "Moderate" if shap_result["probability"] >= 0.40
            else "Low"
        ),
        "shap": shap_result,
        "lime": lime_result,
    }
