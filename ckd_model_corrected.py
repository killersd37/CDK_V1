"""
CKD Benin - Pipeline ML corrigé et orienté usage clinique.

Ce script implémente deux sorties complémentaires:
1) Une classification ordinale des stades CKD (1..5).
2) Un score de risque de sévérité (CKD sévère: stade >= 3), utile pour priorisation.

Principes méthodologiques:
- Split train/test avant tout fit (anti-fuite).
- Prétraitements encapsulés dans un Pipeline/ColumnTransformer.
- Validation croisée stratifiée pour tuning.
- Calibration des probabilités pour l'usage clinique.
- Export des métriques, artefacts et agrégats géographiques.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1a4iZPf93nLejpL7d7LYnF9JliV10etsJS8ia-bw1MXg/export?format=csv"
)

# Variables exclues des features pour limiter la fuite.
LEAKAGE_COLUMNS = {
    "Stade_CKD",
    "stade_ckd",
    "eGFR",
    "egfr",
    "patient_id",
    "id_patient",
}

# Variables optionnelles utilisées pour agrégation géographique.
GEO_CANDIDATES = ["departement", "département", "commune", "ville", "region"]


@dataclass
class TrainOutputs:
    stage_model: CalibratedClassifierCV
    severity_model: CalibratedClassifierCV
    stage_metrics: dict[str, Any]
    severity_metrics: dict[str, Any]


def calculer_egfr_mdrd(creatinine: float, age: float, sexe: str) -> float:
    """Calcule eGFR via MDRD (créatinine mg/dL)."""
    egfr = 175.0 * (float(creatinine) ** -1.154) * (float(age) ** -0.203)
    if str(sexe).strip().upper().startswith("F"):
        egfr *= 0.742
    return float(egfr)


def determiner_stade_ckd(egfr: float) -> int:
    """Mappe eGFR vers stade CKD 1..5."""
    if egfr >= 90:
        return 1
    if egfr >= 60:
        return 2
    if egfr >= 30:
        return 3
    if egfr >= 15:
        return 4
    return 5


def ensure_stage_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit une cible de stade:
    - Utilise Stade_CKD/stade_ckd si présent.
    - Sinon construit la cible depuis créatinine + âge + sexe.
    """
    data = df.copy()

    if "Stade_CKD" in data.columns:
        data["Stade_CKD"] = pd.to_numeric(data["Stade_CKD"], errors="coerce").astype("Int64")
        data = data.dropna(subset=["Stade_CKD"]).copy()
        data["Stade_CKD"] = data["Stade_CKD"].astype(int)
        return data

    if "stade_ckd" in data.columns:
        data["Stade_CKD"] = pd.to_numeric(data["stade_ckd"], errors="coerce").astype("Int64")
        data = data.dropna(subset=["Stade_CKD"]).copy()
        data["Stade_CKD"] = data["Stade_CKD"].astype(int)
        return data

    required = ["creatinine", "age", "sexe"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour construire la cible: {missing}")

    data["eGFR"] = data.apply(
        lambda row: calculer_egfr_mdrd(
            creatinine=row["creatinine"],
            age=row["age"],
            sexe=row["sexe"],
        ),
        axis=1,
    )
    data["Stade_CKD"] = data["eGFR"].apply(determiner_stade_ckd).astype(int)
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construit X et y en retirant les colonnes fuyardes."""
    y = data["Stade_CKD"].astype(int)
    to_drop = [col for col in data.columns if col in LEAKAGE_COLUMNS]
    X = data.drop(columns=to_drop)
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Préprocesseur robuste pour variables numériques et catégorielles."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_calibrated_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[CalibratedClassifierCV, dict[str, Any]]:
    """Entraîne RF avec grid-search puis calibration isotonic."""
    preprocessor = make_preprocessor(X_train)

    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(random_state=random_state)),
        ]
    )

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
        "model__class_weight": ["balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(
        estimator=gs.best_estimator_,
        method="isotonic",
        cv=3,
    )
    calibrated.fit(X_train, y_train)

    return calibrated, gs.best_params_


def evaluate_stage_model(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Métriques pour tâche ordinale multi-classes (stades)."""
    return {
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mae_ordinal": float(mean_absolute_error(y_true, y_pred)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ),
        "classification_report": classification_report(y_true, y_pred, digits=3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_severity_model(
    y_true_binary: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Métriques pour score de risque binaire (CKD sévère >= stade 3)."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f1": float(f1_score(y_true_binary, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_binary, y_pred)),
        "auroc": float(roc_auc_score(y_true_binary, y_proba)),
        "auprc": float(average_precision_score(y_true_binary, y_proba)),
        "brier": float(brier_score_loss(y_true_binary, y_proba)),
        "classification_report": classification_report(y_true_binary, y_pred, digits=3),
        "confusion_matrix": confusion_matrix(y_true_binary, y_pred).tolist(),
    }


def build_geo_risk_table(
    source_df: pd.DataFrame,
    severity_proba: np.ndarray,
) -> pd.DataFrame:
    """Construit un tableau de priorisation géographique (moyenne risque par zone)."""
    geo_col = next((c for c in GEO_CANDIDATES if c in source_df.columns), None)
    if geo_col is None:
        return pd.DataFrame(
            {
                "message": [
                    "Aucune colonne géographique trouvée (departement/commune/ville...)."
                ]
            }
        )

    geo_df = pd.DataFrame(
        {
            "zone": source_df[geo_col].astype(str),
            "risk_score": severity_proba,
        }
    )
    return (
        geo_df.groupby("zone", as_index=False)
        .agg(nb_patients=("risk_score", "size"), risque_moyen=("risk_score", "mean"))
        .sort_values("risque_moyen", ascending=False)
    )


def train_project(
    url: str = DATA_URL,
    out_dir: str = "artifacts",
    random_state: int = 42,
) -> TrainOutputs:
    """Pipeline principal: charge, entraîne, évalue et sauvegarde les artefacts."""
    df_raw = pd.read_csv(url)
    df = ensure_stage_target(df_raw)

    X, y_stage = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_stage,
        test_size=0.2,
        random_state=random_state,
        stratify=y_stage,
    )

    # Modèle 1: Prédiction de stade (ordinal multiclass)
    stage_model, stage_best_params = train_calibrated_random_forest(
        X_train, y_train, random_state=random_state
    )
    stage_pred = stage_model.predict(X_test)
    stage_metrics = evaluate_stage_model(y_test, stage_pred)
    stage_metrics["best_params"] = stage_best_params

    # Modèle 2: Score de risque de sévérité (stade >= 3)
    y_bin = (y_stage >= 3).astype(int)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X,
        y_bin,
        test_size=0.2,
        random_state=random_state,
        stratify=y_bin,
    )
    severity_model, severity_best_params = train_calibrated_random_forest(
        X_train_b, y_train_b, random_state=random_state
    )
    proba_test = severity_model.predict_proba(X_test_b)[:, 1]
    severity_metrics = evaluate_severity_model(y_test_b, proba_test, threshold=0.5)
    severity_metrics["best_params"] = severity_best_params

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "stage_model_calibrated.pkl").open("wb") as f:
        pickle.dump(stage_model, f)

    with (out_path / "severity_model_calibrated.pkl").open("wb") as f:
        pickle.dump(severity_model, f)

    with (out_path / "stage_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(stage_metrics, f, ensure_ascii=False, indent=2)

    with (out_path / "severity_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(severity_metrics, f, ensure_ascii=False, indent=2)

    # Table de priorisation géographique basée sur score de sévérité.
    full_severity_proba = severity_model.predict_proba(X)[:, 1]
    geo_table = build_geo_risk_table(df, full_severity_proba)
    geo_table.to_csv(out_path / "geo_priorisation.csv", index=False)

    metadata = {
        "n_patients": int(len(df)),
        "n_features": int(X.shape[1]),
        "classes_stage": sorted(int(c) for c in np.unique(y_stage)),
        "target_severity_definition": "severe_ckd = 1 if stage >= 3 else 0",
        "leakage_columns_removed": sorted([c for c in LEAKAGE_COLUMNS if c in df.columns]),
    }
    with (out_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return TrainOutputs(
        stage_model=stage_model,
        severity_model=severity_model,
        stage_metrics=stage_metrics,
        severity_metrics=severity_metrics,
    )


def infer_patient(
    stage_model: CalibratedClassifierCV,
    severity_model: CalibratedClassifierCV,
    patient: dict[str, Any],
) -> dict[str, Any]:
    """Inférence patient alignée au pipeline entraîné."""
    x_new = pd.DataFrame([patient])

    stage_pred = int(stage_model.predict(x_new)[0])
    stage_proba = stage_model.predict_proba(x_new)[0]
    stage_classes = [int(c) for c in stage_model.classes_]

    sev_proba = float(severity_model.predict_proba(x_new)[0, 1])

    # eGFR calculé en support clinique si variables disponibles.
    egfr = None
    if {"creatinine", "age", "sexe"}.issubset(set(x_new.columns)):
        egfr = round(
            calculer_egfr_mdrd(
                creatinine=float(x_new.iloc[0]["creatinine"]),
                age=float(x_new.iloc[0]["age"]),
                sexe=str(x_new.iloc[0]["sexe"]),
            ),
            2,
        )

    return {
        "stade_predit": stage_pred,
        "proba_stade": {str(c): float(p) for c, p in zip(stage_classes, stage_proba)},
        "score_risque_severe": sev_proba,
        "priorite": "élevée" if sev_proba >= 0.7 else "moyenne" if sev_proba >= 0.4 else "faible",
        "eGFR_calcule_support": egfr,
    }


if __name__ == "__main__":
    outputs = train_project()
    print("=== Stage metrics ===")
    print(json.dumps(outputs.stage_metrics, ensure_ascii=False, indent=2))
    print("=== Severity metrics ===")
    print(json.dumps(outputs.severity_metrics, ensure_ascii=False, indent=2))
