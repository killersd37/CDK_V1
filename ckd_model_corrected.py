"""
Pipeline corrigé pour la prédiction de stades CKD.

Points clés:
- Prétraitements encapsulés dans un Pipeline/ColumnTransformer (anti-fuite).
- Split stratifié AVANT tout fit.
- GridSearchCV uniquement sur train.
- Calibration des probabilités (CalibratedClassifierCV).
- Évaluation multi-métriques (dont métriques ordinales).
- Inférence alignée avec le pipeline entraîné.
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
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1a4iZPf93nLejpL7d7LYnF9JliV10etsJS8ia-bw1MXg/export?format=csv"
)


def calculer_egfr(creatinine: float, age: float, sexe: str) -> float:
    """Calcule eGFR (MDRD) à partir de créatinine (mg/dL), âge, sexe."""
    egfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
    if str(sexe).upper().startswith("F"):
        egfr *= 0.742
    return float(egfr)


def determiner_stade(egfr: float) -> int:
    """Détermine un stade 1..5 à partir de eGFR."""
    if egfr >= 90:
        return 1
    if egfr >= 60:
        return 2
    if egfr >= 30:
        return 3
    if egfr >= 15:
        return 4
    return 5


def ensure_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantit la présence de la cible 'Stade_CKD'.
    Si absente, la construit via eGFR calculé (MDRD).
    """
    out = df.copy()
    if "Stade_CKD" in out.columns:
        return out

    required = {"creatinine", "age", "sexe"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            "Impossible de construire 'Stade_CKD'. Colonnes manquantes: "
            f"{sorted(missing)}"
        )

    out["eGFR"] = out.apply(
        lambda r: calculer_egfr(r["creatinine"], r["age"], r["sexe"]), axis=1
    )
    out["Stade_CKD"] = out["eGFR"].apply(determiner_stade).astype(int)
    return out


@dataclass
class TrainArtifacts:
    pipeline: Pipeline
    metrics: dict[str, Any]
    feature_columns: list[str]


class CKDStageModel:
    """Modèle de classification des stades CKD avec pipeline complet."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.classes_: np.ndarray | None = None

    def _build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                    ]),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]),
                    categorical_features,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        model = RandomForestClassifier(random_state=self.random_state)

        return Pipeline([
            ("preprocess", preprocess),
            ("model", model),
        ])

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        base_pipeline = self._build_pipeline(X_train)

        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
            "model__class_weight": ["balanced"],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        search.fit(X_train, y_train)

        # Calibration sur le meilleur pipeline trouvé
        calibrated = CalibratedClassifierCV(
            estimator=search.best_estimator_,
            method="isotonic",
            cv=3,
        )
        calibrated.fit(X_train, y_train)

        self.pipeline = calibrated
        self.classes_ = np.array(sorted(pd.unique(y_train)))
        return search

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Le modèle n'est pas entraîné.")
        return self.pipeline.predict_proba(X)


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Évalue avec métriques standards + ordinales."""
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


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construit X/y en évitant les colonnes de fuite évidente."""
    leak_cols = {"Stade_CKD", "eGFR", "patient_id"}
    available_leaks = [c for c in leak_cols if c in df.columns]

    y = df["Stade_CKD"].astype(int)
    X = df.drop(columns=available_leaks)
    return X, y


def train_from_url(url: str = DATA_URL, out_dir: str = "artifacts") -> TrainArtifacts:
    """Entraîne et sauvegarde les artefacts sur un dataset distant CSV."""
    df_raw = pd.read_csv(url)
    df = ensure_target(df_raw)

    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = CKDStageModel(random_state=42)
    search = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    metrics["best_params"] = search.best_params_

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with (out_path / "ckd_pipeline_calibrated.pkl").open("wb") as f:
        pickle.dump(model.pipeline, f)

    with (out_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": int(len(df)),
                "n_features": int(X.shape[1]),
                "classes": sorted([int(c) for c in pd.unique(y)]),
                "best_params": search.best_params_,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with (out_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return TrainArtifacts(
        pipeline=model.pipeline,
        metrics=metrics,
        feature_columns=list(X.columns),
    )


def predire_stade_patient(
    trained_pipeline: Pipeline,
    age: float,
    sexe: str,
    creatinine: float,
    **autres_variables: Any,
) -> dict[str, Any]:
    """
    Prédit le stade CKD avec le pipeline entraîné (et non via règle hardcodée).
    Retourne aussi eGFR calculé pour interprétation clinique complémentaire.
    """
    row = {
        "age": age,
        "sexe": sexe,
        "creatinine": creatinine,
        **autres_variables,
    }
    X_new = pd.DataFrame([row])

    y_pred = int(trained_pipeline.predict(X_new)[0])
    proba = trained_pipeline.predict_proba(X_new)[0]
    classes = [int(c) for c in trained_pipeline.classes_]

    egfr = calculer_egfr(creatinine=creatinine, age=age, sexe=sexe)

    return {
        "stade_predit": y_pred,
        "probabilites_par_stade": {
            str(cls): float(p) for cls, p in zip(classes, proba, strict=True)
        },
        "eGFR_calcule": round(egfr, 2),
    }


if __name__ == "__main__":
    artifacts = train_from_url()
    print("Entraînement terminé.")
    print(json.dumps(artifacts.metrics, indent=2, ensure_ascii=False))
