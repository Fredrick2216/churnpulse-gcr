"""Random Forest churn pipeline used by the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEFAULT_RF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 5,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "random_state": 42,
}

ID_HINTS = ("id", "customerid", "customer_id", "userid", "user_id", "index")
TARGET_HINTS = (
    "churn",
    "exited",
    "attrition",
    "attrited",
    "left",
    "target",
    "label",
    "outcome",
    "class",
)


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    raw: pd.DataFrame
    feature_schema: list[dict[str, Any]]
    encoders: dict[str, LabelEncoder]
    target_name: str
    dropped_columns: list[str]
    class_names: list[str]
    positive_label: int


@dataclass
class TrainResult:
    model: RandomForestClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, float]
    confusion: np.ndarray
    report: dict[str, Any]
    feature_importance: pd.DataFrame
    train_accuracy: float
    test_accuracy: float
    accuracy_gap: float
    params: dict[str, Any] = field(default_factory=dict)


def load_table(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "upload").lower()
    uploaded_file.seek(0)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def detect_target_column(df: pd.DataFrame) -> str | None:
    lookup = {str(col).lower().replace(" ", "").replace("_", ""): col for col in df.columns}
    for hint in TARGET_HINTS:
        for key, original in lookup.items():
            if hint == key or hint in key:
                return original
    binary = [col for col in df.columns if df[col].nunique(dropna=True) == 2]
    if len(binary) == 1:
        return binary[0]
    return None


def generate_demo_dataset(n: int = 1800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.normal(64, 26, n).clip(18, 145), 2)
    tickets = rng.integers(0, 8, n)
    age = rng.integers(19, 80, n)
    senior = (age >= 65).astype(int)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n,
        p=[0.55, 0.27, 0.18],
    )
    payment = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        n,
        p=[0.38, 0.27, 0.22, 0.13],
    )
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.48, 0.35, 0.17])
    tech = rng.choice(["Yes", "No"], n, p=[0.36, 0.64])
    paperless = rng.choice(["Yes", "No"], n, p=[0.62, 0.38])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    streaming = rng.choice(["Yes", "No"], n, p=[0.44, 0.56])
    total = np.round(monthly * tenure * rng.uniform(0.82, 1.08, n), 2)

    logit = (
        -1.35
        + 1.55 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        + 0.95 * (payment == "Electronic check")
        + 0.018 * (monthly - 55)
        - 0.035 * tenure
        + 0.28 * tickets
        + 0.55 * (tech == "No")
        + 0.32 * (internet == "Fiber optic")
        + 0.22 * senior
        - 0.18 * (partner == "Yes")
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n) < prob, "Yes", "No")

    return pd.DataFrame(
        {
            "CustomerID": [f"C{100000 + i}" for i in range(n)],
            "Age": age,
            "SeniorCitizen": senior,
            "Partner": partner,
            "TenureMonths": tenure,
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": tech,
            "StreamingTV": streaming,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "SupportTickets": tickets,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        }
    )


def inspect_dataset(df: pd.DataFrame, target_name: str | None = None) -> dict[str, Any]:
    dtypes = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Missing": df.isnull().sum().values,
            "Unique": [df[c].nunique(dropna=True) for c in df.columns],
        }
    )
    numeric = df.select_dtypes(include=["number"])
    summary = numeric.describe().T if not numeric.empty else pd.DataFrame()
    target_name = target_name or detect_target_column(df)
    churn_rate = None
    if target_name and target_name in df.columns:
        mapped = _map_churn_series(df[target_name])
        churn_rate = float(mapped.mean())
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": dtypes,
        "missing": df.isnull().sum().rename("Missing").to_frame(),
        "summary": summary,
        "head": df.head(8),
        "churn_rate": churn_rate,
        "target_name": target_name,
    }


def _map_churn_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    if pd.api.types.is_numeric_dtype(cleaned):
        numeric = pd.to_numeric(cleaned, errors="coerce")
        uniques = set(numeric.dropna().unique().tolist())
        if uniques <= {0, 1, 0.0, 1.0}:
            return numeric.fillna(0).astype(int)
        if len(uniques) == 2:
            positive = max(uniques)
            return numeric.eq(positive).astype(int)
        return (numeric.fillna(numeric.median()) > numeric.median()).astype(int)

    text = cleaned.astype(str).str.strip()
    lowered = text.str.lower()
    positive = {
        "yes",
        "y",
        "true",
        "1",
        "churn",
        "churned",
        "leave",
        "left",
        "exited",
        "attrited",
        "attrited customer",
        "positive",
    }
    if lowered.isin(positive).any():
        return lowered.isin(positive).astype(int)
    uniques = [value for value in text.dropna().unique().tolist() if value.lower() != "nan"]
    if len(uniques) == 2:
        ranked = sorted(uniques, key=lambda value: any(token in value.lower() for token in ("churn", "exit", "attrit", "leave", "yes")))
        return text.eq(ranked[-1]).astype(int)
    return pd.Series(0, index=series.index, dtype=int)


def map_target_series(series: pd.Series) -> pd.Series:
    return _map_churn_series(series)


def _is_id_column(name: str, series: pd.Series) -> bool:
    key = name.lower().replace(" ", "")
    if any(hint in key for hint in ID_HINTS):
        return True
    if series.nunique(dropna=True) == len(series) and not pd.api.types.is_numeric_dtype(series):
        return True
    return False


def prepare_features(df: pd.DataFrame, target_name: str = "Churn") -> PreparedData:
    if target_name not in df.columns:
        raise ValueError(f'Target column "{target_name}" was not found in the file.')

    working = df.copy()
    dropped = [c for c in working.columns if c != target_name and _is_id_column(c, working[c])]
    working = working.drop(columns=dropped)

    y = _map_churn_series(working[target_name])
    X_raw = working.drop(columns=[target_name])

    for col in X_raw.columns:
        if X_raw[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                X_raw[col] = X_raw[col].fillna(X_raw[col].median())
            else:
                mode = X_raw[col].mode()
                X_raw[col] = X_raw[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    encoders: dict[str, LabelEncoder] = {}
    schema: list[dict[str, Any]] = []
    X = pd.DataFrame(index=X_raw.index)

    for col in X_raw.columns:
        series = X_raw[col]
        if pd.api.types.is_numeric_dtype(series):
            X[col] = series.astype(float)
            schema.append(
                {
                    "name": col,
                    "kind": "numeric",
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median()),
                }
            )
        else:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(series.astype(str))
            encoders[col] = encoder
            X[col] = encoded.astype(float)
            schema.append(
                {
                    "name": col,
                    "kind": "categorical",
                    "options": encoder.classes_.tolist(),
                    "default": str(series.mode().iloc[0]),
                }
            )

    return PreparedData(
        X=X,
        y=y,
        raw=X_raw,
        feature_schema=schema,
        encoders=encoders,
        target_name=target_name,
        dropped_columns=dropped,
        class_names=["Stay", "Churn"],
        positive_label=1,
    )


def train_random_forest(
    prepared: PreparedData,
    params: dict[str, Any] | None = None,
    test_size: float = 0.20,
) -> TrainResult:
    cfg = {**DEFAULT_RF_PARAMS, **(params or {})}
    X, y = prepared.X, prepared.y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=cfg.get("random_state", 42),
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=int(cfg["n_estimators"]),
        criterion=str(cfg["criterion"]),
        max_depth=int(cfg["max_depth"]) if cfg["max_depth"] else None,
        min_samples_split=int(cfg["min_samples_split"]),
        min_samples_leaf=int(cfg["min_samples_leaf"]),
        random_state=int(cfg["random_state"]),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else np.zeros(len(y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    importance = (
        pd.DataFrame(
            {
                "Feature": X.columns,
                "Importance": model.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    labels = sorted(int(v) for v in y.unique())
    names = [prepared.class_names[i] if i < len(prepared.class_names) else str(i) for i in labels]
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )

    return TrainResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics={
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        confusion=confusion_matrix(y_test, y_pred, labels=[0, 1]),
        report=report,
        feature_importance=importance,
        train_accuracy=float(train_accuracy),
        test_accuracy=float(test_accuracy),
        accuracy_gap=float(train_accuracy - test_accuracy),
        params=cfg,
    )


def encode_customer_row(payload: dict[str, Any], prepared: PreparedData) -> pd.DataFrame:
    row = {}
    for field in prepared.feature_schema:
        name = field["name"]
        value = payload[name]
        if field["kind"] == "categorical":
            encoder = prepared.encoders[name]
            label = str(value)
            if label not in encoder.classes_:
                raise ValueError(f'Unknown value "{label}" for {name}.')
            row[name] = float(encoder.transform([label])[0])
        else:
            row[name] = float(value)
    return pd.DataFrame([row], columns=prepared.X.columns)


def predict_customer(
    model: RandomForestClassifier,
    payload: dict[str, Any],
    prepared: PreparedData,
) -> dict[str, Any]:
    encoded = encode_customer_row(payload, prepared)
    proba = float(model.predict_proba(encoded)[0, 1]) if len(model.classes_) > 1 else 0.0
    pred = int(model.predict(encoded)[0])
    contributions = (
        pd.DataFrame(
            {
                "Feature": prepared.X.columns,
                "Importance": model.feature_importances_,
                "Value": encoded.iloc[0].values,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "label": "Churn" if pred == 1 else "Stay",
        "prediction": pred,
        "churn_probability": proba,
        "stay_probability": 1.0 - proba,
        "risk_band": _risk_band(proba),
        "contributions": contributions,
        "encoded": encoded,
    }


def _risk_band(prob: float) -> str:
    if prob >= 0.70:
        return "Critical"
    if prob >= 0.45:
        return "Elevated"
    if prob >= 0.25:
        return "Watch"
    return "Stable"
