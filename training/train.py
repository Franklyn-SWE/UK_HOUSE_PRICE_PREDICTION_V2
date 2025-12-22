import json
from pathlib import Path

import dill
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    median_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

DATA_PATH = Path("data/UK_House_Price_Prediction_dataset_2015_to_2024.csv")
MODEL_PATH = Path("full_pipeline_and_model.pkl")
METRICS_PATH = Path("training/metrics.json")
TARGET_TRANSFORM = "log1p"

DATE_FEATURES = [
    "sale_year",
    "sale_month",
    "sale_quarter",
    "sale_dayofweek",
    "sale_is_month_end",
]

CATEGORICAL_FEATURES = [
    "postcode",
    "property_type",
    "new_build",
    "freehold",
    "street",
    "locality",
    "town",
    "district",
    "county",
]

NUMERIC_FEATURES = DATE_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_quarter"] = df["date"].dt.quarter
    df["sale_dayofweek"] = df["date"].dt.dayofweek
    df["sale_is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def train_test_split_by_date(df: pd.DataFrame):
    df = df.sort_values("date")
    train = df[df["date"].dt.year <= 2022]
    valid = df[df["date"].dt.year == 2023]
    test = df[df["date"].dt.year >= 2024]
    return train, valid, test


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["postcode"] = df["postcode"].astype(str).str.upper().str.strip()
    for flag in ["new_build", "freehold"]:
        df[flag] = (
            df[flag]
            .replace({"Y": 1, "N": 0, "y": 1, "n": 0, True: 1, False: 0})
        )
        df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)
    for column in [
        "property_type",
        "street",
        "locality",
        "town",
        "district",
        "county",
    ]:
        df[column] = df[column].fillna("UNKNOWN").astype(str).str.strip()
    return df


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = np.nanmean(np.abs((y_true - y_pred) / denom))
    y_pred_clipped = np.clip(y_pred, a_min=0, a_max=None)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred_clipped))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": mae,
        "median_ae": median_ae,
        "rmse": rmse,
        "mape": mape,
        "rmsle": rmsle,
        "r2": r2,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df = add_date_features(df)
    df = prepare_features(df)
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]

    train_df, valid_df, test_df = train_test_split_by_date(df)
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Train/valid/test split resulted in an empty dataset.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = np.log1p(train_df["price"])

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = np.log1p(valid_df["price"])

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["price"]
    baseline_prediction = float(train_df["price"].median())

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=1200,
        depth=8,
        learning_rate=0.05,
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=CATEGORICAL_FEATURES,
        use_best_model=True,
    )

    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log)

    metrics = evaluate(y_test, test_pred)
    baseline_metrics = evaluate(
        y_test, np.full_like(y_test, fill_value=baseline_prediction, dtype=float)
    )
    improvement_pct = {
        "mae": ((baseline_metrics["mae"] - metrics["mae"]) / baseline_metrics["mae"]) * 100,
        "median_ae": (
            (baseline_metrics["median_ae"] - metrics["median_ae"]) / baseline_metrics["median_ae"]
        )
        * 100,
    }
    metrics["baseline"] = {
        "prediction": baseline_prediction,
        "mae": baseline_metrics["mae"],
        "median_ae": baseline_metrics["median_ae"],
    }
    metrics["improvement_pct"] = improvement_pct
    metrics["rows"] = {
        "train": int(train_df.shape[0]),
        "valid": int(valid_df.shape[0]),
        "test": int(test_df.shape[0]),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    artifact = {
        "model": model,
        "target_transform": TARGET_TRANSFORM,
        "feature_config": {
            "date_features": DATE_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "feature_columns": FEATURE_COLUMNS,
            "target_transform": TARGET_TRANSFORM,
        },
    }

    with MODEL_PATH.open("wb") as f:
        dill.dump(artifact, f)


if __name__ == "__main__":
    main()
