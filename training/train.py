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
    "years_since_2015",  # NEW: Capture overall time trend
]

CATEGORICAL_FEATURES = [
    "postcode",
    "postcode_area",      # NEW: Extract postcode area (e.g., "LE17" from "LE17 5AP")
    "postcode_district",  # NEW: First part only (e.g., "LE")
    "property_type",
    "new_build",
    "freehold",
    "street",
    "locality",
    "town",
    "district",
    "county",
]

# NEW: Target encoding features for high-cardinality categoricals
# NOTE: Removed postcode_mean_price - too granular, causes overfitting
TARGET_ENCODED_FEATURES = [
    "town_mean_price",
    "district_mean_price",
    "county_mean_price",
    "property_type_mean_price",
]

NUMERIC_FEATURES = DATE_FEATURES + TARGET_ENCODED_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["sale_quarter"] = df["date"].dt.quarter
    df["sale_dayofweek"] = df["date"].dt.dayofweek
    df["sale_is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["years_since_2015"] = df["sale_year"] - 2015  # Linear time trend
    return df


def add_postcode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hierarchical postcode information"""
    df = df.copy()
    df["postcode"] = df["postcode"].astype(str).str.upper().str.strip()
    
    # Extract postcode area (e.g., "SW1A" from "SW1A 1AA")
    df["postcode_area"] = df["postcode"].str.split().str[0]
    
    # Extract postcode district (e.g., "SW" from "SW1A 1AA")
    df["postcode_district"] = df["postcode"].str.extract(r'^([A-Z]+)', expand=False)
    
    return df


def add_target_encoding(train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> tuple:
    """Add target-encoded features based on training data"""
    # Calculate mean prices from training data only
    # NOTE: Excluding postcode - too granular, causes overfitting
    encoding_map = {}
    
    for col in ["town", "district", "county", "property_type"]:
        # Calculate mean with smoothing (add overall mean for rare categories)
        global_mean = train_df["price"].mean()
        
        # Group by category and calculate mean
        col_means = train_df.groupby(col)["price"].agg(["mean", "count"]).reset_index()
        
        # Apply smoothing: weight by count (minimum 10 samples for full weight)
        smoothing_factor = 10
        col_means["smoothed_mean"] = (
            (col_means["mean"] * col_means["count"] + global_mean * smoothing_factor) /
            (col_means["count"] + smoothing_factor)
        )
        
        # CRITICAL FIX: Apply log1p transformation to match target scale
        col_means["smoothed_mean_log"] = np.log1p(col_means["smoothed_mean"])
        
        encoding_map[col] = dict(zip(col_means[col], col_means["smoothed_mean_log"]))
    
    # Global mean in log scale for fallback
    global_mean_log = np.log1p(train_df["price"].mean())
    
    # Apply encoding to all datasets
    for df_set in [train_df, valid_df, test_df]:
        df_set["town_mean_price"] = df_set["town"].map(
            encoding_map["town"]
        ).fillna(global_mean_log)
        
        df_set["district_mean_price"] = df_set["district"].map(
            encoding_map["district"]
        ).fillna(global_mean_log)
        
        df_set["county_mean_price"] = df_set["county"].map(
            encoding_map["county"]
        ).fillna(global_mean_log)
        
        df_set["property_type_mean_price"] = df_set["property_type"].map(
            encoding_map["property_type"]
        ).fillna(global_mean_log)
    
    return train_df, valid_df, test_df, encoding_map


def train_test_split_by_date(df: pd.DataFrame):
    df = df.sort_values("date")
    train = df[df["date"].dt.year <= 2022]
    valid = df[df["date"].dt.year == 2023]
    test = df[df["date"].dt.year >= 2024]
    return train, valid, test


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
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
        df[column] = df[column].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    
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
    print("="*70)
    print("UK HOUSE PRICE PREDICTION - IMPROVED MODEL")
    print("="*70)
    
    df = pd.read_csv(DATA_PATH)
    print(f"\nLoaded {len(df):,} records")
    
    df = add_date_features(df)
    df = add_postcode_features(df)
    df = prepare_features(df)
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]
    
    # Filter outliers
    df = df[(df["price"] >= 10000) & (df["price"] <= 10_000_000)]
    print(f"After filtering: {len(df):,} records")

    train_df, valid_df, test_df = train_test_split_by_date(df)
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Train/valid/test split resulted in an empty dataset.")
    
    print(f"Train: {len(train_df):,} | Valid: {len(valid_df):,} | Test: {len(test_df):,}")

    # Add target encoding features
    print("\nCreating target-encoded features...")
    train_df, valid_df, test_df, encoding_map = add_target_encoding(
        train_df.copy(), valid_df.copy(), test_df.copy()
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = np.log1p(train_df["price"])

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = np.log1p(valid_df["price"])

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["price"]
    baseline_prediction = float(train_df["price"].median())

    # IMPROVED: Better hyperparameters
    print("\nTraining CatBoost model with improved hyperparameters...")
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=2000,           # Increased from 1200
        depth=8,                   # Keep at 8 to avoid overfitting with strong features
        learning_rate=0.03,        # Decreased for better generalization
        l2_leaf_reg=10,            # Increased regularization
        bagging_temperature=1.0,   # Default - less aggressive
        random_strength=1.0,       # Default - less aggressive
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=100, # Increased patience
        verbose=100,
        task_type="CPU",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=CATEGORICAL_FEATURES,
        use_best_model=True,
    )

    # Get predictions
    test_pred_log = model.predict(X_test)
    test_pred = np.expm1(test_pred_log)

    # Evaluate
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
    
    # Feature importance
    feature_importance = dict(zip(
        model.feature_names_,
        model.feature_importances_
    ))
    top_features = dict(sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:15])
    
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
    metrics["top_features"] = top_features
    
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"R² Score:       {metrics['r2']:.4f}")
    print(f"RMSE:          £{metrics['rmse']:,.0f}")
    print(f"MAE:           £{metrics['mae']:,.0f}")
    print(f"Median AE:     £{metrics['median_ae']:,.0f}")
    print(f"MAPE:          {metrics['mape']*100:.2f}%")
    print(f"RMSLE:         {metrics['rmsle']:.4f}")
    print(f"\nBaseline MAE:  £{baseline_metrics['mae']:,.0f}")
    print(f"Improvement:   {improvement_pct['mae']:.1f}%")
    print(f"{'='*70}")
    
    print(f"\nTop 10 Most Important Features:")
    for i, (feat, imp) in enumerate(list(top_features.items())[:10], 1):
        print(f"{i:2d}. {feat:35s} {imp:6.2f}")

    artifact = {
        "model": model,
        "target_transform": TARGET_TRANSFORM,
        "encoding_map": encoding_map,
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
    
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Metrics saved to: {METRICS_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()
