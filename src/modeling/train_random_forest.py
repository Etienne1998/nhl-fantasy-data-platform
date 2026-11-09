
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TRAIN_PATH = Path("data/modeling/splits/train.csv")
VALIDATION_PATH = Path("data/modeling/splits/validation.csv")
METADATA_PATH = Path("data/modeling/training_metadata.json")
BASELINE_PATH = Path("data/modeling/baseline_results.csv")

MODEL_DIR = Path("models/random_forest")
RESULTS_PATH = Path("data/modeling/random_forest_results.csv")

TARGETS = [
    "goals",
    "assists",
    "shots_on_goal",
    "plus_minus",
    "penalty_minutes",
    "power_play_goals",
    "power_play_points",
    "short_handed_goals",
    "short_handed_points",
    "game_winning_goals",
    "time_on_ice_seconds",
]

N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 10
N_JOBS = 2


def train_models():

    print("Loading datasets...", flush=True)

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["features"]
    categorical_features = metadata["categorical_features"]

    numeric_features = [
        column for column in feature_columns
        if column not in categorical_features
    ]

    X_train = train_df[feature_columns]
    X_validation = validation_df[feature_columns]

    print(f"Training records: {len(train_df)}", flush=True)
    print(f"Validation records: {len(validation_df)}", flush=True)
    print(f"Features: {len(feature_columns)}", flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results to allow resuming
    if RESULTS_PATH.exists():
        results_df = pd.read_csv(RESULTS_PATH)
        results = results_df.to_dict("records")
    else:
        results = []

    completed_targets = {result["target"] for result in results}

    for target in TARGETS:

        if target in completed_targets:
            print(f"\nSkipping {target}: already completed.", flush=True)
            continue

        print(f"\nTraining model: {target}", flush=True)
        start_time = time.time()

        y_train = train_df[target]
        y_validation = validation_df[target]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=42,
                n_jobs=N_JOBS
            ))
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        mae = mean_absolute_error(y_validation, predictions)
        rmse = mean_squared_error(y_validation, predictions) ** 0.5

        model_path = MODEL_DIR / f"{target}_model.joblib"
        joblib.dump(model, model_path)

        results.append({
            "target": target,
            "random_forest_mae": mae,
            "random_forest_rmse": rmse,
        })

        # Save after every model
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)

        elapsed = time.time() - start_time

        print(f"MAE: {mae:.3f}", flush=True)
        print(f"RMSE: {rmse:.3f}", flush=True)
        print(f"Training time: {elapsed:.1f} seconds", flush=True)

    # Compare against baseline
    results_df = pd.DataFrame(results)
    baseline_df = pd.read_csv(BASELINE_PATH)

    results_df = results_df.merge(
        baseline_df,
        on="target",
        how="left"
    )

    best_baseline_mae = results_df["mean_baseline_mae"].where(
        results_df["best_baseline"] == "training_mean",
        results_df["last_5_baseline_mae"]
    )

    results_df["improvement_vs_best_baseline_pct"] = (
        (best_baseline_mae - results_df["random_forest_mae"])
        / best_baseline_mae
    ) * 100

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 65)
    print("RANDOM FOREST EVALUATION")
    print("=" * 65)

    print(
        results_df[
            [
                "target",
                "random_forest_mae",
                "best_baseline",
                "improvement_vs_best_baseline_pct",
            ]
        ].round(3).to_string(index=False)
    )

    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    train_models()
