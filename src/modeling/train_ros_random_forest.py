
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


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "modeling" / "ros"
SPLIT_DIR = DATA_DIR / "splits"

MODEL_DIR = ROOT_DIR / "models" / "ros_random_forest"
RESULTS_PATH = DATA_DIR / "ros_random_forest_results.csv"
BASELINE_PATH = DATA_DIR / "ros_baseline_results.csv"
METADATA_PATH = SPLIT_DIR / "metadata.json"

N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 10
N_JOBS = 2
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("Loading ROS datasets...")

train = pd.read_csv(SPLIT_DIR / "train.csv")
validation = pd.read_csv(SPLIT_DIR / "validation.csv")

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

features = metadata["features"]
numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
targets = metadata["targets"]

baseline = pd.read_csv(BASELINE_PATH)
baseline = baseline.set_index("target")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Training records: {len(train)}")
print(f"Validation records: {len(validation)}")
print(f"Features: {len(features)}")
print(f"Targets: {len(targets)}")


# ============================================================
# LOAD PREVIOUS RESULTS (RESUME SUPPORT)
# ============================================================

if RESULTS_PATH.exists():
    previous_results = pd.read_csv(RESULTS_PATH)
    results_by_target = {
        row["target"]: row.to_dict()
        for _, row in previous_results.iterrows()
    }
else:
    results_by_target = {}


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# TRAIN ONE MODEL PER TARGET
# ============================================================

for target in targets:

    model_path = MODEL_DIR / f"{target}_model.joblib"

    # Skip models already completed
    if target in results_by_target and model_path.exists():
        print(f"\nSkipping {target} — already completed.")
        continue

    print("\n" + "=" * 60)
    print(f"Training model: {target}")
    print("=" * 60)

    start_time = time.perf_counter()

    X_train = train[features]
    y_train = train[target]

    X_validation = validation[features]
    y_validation = validation[target]

    # Build pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_validation)

    # Evaluate
    mae = mean_absolute_error(y_validation, predictions)
    rmse = mean_squared_error(
        y_validation,
        predictions
    ) ** 0.5

    baseline_mae = baseline.loc[target, "best_baseline_mae"]

    improvement_pct = (
        (baseline_mae - mae) / baseline_mae
    ) * 100

    elapsed_time = time.perf_counter() - start_time

    # Save model
    joblib.dump(model, model_path)

    # Save results
    results_by_target[target] = {
        "target": target,
        "rf_mae": mae,
        "rf_rmse": rmse,
        "baseline_mae": baseline_mae,
        "improvement_pct": improvement_pct,
        "training_time_seconds": elapsed_time
    }

    results_df = pd.DataFrame(
        [
            results_by_target[t]
            for t in targets
            if t in results_by_target
        ]
    )

    results_df.to_csv(RESULTS_PATH, index=False)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Improvement: {improvement_pct:+.2f}%")
    print(f"Training time: {elapsed_time:.1f} seconds")
    print(f"Model saved: {model_path}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ROS RANDOM FOREST TRAINING COMPLETED")
print("=" * 60)

final_results = pd.DataFrame(
    [
        results_by_target[t]
        for t in targets
        if t in results_by_target
    ]
)

print(
    final_results[
        ["target", "rf_mae", "baseline_mae", "improvement_pct"]
    ].to_string(index=False)
)

print(f"\nResults saved to: {RESULTS_PATH}")
