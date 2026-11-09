
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"
MODEL_DIR = ROOT / "models" / "ros_availability"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train = pd.read_csv(SPLIT_DIR / "train.csv")
validation = pd.read_csv(SPLIT_DIR / "validation.csv")
test = pd.read_csv(SPLIT_DIR / "test.csv")

with open(SPLIT_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

TARGET = "remaining_games"

print(f"Training records: {len(train)}")
print(f"Validation records: {len(validation)}")
print(f"Test records: {len(test)}")


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    feature for feature in FEATURES
    if feature != "position"
]

categorical_features = ["position"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining availability model...")

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=50,
                min_samples_leaf=10,
                n_jobs=2,
                random_state=42,
            ),
        ),
    ]
)

model.fit(
    train[FEATURES],
    train[TARGET],
)


# ============================================================
# EVALUATE
# ============================================================

def evaluate(dataset, name):

    actual = dataset[TARGET].to_numpy()

    predicted = model.predict(dataset[FEATURES])

    # Number of remaining games cannot be negative.
    predicted = np.maximum(predicted, 0)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"\n{name}")
    print(f"MAE: {mae:.3f} games")
    print(f"RMSE: {rmse:.3f} games")

    return predicted


validation_predictions = evaluate(validation, "VALIDATION")
test_predictions = evaluate(test, "TEST")


# ============================================================
# SAVE MODEL AND TEST PREDICTIONS
# ============================================================

joblib.dump(
    model,
    MODEL_DIR / "remaining_games_model.joblib",
)

test_results = test[
    [
        "player_id",
        "player_name",
        "team",
        "position",
        "season_id",
        "cutoff_date",
        "cutoff_percentage",
        TARGET,
    ]
].copy()

test_results["predicted_remaining_games"] = test_predictions

test_results.to_csv(
    MODEL_DIR / "availability_test_predictions.csv",
    index=False,
)

print("\nModel saved to:")
print(MODEL_DIR / "remaining_games_model.joblib")

print("\nTest predictions saved to:")
print(MODEL_DIR / "availability_test_predictions.csv")
