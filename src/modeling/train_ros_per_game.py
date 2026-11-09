
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
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT / "data" / "modeling" / "ros" / "splits"
FANTASY_DIR = ROOT / "data" / "modeling" / "ros" / "fantasy"
MODEL_DIR = ROOT / "models" / "ros_per_game"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

AVAILABILITY_MODEL_FILE = (
    ROOT / "models" / "ros_availability" /
    "remaining_games_model.joblib"
)

FANTASY_PREDICTIONS_FILE = (
    FANTASY_DIR / "ros_fantasy_player_predictions.csv"
)


# ============================================================
# TARGETS
# ============================================================

TARGETS = [
    "remaining_goals",
    "remaining_assists",
    "remaining_shots_on_goal",
    "remaining_plus_minus",
    "remaining_penalty_minutes",
    "remaining_power_play_goals",
    "remaining_power_play_points",
    "remaining_short_handed_goals",
    "remaining_short_handed_points",
    "remaining_game_winning_goals",
    "remaining_time_on_ice_seconds",
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train = pd.read_csv(SPLIT_DIR / "train.csv")
validation = pd.read_csv(SPLIT_DIR / "validation.csv")

with open(SPLIT_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

existing_predictions = pd.read_csv(
    FANTASY_PREDICTIONS_FILE
)

print(f"Training records: {len(train)}")
print(f"Validation records: {len(validation)}")


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    feature for feature in FEATURES
    if feature != "position"
]

categorical_features = ["position"]


def create_model():

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
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
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


# ============================================================
# TRAIN PER-GAME MODELS
# ============================================================

print("\nTraining per-game production models...")

production_models = {}
per_game_predictions = {}

for target in TARGETS:

    print(f"\nTraining: {target}")

    # Only players who played at least one future game.
    train_eligible = train[
        train["remaining_games"] > 0
    ].copy()

    # Production per actual game played.
    y_train = (
        train_eligible[target]
        / train_eligible["remaining_games"]
    )

    model = create_model()

    model.fit(
        train_eligible[FEATURES],
        y_train,
    )

    predictions = model.predict(
        validation[FEATURES]
    )

    # Non-negative statistics cannot have negative rates.
    if target != "remaining_plus_minus":
        predictions = np.maximum(predictions, 0)

    production_models[target] = model
    per_game_predictions[target] = predictions

    print(
        f"Training observations: {len(train_eligible)}"
    )


# ============================================================
# PREDICT REMAINING GAMES
# ============================================================

print("\nLoading availability model...")

availability_model = joblib.load(
    AVAILABILITY_MODEL_FILE
)

predicted_games = availability_model.predict(
    validation[FEATURES]
)

predicted_games = np.maximum(predicted_games, 0)


# ============================================================
# BUILD TWO-STAGE FANTASY PREDICTIONS
# ============================================================

print("\nCalculating two-stage Fantasy projections...")

results = validation[
    [
        "player_id",
        "player_name",
        "team",
        "position",
        "season_id",
        "cutoff_date",
        "cutoff_percentage",
        "remaining_games",
    ]
].copy()

results["predicted_remaining_games"] = predicted_games

for target in TARGETS:

    rate = per_game_predictions[target]

    results[f"pred_rate_{target}"] = rate

    results[f"pred_{target}"] = (
        predicted_games * rate
    )


# ============================================================
# CALCULATE FANTASY POINTS
# ============================================================

def calculate_fantasy_points(df, prefix):

    goals = df[f"{prefix}remaining_goals"]
    assists = df[f"{prefix}remaining_assists"]
    plus_minus = df[f"{prefix}remaining_plus_minus"]
    pim = df[f"{prefix}remaining_penalty_minutes"]

    ppg = df[f"{prefix}remaining_power_play_goals"]
    ppp = df[f"{prefix}remaining_power_play_points"]

    shg = df[f"{prefix}remaining_short_handed_goals"]
    shp = df[f"{prefix}remaining_short_handed_points"]

    gwg = df[f"{prefix}remaining_game_winning_goals"]
    shots = df[f"{prefix}remaining_shots_on_goal"]

    ppa = np.maximum(ppp - ppg, 0)
    sha = np.maximum(shp - shg, 0)

    return (
        goals * 2
        + assists
        + plus_minus
        + pim * 0.3
        + ppg
        + ppa * 0.5
        + shg * 2
        + sha
        + gwg
        + shots * 0.2
    )


results["pred_fantasy_points_two_stage"] = (
    calculate_fantasy_points(results, "pred_")
)


# ============================================================
# COMPARE WITH EXISTING RANDOM FOREST
# ============================================================

print("\nComparing with existing Fantasy Random Forest...")

keys = [
    "player_id",
    "season_id",
    "cutoff_date",
    "cutoff_percentage",
]

results["player_id"] = results["player_id"].astype(str)
existing_predictions["player_id"] = (
    existing_predictions["player_id"].astype(str)
)

results["season_id"] = results["season_id"].astype(str)
existing_predictions["season_id"] = (
    existing_predictions["season_id"].astype(str)
)

results["cutoff_date"] = pd.to_datetime(
    results["cutoff_date"]
)

existing_predictions["cutoff_date"] = pd.to_datetime(
    existing_predictions["cutoff_date"]
)

comparison = results.merge(
    existing_predictions[
        keys + [
            "pred_fantasy_points",
            "actual_fantasy_points",
        ]
    ],
    on=keys,
    how="inner",
    validate="one_to_one",
)

comparison["actual_remaining_games"] = (
    validation["remaining_games"].to_numpy()
)

print(f"Matched validation records: {len(comparison)}")


# ============================================================
# EVALUATION
# ============================================================

print("\n" + "=" * 75)
print("TWO-STAGE MODEL VS EXISTING RANDOM FOREST")
print("=" * 75)

evaluation_results = []

for cutoff, group in comparison.groupby("cutoff_percentage"):

    actual = group["actual_fantasy_points"]

    for model_name, column in [
        (
            "Existing Random Forest",
            "pred_fantasy_points",
        ),
        (
            "Two-stage model",
            "pred_fantasy_points_two_stage",
        ),
    ]:

        predicted = group[column]

        mae = mean_absolute_error(actual, predicted)

        rmse = np.sqrt(
            mean_squared_error(actual, predicted)
        )

        spearman = spearmanr(
            actual,
            predicted
        ).statistic

        actual_rank = actual.rank(
            method="min",
            ascending=False,
        )

        predicted_rank = predicted.rank(
            method="min",
            ascending=False,
        )

        top10 = (
            (actual_rank <= 10)
            & (predicted_rank <= 10)
        ).sum()

        top25 = (
            (actual_rank <= 25)
            & (predicted_rank <= 25)
        ).sum()

        top50 = (
            (actual_rank <= 50)
            & (predicted_rank <= 50)
        ).sum()

        evaluation_results.append({
            "cutoff": cutoff,
            "model": model_name,
            "players": len(group),
            "MAE": mae,
            "RMSE": rmse,
            "spearman": spearman,
            "top10_overlap": top10,
            "top25_overlap": top25,
            "top50_overlap": top50,
        })

evaluation_df = pd.DataFrame(evaluation_results)

print(
    evaluation_df.round(3).to_string(index=False)
)


# ============================================================
# SAVE MODELS AND RESULTS
# ============================================================

joblib.dump(
    production_models,
    MODEL_DIR / "per_game_models.joblib",
)

comparison.to_csv(
    MODEL_DIR / "two_stage_validation_predictions.csv",
    index=False,
)

evaluation_df.to_csv(
    MODEL_DIR / "two_stage_evaluation.csv",
    index=False,
)

print("\nModels and results saved to:")
print(MODEL_DIR)

print("\nExperiment completed.")
