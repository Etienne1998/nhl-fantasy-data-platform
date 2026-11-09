
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_PATH = Path("data/modeling/splits/train.csv")
VALIDATION_PATH = Path("data/modeling/splits/validation.csv")
OUTPUT_PATH = Path("data/modeling/baseline_results.csv")

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


def calculate_metrics(actual, predictions):
    errors = actual - predictions
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    return mae, rmse


def evaluate_baselines():

    print("Loading datasets...")

    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    results = []

    print("\nEvaluating baselines...")

    for target in TARGETS:

        training_mean = train_df[target].mean()
        actual = validation_df[target]

        # Baseline 1: Training average
        mean_predictions = pd.Series(
            training_mean,
            index=validation_df.index
        )

        mean_mae, mean_rmse = calculate_metrics(
            actual,
            mean_predictions
        )

        # Baseline 2: Previous 5-game average
        rolling_feature = f"{target}_last_5_avg"

        if rolling_feature in validation_df.columns:

            rolling_predictions = validation_df[
                rolling_feature
            ].fillna(training_mean)

            rolling_mae, rolling_rmse = calculate_metrics(
                actual,
                rolling_predictions
            )

        else:
            rolling_mae = np.nan
            rolling_rmse = np.nan

        results.append({
            "target": target,
            "training_mean": training_mean,
            "mean_baseline_mae": mean_mae,
            "mean_baseline_rmse": mean_rmse,
            "last_5_baseline_mae": rolling_mae,
            "last_5_baseline_rmse": rolling_rmse,
            "best_baseline": (
                "last_5"
                if pd.notna(rolling_mae) and rolling_mae < mean_mae
                else "training_mean"
            ),
        })

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 75)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 75)

    print(
        results_df[
            [
                "target",
                "mean_baseline_mae",
                "last_5_baseline_mae",
                "best_baseline",
            ]
        ].round(3).to_string(index=False)
    )

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_baselines()
