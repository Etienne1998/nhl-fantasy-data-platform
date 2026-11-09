
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

SPLIT_DIR = ROOT_DIR / "data" / "modeling" / "ros" / "splits"

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "ros_availability"
    / "remaining_games_model.joblib"
)

OUTPUT_DIR = ROOT_DIR / "models" / "ros_availability"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation dataset...")

validation = pd.read_csv(SPLIT_DIR / "validation.csv")

model = joblib.load(MODEL_PATH)

features = list(model.feature_names_in_)

X_validation = validation[features]
y_validation = validation["remaining_games"]

predictions = model.predict(X_validation)

results = validation[
    ["cutoff_percentage", "remaining_games"]
].copy()

results["predicted_remaining_games"] = predictions

results["error"] = (
    results["predicted_remaining_games"] - results["remaining_games"]
)

results["absolute_error"] = results["error"].abs()


# ============================================================
# EVALUATE EACH CUTOFF
# ============================================================

print("\n" + "=" * 75)
print("AVAILABILITY MODEL PERFORMANCE BY CUTOFF")
print("=" * 75)

summary = []

for cutoff in sorted(results["cutoff_percentage"].unique()):

    group = results[
        results["cutoff_percentage"] == cutoff
    ]

    actual = group["remaining_games"]
    predicted = group["predicted_remaining_games"]

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    bias = (predicted - actual).mean()

    spearman = spearmanr(actual, predicted).statistic

    summary.append({
        "cutoff": cutoff,
        "players": len(group),
        "avg_actual_games": actual.mean(),
        "avg_predicted_games": predicted.mean(),
        "MAE": mae,
        "RMSE": rmse,
        "bias": bias,
        "spearman": spearman
    })


summary_df = pd.DataFrame(summary)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "avg_actual_games": "{:.2f}".format,
            "avg_predicted_games": "{:.2f}".format,
            "MAE": "{:.3f}".format,
            "RMSE": "{:.3f}".format,
            "bias": "{:.3f}".format,
            "spearman": "{:.3f}".format
        }
    )
)


# ============================================================
# ERROR DISTRIBUTION BY CUTOFF
# ============================================================

print("\n" + "=" * 75)
print("PREDICTION ERROR ANALYSIS")
print("=" * 75)

for cutoff in sorted(results["cutoff_percentage"].unique()):

    group = results[
        results["cutoff_percentage"] == cutoff
    ]

    print(f"\nCutoff: {cutoff:.0%}")

    print(f"Players: {len(group)}")

    print(
        f"Underestimated by 5+ games: "
        f"{(group['error'] <= -5).sum()}"
    )

    print(
        f"Overestimated by 5+ games: "
        f"{(group['error'] >= 5).sum()}"
    )

    print(
        f"Within 3 games: "
        f"{(group['absolute_error'] <= 3).mean():.1%}"
    )

    print(
        f"Within 5 games: "
        f"{(group['absolute_error'] <= 5).mean():.1%}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

summary_path = OUTPUT_DIR / "availability_by_cutoff.csv"

summary_df.to_csv(summary_path, index=False)

results_path = OUTPUT_DIR / "availability_validation_predictions.csv"

results.to_csv(results_path, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("=" * 75)

print(f"Summary saved to: {summary_path}")
print(f"Predictions saved to: {results_path}")
