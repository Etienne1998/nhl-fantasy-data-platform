
import json
import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_DIR = Path("models/random_forest")
OUTPUT_DIR = Path("data/modeling/analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Focus sur les deux modèles qui battent actuellement
# leur baseline.
FOCUS_TARGETS = [
    "shots_on_goal",
    "time_on_ice_seconds",
]


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------


def analyze_model(target):

    model_path = MODEL_DIR / f"{target}_model.joblib"

    print(f"\nAnalyzing: {target}")

    model = joblib.load(model_path)

    # Extract fitted preprocessing and Random Forest
    preprocessor = model.named_steps["preprocessor"]
    rf_model = model.named_steps["regressor"]

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Get Random Forest feature importance
    importance = rf_model.feature_importances_

    results = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    results = results.sort_values(
        "importance",
        ascending=False
    )

    results["importance_pct"] = (
        results["importance"] * 100
    )

    results.to_csv(
        OUTPUT_DIR / f"{target}_feature_importance.csv",
        index=False
    )

    print("\nTop 15 features:")

    print(
        results.head(15).to_string(
            index=False,
            formatters={
                "importance": "{:.5f}".format,
                "importance_pct": "{:.2f}%".format
            }
        )
    )

    return results



# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    all_results = {}

    for target in TARGETS:

        results = analyze_model(target)

        all_results[target] = results

    # Combine all models into one dataset
    combined = pd.concat(
        [
            results.assign(target=target)
            for target, results in all_results.items()
        ],
        ignore_index=True
    )

    combined.to_csv(
        OUTPUT_DIR / "all_models_feature_importance.csv",
        index=False
    )

    # Focus report
    focus = combined[
        combined["target"].isin(FOCUS_TARGETS)
    ]

    print("\n" + "=" * 65)
    print("FOCUS MODEL COMPARISON")
    print("=" * 65)

    for target in FOCUS_TARGETS:

        print(f"\n{target.upper()}")

        print(
            focus[focus["target"] == target]
            .head(10)[["feature", "importance_pct"]]
            .to_string(
                index=False,
                formatters={
                    "importance_pct": "{:.2f}%".format
                }
            )
        )

    print("\nAnalysis completed!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
