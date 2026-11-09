
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/modeling/ros/splits")
OUTPUT_DIR = Path("data/modeling/ros")

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
    "remaining_time_on_ice_seconds"
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading ROS datasets...")

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    print(f"Training records: {len(train)}")
    print(f"Validation records: {len(validation)}")

    results = []

    # --------------------------------------------------------
    # Evaluate each target
    # --------------------------------------------------------

    for target in TARGETS:

        print(f"\nEvaluating: {target}")

        actual = validation[target]

        # ----------------------------------------------------
        # Baseline 1: Global training mean
        # ----------------------------------------------------

        global_mean = train[target].mean()

        global_predictions = [
            global_mean
        ] * len(validation)

        global_mae = mean_absolute_error(
            actual,
            global_predictions
        )

        global_rmse = mean_squared_error(
            actual,
            global_predictions
        ) ** 0.5

        # ----------------------------------------------------
        # Baseline 2: Mean by cutoff percentage
        # ----------------------------------------------------

        cutoff_means = (
            train.groupby("cutoff_percentage")[target]
            .mean()
        )

        cutoff_predictions = (
            validation["cutoff_percentage"]
            .map(cutoff_means)
        )

        if cutoff_predictions.isna().any():
            raise ValueError(
                f"Missing cutoff mean for {target}"
            )

        cutoff_mae = mean_absolute_error(
            actual,
            cutoff_predictions
        )

        cutoff_rmse = mean_squared_error(
            actual,
            cutoff_predictions
        ) ** 0.5

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        results.append({
            "target": target,
            "global_mean": global_mean,
            "global_mae": global_mae,
            "global_rmse": global_rmse,
            "cutoff_mae": cutoff_mae,
            "cutoff_rmse": cutoff_rmse,
            "best_baseline": (
                "cutoff_mean"
                if cutoff_mae < global_mae
                else "global_mean"
            ),
            "best_baseline_mae": min(
                global_mae,
                cutoff_mae
            )
        })

        print(f"Global mean MAE: {global_mae:.4f}")
        print(f"Cutoff mean MAE: {cutoff_mae:.4f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "ros_baseline_results.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("ROS BASELINE RESULTS")
    print("=" * 60)

    print(
        results_df[
            [
                "target",
                "global_mae",
                "cutoff_mae",
                "best_baseline",
                "best_baseline_mae"
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:", output_file)


if __name__ == "__main__":
    main()
