
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "modeling" / "ros" /
    "prediction_errors" / "ros_player_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "modeling" / "ros" / "ranking"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "remaining_goals",
    "remaining_assists",
    "remaining_shots_on_goal",
    "remaining_power_play_points",
    "remaining_time_on_ice_seconds",
]

TOP_N_VALUES = [10, 25, 50]


print("Loading player predictions...")
df = pd.read_csv(INPUT_FILE)

results = []

for target in TARGETS:

    print("\n" + "=" * 65)
    print(f"Analyzing ranking: {target}")
    print("=" * 65)

    target_df = df[df["target"] == target]

    for cutoff, group in target_df.groupby("cutoff_percentage"):

        # Spearman rank correlation
        spearman = group["actual"].corr(
            group["predicted"],
            method="spearman"
        )

        row = {
            "target": target,
            "cutoff": cutoff,
            "players": len(group),
            "spearman_correlation": spearman,
        }

        # Compare the top N actual and predicted players
        for n in TOP_N_VALUES:

            actual_top = set(
                group.nlargest(n, "actual")["player_id"]
            )

            predicted_top = set(
                group.nlargest(n, "predicted")["player_id"]
            )

            overlap = len(actual_top.intersection(predicted_top))

            row[f"top_{n}_overlap"] = overlap
            row[f"top_{n}_precision"] = overlap / n

        results.append(row)

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "ros_ranking_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("RANKING EVALUATION RESULTS")
print("=" * 65)

print(
    results_df.round(3).to_string(index=False)
)

print("\nResults saved to:")
print(OUTPUT_DIR / "ros_ranking_results.csv")
