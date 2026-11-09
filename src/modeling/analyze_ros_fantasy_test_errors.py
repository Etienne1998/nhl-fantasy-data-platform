
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "modeling" / "ros" /
    "fantasy" / "ros_fantasy_test_predictions.csv"
)

OUTPUT_DIR = ROOT / "data" / "modeling" / "ros" / "fantasy"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df)} predictions")


# ============================================================
# ANALYZE TOP 10 ERRORS
# ============================================================

for cutoff, group in df.groupby("cutoff_percentage"):

    group = group.copy()

    group["actual_rank"] = (
        group["actual_fantasy_points"]
        .rank(method="min", ascending=False)
    )

    group["predicted_rank"] = (
        group["pred_fantasy_points"]
        .rank(method="min", ascending=False)
    )

    predicted_top10 = group[group["predicted_rank"] <= 10]
    actual_top10 = group[group["actual_rank"] <= 10]

    false_positives = predicted_top10[
        predicted_top10["actual_rank"] > 10
    ].sort_values("predicted_rank")

    false_negatives = actual_top10[
        actual_top10["predicted_rank"] > 10
    ].sort_values("actual_rank")

    print("\n" + "=" * 80)
    print(f"CUTOFF: {cutoff:.0%}")
    print("=" * 80)

    print("\nFALSE POSITIVES")
    print("Players predicted Top 10 but actually outside Top 10")

    print(
        false_positives[
            [
                "player_name",
                "team",
                "position",
                "predicted_rank",
                "actual_rank",
                "pred_fantasy_points",
                "actual_fantasy_points",
            ]
        ].round(1).to_string(index=False)
    )

    print("\nFALSE NEGATIVES")
    print("Players actually Top 10 but predicted outside Top 10")

    print(
        false_negatives[
            [
                "player_name",
                "team",
                "position",
                "predicted_rank",
                "actual_rank",
                "pred_fantasy_points",
                "actual_fantasy_points",
            ]
        ].round(1).to_string(index=False)
    )


# ============================================================
# SAVE RANKED DATA
# ============================================================

df["actual_rank"] = (
    df.groupby("cutoff_percentage")["actual_fantasy_points"]
    .rank(method="min", ascending=False)
)

df["predicted_rank"] = (
    df.groupby("cutoff_percentage")["pred_fantasy_points"]
    .rank(method="min", ascending=False)
)

df["rank_error"] = df["predicted_rank"] - df["actual_rank"]

df.to_csv(
    OUTPUT_DIR / "ros_fantasy_test_rank_analysis.csv",
    index=False
)

print("\nRank analysis saved.")
