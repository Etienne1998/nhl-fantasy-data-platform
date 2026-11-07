import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = PROJECT_ROOT / "data" / "gold"


def load_rankings(profile_id: str, player_type: str) -> pd.DataFrame:
    """Load a player's ranking for a scoring profile."""

    filename = f"{profile_id}_{player_type}_rankings.json"
    file_path = GOLD_PATH / filename

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame(data)

    df["rank"] = range(1, len(df) + 1)

    return df[["player_id", "player_name", "team", "rank", "fantasy_points"]]


def compare_profiles(player_type: str):
    """Compare player rankings across all scoring profiles."""

    profiles = ["espn_custom", "offensive", "peripheral"]

    comparison = None

    for profile_id in profiles:

        df = load_rankings(profile_id, player_type)

        df = df.rename(
            columns={
                "rank": f"{profile_id}_rank",
                "fantasy_points": f"{profile_id}_points"
            }
        )

        if comparison is None:
            comparison = df
        else:
            comparison = comparison.merge(
                df,
                on=["player_id", "player_name", "team"],
                how="outer"
            )

    return comparison.sort_values("espn_custom_rank")


def main():

    for player_type in ["skaters", "goalies"]:

        print("\n" + "=" * 70)
        print(f"PROFILE COMPARISON: {player_type.upper()}")
        print("=" * 70)

        comparison = compare_profiles(player_type)

        output_path = GOLD_PATH / f"{player_type}_profile_comparison.json"

        comparison.to_json(
            output_path,
            orient="records",
            indent=4,
            force_ascii=False
        )

        print("\nTOP 10 BY ESPN RANKING")

        print(
            comparison[
                [
                    "player_name",
                    "espn_custom_rank",
                    "offensive_rank",
                    "peripheral_rank"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        print(f"\nComparison saved: {output_path}")


if __name__ == "__main__":
    main()