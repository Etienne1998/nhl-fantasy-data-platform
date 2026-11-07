
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_scoring_profile(profile_id: str) -> dict:
    """Load a scoring profile from the configuration file."""

    config_path = PROJECT_ROOT / "config" / "scoring_profiles.json"

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    for profile in config["profiles"]:
        if profile["profile_id"] == profile_id:
            return profile

    raise ValueError(f"Scoring profile '{profile_id}' not found.")


def calculate_fantasy_points(
    players: pd.DataFrame,
    scoring_rules: dict,
    player_type: str
) -> pd.DataFrame:

    rules = scoring_rules["scoring"][player_type]

    available_stats = [
        stat for stat in rules
        if stat in players.columns
    ]

    unsupported_stats = [
        stat for stat in rules
        if stat not in players.columns
    ]

    if unsupported_stats:
        print(f"\nWARNING: Unsupported {player_type} statistics:")
        for stat in unsupported_stats:
            print(f" - {stat}")

    result = players.copy()
    result["fantasy_points"] = 0.0

    for stat in available_stats:
        result["fantasy_points"] += (
            result[stat].fillna(0) * rules[stat]
        )

    print(f"\n{player_type.upper()} SCORING")
    print(f"Supported statistics: {len(available_stats)}")
    print(f"Unsupported statistics: {len(unsupported_stats)}")

    return result.sort_values(
        by="fantasy_points",
        ascending=False
    )
