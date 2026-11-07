
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_PLAYER_TYPES = {"skaters", "goalies"}


def load_scoring_profiles() -> list:
    """Load and validate all scoring profiles."""

    config_path = PROJECT_ROOT / "config" / "scoring_profiles.json"

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    profiles = config.get("profiles", [])

    if not profiles:
        raise ValueError("No scoring profiles found.")

    profile_ids = []

    for profile in profiles:
        profile_id = profile.get("profile_id")

        if not profile_id:
            raise ValueError("A profile is missing its profile_id.")

        if profile_id in profile_ids:
            raise ValueError(f"Duplicate profile_id: {profile_id}")

        profile_ids.append(profile_id)

        scoring = profile.get("scoring", {})

        if set(scoring.keys()) != REQUIRED_PLAYER_TYPES:
            raise ValueError(
                f"Profile '{profile_id}' must contain skaters and goalies."
            )

        for player_type in REQUIRED_PLAYER_TYPES:
            for stat, points in scoring[player_type].items():
                if not isinstance(points, (int, float)):
                    raise ValueError(
                        f"Invalid scoring value for {stat} in {profile_id}."
                    )

    print(f"Validated scoring profiles: {len(profiles)}")

    return profiles


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
