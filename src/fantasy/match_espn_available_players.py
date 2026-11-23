
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEASON_ID = 20262027
ESPN_SEASON_ID = 2027

REFERENCE_FILE = (
    PROJECT_ROOT / "data" / "reference"
    / f"nhl_players_{SEASON_ID}.json"
)

ESPN_FILE = (
    PROJECT_ROOT / "data" / "bronze"
    / f"espn_available_players_{ESPN_SEASON_ID}.json"
)

GOLD_FILE = (
    PROJECT_ROOT / "data" / "gold"
    / f"player_performance_{SEASON_ID}.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "gold"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_players(path, wrapped=False):
    data = load_json(path)

    if wrapped:
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object: {path.name}")

        players = data.get("data")
    else:
        players = data

    if not isinstance(players, list):
        raise ValueError(f"Expected player list: {path.name}")

    return players


def normalize_name(name):
    normalized = unicodedata.normalize("NFKD", name or "")

    normalized = "".join(
        char for char in normalized
        if not unicodedata.combining(char)
    )

    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def main():
    print("Loading NHL reference...")
    nhl_players = load_players(REFERENCE_FILE)

    print("Loading ESPN available players...")
    espn_players = load_players(ESPN_FILE, wrapped=True)

    print("Loading Gold performance...")
    gold_players = load_players(GOLD_FILE, wrapped=True)

    # Index NHL players by normalized name.
    nhl_by_name = {}

    for player in nhl_players:
        key = normalize_name(player.get("player_name"))

        if key:
            nhl_by_name.setdefault(key, []).append(player)

    gold_by_id = {
        int(player["player_id"]): player
        for player in gold_players
    }

    matched = []
    unmatched = []
    ambiguous = []

    for espn_player in espn_players:
        key = normalize_name(espn_player.get("player_name"))
        candidates = nhl_by_name.get(key, [])

        if len(candidates) == 0:
            unmatched.append({
                **espn_player,
                "match_status": "not_found"
            })
            continue

        if len(candidates) > 1:
            ambiguous.append({
                **espn_player,
                "match_status": "ambiguous",
                "candidates": candidates
            })
            continue

        nhl_player = candidates[0]
        nhl_player_id = int(nhl_player["player_id"])

        gold_performance = gold_by_id.get(nhl_player_id)

        matched.append({
            **espn_player,
            "nhl_player_id": nhl_player_id,
            "nhl_player_name": nhl_player["player_name"],
            "nhl_team": nhl_player.get("team"),
            "nhl_position": nhl_player.get("position"),
            "has_gold_stats": gold_performance is not None,
            "gold_performance": gold_performance
        })

    matched_file = (
        OUTPUT_DIR
        / f"espn_available_players_enriched_{ESPN_SEASON_ID}.json"
    )

    unmatched_file = (
        OUTPUT_DIR
        / f"espn_available_players_unmatched_{ESPN_SEASON_ID}.json"
    )

    ambiguous_file = (
        OUTPUT_DIR
        / f"espn_available_players_ambiguous_{ESPN_SEASON_ID}.json"
    )

    save_json(matched_file, matched)
    save_json(unmatched_file, unmatched)
    save_json(ambiguous_file, ambiguous)

    with_gold = sum(
        player["has_gold_stats"] for player in matched
    )

    print("\n" + "=" * 50)
    print("ESPN PLAYER MATCHING COMPLETED")
    print("=" * 50)

    print(f"ESPN available players: {len(espn_players)}")
    print(f"NHL reference players: {len(nhl_players)}")
    print(f"Gold performance players: {len(gold_players)}")

    print(f"\nMatched: {len(matched)}")
    print(f"Unmatched: {len(unmatched)}")
    print(f"Ambiguous: {len(ambiguous)}")

    print(f"\nMatched with Gold stats: {with_gold}")
    print(f"Matched without Gold stats: {len(matched) - with_gold}")

    print("\nOUTPUT FILES")
    print(matched_file)
    print(unmatched_file)
    print(ambiguous_file)

    print("\nDONE!")


if __name__ == "__main__":
    main()
