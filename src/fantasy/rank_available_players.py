
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ESPN_SEASON_ID = 2027

INPUT_FILE = (
    PROJECT_ROOT / "data" / "gold"
    / f"espn_available_players_enriched_{ESPN_SEASON_ID}.json"
)

OUTPUT_JSON = (
    PROJECT_ROOT / "data" / "gold"
    / f"available_player_rankings_{ESPN_SEASON_ID}.json"
)

OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "gold"
    / f"available_player_rankings_{ESPN_SEASON_ID}.csv"
)

MIN_GAMES_PLAYED = 5

WEIGHTS = {
    "season": 0.50,
    "l10": 0.30,
    "l5": 0.20,
}


def load_players():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        players = json.load(file)

    if not isinstance(players, list):
        raise ValueError("Expected a JSON list of enriched players.")

    return players


def calculate_ranking_score(stats):
    season_avg = float(stats.get("fantasy_points_per_game") or 0)
    l10_avg = float(stats.get("fantasy_points_l10") or 0)
    l5_avg = float(stats.get("fantasy_points_l5") or 0)

    score = (
        season_avg * WEIGHTS["season"]
        + l10_avg * WEIGHTS["l10"]
        + l5_avg * WEIGHTS["l5"]
    )

    return round(score, 3)


def build_rankings(players):
    rankings = []
    seen_player_ids = set()

    excluded_low_games = 0
    excluded_duplicates = 0
    excluded_without_gold = 0

    for player in players:
        stats = player.get("gold_performance")

        if not player.get("has_gold_stats") or not isinstance(stats, dict):
            excluded_without_gold += 1
            continue

        player_id = player.get("nhl_player_id")

        if player_id is None:
            continue

        player_id = int(player_id)

        # Prevent the same NHL player from appearing twice.
        if player_id in seen_player_ids:
            excluded_duplicates += 1
            continue

        seen_player_ids.add(player_id)

        games_played = int(stats.get("games_played") or 0)

        # Avoid ranking players with very small samples.
        if games_played < MIN_GAMES_PLAYED:
            excluded_low_games += 1
            continue

        ranking_score = calculate_ranking_score(stats)

        rankings.append({
            "player_name": player.get("player_name"),
            "nhl_player_id": player_id,
            "team": player.get("nhl_team"),
            "position": player.get("nhl_position"),
            "games_played": games_played,
            "fantasy_points_total": stats.get("fantasy_points_total"),
            "fantasy_points_per_game": stats.get(
                "fantasy_points_per_game"
            ),
            "fantasy_points_l5": stats.get("fantasy_points_l5"),
            "games_in_l5": stats.get("games_in_l5"),
            "fantasy_points_l10": stats.get("fantasy_points_l10"),
            "games_in_l10": stats.get("games_in_l10"),
            "ranking_score": ranking_score,
        })

    rankings.sort(
        key=lambda player: player["ranking_score"],
        reverse=True
    )

    for rank, player in enumerate(rankings, start=1):
        player["rank"] = rank

    exclusions = {
        "excluded_without_gold": excluded_without_gold,
        "excluded_low_games": excluded_low_games,
        "excluded_duplicates": excluded_duplicates,
    }

    return rankings, exclusions


def save_json(rankings):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(
            rankings,
            file,
            indent=4,
            ensure_ascii=False
        )


def save_csv(rankings):
    if not rankings:
        print("No players to export.")
        return

    fieldnames = list(rankings[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rankings)


def main():
    print("Loading enriched ESPN available players...")

    players = load_players()

    print(f"Available players loaded: {len(players)}")

    print("Calculating Fantasy rankings...")

    rankings, exclusions = build_rankings(players)

    save_json(rankings)
    save_csv(rankings)

    print("\n" + "=" * 75)
    print("TOP 10 AVAILABLE FANTASY PLAYERS")
    print("=" * 75)

    print(
        f"{'RANK':<6}"
        f"{'PLAYER':<25}"
        f"{'TEAM':<7}"
        f"{'POS':<6}"
        f"{'GP':<5}"
        f"{'SEASON':<9}"
        f"{'L10':<9}"
        f"{'L5':<9}"
        f"{'SCORE':<9}"
    )

    print("-" * 85)

    for player in rankings[:10]:
        print(
            f"{player['rank']:<6}"
            f"{(player['player_name'] or ''):<25}"
            f"{(player['team'] or ''):<7}"
            f"{(player['position'] or ''):<6}"
            f"{player['games_played']:<5}"
            f"{float(player['fantasy_points_per_game'] or 0):<9.2f}"
            f"{float(player['fantasy_points_l10'] or 0):<9.2f}"
            f"{float(player['fantasy_points_l5'] or 0):<9.2f}"
            f"{player['ranking_score']:<9.3f}"
        )

    print("\nRANKING SUMMARY")
    print(f"Available ESPN players: {len(players)}")
    print(f"Ranked players: {len(rankings)}")
    print(f"Minimum games required: {MIN_GAMES_PLAYED}")
    print(f"Excluded without Gold stats: {exclusions['excluded_without_gold']}")
    print(f"Excluded below minimum games: {exclusions['excluded_low_games']}")
    print(f"Excluded duplicate NHL IDs: {exclusions['excluded_duplicates']}")

    print("\nOUTPUT FILES")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)

    print("\nRANKING COMPLETED!")


if __name__ == "__main__":
    main()
