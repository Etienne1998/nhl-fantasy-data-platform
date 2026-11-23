
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


# ============================================================
# CONFIGURATION
# ============================================================

parser = argparse.ArgumentParser(
    description="Fetch available players from ESPN Fantasy Hockey."
)

parser.add_argument("--season", type=int, default=2027)
parser.add_argument("--league-id", type=int, default=107851603)

args = parser.parse_args()

SEASON_ID = args.season
LEAGUE_ID = args.league_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / f"espn_available_players_{SEASON_ID}.json"
)

BASE_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/"
    f"fhl/seasons/{SEASON_ID}/segments/0/leagues/{LEAGUE_ID}"
)

PAGE_SIZE = 200


# ============================================================
# ESPN AUTHENTICATION
# ============================================================

ESPN_S2 = os.getenv("ESPN_S2")
ESPN_SWID = os.getenv("ESPN_SWID")

session = requests.Session()

session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
})

if ESPN_S2 and ESPN_SWID:
    session.cookies.update({
        "espn_s2": ESPN_S2,
        "SWID": ESPN_SWID,
    })
    print("ESPN authentication cookies loaded.")
else:
    print("WARNING: ESPN authentication cookies not found.")


# ============================================================
# API REQUEST
# ============================================================

def make_request(params, headers=None):

    response = session.get(
        BASE_URL,
        params=params,
        headers=headers,
        timeout=30,
    )

    if response.status_code == 401:
        raise PermissionError(
            "HTTP 401: ESPN authentication failed. "
            "Verify your ESPN cookies."
        )

    if response.status_code == 403:
        raise PermissionError(
            "HTTP 403: ESPN denied access to this league."
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# CHECK LEAGUE ACCESS
# ============================================================

def get_league_settings():

    print("\nChecking ESPN league access...")

    data = make_request({
        "view": "mSettings"
    })

    settings = data.get("settings", {})

    print(f"League ID: {data.get('id', LEAGUE_ID)}")
    print(f"Season: {data.get('seasonId', SEASON_ID)}")

    return data


# ============================================================
# FETCH AVAILABLE PLAYERS
# ============================================================

def fetch_available_players():

    print("\nFetching ESPN available players...")

    all_players = []
    offset = 0

    while True:

        fantasy_filter = {
            "players": {
                "filterStatus": {
                    "value": [
                        "FREEAGENT",
                        "WAIVERS"
                    ]
                },
                "limit": PAGE_SIZE,
                "offset": offset,
                "sortPercOwned": {
                    "sortPriority": 1,
                    "sortAsc": False
                }
            }
        }

        headers = {
            "X-Fantasy-Filter": json.dumps(fantasy_filter)
        }

        data = make_request(
            params={"view": "kona_player_info"},
            headers=headers,
        )

        players = data.get("players", [])

        if not players:
            break

        all_players.extend(players)

        print(
            f"Retrieved {len(players)} players "
            f"(total: {len(all_players)})"
        )

        if len(players) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return all_players


# ============================================================
# STANDARDIZE PLAYER RECORDS
# ============================================================

def standardize_players(raw_players):

    standardized = []
    seen_ids = set()

    for entry in raw_players:

        player = entry.get("player", {})
        pool = entry.get("playerPoolEntry", {})

        player_id = entry.get("id")

        if player_id is None:
            continue

        player_id = int(player_id)

        if player_id in seen_ids:
            continue

        seen_ids.add(player_id)

        standardized.append({
            "espn_player_id": player_id,
            "player_name": player.get("fullName"),
            "active": player.get("active"),
            "pro_team_id": player.get("proTeamId"),
            "position_id": player.get("defaultPositionId"),
            "eligible_slots": player.get("eligibleSlots", []),
            "injury_status": player.get("injuryStatus"),
            "availability_status": (
                entry.get("status")
                or pool.get("status")
                or "AVAILABLE"
            ),
            "on_team_id": entry.get("onTeamId", 0),
            "percent_owned": pool.get("percentOwned"),
            "percent_started": pool.get("percentStarted"),
            "fantasy_points_total": pool.get("totalPoints"),
        })

    return standardized


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ESPN FANTASY HOCKEY INGESTION")
    print("=" * 60)

    print(f"League ID: {LEAGUE_ID}")
    print(f"Season: {SEASON_ID}")

    try:

        league_data = get_league_settings()

        raw_players = fetch_available_players()

        players = standardize_players(raw_players)

        if not players:
            raise ValueError(
                "No available players returned by ESPN."
            )

        output = {
            "league_id": LEAGUE_ID,
            "season_id": SEASON_ID,
            "league_name": league_data.get("settings", {}).get(
                "name"
            ),
            "source": "ESPN Fantasy API",
            "extraction_timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "total_available_players": len(players),
            "data": players,
        }

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_file = OUTPUT_FILE.with_suffix(".tmp")

        with open(
            temporary_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False
            )

        temporary_file.replace(OUTPUT_FILE)

        print("\n" + "=" * 60)
        print("ESPN INGESTION COMPLETED!")
        print("=" * 60)

        print(f"Available players: {len(players)}")
        print(f"Output: {OUTPUT_FILE}")

        print("\nTOP 10 AVAILABLE PLAYERS")

        for player in players[:10]:

            print(
                f"{player['player_name']} "
                f"| ESPN ID: {player['espn_player_id']} "
                f"| Owned: {player['percent_owned']}%"
            )

    except Exception as error:

        print("\nESPN INGESTION FAILED")
        print(f"Error: {error}")

        sys.exit(1)


if __name__ == "__main__":
    main()
