
import json
import os
from datetime import datetime, timezone

from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


# ==================================================
# 1. CONFIGURATION
# ==================================================

ACCOUNT_NAME = "stnhlfantasydev2026"

BRONZE_CONTAINER = "bronze"
SILVER_CONTAINER = "silver"

LOCAL_OUTPUT = "data/silver/skaters.json"


# ==================================================
# 2. AZURE CONNECTION
# ==================================================

account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"

credential = AzureCliCredential()

blob_service_client = BlobServiceClient(
    account_url=account_url,
    credential=credential
)

print("Azure connection successful!")


# ==================================================
# 3. FIND LATEST BRONZE FILES
# ==================================================

bronze_client = blob_service_client.get_container_client(
    BRONZE_CONTAINER
)


def get_latest_blob(prefix):

    blobs = list(
        bronze_client.list_blobs(name_starts_with=prefix)
    )

    if not blobs:
        raise FileNotFoundError(
            f"No files found with prefix: {prefix}"
        )

    return max(blobs, key=lambda blob: blob.name)


summary_blob = get_latest_blob("nhl/skater_summary_")
realtime_blob = get_latest_blob("nhl/skater_realtime_")


print(f"Summary file: {summary_blob.name}")
print(f"Realtime file: {realtime_blob.name}")


# ==================================================
# 4. VALIDATE SOURCE FILE PAIR
# ==================================================

summary_timestamp = summary_blob.name.removeprefix(
    "nhl/skater_summary_"
).removesuffix(".json")

realtime_timestamp = realtime_blob.name.removeprefix(
    "nhl/skater_realtime_"
).removesuffix(".json")

if summary_timestamp != realtime_timestamp:
    raise ValueError(
        "Summary and Realtime files do not have matching timestamps."
    )


# ==================================================
# 5. DOWNLOAD BRONZE DATA
# ==================================================

def download_json(blob_name):

    blob_client = bronze_client.get_blob_client(blob_name)

    data = blob_client.download_blob().readall()

    return json.loads(data)


summary_data = download_json(summary_blob.name)
realtime_data = download_json(realtime_blob.name)

summary_players = summary_data["data"]
realtime_players = realtime_data["data"]

print(f"Summary records: {len(summary_players)}")
print(f"Realtime records: {len(realtime_players)}")


# ==================================================
# 6. DEDUPLICATE AND INDEX PLAYER IDs
# ==================================================

def deduplicate_players(players, report_name):

    players_by_id = {}

    for player in players:

        player_id = player["playerId"]

        if player_id in players_by_id:

            if player != players_by_id[player_id]:

                raise ValueError(
                    f"Conflicting duplicate for player {player_id} "
                    f"in {report_name}."
                )

        else:
            players_by_id[player_id] = player

    duplicates_removed = len(players) - len(players_by_id)

    print(f"\n{report_name}:")
    print(f"Original records: {len(players)}")
    print(f"Unique players: {len(players_by_id)}")
    print(f"Duplicates removed: {duplicates_removed}")

    return players_by_id


summary_by_id = deduplicate_players(
    summary_players,
    "Summary"
)

realtime_by_id = deduplicate_players(
    realtime_players,
    "Realtime"
)


# ==================================================
# 7. JOIN DIAGNOSTICS
# ==================================================

summary_ids = set(summary_by_id)
realtime_ids = set(realtime_by_id)

common_ids = summary_ids & realtime_ids

summary_only = summary_ids - realtime_ids
realtime_only = realtime_ids - summary_ids

print("\n--- JOIN DIAGNOSTICS ---")

print(f"Players in both reports: {len(common_ids)}")
print(f"Players only in Summary: {len(summary_only)}")
print(f"Players only in Realtime: {len(realtime_only)}")


# ==================================================
# 8. FULL OUTER JOIN
# ==================================================

all_player_ids = summary_ids | realtime_ids

skaters = []

for player_id in all_player_ids:

    summary = summary_by_id.get(player_id, {})
    realtime = realtime_by_id.get(player_id, {})

    # Power play statistics
    pp_goals = summary.get("ppGoals")
    pp_points = summary.get("ppPoints")

    # Short-handed statistics
    sh_goals = summary.get("shGoals")
    sh_points = summary.get("shPoints")

    player = {

        # ------------------------------------------
        # Player information
        # ------------------------------------------

        "player_id": player_id,

        "player_name": (
            summary.get("skaterFullName")
            or realtime.get("skaterFullName")
        ),

        "team": (
            summary.get("teamAbbrevs")
            or realtime.get("teamAbbrevs")
        ),

        "position": (
            summary.get("positionCode")
            or realtime.get("positionCode")
        ),

        "season_id": (
            summary.get("seasonId")
            or realtime.get("seasonId")
        ),

        "games_played": (
            summary.get("gamesPlayed")
            if summary.get("gamesPlayed") is not None
            else realtime.get("gamesPlayed")
        ),

        # ------------------------------------------
        # Scoring statistics
        # ------------------------------------------

        "goals": summary.get("goals"),
        "assists": summary.get("assists"),
        "points": summary.get("points"),
        "plus_minus": summary.get("plusMinus"),
        "penalty_minutes": summary.get("penaltyMinutes"),

        # ------------------------------------------
        # Power play statistics
        # ------------------------------------------

        "power_play_goals": pp_goals,
        "power_play_points": pp_points,

        "power_play_assists": (
            pp_points - pp_goals
            if pp_points is not None and pp_goals is not None
            else None
        ),

        # ------------------------------------------
        # Short-handed statistics
        # ------------------------------------------

        "short_handed_goals": sh_goals,
        "short_handed_points": sh_points,

        "short_handed_assists": (
            sh_points - sh_goals
            if sh_points is not None and sh_goals is not None
            else None
        ),

        # ------------------------------------------
        # Other summary statistics
        # ------------------------------------------

        "game_winning_goals": summary.get("gameWinningGoals"),
        "shots_on_goal": summary.get("shots"),
        "shooting_pct": summary.get("shootingPct"),
        "faceoff_win_pct": summary.get("faceoffWinPct"),

        # ------------------------------------------
        # Realtime statistics
        # ------------------------------------------

        "hits": realtime.get("hits"),
        "blocked_shots": realtime.get("blockedShots"),
        "giveaways": realtime.get("giveaways"),
        "takeaways": realtime.get("takeaways"),
        "total_shot_attempts": realtime.get("totalShotAttempts"),

        # ------------------------------------------
        # Time on ice
        # ------------------------------------------

        "time_on_ice_per_game": summary.get("timeOnIcePerGame")
    }

    skaters.append(player)


# ==================================================
# 9. SORT AND VALIDATE SILVER DATA
# ==================================================

skaters.sort(
    key=lambda player: player["player_id"]
)

print("\n--- SILVER VALIDATION ---")

print(f"Total unique players: {len(skaters)}")

if len(skaters) != len(all_player_ids):

    raise ValueError(
        "Unexpected player count after transformation."
    )

if len({player["player_id"] for player in skaters}) != len(skaters):

    raise ValueError(
        "Duplicate player IDs found in Silver output."
    )

missing_names = [
    player["player_id"]
    for player in skaters
    if not player["player_name"]
]

if missing_names:

    raise ValueError(
        f"Players missing names: {missing_names}"
    )


# ==================================================
# 10. SAVE LOCAL SILVER DATA
# ==================================================

os.makedirs(
    os.path.dirname(LOCAL_OUTPUT),
    exist_ok=True
)

with open(
    LOCAL_OUTPUT,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        skaters,
        file,
        indent=2,
        ensure_ascii=False
    )

print(f"\nLocal Silver file saved: {LOCAL_OUTPUT}")


# ==================================================
# 11. UPLOAD TO AZURE SILVER
# ==================================================

timestamp = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S"
)

silver_blob_name = f"nhl/skaters_{timestamp}.json"

silver_client = blob_service_client.get_container_client(
    SILVER_CONTAINER
)

blob_client = silver_client.get_blob_client(
    silver_blob_name
)

json_data = json.dumps(
    skaters,
    indent=2,
    ensure_ascii=False
)

blob_client.upload_blob(
    json_data.encode("utf-8"),
    overwrite=True
)

print(
    f"Azure Silver upload successful: {silver_blob_name}"
)


# ==================================================
# 12. COMPLETION SUMMARY
# ==================================================

print("\n========================================")
print("SILVER TRANSFORMATION COMPLETED!")
print("========================================")

print(f"Summary records: {len(summary_players)}")
print(f"Realtime records: {len(realtime_players)}")

print(f"Summary unique players: {len(summary_by_id)}")
print(f"Realtime unique players: {len(realtime_by_id)}")

print(f"Common players: {len(common_ids)}")
print(f"Summary-only players: {len(summary_only)}")
print(f"Realtime-only players: {len(realtime_only)}")

print(f"Total unique players in Silver: {len(skaters)}")

print(f"Season: {skaters[0]['season_id'] if skaters else 'N/A'}")

print("========================================")
