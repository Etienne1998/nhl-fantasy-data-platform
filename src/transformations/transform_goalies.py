import json
from pathlib import Path
from datetime import datetime

from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STORAGE_ACCOUNT_NAME = "stnhlfantasydev2026"
BRONZE_CONTAINER = "bronze"
SILVER_CONTAINER = "silver"


def get_azure_clients():
    credential = AzureCliCredential()

    account_url = (
        f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    )

    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential
    )

    return blob_service_client


def load_latest_bronze_file(container_client):
    blobs = list(
        container_client.list_blobs(
            name_starts_with="nhl/goalie_summary_"
        )
    )

    if not blobs:
        raise FileNotFoundError(
            "No goalie summary files found in Bronze."
        )

    latest_blob = max(blobs, key=lambda blob: blob.name)

    print(f"Bronze source: {latest_blob.name}")

    blob_client = container_client.get_blob_client(
        latest_blob.name
    )

    raw_data = blob_client.download_blob().readall()

    return json.loads(raw_data)


def deduplicate_goalies(goalies):
    """Remove identical duplicate records and reject conflicts."""

    goalies_by_id = {}

    for goalie in goalies:
        player_id = goalie["playerId"]

        if player_id in goalies_by_id:
            if goalie != goalies_by_id[player_id]:
                raise ValueError(
                    f"Conflicting duplicate records for player ID: "
                    f"{player_id}"
                )
        else:
            goalies_by_id[player_id] = goalie

    print(f"Original records: {len(goalies)}")
    print(f"Unique goalies: {len(goalies_by_id)}")
    print(
        f"Duplicates removed: "
        f"{len(goalies) - len(goalies_by_id)}"
    )

    return goalies_by_id.values()


def transform_goalies(goalies):
    """Standardize NHL goalie statistics into Silver format."""

    silver_goalies = []

    for goalie in goalies:
        silver_goalie = {
            "player_id": goalie["playerId"],
            "player_name": goalie["goalieFullName"],
            "team": goalie["teamAbbrevs"],
            "position": "G",
            "season_id": goalie["seasonId"],

            "games_played": goalie["gamesPlayed"],
            "games_started": goalie["gamesStarted"],

            "wins": goalie["wins"],
            "losses": goalie["losses"],
            "overtime_losses": goalie["otLosses"],

            "goals_against": goalie["goalsAgainst"],
            "saves": goalie["saves"],
            "shots_against": goalie["shotsAgainst"],
            "save_pct": goalie["savePct"],
            "goals_against_average": goalie["goalsAgainstAverage"],
            "shutouts": goalie["shutouts"],

            "goals": goalie["goals"],
            "assists": goalie["assists"],
            "points": goalie["points"],
            "penalty_minutes": goalie["penaltyMinutes"],

            "ties": goalie["ties"],
            "time_on_ice": goalie["timeOnIce"]
        }

        silver_goalies.append(silver_goalie)

    return sorted(
        silver_goalies,
        key=lambda goalie: goalie["player_id"]
    )


def main():
    # 1. Connect to Azure
    blob_service_client = get_azure_clients()

    bronze_client = blob_service_client.get_container_client(
        BRONZE_CONTAINER
    )

    silver_client = blob_service_client.get_container_client(
        SILVER_CONTAINER
    )

    print("Azure connection successful!")

    # 2. Load Bronze data
    raw_goalies = load_latest_bronze_file(bronze_client)

    # 3. Deduplicate records
    unique_goalies = deduplicate_goalies(raw_goalies)

    # 4. Transform data
    silver_goalies = transform_goalies(unique_goalies)

    # 5. Validate data
    player_ids = [
        goalie["player_id"] for goalie in silver_goalies
    ]

    if len(player_ids) != len(set(player_ids)):
        raise ValueError("Duplicate player IDs found in Silver.")

    if any(not goalie["player_name"] for goalie in silver_goalies):
        raise ValueError("Missing goalie names in Silver.")

    print("\nSILVER VALIDATION")
    print(f"Total unique goalies: {len(silver_goalies)}")

    # 6. Save locally
    local_path = PROJECT_ROOT / "data" / "silver" / "goalies.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    with open(local_path, "w", encoding="utf-8") as file:
        json.dump(
            silver_goalies,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"Local Silver file saved: {local_path}")

    # 7. Upload to Azure Silver
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    blob_name = f"nhl/goalies_{timestamp}.json"

    blob_client = silver_client.get_blob_client(
        blob_name
    )

    blob_client.upload_blob(
        json.dumps(
            silver_goalies,
            indent=4,
            ensure_ascii=False
        ).encode("utf-8"),
        overwrite=True
    )

    print(f"Azure Silver upload successful: {blob_name}")
    print("\nGOALIE SILVER TRANSFORMATION COMPLETED!")


if __name__ == "__main__":
    main()