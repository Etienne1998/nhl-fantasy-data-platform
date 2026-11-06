import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient

from src.fantasy.calculate_fantasy_points import (
    calculate_fantasy_points,
    load_scoring_profile
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STORAGE_ACCOUNT_NAME = "stnhlfantasydev2026"
CONTAINER_NAME = "gold"


def upload_to_azure(file_path: Path, blob_name: str):
    """Upload a Gold file to Azure Data Lake Storage."""

    credential = AzureCliCredential()

    account_url = (
        f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
    )

    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=credential
    )

    blob_client = blob_service_client.get_blob_client(
        container=CONTAINER_NAME,
        blob=blob_name
    )

    with open(file_path, "rb") as file:
        blob_client.upload_blob(file, overwrite=True)

    print(f"Azure upload successful: {blob_name}")


def main():
    # 1. Load Silver data
    silver_path = PROJECT_ROOT / "data" / "silver" / "skaters.json"

    with open(silver_path, "r", encoding="utf-8") as file:
        players_data = json.load(file)

    players = pd.DataFrame(players_data)

    # 2. Load scoring profile
    profile = load_scoring_profile("espn_custom")

    # 3. Calculate fantasy points
    ranked_players = calculate_fantasy_points(players, profile)

    # 4. Create Gold directory
    gold_path = PROJECT_ROOT / "data" / "gold"
    gold_path.mkdir(parents=True, exist_ok=True)

    # 5. Export complete rankings locally
    output_path = gold_path / "fantasy_rankings.json"

    ranked_players.to_json(
        output_path,
        orient="records",
        indent=4,
        force_ascii=False
    )

    # 6. Generate timestamped Azure filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    blob_name = f"nhl/fantasy_rankings_{timestamp}.json"

    # 7. Upload to Azure Gold
    upload_to_azure(output_path, blob_name)

    # 8. Display top 20 players
    columns = [
        "player_name",
        "team",
        "position",
        "games_played",
        "goals",
        "assists",
        "fantasy_points"
    ]

    print("\nTOP 20 FANTASY SKATERS\n")
    print(
        ranked_players[columns]
        .head(20)
        .to_string(index=False)
    )

    print(f"\nTotal players ranked: {len(ranked_players)}")
    print(f"Local Gold file: {output_path}")


if __name__ == "__main__":
    main()