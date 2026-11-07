
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


def load_silver_data(filename: str) -> pd.DataFrame:
    """Load standardized Silver data."""

    file_path = PROJECT_ROOT / "data" / "silver" / filename

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return pd.DataFrame(data)


def save_gold_data(data: pd.DataFrame, filename: str):
    """Save rankings locally and upload them to Azure Gold."""

    gold_path = PROJECT_ROOT / "data" / "gold"
    gold_path.mkdir(parents=True, exist_ok=True)

    output_path = gold_path / filename

    data.to_json(
        output_path,
        orient="records",
        indent=4,
        force_ascii=False
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob_name = f"nhl/{filename.replace('.json', '')}_{timestamp}.json"

    upload_to_azure(output_path, blob_name)

    print(f"Local Gold file: {output_path}")


def main():

    # 1. Load Silver data
    skaters = load_silver_data("skaters.json")
    goalies = load_silver_data("goalies.json")

    print(f"Skaters loaded: {len(skaters)}")
    print(f"Goalies loaded: {len(goalies)}")

    # 2. Load scoring profile
    profile = load_scoring_profile("espn_custom")

    # 3. Calculate fantasy points
    ranked_skaters = calculate_fantasy_points(
        skaters,
        profile,
        player_type="skaters"
    )

    ranked_goalies = calculate_fantasy_points(
        goalies,
        profile,
        player_type="goalies"
    )

    # 4. Save and upload Gold rankings
    save_gold_data(
        ranked_skaters,
        "fantasy_skaters_rankings.json"
    )

    save_gold_data(
        ranked_goalies,
        "fantasy_goalies_rankings.json"
    )

    # 5. Display top 20 skaters
    skater_columns = [
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
        ranked_skaters[skater_columns]
        .head(20)
        .to_string(index=False)
    )

    # 6. Display top 20 goalies
    goalie_columns = [
        "player_name",
        "team",
        "games_played",
        "wins",
        "saves",
        "goals_against",
        "fantasy_points"
    ]

    print("\nTOP 20 FANTASY GOALIES\n")
    print(
        ranked_goalies[goalie_columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nFANTASY ENGINE COMPLETED!")
    print(f"Skaters ranked: {len(ranked_skaters)}")
    print(f"Goalies ranked: {len(ranked_goalies)}")


if __name__ == "__main__":
    main()
