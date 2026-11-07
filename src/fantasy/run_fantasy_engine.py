
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient

from src.fantasy.calculate_fantasy_points import (
    calculate_fantasy_points,
    load_scoring_profiles
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STORAGE_ACCOUNT_NAME = "stnhlfantasydev2026"
CONTAINER_NAME = "gold"


def upload_to_azure(file_path: Path, blob_name: str):
    """Upload a Gold file to Azure."""

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


def save_gold_data(data: pd.DataFrame, filename: str, timestamp: str):
    """Save rankings locally and upload to Azure Gold."""

    gold_path = PROJECT_ROOT / "data" / "gold"
    gold_path.mkdir(parents=True, exist_ok=True)

    output_path = gold_path / filename

    data.to_json(
        output_path,
        orient="records",
        indent=4,
        force_ascii=False
    )

    blob_name = f"nhl/{filename.replace('.json', '')}_{timestamp}.json"

    upload_to_azure(output_path, blob_name)


def main():

    # 1. Load Silver data
    skaters = load_silver_data("skaters.json")
    goalies = load_silver_data("goalies.json")

    print(f"Skaters loaded: {len(skaters)}")
    print(f"Goalies loaded: {len(goalies)}")

    # 2. Load and validate all scoring profiles
    profiles = load_scoring_profiles()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 3. Process every profile
    for profile in profiles:

        profile_id = profile["profile_id"]
        profile_name = profile["profile_name"]

        print("\n" + "=" * 60)
        print(f"PROCESSING PROFILE: {profile_name}")
        print("=" * 60)

        # Calculate skater rankings
        ranked_skaters = calculate_fantasy_points(
            skaters,
            profile,
            player_type="skaters"
        )

        # Calculate goalie rankings
        ranked_goalies = calculate_fantasy_points(
            goalies,
            profile,
            player_type="goalies"
        )

        # Save skater rankings
        save_gold_data(
            ranked_skaters,
            f"{profile_id}_skaters_rankings.json",
            timestamp
        )

        # Save goalie rankings
        save_gold_data(
            ranked_goalies,
            f"{profile_id}_goalies_rankings.json",
            timestamp
        )

        # Display top 5 skaters
        print("\nTOP 5 SKATERS")
        print(
            ranked_skaters[
                ["player_name", "team", "fantasy_points"]
            ]
            .head(5)
            .to_string(index=False)
        )

        # Display top 5 goalies
        print("\nTOP 5 GOALIES")
        print(
            ranked_goalies[
                ["player_name", "team", "fantasy_points"]
            ]
            .head(5)
            .to_string(index=False)
        )

    print("\nFANTASY ENGINE COMPLETED!")
    print(f"Profiles processed: {len(profiles)}")
    print(f"Total ranking files generated: {len(profiles) * 2}")


if __name__ == "__main__":
    main()
