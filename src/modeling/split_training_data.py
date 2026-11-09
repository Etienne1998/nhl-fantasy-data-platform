
import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/modeling/skater_training_data.csv")
OUTPUT_DIR = Path("data/modeling/splits")

TRAIN_SEASONS = ["20212022", "20222023", "20232024"]
VALIDATION_SEASONS = ["20242025"]
TEST_SEASONS = ["20252026"]


def split_training_data():

    print("Loading training dataset...")

    df = pd.read_csv(INPUT_PATH)

    # Normalize season IDs to strings
    df["season_id"] = df["season_id"].astype(str)

    print(f"Total records: {len(df)}")

    # --------------------------------------------------
    # 1. TEMPORAL SPLIT
    # --------------------------------------------------

    train_df = df[df["season_id"].isin(TRAIN_SEASONS)].copy()

    validation_df = df[
        df["season_id"].isin(VALIDATION_SEASONS)
    ].copy()

    test_df = df[
        df["season_id"].isin(TEST_SEASONS)
    ].copy()

    # --------------------------------------------------
    # 2. VALIDATION
    # --------------------------------------------------

    assert len(train_df) > 0
    assert len(validation_df) > 0
    assert len(test_df) > 0

    assert set(train_df["season_id"]).isdisjoint(
        validation_df["season_id"]
    )

    assert set(train_df["season_id"]).isdisjoint(
        test_df["season_id"]
    )

    assert set(validation_df["season_id"]).isdisjoint(
        test_df["season_id"]
    )

    train_max_date = pd.to_datetime(train_df["game_date"]).max()
    validation_min_date = pd.to_datetime(
        validation_df["game_date"]
    ).min()

    validation_max_date = pd.to_datetime(
        validation_df["game_date"]
    ).max()

    test_min_date = pd.to_datetime(test_df["game_date"]).min()

    assert train_max_date < validation_min_date
    assert validation_max_date < test_min_date

    # --------------------------------------------------
    # 3. EXPORT
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    metadata = {
        "strategy": "Temporal split by NHL season",
        "train_seasons": TRAIN_SEASONS,
        "validation_seasons": VALIDATION_SEASONS,
        "test_seasons": TEST_SEASONS,
        "train_records": len(train_df),
        "validation_records": len(validation_df),
        "test_records": len(test_df),
        "train_last_date": str(train_max_date.date()),
        "validation_first_date": str(validation_min_date.date()),
        "validation_last_date": str(validation_max_date.date()),
        "test_first_date": str(test_min_date.date()),
        "random_split": False,
    }

    with open(
        OUTPUT_DIR / "split_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    # --------------------------------------------------
    # 4. SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("TEMPORAL DATA SPLIT COMPLETED")
    print("=" * 55)

    for name, dataset in [
        ("TRAIN", train_df),
        ("VALIDATION", validation_df),
        ("TEST", test_df),
    ]:
        print(f"\n{name}")
        print(f"Records: {len(dataset)}")
        print(f"Seasons: {sorted(dataset['season_id'].unique())}")
        print(f"First game: {dataset['game_date'].min()}")
        print(f"Last game: {dataset['game_date'].max()}")

    print("\nChronological validation passed!")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    split_training_data()
