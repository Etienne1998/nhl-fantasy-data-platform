
import json
from pathlib import Path

files = sorted(Path("data/bronze").glob("skater_stats_*.json"))
latest_file = files[-1]

with latest_file.open("r", encoding="utf-8") as file:
    data = json.load(file)

for category, players in data.items():
    print(f"\n{'=' * 50}")
    print(f"CATEGORY: {category}")
    print(f"Number of records: {len(players)}")

    if players:
        print("Fields:", list(players[0].keys()))
        print("Sample:", json.dumps(players[0], indent=2, ensure_ascii=False))
