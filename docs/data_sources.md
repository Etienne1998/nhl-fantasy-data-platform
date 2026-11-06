# NHL API — Data Sources

## 1. API Overview

**Primary API:** `https://api-web.nhle.com`

**Secondary API:** `https://api.nhle.com/stats/rest`

**Documentation:** https://github.com/Zmalski/NHL-API-Reference

The project will initially use the NHL Web API to collect player statistics, rosters and game data.

## 2. Initial Endpoints

| Endpoint | Purpose |
|---|---|
| `/v1/skater-stats-leaders/{season}/{game-type}` | Skater statistics |
| `/v1/goalie-stats-leaders/{season}/{game-type}` | Goalie statistics |
| `/v1/player/{player}/landing` | Individual player information |
| `/v1/player/{player}/game-log/{season}/{game-type}` | Individual player game logs |

## 3. Initial Data Requirements

- Player identification and biographical information
- Team and position
- Season statistics
- Individual game statistics
- Goalie statistics

## 4. Development Notes

- Start with the NHL Web API.
- Inspect actual API responses before defining the data schema.
- Store raw API responses in the Bronze layer.
- Keep API extraction code separate from transformation logic.