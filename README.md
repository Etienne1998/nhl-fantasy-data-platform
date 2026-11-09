# NHL Fantasy Intelligence Platform

An end-to-end data engineering and machine learning project that transforms NHL data into rest-of-season fantasy hockey projections and player rankings.

The platform combines historical player performance, game-level statistics, feature engineering and predictive models to estimate the remaining fantasy value of NHL players.

## Project Overview

The objective is to build a data platform that helps fantasy hockey managers identify valuable players throughout the NHL season.

The system uses a two-stage machine learning approach:

1. **Player availability model:** estimates how many games a player is expected to play for the remainder of the season.
2. **Per-game performance models:** predict the player's expected statistical production per game.
3. **Fantasy value engine:** combines both predictions and applies configurable fantasy scoring rules to calculate projected rest-of-season fantasy points.

The resulting projections can be used to generate and evaluate fantasy hockey player rankings.

## Architecture

```text
NHL API
   |
   v
Data Ingestion
   |
   v
Azure Blob Storage
   |
   v
Bronze Layer (Raw Data)
   |
   v
Silver Layer (Cleaned Data)
   |
   v
Feature Engineering
   |
   v
Machine Learning
   |
   +--> Player Availability Model
   |
   +--> Per-Game Performance Models
   |
   v
Fantasy Value Engine
   |
   v
Rest-of-Season Player Rankings
```

## Technology Stack

| Area | Technologies |
|---|---|
| Data sources | NHL Web API |
| Cloud storage | Microsoft Azure Blob Storage |
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, Random Forest |
| Model serialization | Joblib |
| Data formats | JSON, CSV |
| Version control | Git, GitHub |

## Machine Learning Approach

The platform uses historical NHL data from the 2021–22 through 2025–26 seasons.

The modeling dataset contains approximately 236,000 player-game observations.

### Two-Stage Prediction Model

**Stage 1 — Remaining Games Prediction**

A Random Forest model estimates the number of games each player is expected to play before the end of the season.

**Stage 2 — Per-Game Performance Prediction**

A collection of Random Forest models predicts expected per-game production across 11 statistical targets.

Predicted per-game production is multiplied by expected remaining games to estimate rest-of-season statistics.

The fantasy engine then converts projected statistics into fantasy points using configurable scoring profiles.

### Model Evaluation

Chronological datasets are used to evaluate the system:

| Dataset | Seasons |
|---|---|
| Training | 2021–22 to 2023–24 |
| Validation | 2024–25 |
| Out-of-sample test | 2025–26 |

Test performance for the 2025–26 season, using a 25% season cutoff:

| Metric | Result |
|---|---:|
| Mean Absolute Error | 23.11 |
| Root Mean Squared Error | 30.69 |
| Spearman Rank Correlation | 0.735 |
| Actual top-10 players captured | 4/10 |
| Actual top-25 players captured | 13/25 |
| Actual top-50 players captured | 27/50 |

The evaluation considers both prediction accuracy and ranking quality.

## Repository Structure

```text
.
├── config/                 # Fantasy scoring profiles
├── docs/                   # Architecture and data documentation
├── models/
│   ├── ros_availability/   # Remaining-games model
│   └── ros_per_game/       # Per-game prediction models
├── notebooks/              # Exploratory analysis
├── sql/                    # SQL transformations and queries
├── src/
│   ├── ingestion/           # NHL API data extraction
│   ├── transformations/     # Bronze-to-Silver processing
│   ├── features/            # Feature engineering
│   ├── modeling/            # Training and evaluation
│   ├── fantasy/             # Fantasy scoring and ranking engine
│   └── utils/               # Shared utilities
├── tests/                   # Tests
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Etienne1998/nhl-fantasy-data-platform.git
cd nhl-fantasy-data-platform
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

Windows:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate fantasy rankings

The ranking generation script uses the trained models and prepared feature data.

```bash
python src/modeling/generate_fantasy_rankings.py
```

Note: The trained models are included in the repository. Historical raw and processed datasets are not included.

## Current Limitations

- Player availability predictions do not incorporate a dedicated injury data source.
- Fantasy scoring depends on the statistics available in the underlying dataset; some league categories may not yet be modeled.
- Rankings are historical projections generated from season snapshots, not live forecasts.
- Automated scheduling and production orchestration remain future development opportunities.

## Project Goals

This project demonstrates practical data engineering, cloud storage integration, feature engineering, predictive modeling and the translation of machine learning outputs into a usable fantasy hockey application.

**Author:** Étienne Fafard