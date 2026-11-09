from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/modeling/ros/splits/train.csv"
MODEL_PATH = ROOT / "models/ros_availability/remaining_games_model.joblib"

OUTPUT_DIR = ROOT / "models/ros_availability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "remaining_games"


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

print("Loading training data and availability model...")

df = pd.read_csv(TRAIN_PATH)
model = joblib.load(MODEL_PATH)

# Retrieve the exact features expected by the trained pipeline.
if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "The model does not expose feature_names_in_. "
        "Check the saved model pipeline."
    )

model_features = list(model.feature_names_in_)

missing = [
    feature for feature in model_features
    if feature not in df.columns
]

if missing:
    raise ValueError(f"Training dataset is missing model features: {missing}")

X = df[model_features].copy()
y = df[TARGET].copy()

predictions = model.predict(X)

df["predicted_remaining_games"] = predictions
df["actual_remaining_games"] = y
df["prediction_error"] = predictions - y
df["absolute_error"] = np.abs(predictions - y)

print(f"\nTraining records: {len(df):,}")
print(f"Model features: {len(model_features)}")
print(f"Overall MAE: {mean_absolute_error(y, predictions):.3f}")
print(
    f"Overall RMSE: "
    f"{np.sqrt(mean_squared_error(y, predictions)):.3f}"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def analyze_groups(data, group_column, group_name):

    if group_column not in data.columns:
        print(f"\nSkipping {group_name}: column unavailable.")
        return

    temp = data.dropna(subset=[group_column]).copy()

    results = (
        temp.groupby(group_column, observed=True)
        .agg(
            players=("actual_remaining_games", "size"),
            avg_actual_remaining=("actual_remaining_games", "mean"),
            avg_predicted_remaining=("predicted_remaining_games", "mean"),
            avg_prediction_error=("prediction_error", "mean"),
            MAE=("absolute_error", "mean"),
        )
        .reset_index()
    )

    print(f"\n{group_name.upper()}")
    print(results.round(3).to_string(index=False))


# ============================================================
# 1. RECENT GAME PARTICIPATION
# ============================================================

print("\n" + "=" * 75)
print("1. RECENT GAME PARTICIPATION")
print("=" * 75)

if "games_in_last_10" in df.columns:

    df["participation_group"] = pd.cut(
        df["games_in_last_10"],
        bins=[-1, 5, 7, 9, 10],
        labels=["0-5 games", "6-7 games", "8-9 games", "10 games"],
    )

    analyze_groups(df, "participation_group", "Participation")


# ============================================================
# 2. DAYS SINCE LAST GAME
# ============================================================

print("\n" + "=" * 75)
print("2. DAYS SINCE LAST GAME")
print("=" * 75)

if "days_since_last_game" in df.columns:

    df["absence_group"] = pd.cut(
        df["days_since_last_game"],
        bins=[-1, 3, 7, 14, 30, np.inf],
        labels=["0-3 days", "4-7 days", "8-14 days", "15-30 days", "30+ days"],
    )

    analyze_groups(df, "absence_group", "Absence")


# ============================================================
# 3. RECENT ICE TIME
# ============================================================

print("\n" + "=" * 75)
print("3. RECENT ICE TIME")
print("=" * 75)

if "time_on_ice_seconds_last_10_avg" in df.columns:

    df["toi_group"] = pd.qcut(
        df["time_on_ice_seconds_last_10_avg"],
        q=4,
        labels=["Lowest TOI", "Low-medium TOI", "High-medium TOI", "Highest TOI"],
        duplicates="drop",
    )

    analyze_groups(df, "toi_group", "Ice time")


# ============================================================
# 4. ICE TIME TREND
# ============================================================

print("\n" + "=" * 75)
print("4. ICE TIME TREND")
print("=" * 75)

if "time_on_ice_seconds_trend_5_vs_20" in df.columns:

    df["toi_trend_group"] = pd.cut(
        df["time_on_ice_seconds_trend_5_vs_20"],
        bins=[-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf],
        labels=[
            "Major decrease",
            "Moderate decrease",
            "Stable",
            "Moderate increase",
            "Major increase",
        ],
    )

    analyze_groups(df, "toi_trend_group", "TOI trend")


# ============================================================
# 5. ERROR BY CUTOFF
# ============================================================

print("\n" + "=" * 75)
print("5. ERROR BY CUTOFF")
print("=" * 75)

if "cutoff" in df.columns:

    results = (
        df.groupby("cutoff")
        .agg(
            players=("actual_remaining_games", "size"),
            MAE=("absolute_error", "mean"),
            RMSE=("prediction_error", lambda x: np.sqrt(np.mean(x ** 2))),
            mean_error=("prediction_error", "mean"),
        )
        .reset_index()
    )

    print(results.round(3).to_string(index=False))


# ============================================================
# 6. SAVE RESULTS
# ============================================================

output_path = OUTPUT_DIR / "availability_pattern_analysis.csv"

df.to_csv(output_path, index=False)

print("\n" + "=" * 75)
print("ANALYSIS COMPLETED")
print("=" * 75)

print(f"Detailed results: {output_path}")
print("No model was retrained.")
print("Only training data was used.")