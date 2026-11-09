
import pandas as pd
import joblib
import time

from pathlib import Path
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_PATH = Path("data/modeling/splits/validation.csv")
MODEL_DIR = Path("models/random_forest")
OUTPUT_DIR = Path("data/modeling/analysis")

TARGETS = [
    "shots_on_goal",
    "time_on_ice_seconds"
]

SAMPLE_SIZE = 10000
N_REPEATS = 5
RANDOM_STATE = 42
N_JOBS = 2


# ============================================================
# MAIN
# ============================================================

def main():

    print("Loading validation dataset...")

    validation = pd.read_csv(VALIDATION_PATH)

    print(f"Validation records: {len(validation)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for target in TARGETS:

        print("\n" + "=" * 60)
        print(f"Analyzing: {target}")
        print("=" * 60)

        start_time = time.time()

        model_path = MODEL_DIR / f"{target}_model.joblib"

        print("Loading model...")
        model = joblib.load(model_path)

        # Use exactly the raw features expected by the pipeline
        feature_names = list(model.feature_names_in_)

        X = validation[feature_names]
        y = validation[target]

        # Representative sample to limit computation
        sample_size = min(SAMPLE_SIZE, len(X))

        X_sample = X.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

        y_sample = y.loc[X_sample.index]

        print(f"Sample records: {len(X_sample)}")
        print(f"Features: {len(feature_names)}")
        print("Calculating permutation importance...")
        print("This may take a few minutes.")

        results = permutation_importance(
            estimator=model,
            X=X_sample,
            y=y_sample,
            scoring="neg_mean_absolute_error",
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )

        # Positive importance means shuffling the feature
        # increases MAE, indicating useful predictive information.
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": results.importances_mean,
            "importance_std": results.importances_std
        })

        importance_df = importance_df.sort_values(
            by="importance_mean",
            ascending=False
        )

        importance_df["target"] = target

        # Save individual target results
        output_path = OUTPUT_DIR / (
            f"{target}_permutation_importance.csv"
        )

        importance_df.to_csv(output_path, index=False)

        all_results.append(importance_df)

        print("\nTop 15 features:")

        print(
            importance_df[
                ["feature", "importance_mean", "importance_std"]
            ].head(15).to_string(index=False)
        )

        elapsed = time.time() - start_time

        print(f"\nCompleted in {elapsed:.1f} seconds")
        print(f"Saved: {output_path}")

    # Save combined results
    combined = pd.concat(all_results, ignore_index=True)

    combined_path = OUTPUT_DIR / "permutation_importance_results.csv"

    combined.to_csv(combined_path, index=False)

    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"Results saved to: {combined_path}")


if __name__ == "__main__":
    main()
