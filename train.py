"""Standalone training script for CardioSense.

Usage
-----
    python train.py                          # train + hold-out eval
    python train.py --cv                     # train + 5-fold cross-validation
    python train.py --cv --folds 10          # train + 10-fold CV
    python train.py --dataset /path/to.csv   # custom dataset path
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model import (
    DATASET_PATH,
    cross_validate_model,
    load_dataset,
    train_and_save_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CardioSense ensemble model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the cardiovascular disease CSV dataset.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run stratified k-fold cross-validation after training.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5).",
    )
    args = parser.parse_args()

    dataset_path: Path = args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Dataset : {dataset_path}")
    print(f"CV      : {'yes — ' + str(args.folds) + ' folds' if args.cv else 'no'}")
    print("=" * 50)

    if args.cv:
        print(f"\n[1/2] Running {args.folds}-fold cross-validation…\n")
        df = load_dataset(dataset_path)
        cv_results = cross_validate_model(df, n_splits=args.folds)
        print(f"\nCV Mean AUC : {cv_results['mean_auc']:.4f}")
        print(f"CV Mean F1  : {cv_results['mean_f1']:.4f}")
        print("\n[2/2] Training final model on full dataset…\n")
    else:
        print("\n[1/1] Training model…\n")

    bundle = train_and_save_artifacts(dataset_path)
    print("\nArtifacts saved to ./artifacts/")
    print("  - ensemble_model.joblib")
    print("  - scaler.joblib")
    print("  - metadata.joblib")
    print("\nRun the API with:  uvicorn app:app --reload")


if __name__ == "__main__":
    main()
