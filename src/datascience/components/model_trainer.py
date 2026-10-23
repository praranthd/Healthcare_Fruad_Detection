
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.datascience.exception import HealthcareFraudException
from src.datascience.logging import logger
from src.datascience.utils.common import read_yaml


class ModelTrainer:
    """Train and persist a machine-learning classification model."""

    def __init__(
        self,
        X_train_path: str,
        y_train_path: str,
        artifact_dir: str,
        model_name: str,
        params_path: str,
    ):
        self.X_train_path = Path(X_train_path)
        self.y_train_path = Path(y_train_path)
        self.artifact_dir = Path(artifact_dir)
        self.model_name = model_name
        self.params_path = Path(params_path)

    def initiate_model_training(self) -> dict:
        try:
            logger.info("Starting model training")

            # 1. Verify input artifacts
            for path in (
                self.X_train_path,
                self.y_train_path,
                self.params_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Required training input not found: {path}"
                    )

            # 2. Load training data and parameters
            X_train = pd.read_csv(self.X_train_path)
            y_train_df = pd.read_csv(self.y_train_path)
            params_config = read_yaml(str(self.params_path))
            params = params_config["model_training"]

            if X_train.empty or y_train_df.empty:
                raise ValueError("Training data cannot be empty")

            if len(X_train) != len(y_train_df):
                raise ValueError(
                    "Training features and labels have different row counts"
                )

            if X_train.isna().any().any():
                raise ValueError(
                    "Training features contain missing values"
                )

            if y_train_df.shape[1] != 1:
                raise ValueError(
                    "Expected exactly one target column in y_train"
                )

            y_train = y_train_df.iloc[:, 0]

            if y_train.isna().any():
                raise ValueError("Training target contains missing values")

            if y_train.nunique() < 2:
                raise ValueError(
                    "Training requires at least two target classes"
                )

            # 3. Build model from YAML parameters
            model = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=(
                    None
                    if params.get("max_depth") is None
                    else int(params["max_depth"])
                ),
                min_samples_split=int(
                    params["min_samples_split"]
                ),
                min_samples_leaf=int(params["min_samples_leaf"]),
                class_weight=params["class_weight"],
                random_state=int(params["random_state"]),
                n_jobs=int(params["n_jobs"]),
            )

            # 4. Fit only on training data
            model.fit(X_train, y_train)

            # 5. Save trained model
            self.artifact_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model_path = (
                self.artifact_dir / f"{self.model_name}.joblib"
            )
            temporary_model_path = (
                self.artifact_dir / f"{self.model_name}.tmp"
            )

            try:
                joblib.dump(model, temporary_model_path)
                os.replace(temporary_model_path, model_path)
            finally:
                if temporary_model_path.exists():
                    temporary_model_path.unlink()

            # 6. Save training metadata
            metadata = {
                "model_name": self.model_name,
                "model_type": "RandomForestClassifier",
                "trained_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "training_rows": int(len(X_train)),
                "feature_count": int(X_train.shape[1]),
                "feature_names": X_train.columns.tolist(),
                "target_distribution": {
                    str(key): int(value)
                    for key, value in y_train.value_counts().items()
                },
                "parameters": model.get_params(),
                "model_path": str(model_path),
            }

            metadata_path = (
                self.artifact_dir / "training_metadata.json"
            )
            temporary_metadata_path = metadata_path.with_suffix(
                ".tmp"
            )

            try:
                with temporary_metadata_path.open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(metadata, file, indent=4)

                os.replace(temporary_metadata_path, metadata_path)
            finally:
                if temporary_metadata_path.exists():
                    temporary_metadata_path.unlink()

            logger.info("Model training completed successfully")
            logger.info("Trained model saved to: %s", model_path)

            return {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "training_rows": len(X_train),
                "feature_count": X_train.shape[1],
            }

        except Exception as error:
            logger.exception("Model training failed")

            raise HealthcareFraudException(
                f"Model training failed: {error}",
                sys,
            ) from error
