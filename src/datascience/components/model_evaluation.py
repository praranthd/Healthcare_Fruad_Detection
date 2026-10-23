
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.datascience.exception import HealthcareFraudException
from src.datascience.logging import logger
from src.datascience.utils.common import read_yaml


class ModelEvaluation:
    """Evaluate a trained classifier on held-out test data."""

    def __init__(
        self,
        model_path: str,
        X_test_path: str,
        y_test_path: str,
        artifact_dir: str,
        params_path: str,
    ):
        self.model_path = Path(model_path)
        self.X_test_path = Path(X_test_path)
        self.y_test_path = Path(y_test_path)
        self.artifact_dir = Path(artifact_dir)
        self.params_path = Path(params_path)

    def initiate_model_evaluation(self) -> dict:
        try:
            logger.info("Starting model evaluation")

            # 1. Verify required files exist.
            required_paths = [
                self.model_path,
                self.X_test_path,
                self.y_test_path,
                self.params_path,
            ]

            for path in required_paths:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Required evaluation file not found: {path}"
                    )

            # 2. Load model, test data, and parameters.
            model = joblib.load(self.model_path)
            X_test = pd.read_csv(self.X_test_path)
            y_test_df = pd.read_csv(self.y_test_path)

            params = read_yaml(
                str(self.params_path)
            )["model_evaluation"]

            threshold = float(
                params.get("prediction_threshold", 0.5)
            )
            positive_label = int(
                params.get("positive_label", 1)
            )

            if not 0 < threshold < 1:
                raise ValueError(
                    "prediction_threshold must be between 0 and 1"
                )

            if X_test.empty or y_test_df.empty:
                raise ValueError("Test data cannot be empty")

            if len(X_test) != len(y_test_df):
                raise ValueError(
                    "Test features and labels have different row counts"
                )

            if y_test_df.shape[1] != 1:
                raise ValueError(
                    "Expected exactly one target column in y_test"
                )

            y_test = y_test_df.iloc[:, 0]

            if y_test.isna().any():
                raise ValueError("Test target contains missing values")

            if set(y_test.unique()) != {0, 1}:
                raise ValueError(
                    "This evaluation expects both target classes: 0 and 1"
                )

            # 3. Confirm the model expects the same feature schema.
            expected_features = getattr(
                model, "feature_names_in_", None
            )

            if expected_features is not None:
                if list(expected_features) != list(X_test.columns):
                    raise ValueError(
                        "Test feature names or order do not match "
                        "the features used to train the model"
                    )

            # 4. Predict using the held-out test set.
            predictions = model.predict(X_test)

            # Obtain the probability associated with the fraud class.
            classes = list(model.classes_)

            if positive_label not in classes:
                raise ValueError(
                    f"Positive class {positive_label} not found "
                    "in the trained model"
                )

            positive_index = classes.index(positive_label)
            probabilities = model.predict_proba(X_test)[
                :, positive_index
            ]

            threshold_predictions = (
                probabilities >= threshold
            ).astype(int)

            # 5. Calculate classification metrics.
            metrics = {
                "accuracy": float(
                    accuracy_score(y_test, predictions)
                ),
                "precision": float(
                    precision_score(
                        y_test,
                        threshold_predictions,
                        pos_label=positive_label,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        threshold_predictions,
                        pos_label=positive_label,
                        zero_division=0,
                    )
                ),
                "f1_score": float(
                    f1_score(
                        y_test,
                        threshold_predictions,
                        pos_label=positive_label,
                        zero_division=0,
                    )
                ),
                "average_precision": float(
                    average_precision_score(
                        y_test == positive_label,
                        probabilities,
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(
                        y_test == positive_label,
                        probabilities,
                    )
                ),
            }

            # Also record metrics at the model's default threshold.
            default_metrics = {
                "precision": float(
                    precision_score(
                        y_test,
                        predictions,
                        pos_label=positive_label,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test,
                        predictions,
                        pos_label=positive_label,
                        zero_division=0,
                    )
                ),
                "f1_score": float(
                    f1_score(
                        y_test,
                        predictions,
                        pos_label=positive_label,
                        zero_division=0,
                    )
                ),
            }

            matrix = confusion_matrix(
                y_test,
                threshold_predictions,
                labels=[0, 1],
            )

            report = classification_report(
                y_test,
                threshold_predictions,
                labels=[0, 1],
                target_names=["not_fraud", "fraud"],
                output_dict=True,
                zero_division=0,
            )

            # 6. Save evaluation results.
            self.artifact_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            evaluation = {
                "evaluated_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "model_path": str(self.model_path),
                "test_rows": int(len(X_test)),
                "prediction_threshold": threshold,
                "positive_label": positive_label,
                "metrics": metrics,
                "default_threshold_metrics": default_metrics,
                "confusion_matrix": matrix.tolist(),
                "classification_report": report,
            }

            report_path = (
                self.artifact_dir / "evaluation_report.json"
            )
            metrics_path = (
                self.artifact_dir / "metrics.json"
            )

            self._write_json_atomically(
                evaluation,
                report_path,
            )
            self._write_json_atomically(
                metrics,
                metrics_path,
            )

            logger.info("Model evaluation completed")
            logger.info("Evaluation metrics: %s", metrics)
            logger.info("Evaluation report: %s", report_path)

            return {
                "report_path": str(report_path),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
                "confusion_matrix": matrix.tolist(),
            }

        except Exception as error:
            logger.exception("Model evaluation failed")

            raise HealthcareFraudException(
                f"Model evaluation failed: {error}",
                sys,
            ) from error

    @staticmethod
    def _write_json_atomically(
        data: dict,
        output_path: Path,
    ) -> None:
        temporary_path = output_path.with_suffix(".tmp")

        try:
            with temporary_path.open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(data, file, indent=4)

            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
