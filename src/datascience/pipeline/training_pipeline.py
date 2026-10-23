
import sys
from pathlib import Path

import mlflow
import yaml

from src.datascience.components.data_ingestion import DataIngestion
from src.datascience.components.data_validation import DataValidation
from src.datascience.components.data_transformation import DataTransformation
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience.components.model_evaluation import ModelEvaluation

from src.datascience.exception import HealthcareFraudException
from src.datascience.logging import logger
from src.datascience.utils.common import read_yaml


class TrainingPipeline:

    def __init__(self):
        self.project_root = Path.cwd()

        self.config = read_yaml(
            self.project_root / "config" / "config.yaml"
        )

        self.params_path = self.project_root / "params.yaml"

    def _log_parameters(self):
        """Log model-training and evaluation parameters to MLflow."""

        with open(self.params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        for section, values in params.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    mlflow.log_param(f"{section}.{key}", str(value))
            else:
                mlflow.log_param(section, str(values))

    def _log_artifact_if_exists(self, path):
        """Log an existing file without logging raw or transformed datasets."""

        path = Path(path)

        if path.is_file():
            mlflow.log_artifact(str(path))

    def run_pipeline(self):

        try:
            logger.info("Healthcare Fraud Detection Pipeline Started")

            mlflow_config = self.config["mlflow"]

            mlflow.set_tracking_uri(mlflow_config["tracking_uri"])



            mlflow.set_experiment(
                mlflow_config["experiment_name"]
            )

            with mlflow.start_run(
                run_name="healthcare-fraud-training"
            ) as run:

                self._log_parameters()

                # Step 1: Data ingestion
                logger.info("Step 1: Data ingestion")

                ingestion_config = self.config["data_ingestion"]

                ingestion = DataIngestion(
                    raw_data_path=ingestion_config["raw_data_path"],
                    artifact_dir=ingestion_config["artifact_dir"],
                )

                ingestion_artifact = (
                    ingestion.initiate_data_ingestion()
                )

                # Step 2: Data validation
                logger.info("Step 2: Data validation")

                validation = DataValidation(
                    data_path=str(ingestion_artifact),
                    schema_path=self.config["data_validation"][
                        "schema_path"
                    ],
                    artifact_dir="artifacts/data_validation",
                )

                validation_artifact = (
                    validation.initiate_data_validation()
                )

                # Step 3: Data transformation
                logger.info("Step 3: Data transformation")

                transformation_config = self.config[
                    "data_transformation"
                ]

                transformation = DataTransformation(
                    data_path=str(ingestion_artifact),
                    schema_path=self.config["data_validation"][
                        "schema_path"
                    ],
                    artifact_dir=transformation_config["artifact_dir"],
                    test_size=transformation_config["test_size"],
                    random_state=transformation_config["random_state"],
                )

                transformation_artifact = (
                    transformation.initiate_data_transformation()
                )

                # Step 4: Model training
                logger.info("Step 4: Model training")

                training_config = self.config["model_training"]

                trainer = ModelTrainer(
                    X_train_path=transformation_artifact["X_train_path"],
                    y_train_path=transformation_artifact["y_train_path"],
                    params_path=str(self.params_path),
                    artifact_dir=training_config["artifact_dir"],
                    model_name=training_config["model_name"],
                )

                model_artifact = trainer.initiate_model_training()

                model_path = model_artifact["model_path"]

                # Step 5: Model evaluation
                logger.info("Step 5: Model evaluation")

                evaluation_config = self.config["model_evaluation"]

                evaluator = ModelEvaluation(
                    model_path=model_path,
                    X_test_path=transformation_artifact["X_test_path"],
                    y_test_path=transformation_artifact["y_test_path"],
                    artifact_dir=evaluation_config["artifact_dir"],
                    params_path=str(self.params_path),
                )

                evaluation_artifact = (
                    evaluator.initiate_model_evaluation()
                )

                # Log evaluation metrics
                metrics = evaluation_artifact["metrics"]

                for name, value in metrics.items():
                        mlflow.log_metric(name, float(value))

                # Log model and useful reports, not the dataset.
                self._log_artifact_if_exists(model_path)
                self._log_artifact_if_exists(validation_artifact)
                self._log_artifact_if_exists(
                    evaluation_artifact["report_path"]
                )
                self._log_artifact_if_exists(
                    evaluation_artifact["metrics_path"]
                )

                self._log_artifact_if_exists(
                    model_artifact.get("metadata_path", "")
                )
                self._log_artifact_if_exists(
                    transformation_artifact.get("metadata_path", "")
                )

                logger.info("MLflow tracking completed")
                logger.info("MLflow Run ID: %s", run.info.run_id)
                logger.info("Evaluation metrics: %s", metrics)

                return {
                    "run_id": run.info.run_id,
                    "model": model_artifact,
                    "evaluation": evaluation_artifact,
                    "metrics": metrics,
                }

        except Exception as error:
            logger.exception("Training pipeline failed")
            raise HealthcareFraudException(
                str(error), sys
            ) from error


if __name__ == "__main__":

    pipeline = TrainingPipeline()
    result = pipeline.run_pipeline()

    print("\nPipeline completed successfully!")
    print("MLflow Run ID:", result["run_id"])
    print("Model:", result["model"])
    print("Metrics:", result["metrics"])
