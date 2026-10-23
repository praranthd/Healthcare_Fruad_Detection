
import json
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root: healthcare-fraud-detection/
PROJECT_ROOT = Path.cwd()

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

TRANSFORMED_DIR = (
    PROJECT_ROOT / "artifacts" / "data_transformation"
)

RESULTS_DIR = (
    PROJECT_ROOT / "artifacts" / "model_evaluation"
)

RESULTS_PATH = RESULTS_DIR / "model_comparison_results.json"

REGISTERED_MODEL_NAME = "HealthcareFraudDetectionModel"


def load_yaml(path: Path) -> dict:
    """Read a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return data


def load_target(path: Path, target_column: str) -> pd.Series:
    """Load target labels from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Target file not found: {path}")

    df = pd.read_csv(path)

    if target_column in df.columns:
        target = df[target_column]
    elif df.shape[1] == 1:
        target = df.iloc[:, 0]
    else:
        raise ValueError(
            f"Could not identify target column '{target_column}' "
            f"in {path}. Available columns: {list(df.columns)}"
        )

    return target


def calculate_metrics(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    positive_label: int,
) -> tuple[dict, list]:
    """Calculate classification metrics using the configured threshold."""
    probabilities = model.predict_proba(X_test)

    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(
            f"Positive label {positive_label} not found in model classes: "
            f"{classes}"
        )

    positive_index = classes.index(positive_label)
    positive_probabilities = probabilities[:, positive_index]

    negative_label = next(
        label for label in classes if label != positive_label
    )

    predictions = [
        positive_label if probability >= threshold else negative_label
        for probability in positive_probabilities
    ]

    y_binary = (y_test == positive_label).astype(int)
    pred_binary = (pd.Series(predictions, index=y_test.index) == positive_label).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
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
        "average_precision": float(
            average_precision_score(y_binary, positive_probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_binary, positive_probabilities)
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[negative_label, positive_label],
    ).tolist()

    return metrics, matrix


def main():
    """Train, compare, register, and promote the best accepted model."""
    logger.info("Starting model comparison pipeline.")

    config = load_yaml(CONFIG_PATH)
    params = load_yaml(PARAMS_PATH)

    mlflow_config = config["mlflow"]
    training_params = params["model_training"]
    evaluation_params = params["model_evaluation"]
    quality_params = params["model_quality"]

    target_column = config["data_transformation"]["target_column"]
    threshold = float(
        evaluation_params.get("prediction_threshold", 0.5)
    )
    positive_label = int(
        evaluation_params.get("positive_label", 1)
    )

    min_f1_score = float(quality_params["min_f1_score"])
    min_recall = float(quality_params["min_recall"])

    # ---------------------------------------------------------
    # 1. Configure MLflow
    # ---------------------------------------------------------
    tracking_uri = mlflow_config["tracking_uri"]
    experiment_name = mlflow_config["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow tracking URI: %s", tracking_uri)

    # ---------------------------------------------------------
    # 2. Load transformed datasets
    # ---------------------------------------------------------
    X_train = pd.read_csv(TRANSFORMED_DIR / "X_train.csv")
    X_test = pd.read_csv(TRANSFORMED_DIR / "X_test.csv")

    y_train = load_target(
        TRANSFORMED_DIR / "y_train.csv",
        target_column,
    )
    y_test = load_target(
        TRANSFORMED_DIR / "y_test.csv",
        target_column,
    )

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different row counts.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different row counts.")

    # Ensure the same features and feature order are used.
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train and X_test feature columns do not match."
        )

    logger.info("Training rows: %s", len(X_train))
    logger.info("Testing rows: %s", len(X_test))
    logger.info("Feature count: %s", X_train.shape[1])

    # ---------------------------------------------------------
    # 3. Define candidate models
    # ---------------------------------------------------------
    random_forest = RandomForestClassifier(
        n_estimators=int(training_params["n_estimators"]),
        max_depth=training_params.get("max_depth"),
        min_samples_split=int(
            training_params["min_samples_split"]
        ),
        min_samples_leaf=int(
            training_params["min_samples_leaf"]
        ),
        class_weight=training_params.get(
            "class_weight", "balanced"
        ),
        random_state=int(training_params["random_state"]),
        n_jobs=int(training_params.get("n_jobs", -1)),
    )

    logistic_regression = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(
                        training_params["random_state"]
                    ),
                ),
            ),
        ]
    )

    candidates = [
        ("RandomForest", random_forest),
        ("LogisticRegression", logistic_regression),
    ]

    results = []

    # ---------------------------------------------------------
    # 4. Train, evaluate, and log each candidate
    # ---------------------------------------------------------
    for model_name, model in candidates:
        logger.info("Training model: %s", model_name)

        with mlflow.start_run(run_name=model_name) as run:
            model.fit(X_train, y_train)

            metrics, matrix = calculate_metrics(
                model=model,
                X_test=X_test,
                y_test=y_test,
                threshold=threshold,
                positive_label=positive_label,
            )

            passed = (
                metrics["f1_score"] >= min_f1_score
                and metrics["recall"] >= min_recall
            )

            status = "PASSED" if passed else "REJECTED"

            # Log model parameters.
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("prediction_threshold", threshold)
            mlflow.log_param("positive_label", positive_label)
            mlflow.log_param("min_f1_score", min_f1_score)
            mlflow.log_param("min_recall", min_recall)

            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))

            mlflow.log_metric("quality_gate_passed", int(passed))
            mlflow.log_param("quality_gate_status", status)

            # Log the fitted model in MLflow.
            logged_model = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle",
            )

            result = {
                "model_name": model_name,
                "metrics": metrics,
                "confusion_matrix": matrix,
                "quality_gate": status,
                "run_id": run.info.run_id,
                "model_uri": logged_model.model_uri,
            }

            results.append(result)

            logger.info(
                "%s | F1: %.4f | Recall: %.4f | Quality gate: %s",
                model_name,
                metrics["f1_score"],
                metrics["recall"],
                status,
            )

            logger.info("Confusion matrix: %s", matrix)

    # ---------------------------------------------------------
    # 5. Save comparison results locally
    # ---------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    logger.info("Comparison results saved to: %s", RESULTS_PATH)

    # ---------------------------------------------------------
    # 6. Select the best model that passed the quality gate
    # ---------------------------------------------------------
    accepted_models = [
        result
        for result in results
        if result["quality_gate"] == "PASSED"
    ]

    if not accepted_models:
        raise RuntimeError(
            "No model passed the quality gate. "
            "No model was registered or promoted."
        )

    best_model = max(
        accepted_models,
        key=lambda result: result["metrics"]["f1_score"],
    )

    logger.info("Best accepted model: %s", best_model["model_name"])
    logger.info(
        "Best accepted F1 score: %.4f",
        best_model["metrics"]["f1_score"],
    )

    # ---------------------------------------------------------
    # 7. Register the selected model
    # ---------------------------------------------------------
    client = MlflowClient()

    registered_model = mlflow.register_model(
        model_uri=best_model["model_uri"],
        name=REGISTERED_MODEL_NAME,
    )

    # Wait until MLflow finishes creating the model version.
    client = MlflowClient()
    client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=registered_model.version,
    )

    # ---------------------------------------------------------
    # 8. Promote the registered version using an alias
    # ---------------------------------------------------------
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    # ---------------------------------------------------------
    # 9. Print final summary
    # ---------------------------------------------------------
    print("\n" + "=" * 55)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 55)

    for result in results:
        print(f"\nModel: {result['model_name']}")
        print(f"Accuracy:  {result['metrics']['accuracy']:.4f}")
        print(f"Precision: {result['metrics']['precision']:.4f}")
        print(f"Recall:    {result['metrics']['recall']:.4f}")
        print(f"F1 score:  {result['metrics']['f1_score']:.4f}")
        print(f"ROC AUC:   {result['metrics']['roc_auc']:.4f}")
        print(f"Quality gate: {result['quality_gate']}")

    print("\n" + "=" * 55)
    print("MODEL REGISTRATION SUMMARY")
    print("=" * 55)
    print(f"Selected model: {best_model['model_name']}")
    print(f"Registered name: {REGISTERED_MODEL_NAME}")
    print(f"Registered version: {registered_model.version}")
    print("Alias: champion")
    print(f"F1 score: {best_model['metrics']['f1_score']:.4f}")
    print(f"Recall: {best_model['metrics']['recall']:.4f}")
    print(f"Results file: {RESULTS_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()
