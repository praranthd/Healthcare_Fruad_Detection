
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Project root: healthcare-fraud-detection
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "data_transformation"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_yaml(CONFIG_PATH)
    params = load_yaml(PARAMS_PATH)

    # Load the same transformed data used by your existing pipeline.
    X_train = pd.read_csv(ARTIFACT_DIR / "X_train.csv")
    X_test = pd.read_csv(ARTIFACT_DIR / "X_test.csv")

    y_train = pd.read_csv(ARTIFACT_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(ARTIFACT_DIR / "y_test.csv").squeeze("columns")

    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    model_params = params["model_training"]

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=model_params.get("random_state", 42),
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 200),
            max_depth=model_params.get("max_depth", 15),
            min_samples_split=model_params.get("min_samples_split", 2),
            min_samples_leaf=model_params.get("min_samples_leaf", 2),
            class_weight="balanced",
            random_state=model_params.get("random_state", 42),
            n_jobs=model_params.get("n_jobs", -1),
        ),
    }

    results = []

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"comparison-{model_name}"):
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }

            mlflow.log_param("model_name", model_name)
            mlflow.log_params({
                key: str(value)
                for key, value in model.get_params().items()
                if isinstance(value, (str, int, float, bool))
                or value is None
            })
            mlflow.log_metrics({
                key: float(value)
                for key, value in metrics.items()
            })
            mlflow.sklearn.log_model(sk_model=model,
                                      name="model",
                                      serialization_format="pickle",)

            print(f"\n{model_name}")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

            results.append({
                "model_name": model_name,
                **metrics,
            })

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score", ascending=False
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
