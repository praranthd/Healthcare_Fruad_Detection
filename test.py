
import json
from pathlib import Path

from src.datascience.components.model_evaluation import (
    ModelEvaluation,
)
from src.datascience.utils.common import read_yaml


config = read_yaml("config/config.yaml")
evaluation_config = config["model_evaluation"]
training_config = config["model_training"]

evaluator = ModelEvaluation(
    model_path=(
        f"{training_config['artifact_dir']}/"
        f"{training_config['model_name']}.joblib"
    ),
    X_test_path="artifacts/data_transformation/X_test.csv",
    y_test_path="artifacts/data_transformation/y_test.csv",
    artifact_dir=evaluation_config["artifact_dir"],
    params_path="params.yaml",
)

result = evaluator.initiate_model_evaluation()

print("\nModel evaluation completed!")
print("\nMetrics:")

for name, value in result["metrics"].items():
    print(f"{name}: {value:.4f}")

print("\nConfusion matrix:")
print(result["confusion_matrix"])

print("\nReport saved at:", result["report_path"])

with open(
    result["report_path"],
    "r",
    encoding="utf-8",
) as file:
    report = json.load(file)

assert report["test_rows"] == 2000
assert 0 <= report["metrics"]["f1_score"] <= 1

print("\nAll evaluation checks passed!")
