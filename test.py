
import json
from pathlib import Path

import pandas as pd

from src.datascience.components.data_transformation import (
    DataTransformation,
)

config_path = Path("config/config.yaml")

from src.datascience.utils.common import read_yaml

config = read_yaml(str(config_path))

transformation_config = config["data_transformation"]

transformer = DataTransformation(
    data_path="artifacts/data_ingestion/healthcare_data.csv",
    schema_path="config/schema.yaml",
    artifact_dir=transformation_config["artifact_dir"],
    test_size=transformation_config["test_size"],
    random_state=transformation_config["random_state"],
)

artifacts = transformer.initiate_data_transformation()

output_dir = Path(artifacts["artifact_dir"])

X_train = pd.read_csv(output_dir / "X_train.csv")
X_test = pd.read_csv(output_dir / "X_test.csv")
y_train = pd.read_csv(output_dir / "y_train.csv")
y_test = pd.read_csv(output_dir / "y_test.csv")

with open(
    output_dir / "transformation_metadata.json",
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

print("Data transformation passed!")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)
print("Train class distribution:")
print(y_train["Is_Fraud"].value_counts())
print("Test class distribution:")
print(y_test["Is_Fraud"].value_counts())
print("Preprocessor:", artifacts["preprocessor_path"])

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)
assert X_train.shape[1] == X_test.shape[1]
assert not X_train.isna().any().any()
assert not X_test.isna().any().any()
assert not y_train["Is_Fraud"].isna().any()
assert not y_test["Is_Fraud"].isna().any()
assert "Claim_ID" not in X_train.columns
assert "Provider_ID" not in X_train.columns
assert "Claim_Status" not in X_train.columns
assert "Approved_Amount" not in X_train.columns

print("All transformation checks passed!")
