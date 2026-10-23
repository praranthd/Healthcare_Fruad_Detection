
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datascience.exception import HealthcareFraudException
from src.datascience.logging import logger
from src.datascience.utils.common import read_yaml


class DataTransformation:
    """Transform validated data and save reusable ML artifacts."""

    # Exclude identifiers and fields that may reveal the claim outcome.
    EXCLUDED_COLUMNS = {
        "Claim_ID",
        "Provider_ID",
        "Claim_Status",
        "Approved_Amount",
    }

    def __init__(
        self,
        data_path: str,
        schema_path: str,
        artifact_dir: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path)
        self.artifact_dir = Path(artifact_dir)
        self.test_size = test_size
        self.random_state = random_state

    def initiate_data_transformation(self) -> dict:
        try:
            logger.info("Starting data transformation")

            if not self.data_path.is_file():
                raise FileNotFoundError(
                    f"Input dataset not found: {self.data_path}"
                )

            schema = read_yaml(str(self.schema_path))
            df = pd.read_csv(self.data_path, low_memory=False)

            target_column = schema["target_column"]

            if target_column not in df.columns:
                raise ValueError(
                    f"Target column is missing: {target_column}"
                )

            if df[target_column].isna().any():
                raise ValueError(
                    "Target contains missing values. "
                    "Resolve them before training."
                )

            allowed_targets = schema.get("target_values", [0, 1])
            target = pd.to_numeric(
                df[target_column], errors="raise"
            )

            if not target.isin(allowed_targets).all():
                raise ValueError(
                    "Target contains values outside the configured "
                    f"set: {allowed_targets}"
                )

            # Separate features and target.
            X = df.drop(columns=[target_column]).copy()
            y = target.astype(int)

            # Remove identifiers and potentially post-outcome fields.
            excluded = self.EXCLUDED_COLUMNS.intersection(X.columns)
            X = X.drop(columns=list(excluded))

            # Convert submission date into useful numeric features.
            # The original date string is not passed to the model.
            date_columns = [
                column
                for column in schema.get("date_columns", [])
                if column in X.columns
            ]

            for column in date_columns:
                dates = pd.to_datetime(X[column], errors="coerce")
                X[f"{column}_month"] = dates.dt.month
                X[f"{column}_day_of_week"] = dates.dt.dayofweek
                X = X.drop(columns=[column])

            # Use the schema to select supported model features.
            numeric_columns = [
                column
                for column in schema.get("numerical_columns", [])
                if column in X.columns
            ]

            categorical_columns = [
                column
                for column in schema.get("categorical_columns", [])
                if column in X.columns
            ]

            # Add engineered date features to numeric inputs.
            numeric_columns.extend(
                [
                    column
                    for column in X.columns
                    if column.endswith(
                        ("_month", "_day_of_week")
                    )
                ]
            )

            numeric_columns = list(dict.fromkeys(numeric_columns))
            categorical_columns = list(
                dict.fromkeys(categorical_columns)
            )

            selected_columns = numeric_columns + categorical_columns

            if not selected_columns:
                raise ValueError("No model features were selected")

            # Fail early if the schema does not account for all features.
            unassigned_columns = sorted(
                set(X.columns) - set(selected_columns)
            )

            if unassigned_columns:
                raise ValueError(
                    "Features are not assigned to numeric or "
                    f"categorical groups: {unassigned_columns}"
                )

            X = X[selected_columns]

            # Split before fitting any preprocessing steps.
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )

            # Numeric features: median imputation and scaling.
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                    ("scaler", StandardScaler()),
                ]
            )

            # Categorical features: most-frequent imputation
            # and one-hot encoding.
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        numeric_pipeline,
                        numeric_columns,
                    ),
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_columns,
                    ),
                ],
                remainder="drop",
            )

            # Fit on training data only. Never fit on X_test.
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            feature_names = preprocessor.get_feature_names_out()

            X_train_transformed = pd.DataFrame(
                X_train_transformed,
                columns=feature_names,
                index=X_train.index,
            )
            X_test_transformed = pd.DataFrame(
                X_test_transformed,
                columns=feature_names,
                index=X_test.index,
            )

            # Prepare output directory.
            self.artifact_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Save artifacts.
            joblib.dump(
                preprocessor,
                self.artifact_dir / "preprocessor.joblib",
            )

            X_train_transformed.to_csv(
                self.artifact_dir / "X_train.csv",
                index=False,
            )
            X_test_transformed.to_csv(
                self.artifact_dir / "X_test.csv",
                index=False,
            )

            y_train.rename(target_column).to_csv(
                self.artifact_dir / "y_train.csv",
                index=False,
            )
            y_test.rename(target_column).to_csv(
                self.artifact_dir / "y_test.csv",
                index=False,
            )

            metadata = {
                "created_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "source_rows": int(len(df)),
                "target_column": target_column,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "transformed_feature_count": int(
                    len(feature_names)
                ),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "excluded_columns": sorted(excluded),
                "train_target_distribution": {
                    str(k): int(v)
                    for k, v in y_train.value_counts().items()
                },
                "test_target_distribution": {
                    str(k): int(v)
                    for k, v in y_test.value_counts().items()
                },
            }

            metadata_path = (
                self.artifact_dir / "transformation_metadata.json"
            )
            temporary_path = metadata_path.with_suffix(".tmp")

            try:
                with temporary_path.open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(metadata, file, indent=4)

                os.replace(temporary_path, metadata_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

            logger.info(
                "Transformation completed: train=%s, test=%s, "
                "features=%s",
                len(X_train),
                len(X_test),
                len(feature_names),
            )

            return {
                "artifact_dir": str(self.artifact_dir),
                "preprocessor_path": str(
                    self.artifact_dir / "preprocessor.joblib"
                ),
                "metadata_path": str(metadata_path),
                "X_train_path": str(
                    self.artifact_dir / "X_train.csv"
                ),
                "X_test_path": str(
                    self.artifact_dir / "X_test.csv"
                ),
                "y_train_path": str(
                    self.artifact_dir / "y_train.csv"
                ),
                "y_test_path": str(
                    self.artifact_dir / "y_test.csv"
                ),
            }

        except Exception as error:
            logger.exception("Data transformation failed")

            raise HealthcareFraudException(
                f"Data transformation failed: {error}",
                sys,
            ) from error
