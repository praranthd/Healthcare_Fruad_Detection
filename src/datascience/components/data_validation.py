import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.datascience.exception import HealthcareFraudException
from src.datascience.logging import logger
from src.datascience.utils.common import read_yaml


class DataValidation:
    """Validate an ingested dataset against the configured schema."""

    def __init__(
        self,
        data_path: str,
        schema_path: str,
        artifact_dir: str,
    ):
        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path)
        self.artifact_dir = Path(artifact_dir)

    def initiate_data_validation(self) -> Path:
        try:
            logger.info("Starting data validation")

            # 1. Check input files
            if not self.data_path.is_file():
                raise FileNotFoundError(
                    f"Dataset not found: {self.data_path}"
                )

            if not self.schema_path.is_file():
                raise FileNotFoundError(
                    f"Schema not found: {self.schema_path}"
                )

            # 2. Load the schema and dataset
            schema = read_yaml(str(self.schema_path))
            df = pd.read_csv(self.data_path, low_memory=False)

            errors = []
            warnings = []

            if df.empty:
                errors.append("Dataset contains no data rows")

            if not df.columns.is_unique:
                duplicates = (
                    df.columns[df.columns.duplicated()].tolist()
                )
                errors.append(
                    f"Duplicate column names: {duplicates}"
                )

            # 3. Check required columns
            required_columns = schema.get("required_columns", [])
            missing_required = sorted(
                set(required_columns) - set(df.columns)
            )

            if missing_required:
                errors.append(
                    f"Missing required columns: {missing_required}"
                )

            # 4. Check configured column groups
            declared_columns = (
                schema.get("numerical_columns", [])
                + schema.get("categorical_columns", [])
                + schema.get("date_columns", [])
            )

            missing_declared = sorted(
                set(declared_columns) - set(df.columns)
            )

            if missing_declared:
                errors.append(
                    "Schema references missing columns: "
                    f"{missing_declared}"
                )

            # 5. Check numeric columns
            numeric_issues = {}

            for column in schema.get("numerical_columns", []):
                if column not in df.columns:
                    continue

                original = df[column]
                converted = pd.to_numeric(
                    original, errors="coerce"
                )

                invalid_mask = original.notna() & converted.isna()
                invalid_count = int(invalid_mask.sum())

                if invalid_count:
                    numeric_issues[column] = invalid_count

            if numeric_issues:
                errors.append(
                    "Non-numeric values found: "
                    f"{numeric_issues}"
                )

            # 6. Check date columns
            date_issues = {}

            for column in schema.get("date_columns", []):
                if column not in df.columns:
                    continue

                original = df[column]
                converted = pd.to_datetime(
                    original, errors="coerce"
                )

                invalid_mask = original.notna() & converted.isna()
                invalid_count = int(invalid_mask.sum())

                if invalid_count:
                    date_issues[column] = invalid_count

            if date_issues:
                errors.append(
                    f"Invalid date values: {date_issues}"
                )

            # 7. Check target column and its values
            target_column = schema.get("target_column")

            target_distribution = {}

            if not target_column:
                errors.append(
                    "Schema does not define target_column"
                )
            elif target_column not in df.columns:
                errors.append(
                    f"Target column is missing: {target_column}"
                )
            else:
                target = df[target_column]
                allowed_values = schema.get("target_values", [])

                if target.isna().any():
                    errors.append(
                        f"Target column '{target_column}' contains "
                        f"{int(target.isna().sum())} missing values"
                    )

                invalid_targets = target[
                    target.notna()
                    & ~target.isin(allowed_values)
                ].unique().tolist()

                if invalid_targets:
                    errors.append(
                        f"Unexpected target values: {invalid_targets}"
                    )

                target_distribution = {
                    str(key): int(value)
                    for key, value in target.value_counts(
                        dropna=False
                    ).items()
                }

            # 8. Record acceptable data-quality warnings
            missing_values = {
                column: int(count)
                for column, count in df.isna().sum().items()
                if count > 0
            }

            if missing_values:
                warnings.append(
                    "Missing values found; review before training"
                )

            duplicate_rows = int(df.duplicated().sum())

            if duplicate_rows:
                warnings.append(
                    f"Found {duplicate_rows} completely duplicated rows"
                )

            # 9. Create and save the validation report
            self.artifact_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            report = {
                "is_valid": len(errors) == 0,
                "checked_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "dataset_path": str(self.data_path),
                "schema_path": str(self.schema_path),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "missing_values": missing_values,
                "duplicate_rows": duplicate_rows,
                "target_distribution": target_distribution,
                "errors": errors,
                "warnings": warnings,
            }

            report_path = (
                self.artifact_dir / "validation_report.json"
            )
            temporary_path = report_path.with_suffix(".tmp")

            try:
                with temporary_path.open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(report, file, indent=4)

                os.replace(temporary_path, report_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

            if errors:
                raise ValueError(
                    "Dataset validation failed: "
                    + "; ".join(errors)
                )

            logger.info(
                "Data validation passed: rows=%s, columns=%s",
                len(df),
                len(df.columns),
            )
            logger.info("Validation report: %s", report_path)

            return report_path

        except Exception as error:
            logger.exception("Data validation failed")

            raise HealthcareFraudException(
                f"Data validation failed: {error}",
                sys,
            ) from error
