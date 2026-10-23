
from pathlib import Path

import pandas as pd

from src.datascience.logging import logger
from src.datascience.exception import HealthcareFraudException


class DataIngestion:
    def __init__(self, raw_data_path: str, artifact_dir: str):
        self.raw_data_path = Path(raw_data_path)
        self.artifact_dir = Path(artifact_dir)

    def initiate_data_ingestion(self) -> Path:
        try:
            # 1. Check whether the source CSV exists
            if not self.raw_data_path.is_file():
                raise FileNotFoundError(
                    f"Dataset not found: {self.raw_data_path}"
                )

            logger.info("Starting data ingestion")

            # 2. Read the local CSV
            df = pd.read_csv(self.raw_data_path)

            # 3. Perform basic checks
            if df.empty:
                raise ValueError("The dataset is empty")

            if not df.columns.is_unique:
                raise ValueError("Duplicate column names found")

            logger.info("Dataset loaded successfully")
            logger.info("Dataset shape: %s", df.shape)

            # 4. Create the output directory
            self.artifact_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # 5. Save the ingestion artifact
            output_path = (
                self.artifact_dir / "healthcare_data.csv"
            )

            df.to_csv(output_path, index=False)

            logger.info(
                "Ingestion artifact saved to: %s",
                output_path,
            )

            return output_path

        except Exception as error:
            logger.exception("Data ingestion failed")
            raise HealthcareFraudException(
                str(error),
                __import__("sys"),
            ) from error
