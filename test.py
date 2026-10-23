
from src.datascience.components.data_ingestion import DataIngestion

ingestion = DataIngestion(
    raw_data_path="data/raw/healthcare_fraud_detection.csv",
    artifact_dir="artifacts/data_ingestion",
)

artifact_path = ingestion.initiate_data_ingestion()

print("Ingestion completed successfully!")
print("Artifact saved at:", artifact_path)
