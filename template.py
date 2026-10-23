
import logging
from pathlib import Path

# Main Python package
PROJECT_NAME = "datascience"

# Folders and files to generate
list_of_files = [
    # Configuration
    "config/config.yaml",
    "config/schema.yaml",
    "params.yaml",

    # Research
    "research/.gitkeep",

    # Main package
    f"src/{PROJECT_NAME}/__init__.py",

    # ETL and ML components
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/components/data_ingestion.py",
    f"src/{PROJECT_NAME}/components/data_validation.py",
    f"src/{PROJECT_NAME}/components/data_transformation.py",
    f"src/{PROJECT_NAME}/components/model_trainer.py",
    f"src/{PROJECT_NAME}/components/model_evaluation.py",

    # Configuration and MongoDB
    f"src/{PROJECT_NAME}/configuration/__init__.py",
    f"src/{PROJECT_NAME}/configuration/mongo_connection.py",

    # Data access
    f"src/{PROJECT_NAME}/data_access/__init__.py",
    f"src/{PROJECT_NAME}/data_access/healthcare_data.py",

    # Entities
    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/entity/config_entity.py",
    f"src/{PROJECT_NAME}/entity/artifact_entity.py",

    # Pipelines
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/pipeline/training_pipeline.py",
    f"src/{PROJECT_NAME}/pipeline/prediction_pipeline.py",

    # Utilities
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/utils/common.py",
    f"src/{PROJECT_NAME}/utils/logger.py",
    f"src/{PROJECT_NAME}/exception/__init__.py",

    # API and interface
    "api/__init__.py",
    "api/main.py",
    "streamlit_app/Home.py",

    # Tests
    "tests/__init__.py",
    "tests/test_pipeline.py",

    # Project configuration
    "main.py",
    "setup.py",
    "requirements.txt",
    "README.md",
    ".gitignore",
    ".env.example",

    # Docker and CI/CD
    "Dockerfile",
    "docker-compose.yml",
    ".github/workflows/ci.yml",

    # Local data and artifacts
    "data/raw/.gitkeep",
    "artifacts/.gitkeep",
]


def create_project_structure():
    for filepath in list_of_files:
        path = Path(filepath)

        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Never overwrite existing files
            if not path.exists():
                path.touch()
                logging.info("Created: %s", path)
            else:
                logging.info("Already exists: %s", path)

        except OSError:
            logging.exception("Failed to create: %s", path)
            raise

    print("\nProject structure created successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_project_structure()
