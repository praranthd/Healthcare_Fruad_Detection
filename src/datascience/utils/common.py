
from pathlib import Path

import yaml

from src.datascience.logging import logger


def read_yaml(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if content is None:
        return {}

    if not isinstance(content, dict):
        raise ValueError(
            f"Expected a YAML dictionary in {file_path}"
        )

    logger.info("Successfully read YAML file: %s", file_path)
    return content


def create_directories(paths: list[str]) -> None:
    """
    Create multiple directories if they do not already exist.
    """
    for path in paths:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        logger.info("Directory created or already exists: %s", directory)
