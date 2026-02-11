"""Pipeline configuration using Pydantic BaseSettings with EPSTEIN_ env prefix."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pipeline settings loaded from environment variables prefixed with EPSTEIN_.

    Example:
        EPSTEIN_DATA_DIR=/mnt/data
        EPSTEIN_SPACY_MODEL=en_core_web_lg
        EPSTEIN_DEDUP_THRESHOLD=0.95
    """

    model_config = {"env_prefix": "EPSTEIN_"}

    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    cache_dir: Path = Path("./.cache")
    persons_registry_path: Path = Path("./data/persons-registry.json")

    spacy_model: str = "en_core_web_sm"
    dedup_threshold: float = 0.90
    ocr_batch_size: int = 50
    max_workers: int = 4

    def ensure_dirs(self) -> None:
        """Create data, output, and cache directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
