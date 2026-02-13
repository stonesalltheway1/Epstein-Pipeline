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

    # Processing settings
    spacy_model: str = "en_core_web_sm"
    dedup_threshold: float = 0.90
    ocr_batch_size: int = 50
    max_workers: int = 4
    ocr_backend: str = "docling"  # "docling", "pymupdf", or "both"

    # Sea_Doughnut import
    sea_doughnut_dir: Path | None = None

    # Site sync
    site_dir: Path | None = None

    # AI / Vision settings
    vision_model: str = "llava"  # for image description
    summarizer_provider: str = "ollama"
    summarizer_model: str = "llama3.2"

    # Transcription
    whisper_model: str = "large-v3"

    # Embedding settings
    embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe"
    embedding_dimensions: int = 768  # 768 full, 256 Matryoshka
    embedding_chunk_size: int = 3200  # chars (~800 tokens)
    embedding_chunk_overlap: int = 800  # chars (~200 tokens)
    embedding_batch_size: int | None = None  # None = auto-detect
    embedding_device: str | None = None  # None = auto-detect

    def ensure_dirs(self) -> None:
        """Create data, output, and cache directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
