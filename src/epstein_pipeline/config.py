"""Pipeline configuration using Pydantic BaseSettings with EPSTEIN_ env prefix."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class OcrBackend(str, Enum):
    """Available OCR backends, ordered by speed (fastest first)."""

    AUTO = "auto"
    PYMUPDF = "pymupdf"
    SMOLDOCLING = "smoldocling"  # SmolDocling-256M — fast VLM, 0.35s/page, 500MB VRAM
    SURYA = "surya"
    OLMOCR = "olmocr"
    DOCLING = "docling"


class NerBackend(str, Enum):
    """Available NER backends."""

    SPACY = "spacy"
    GLINER = "gliner"
    GLINER2 = "gliner2"
    BOTH = "both"


class DedupMode(str, Enum):
    """Dedup strategies — can be combined via 'all'."""

    EXACT = "exact"  # content hash + title fuzzy + Bates overlap
    MINHASH = "minhash"  # MinHash/LSH near-duplicate
    SEMANTIC = "semantic"  # embedding cosine similarity
    ALL = "all"  # exact → minhash → semantic


class Settings(BaseSettings):
    """Pipeline settings loaded from environment variables prefixed with EPSTEIN_.

    Example:
        EPSTEIN_DATA_DIR=/mnt/data
        EPSTEIN_SPACY_MODEL=en_core_web_trf
        EPSTEIN_DEDUP_THRESHOLD=0.95
        EPSTEIN_NEON_DATABASE_URL=postgresql://...
    """

    model_config = {"env_prefix": "EPSTEIN_"}

    # ── Directory paths ──────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    cache_dir: Path = Path("./.cache")
    persons_registry_path: Path = Path("./data/persons-registry.json")

    # ── General processing ───────────────────────────────────────────────
    max_workers: int = 4

    # ── OCR settings ─────────────────────────────────────────────────────
    ocr_backend: OcrBackend = OcrBackend.AUTO
    ocr_batch_size: int = 50
    ocr_confidence_threshold: float = 0.7  # flag pages below this
    ocr_fallback_chain: list[str] = ["pymupdf", "surya", "olmocr", "docling"]

    # ── NER settings ─────────────────────────────────────────────────────
    spacy_model: str = "en_core_web_trf"  # upgraded from en_core_web_sm
    ner_backend: NerBackend = NerBackend.BOTH
    gliner_model: str = "urchade/gliner_multi_pii-v1"
    gliner2_model: str = "fastino/gliner2-base-v1"
    ner_confidence_threshold: float = 0.5

    # ── Coreference resolution settings ──────────────────────────────────
    enable_coref: bool = False  # opt-in: resolve pronouns before NER
    coref_model: str = "FCoref"  # "FCoref" (fast) or "LingMessCoref" (accurate)

    # ── Dedup settings ───────────────────────────────────────────────────
    dedup_mode: DedupMode = DedupMode.ALL
    dedup_threshold: float = 0.90  # title fuzzy match threshold
    dedup_jaccard_threshold: float = 0.80  # MinHash Jaccard threshold
    dedup_semantic_threshold: float = 0.95  # embedding cosine similarity
    dedup_shingle_size: int = 5  # n-gram size for MinHash
    dedup_num_perm: int = 128  # MinHash permutation count

    # ── Embedding settings ───────────────────────────────────────────────
    embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe"
    embedding_dimensions: int = 768  # 768 full, 256 Matryoshka
    embedding_chunk_size: int = 3200  # chars (~800 tokens)
    embedding_chunk_overlap: int = 800  # chars (~200 tokens)
    embedding_batch_size: int | None = None  # None = auto-detect
    embedding_device: str | None = None  # None = auto-detect

    # ── Chunker settings ─────────────────────────────────────────────────
    chunker_mode: Literal["fixed", "semantic"] = "semantic"
    chunker_target_tokens: int = 512  # target chunk size in tokens
    chunker_min_tokens: int = 100  # minimum chunk size
    chunker_max_tokens: int = 1024  # maximum chunk size

    # ── Neon Postgres settings ───────────────────────────────────────────
    neon_database_url: str | None = None  # postgresql://...@...neon.tech/...
    neon_pool_size: int = 10
    neon_batch_size: int = 500  # rows per upsert batch (tuned for Neon Scale)
    neon_retry_max: int = 5  # max retries for transient errors
    neon_retry_base_delay: float = 1.0  # base delay in seconds (exponential backoff)

    # ── Document classifier settings ─────────────────────────────────────
    classifier_model: str = "knowledgator/gliclass-modern-base-v3.0"
    classifier_confidence_threshold: float = 0.6

    # ── Knowledge graph settings ─────────────────────────────────────────
    kg_llm_provider: str = "openai"  # "openai" or "anthropic"
    kg_llm_model: str = "gpt-4o-mini"
    kg_extract_relationships: bool = False  # LLM extraction is opt-in

    # ── Neo4j settings ────────────────────────────────────────────────────
    neo4j_uri: str | None = None  # bolt://localhost:7687 or neo4j+s://...
    neo4j_username: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"
    neo4j_batch_size: int = 500  # nodes/edges per UNWIND batch
    neo4j_retry_max: int = 3
    neo4j_retry_base_delay: float = 1.0

    # ── Entity resolution (Splink) settings ─────────────────────────────
    splink_match_probability_threshold: float = 0.85
    splink_max_pairs: int = 1_000_000  # max comparison pairs (memory guard)

    # ── Temporal extraction settings ─────────────────────────────────────
    temporal_llm_provider: str = "ollama"  # "ollama", "openai", or "anthropic"
    temporal_llm_model: str | None = None  # None = auto-select per provider
    temporal_chunk_size: int = 3000  # chars per chunk for extraction
    temporal_chunk_overlap: int = 500  # overlap between chunks
    temporal_max_events_per_chunk: int = 20  # cap per LLM call
    temporal_confidence_threshold: float = 0.3  # minimum confidence to keep

    # ── Sea Doughnut import ──────────────────────────────────────────────
    sea_doughnut_dir: Path | None = None

    # ── Site sync ────────────────────────────────────────────────────────
    site_dir: Path | None = None

    # ── AI / Vision settings ─────────────────────────────────────────────
    vision_model: str = "llava"
    summarizer_provider: str = "ollama"
    summarizer_model: str = "llama3.2"

    # ── Transcription ────────────────────────────────────────────────────
    whisper_model: str = "large-v3-turbo"

    # ── OpenSanctions settings ──────────────────────────────────────────
    opensanctions_api_key: str | None = None
    opensanctions_match_threshold: float = 0.5  # minimum score to flag

    # ── ICIJ Offshore Leaks settings ─────────────────────────────────
    icij_data_dir: Path | None = None  # Path to extracted ICIJ CSVs
    icij_fuzzy_threshold: int = 85  # rapidfuzz token_sort_ratio minimum
    icij_min_name_length: int = 5  # skip names shorter than this
    icij_traverse_relationships: bool = True  # follow officer→entity edges

    # ── FEC Political Donations settings ─────────────────────────────
    fec_api_key: str | None = None  # FEC API key from api.data.gov
    fec_min_amount: int = 200  # minimum contribution in dollars
    fec_max_pages: int = 5  # max API pages per person search
    fec_rate_limit: int = 120  # requests per minute

    # ── IRS 990 Nonprofit settings ─────────────────────────────────────
    nonprofit_seed_eins: list[str] = [
        # Phase 1: Original seed orgs
        "223496220", "133996471", "660789697", "455091884",
        "133947890", "237320631", "311306419", "311318013",
        "137265141", "474634539", "454757735", "850325494",
        # Phase 2: Direct Epstein entities
        "133643429", "261605864", "133528667", "811263733",
        # Phase 2: Documented grant recipients
        "030213226", "261636099", "542079811", "330783933",
        "041425590", "810902953", "364793898",
        # Phase 2: Associate foundations
        "413185341", "466230671", "357119449", "134028567",
        # Phase 2: Smaller documented recipients
        "133156445", "582565917", "463485787",
        # Phase 2: Mark Epstein Foundation
        "387193282",
    ]
    nonprofit_max_filings: int = 10
    nonprofit_fuzzy_threshold: int = 85

    # ── Person Auditor settings ────────────────────────────────────────
    auditor_anthropic_api_key: str | None = None
    auditor_anthropic_model: str = "claude-sonnet-4-6"  # Primary model for verification
    auditor_fast_model: str = "claude-haiku-4-5-20251001"  # Quick screening/decomposition
    auditor_deep_model: str = "claude-sonnet-4-6"  # Deep analysis of flagged records
    auditor_voyage_api_key: str | None = None
    auditor_cohere_api_key: str | None = None
    auditor_use_batch_api: bool = True
    auditor_dedup_threshold: float = 0.85
    auditor_name_fuzzy_threshold: int = 85
    auditor_wikidata_rate_limit: float = 1.0  # seconds between requests
    auditor_max_claims_per_person: int = 10
    auditor_max_doc_chunks: int = 5
    auditor_severity_critical: int = 70
    auditor_severity_high: int = 40
    auditor_severity_medium: int = 20

    def ensure_dirs(self) -> None:
        """Create data, output, and cache directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
