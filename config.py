"""Central configuration for the Arabic OCR pipeline.

All connection parameters and tunables are driven by environment variables so the
same code runs on the local node (Dell Pro Max GB10) and against the remote host
(`latitude`) without code changes. Every remote dependency is optional and
fail-soft: if a service is unreachable the pipeline still produces local output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass
class OllamaConfig:
    """Local Ollama server (Dell Pro Max GB10)."""

    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    # NOTE: llama3.2-vision:11b is broken on this Ollama build (mllama arch).
    # qwen2.5vl:7b (6GB) is the primary model: excellent Arabic OCR, ~5-15s/page,
    # and a fraction of the GPU heat of the 35GB qwen3-vl:32b (which caused
    # thermal crashes at ~86s/page of sustained 95% GPU utilization).
    vision_model: str = field(default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b"))
    vision_fallback: str = field(default_factory=lambda: os.getenv("OLLAMA_VISION_FALLBACK", "qwen3-vl:latest"))
    postprocess_model: str = field(default_factory=lambda: os.getenv("OLLAMA_POSTPROCESS_MODEL", "qwen2.5-coder:14b"))
    postprocess_enabled: bool = _env_bool("OLLAMA_POSTPROCESS_ENABLED", False)
    embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"))
    timeout: float = _env_float("OLLAMA_TIMEOUT", 300.0)
    max_retries: int = _env_int("OLLAMA_MAX_RETRIES", 3)
    backoff_base: float = _env_float("OLLAMA_BACKOFF_BASE", 2.0)
    temperature: float = _env_float("OLLAMA_TEMPERATURE", 0.0)
    num_ctx: int = _env_int("OLLAMA_NUM_CTX", 8192)
    # Keep models resident in GPU memory (default Ollama keep-alive) so we don't
    # waste time reloading. Inference calls are serialized via a queue instead.
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    # Serialize all Ollama inference calls through a single global queue so only
    # one runs at a time (prevents GPU overload/overheating).
    queue_inference: bool = _env_bool("OLLAMA_QUEUE_INFERENCE", True)
    # Downscale page images to this max long-edge dimension before sending.
    max_image_dim: int = _env_int("OLLAMA_MAX_IMAGE_DIM", 1500)


@dataclass
class PreprocessConfig:
    """OpenCV image preprocessing applied before vision OCR."""

    enabled: bool = _env_bool("PREPROCESS_ENABLED", False)
    deskew: bool = _env_bool("PREPROCESS_DESKEW", True)
    binarize: bool = _env_bool("PREPROCESS_BINARIZE", False)
    denoise: bool = _env_bool("PREPROCESS_DENOISE", False)
    normalize_contrast: bool = _env_bool("PREPROCESS_NORMALIZE_CONTRAST", False)


@dataclass
class ThermalConfig:
    """GPU thermal throttling to prevent overheating / system crashes."""

    enabled: bool = _env_bool("THERMAL_GUARD_ENABLED", True)
    max_temp: float = _env_float("THERMAL_MAX_TEMP", 77.5)  # Celsius (conservative)
    poll_interval: float = _env_float("THERMAL_POLL_INTERVAL", 10.0)  # seconds
    inter_page_delay: float = _env_float("THERMAL_INTER_PAGE_DELAY", 2.0)  # seconds


@dataclass
class PipelineConfig:
    """Pipeline-wide tunables."""

    input_dir: Path = field(default_factory=lambda: Path(os.getenv("INPUT_DIR", "./input")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    dpi: int = _env_int("OCR_DPI", 300)
    workers: int = _env_int("OCR_WORKERS", 1)  # default 1 to avoid GPU overload
    max_pages: int = _env_int("OCR_MAX_PAGES", 0)  # 0 = all pages
    # OpenCV preprocessing (applied before vision OCR)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    # GPU thermal throttling
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    # Per-page cache directory (avoids re-OCR on resume)
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("OCR_CACHE_DIR", "./.ocr_cache")))
    use_cache: bool = _env_bool("OCR_USE_CACHE", True)
    # State tracking
    state_file: Path = field(default_factory=lambda: Path(os.getenv("OCR_STATE_FILE", "./.ocr_state.json")))
    # RTL / typography
    rtl: bool = _env_bool("OCR_RTL", True)
    font_family: str = os.getenv("OCR_FONT_FAMILY", "'Amiri', 'Traditional Arabic', 'Scheherazade New', serif")
    # Calibre ebook-convert binary (for MOBI). Empty = auto-detect via PATH.
    ebook_convert: str = os.getenv("EBOOK_CONVERT", "")


@dataclass
class RedisConfig:
    """Redis on `latitude` — used for idempotent state tracking."""

    enabled: bool = _env_bool("REDIS_ENABLED", False)
    host: str = os.getenv("REDIS_HOST", "latitude")
    port: int = _env_int("REDIS_PORT", 6379)
    db: int = _env_int("REDIS_DB", 0)
    password: str | None = os.getenv("REDIS_PASSWORD")
    key_prefix: str = os.getenv("REDIS_KEY_PREFIX", "arabic_ocr")


@dataclass
class PostgresConfig:
    """PostgreSQL on `latitude` — execution logs, metadata, ToC."""

    enabled: bool = _env_bool("POSTGRES_ENABLED", False)
    host: str = os.getenv("POSTGRES_HOST", "latitude")
    port: int = _env_int("POSTGRES_PORT", 5432)
    dbname: str = os.getenv("POSTGRES_DB", "arabic_ocr")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")


@dataclass
class ElasticsearchConfig:
    """Elasticsearch on `latitude` — full-text Arabic search."""

    enabled: bool = _env_bool("ELASTICSEARCH_ENABLED", False)
    hosts: list[str] = field(default_factory=lambda: [os.getenv("ELASTICSEARCH_HOST", "http://latitude:9200")])
    index: str = os.getenv("ELASTICSEARCH_INDEX", "arabic_ocr_pages")


@dataclass
class QdrantConfig:
    """Qdrant (local) — vector embeddings for RAG."""

    enabled: bool = _env_bool("QDRANT_ENABLED", False)
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = _env_int("QDRANT_PORT", 6333)
    collection: str = os.getenv("QDRANT_COLLECTION", "arabic_ocr_pages")


@dataclass
class Neo4jConfig:
    """Neo4j (local) — knowledge graph Author->Book->Volume->Chapter->Page."""

    enabled: bool = _env_bool("NEO4J_ENABLED", False)
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "")


@dataclass
class MySQLConfig:
    """MySQL on `latitude` — optional secondary relational store."""

    enabled: bool = _env_bool("MYSQL_ENABLED", False)
    host: str = os.getenv("MYSQL_HOST", "latitude")
    port: int = _env_int("MYSQL_PORT", 3306)
    dbname: str = os.getenv("MYSQL_DB", "arabic_ocr")
    user: str = os.getenv("MYSQL_USER", "root")
    password: str = os.getenv("MYSQL_PASSWORD", "")


@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    elasticsearch: ElasticsearchConfig = field(default_factory=ElasticsearchConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)

    def ensure_dirs(self) -> None:
        self.pipeline.input_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline.cache_dir.mkdir(parents=True, exist_ok=True)


# Singleton accessor
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
