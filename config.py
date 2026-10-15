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


def _load_dotenv(path: Path | None = None) -> None:
    """Load a `.env` file into the environment (no external dependency).

    Reads `KEY=VALUE` lines from the given path (default: `.env` in the current
    working directory) and sets them via `os.environ` only if not already set.
    Lines starting with `#` are comments; blank lines are ignored. Values may be
    optionally quoted. This runs at import time so all config defaults pick up
    the values.
    """
    env_path = path or Path(".env")
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            # Do not override variables already set in the real environment.
            os.environ.setdefault(key, value)
    except OSError:
        pass


# Load .env (if present) before any config defaults are evaluated.
_load_dotenv()


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
    # Image transform used when the model refuses a page (unreadable/overexposed):
    # "contrast" (default) | "sharpen" | "both". Sharpening is preferred for
    # vision LLMs (arXiv 2608.22366); "both" tries contrast then sharpening.
    retry_mode: str = os.getenv("OLLAMA_RETRY_MODE", "contrast")
    # Keep models resident in GPU memory (default Ollama keep-alive) so we don't
    # waste time reloading. Inference calls are serialized via a queue instead.
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
    # Serialize all Ollama inference calls through a single global queue so only
    # one runs at a time (prevents GPU overload/overheating).
    queue_inference: bool = _env_bool("OLLAMA_QUEUE_INFERENCE", True)
    # Downscale page images to this max long-edge dimension before sending.
    max_image_dim: int = _env_int("OLLAMA_MAX_IMAGE_DIM", 1500)
    # Larger cap used for heuristic-flagged "heavy" (diacritized / small-text)
    # pages so diacritics survive downscaling. See PipelineConfig.heavy_heuristic.
    max_image_dim_heavy: int = _env_int("OLLAMA_MAX_IMAGE_DIM_HEAVY", 2500)


@dataclass
class VisionConfig:
    """Vision OCR backend selection.

    `backend` chooses the engine: "kraken" (default; the in-repo recognition
    model trained with `ketos train` — highest accuracy and ~30x faster than the
    VLM paths on our corpus), "ollama" (the local Ollama server), or "hf" (a
    HuggingFace vision-language model loaded in-process). All three are
    fail-soft: if a backend's packages or weights are unavailable, the engine
    factory falls back to Ollama so the pipeline still runs.
    """

    backend: str = os.getenv("VISION_BACKEND", "kraken").strip().lower()  # "ollama" | "hf" | "kraken"
    # Arabic-fine-tuned VLM candidate. Options include:
    #   NAMAA-Space/Qari-OCR-v0.2-VL-2B-Instruct   (2B Qwen2-VL, CER 0.061)
    #   NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct   (2B, layout-aware)
    #   AhmedZaky1/DIMI-Arabic-OCR-V2              (Qwen2.5-VL-7B, CER 0.112)
    #   QCRI/Fanar-2-Oryx-IVU                      (Qwen2.5-VL-7B, Apache-2.0)
    hf_model: str = os.getenv("VISION_HF_MODEL", "NAMA A-Space/Qari-OCR-v0.2-VL-2B-Instruct".replace(" ", ""))
    hf_device: str = os.getenv("VISION_HF_DEVICE", "cuda")
    hf_max_new_tokens: int = _env_int("VISION_HF_MAX_NEW_TOKENS", 2048)
    hf_load_in_4bit: bool = _env_bool("VISION_HF_LOAD_IN_4BIT", False)
    # Kraken recognition model (Route A: the in-repo typeface-specific model trained
    # with `ketos train`). Default is the PP-OCRv6-medium run (99.40% char / 96.83%
    # word on the 103-line holdout, 2026-10-15); the earlier VGSL model
    # (models/run_x_3e4/best_0.9690.safetensors, 96.92% / 86.36%) is kept as fallback.
    kraken_model: str = os.getenv("VISION_KRAKEN_MODEL", "models/run_ppocrv6_m/best_0.9948.safetensors")
    kraken_device: str = os.getenv("VISION_KRAKEN_DEVICE", "auto")
    # Optional custom BLLA line-segmentation model; empty = kraken's bundled blla.mlmodel.
    kraken_seg_model: str = os.getenv("VISION_KRAKEN_SEG_MODEL", "")
    # Principal page direction for line reading order ("horizontal-rl" for Arabic).
    kraken_text_direction: str = os.getenv("VISION_KRAKEN_TEXT_DIRECTION", "horizontal-rl")
    kraken_batch_size: int = _env_int("VISION_KRAKEN_BATCH_SIZE", 16)


@dataclass
class PreprocessConfig:
    """OpenCV image preprocessing applied before vision OCR.

    `mode` selects a preset:
      - "off"    : no preprocessing (default; vision-LLM handles clean scans).
      - "light"  : cheap global contrast normalization only (fast, low-risk).
      - "sharpen": mild unsharp mask only (VLM-friendly; arXiv 2608.22366).
      - "full"   : deskew + binarize + denoise + contrast (heavy pipeline).
                   Set `vlm_safe=True` to keep only deskew/sharpen on this path.
    """

    mode: str = os.getenv("PREPROCESS_MODE", "off")
    enabled: bool = _env_bool("PREPROCESS_ENABLED", False)
    deskew: bool = _env_bool("PREPROCESS_DESKEW", True)
    binarize: bool = _env_bool("PREPROCESS_BINARIZE", False)
    denoise: bool = _env_bool("PREPROCESS_DENOISE", False)
    normalize_contrast: bool = _env_bool("PREPROCESS_NORMALIZE_CONTRAST", False)
    # Mild unsharp mask (VLM-friendly). Off by default.
    sharpen: bool = _env_bool("PREPROCESS_SHARPEN", False)
    # When True, "full" mode skips binarize/denoise/contrast on the vision-LLM
    # path: those operations degrade VLM Arabic OCR (arXiv 2608.22366).
    vlm_safe: bool = _env_bool("PREPROCESS_VLM_SAFE", False)


@dataclass
class ThermalConfig:
    """GPU thermal throttling to prevent overheating / system crashes."""

    enabled: bool = _env_bool("THERMAL_GUARD_ENABLED", True)
    max_temp: float = _env_float("THERMAL_MAX_TEMP", 70.0)  # Celsius — pause threshold
    critical_temp: float = _env_float("THERMAL_CRITICAL_TEMP", 75.0)  # Celsius — peak that forces cooldown
    poll_interval: float = _env_float("THERMAL_POLL_INTERVAL", 5.0)  # seconds (cooldown wait)
    monitor_interval: float = _env_float("THERMAL_MONITOR_INTERVAL", 2.0)  # seconds (background sampling)
    inter_page_delay: float = _env_float("THERMAL_INTER_PAGE_DELAY", 2.0)  # seconds


@dataclass
class PipelineConfig:
    """Pipeline-wide tunables."""

    input_dir: Path = field(default_factory=lambda: Path(os.getenv("INPUT_DIR", "./input")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    dpi: int = _env_int("OCR_DPI", 300)
    workers: int = _env_int("OCR_WORKERS", 1)  # default 1 to avoid GPU overload
    max_pages: int = _env_int("OCR_MAX_PAGES", 0)  # 0 = all pages
    # Higher-resolution path for diacritized / small-text pages (opt-in). When
    # `heavy_heuristic` marks a page "heavy" (many tiny marks), it is re-rendered
    # at `dpi_heavy` and the vision model's image cap is raised to
    # `max_image_dim_heavy`. Defaults are off, so base behaviour is unchanged.
    dpi_heavy: int = _env_int("OCR_DPI_HEAVY", 400)
    heavy_heuristic: bool = _env_bool("OCR_HEAVY_HEURISTIC", False)
    heavy_threshold: float = _env_float("OCR_HEAVY_THRESHOLD", 0.35)
    # Page range to process, e.g. "31-35" or "1,3,5-7". Empty = all pages.
    # Useful for focusing on the main text and skipping cover/intro pages.
    pages: str = os.getenv("OCR_PAGES", "")
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
    # Normalize the Arabic text copy sent to search/RAG backends
    # (Elasticsearch/Qdrant): Unicode NFC, presentation-form/orthographic folding,
    # digit normalization, and diacritic stripping. Opt-in (default off) so
    # indexing behaviour does not change silently; enabling it requires a reindex.
    # The canonical Markdown output is NEVER normalized, so diacritics and exact
    # orthography are preserved in the books themselves.
    normalize_search: bool = _env_bool("OCR_NORMALIZE_SEARCH", False)
    # Running header (book title) repeated on every page — stripped from output.
    running_header: str = os.getenv("OCR_RUNNING_HEADER", "")
    # Calibre ebook-convert binary (for MOBI). Empty = auto-detect via PATH.
    ebook_convert: str = os.getenv("EBOOK_CONVERT", "")
    # Searchable PDF overlay: generate `{stem}_overlay.pdf` with the original scan
    # as the visible image and an invisible OCR text layer on top (render mode 3).
    # mode: "off" | "page" (full-page text layer) | "word" (word-aligned, future).
    overlay_mode: str = os.getenv("OVERLAY_MODE", "off")
    # Arabic font file used for the invisible text layer (must support Arabic).
    overlay_font: str = os.getenv("OVERLAY_FONT", "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf")
    # Debug mode: render overlay text in RED (visible) to verify alignment.
    overlay_debug: bool = _env_bool("OVERLAY_DEBUG", False)
    # Use Surya (best Arabic line detection, mAP 79.67%) for overlay line boxes
    # instead of OpenCV. Runs on GPU by default (thermal guard gates inference).
    overlay_use_surya: bool = _env_bool("OVERLAY_USE_SURYA", True)
    # Detection engine for overlay line boxes: "surya" (default, best line
    # detection) | "easyocr" (best Arabic recognition) | "opencv" (fallback).
    overlay_detector: str = os.getenv("OVERLAY_DETECTOR", "surya")
    # Device for Surya/EasyOCR detection: "cuda" (default, fast) or "cpu".
    overlay_device: str = os.getenv("OVERLAY_DEVICE", "cuda")


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
    vision: VisionConfig = field(default_factory=VisionConfig)
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
