"""Idempotent state tracking with Redis (remote) + local JSON fallback.

Tracks per-file status (pending/processing/completed/failed) so the pipeline can
be safely re-run and resume where it left off. Redis on `latitude` is the
primary store when enabled; a local JSON file is always maintained as a
portable fallback and for offline operation.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from config import Config, get_config
from models import FileStatus


class StateTracker:
    """Thread-safe state store backed by Redis (optional) and a local JSON file."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._lock = threading.Lock()
        self._local: dict[str, dict[str, Any]] = {}
        self._redis = None
        self._load_local()
        if self.config.redis.enabled:
            self._connect_redis()

    # ---- local JSON persistence -------------------------------------------
    def _load_local(self) -> None:
        path = self.config.pipeline.state_file
        if path.exists():
            try:
                self._local = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._local = {}

    def _save_local(self) -> None:
        path = self.config.pipeline.state_file
        try:
            path.write_text(json.dumps(self._local, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ---- redis ------------------------------------------------------------
    def _connect_redis(self) -> None:
        try:
            import redis  # type: ignore

            self._redis = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                password=self.config.redis.password,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            self._redis.ping()
        except Exception:
            self._redis = None  # fail-soft

    def _redis_key(self, file_hash: str) -> str:
        return f"{self.config.redis.key_prefix}:{file_hash}"

    # ---- public API -------------------------------------------------------
    def get(self, file_hash: str) -> dict[str, Any] | None:
        with self._lock:
            if self._redis is not None:
                try:
                    raw = self._redis.get(self._redis_key(file_hash))
                    if raw:
                        return json.loads(raw)
                except Exception:
                    pass
            return self._local.get(file_hash)

    def set(self, file_hash: str, state: dict[str, Any]) -> None:
        with self._lock:
            self._local[file_hash] = state
            self._save_local()
            if self._redis is not None:
                try:
                    self._redis.set(self._redis_key(file_hash), json.dumps(state, ensure_ascii=False))
                except Exception:
                    pass

    def get_status(self, file_hash: str) -> FileStatus:
        state = self.get(file_hash)
        if state is None:
            return FileStatus.PENDING
        try:
            return FileStatus(state.get("status", FileStatus.PENDING.value))
        except ValueError:
            return FileStatus.PENDING

    def mark(self, file_hash: str, status: FileStatus, **extra: Any) -> None:
        state = self.get(file_hash) or {}
        state["status"] = status.value
        state.update(extra)
        self.set(file_hash, state)

    def is_completed(self, file_hash: str) -> bool:
        return self.get_status(file_hash) == FileStatus.COMPLETED

    def completed_pages(self, file_hash: str) -> set[int]:
        state = self.get(file_hash) or {}
        return set(state.get("completed_pages", []))

    def record_page(self, file_hash: str, page_number: int) -> None:
        state = self.get(file_hash) or {}
        pages = set(state.get("completed_pages", []))
        pages.add(page_number)
        state["completed_pages"] = sorted(pages)
        self.set(file_hash, state)
