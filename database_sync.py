"""Multi-database & vector indexing pipeline (all optional, fail-soft).

Each backend is gated by an `*_ENABLED` env flag and wrapped in try/except so a
missing or unreachable service never blocks local output generation.

- PostgreSQL (`latitude`): execution logs, document metadata, ToC.
- Elasticsearch (`latitude`): full-text Arabic page content.
- Qdrant (local): vector embeddings for semantic RAG.
- Neo4j (local): knowledge graph Author->Book->Volume->Chapter->Page.
- MySQL (`latitude`): optional secondary relational store.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Config, get_config
from models import DocumentResult

logger = logging.getLogger(__name__)


class DatabaseSync:
    """Best-effort ingestion into all configured backends."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()

    # ---- PostgreSQL -------------------------------------------------------
    def _sync_postgres(self, result: DocumentResult, toc: list[dict[str, Any]]) -> None:
        if not self.config.postgres.enabled:
            return
        try:
            import psycopg2  # type: ignore

            conn = psycopg2.connect(
                host=self.config.postgres.host,
                port=self.config.postgres.port,
                dbname=self.config.postgres.dbname,
                user=self.config.postgres.user,
                password=self.config.postgres.password,
                connect_timeout=5,
            )
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        rel_path TEXT UNIQUE,
                        file_hash TEXT,
                        total_pages INT,
                        status TEXT,
                        toc JSONB,
                        created_at TIMESTAMP DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO documents (rel_path, file_hash, total_pages, status, toc)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (rel_path) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        total_pages = EXCLUDED.total_pages,
                        status = EXCLUDED.status,
                        toc = EXCLUDED.toc
                    """,
                    (
                        result.rel_path,
                        result.file_hash,
                        result.total_pages,
                        result.status.value,
                        psycopg2.extras.Json(toc) if hasattr(psycopg2, "extras") else str(toc),
                    ),
                )
            conn.close()
            logger.info("PostgreSQL: synced %s", result.rel_path)
        except Exception as exc:
            logger.warning("PostgreSQL sync skipped (%s)", exc)

    # ---- Elasticsearch ----------------------------------------------------
    def _sync_elasticsearch(self, result: DocumentResult) -> None:
        if not self.config.elasticsearch.enabled:
            return
        try:
            from elasticsearch import Elasticsearch  # type: ignore

            es = Elasticsearch(self.config.elasticsearch.hosts, request_timeout=10)
            index = self.config.elasticsearch.index
            if not es.indices.exists(index=index):
                es.indices.create(
                    index=index,
                    body={
                        "settings": {
                            "analysis": {
                                "analyzer": {
                                    "arabic_analyzer": {
                                        "type": "arabic",
                                    }
                                }
                            }
                        },
                        "mappings": {
                            "properties": {
                                "rel_path": {"type": "keyword"},
                                "page": {"type": "integer"},
                                "text": {"type": "text", "analyzer": "arabic_analyzer"},
                            }
                        },
                    },
                )
            for page in result.pages:
                if page.error or not page.text.strip():
                    continue
                es.index(
                    index=index,
                    body={
                        "rel_path": result.rel_path,
                        "page": page.page_number,
                        "text": page.text,
                    },
                )
            logger.info("Elasticsearch: indexed %s", result.rel_path)
        except Exception as exc:
            logger.warning("Elasticsearch sync skipped (%s)", exc)

    # ---- Qdrant -----------------------------------------------------------
    async def _sync_qdrant(self, result: DocumentResult, engine) -> None:
        if not self.config.qdrant.enabled:
            return
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            client = QdrantClient(host=self.config.qdrant.host, port=self.config.qdrant.port)
            collection = self.config.qdrant.collection
            if not client.collection_exists(collection):
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
            points = []
            for page in result.pages:
                if page.error or not page.text.strip():
                    continue
                vec = await engine.embed(page.text[:2000])
                points.append(
                    {
                        "id": f"{result.file_hash[:16]}-{page.page_number}",
                        "vector": vec,
                        "payload": {
                            "rel_path": result.rel_path,
                            "page": page.page_number,
                            "text": page.text[:2000],
                        },
                    }
                )
            if points:
                client.upsert(collection_name=collection, points=points)
            logger.info("Qdrant: indexed %s", result.rel_path)
        except Exception as exc:
            logger.warning("Qdrant sync skipped (%s)", exc)

    # ---- Neo4j ------------------------------------------------------------
    def _sync_neo4j(self, result: DocumentResult, toc: list[dict[str, Any]]) -> None:
        if not self.config.neo4j.enabled:
            return
        try:
            from neo4j import GraphDatabase  # type: ignore

            driver = GraphDatabase.driver(
                self.config.neo4j.uri,
                auth=(self.config.neo4j.user, self.config.neo4j.password),
            )
            # Derive Author/Book/Volume from the relative path.
            parts = result.rel_path.split("/")
            author = parts[0] if len(parts) > 0 else "Unknown"
            book = parts[1] if len(parts) > 1 else result.source_path.stem
            volume = result.source_path.stem

            with driver.session() as session:
                session.run(
                    """
                    MERGE (a:Author {name: $author})
                    MERGE (b:Book {name: $book})
                    MERGE (v:Volume {name: $volume})
                    MERGE (a)-[:AUTHORED]->(b)
                    MERGE (b)-[:HAS_VOLUME]->(v)
                    """,
                    author=author, book=book, volume=volume,
                )
                for entry in toc:
                    session.run(
                        """
                        MATCH (v:Volume {name: $volume})
                        MERGE (c:Chapter {name: $title, page: $page})
                        MERGE (v)-[:HAS_CHAPTER]->(c)
                        """,
                        volume=volume, title=entry.get("title", ""), page=entry.get("page", 1),
                    )
                for page in result.pages:
                    if page.error:
                        continue
                    session.run(
                        """
                        MATCH (v:Volume {name: $volume})
                        MERGE (p:Page {number: $number})
                        MERGE (v)-[:HAS_PAGE]->(p)
                        """,
                        volume=volume, number=page.page_number,
                    )
            driver.close()
            logger.info("Neo4j: synced %s", result.rel_path)
        except Exception as exc:
            logger.warning("Neo4j sync skipped (%s)", exc)

    # ---- MySQL ------------------------------------------------------------
    def _sync_mysql(self, result: DocumentResult) -> None:
        if not self.config.mysql.enabled:
            return
        try:
            import pymysql  # type: ignore

            conn = pymysql.connect(
                host=self.config.mysql.host,
                port=self.config.mysql.port,
                database=self.config.mysql.dbname,
                user=self.config.mysql.user,
                password=self.config.mysql.password,
                connect_timeout=5,
                charset="utf8mb4",
            )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        rel_path VARCHAR(512) UNIQUE,
                        file_hash VARCHAR(64),
                        total_pages INT,
                        status VARCHAR(32)
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO documents (rel_path, file_hash, total_pages, status)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        file_hash = VALUES(file_hash),
                        total_pages = VALUES(total_pages),
                        status = VALUES(status)
                    """,
                    (result.rel_path, result.file_hash, result.total_pages, result.status.value),
                )
            conn.commit()
            conn.close()
            logger.info("MySQL: synced %s", result.rel_path)
        except Exception as exc:
            logger.warning("MySQL sync skipped (%s)", exc)

    # ---- public API -------------------------------------------------------
    async def sync(self, result: DocumentResult, toc: list[dict[str, Any]], engine) -> None:
        """Run all enabled backend syncs (each independently fail-soft)."""
        self._sync_postgres(result, toc)
        self._sync_elasticsearch(result)
        await self._sync_qdrant(result, engine)
        self._sync_neo4j(result, toc)
        self._sync_mysql(result)
