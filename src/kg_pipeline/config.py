from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    openai_api_key: str
    openai_vision_model: str = "gpt-5-mini"
    openai_query_model: str = "gpt-4.1-mini"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_instance_name: Optional[str] = None
    neo4j_database: str = "neo4j"
    neo4j_allow_db_fallback: bool = True
    neo4j_fallback_database: str = "neo4j"
    pid_pdf_path: str = "data/pid/diagram.pdf"
    sop_docx_path: str = "data/sop/sop.docx"
    output_dir: str = "artifacts"
    max_zoom_loops: int = 5
    max_views_per_loop: int = 4
    render_dpi: int = 350
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_chunk_sentences: int = 5
    embedding_chunk_overlap: int = 2
    embedding_top_k_chunks: int = 24
    embedding_top_k_entities: int = 8
    lexical_top_k_entities: int = 8

    @classmethod
    def from_env(cls, env_path: str | None = ".env") -> "Settings":
        if env_path:
            load_dotenv(env_path)

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        neo4j_password = os.getenv("DB_PASSWORD") or os.getenv("NEO4J_PASSWORD") or ""
        if not neo4j_password:
            raise ValueError("DB_PASSWORD or NEO4J_PASSWORD is required.")

        return cls(
            openai_api_key=openai_api_key,
            openai_vision_model=os.getenv("OPENAI_VISION_MODEL", "gpt-5-mini"),
            openai_query_model=os.getenv("OPENAI_QUERY_MODEL", "gpt-4.1-mini"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=neo4j_password,
            neo4j_instance_name=os.getenv("DB_NAME"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            neo4j_allow_db_fallback=_parse_bool(os.getenv("NEO4J_ALLOW_DB_FALLBACK", "true")),
            neo4j_fallback_database=os.getenv("NEO4J_FALLBACK_DATABASE", "neo4j"),
            pid_pdf_path=os.getenv("PID_PDF_PATH", "data/pid/diagram.pdf"),
            sop_docx_path=os.getenv("SOP_DOCX_PATH", "data/sop/sop.docx"),
            output_dir=os.getenv("OUTPUT_DIR", "artifacts"),
            max_zoom_loops=int(os.getenv("MAX_ZOOM_LOOPS", "5")),
            max_views_per_loop=int(os.getenv("MAX_VIEWS_PER_PAGE", "4")),
            render_dpi=int(os.getenv("RENDER_DPI", "350")),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            embedding_chunk_sentences=int(os.getenv("EMBEDDING_CHUNK_SENTENCES", "5")),
            embedding_chunk_overlap=int(os.getenv("EMBEDDING_CHUNK_OVERLAP", "2")),
            embedding_top_k_chunks=int(os.getenv("EMBEDDING_TOP_K_CHUNKS", "24")),
            embedding_top_k_entities=int(os.getenv("EMBEDDING_TOP_K_ENTITIES", "8")),
            lexical_top_k_entities=int(os.getenv("LEXICAL_TOP_K_ENTITIES", "8")),
        )

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
