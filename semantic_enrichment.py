"""
Enhanced Semantic enrichment with hierarchical docstring generation:
• generates docstrings with GPT for functions, files, folders, and branches
• builds embeddings with sentence-transformers for all levels
• handles large files through intelligent chunking and summarization
• creates layered summaries (function → file → folder → branch)
• writes results back to Neo4j

Key Changes:
1. Added hierarchical processing methods
2. Implemented smart chunking for large files
3. Added file size analysis and adaptive processing
4. Created aggregation methods for folder and branch summaries
5. Enhanced the main processing pipeline
"""

from __future__ import annotations

import os, re, json, time, logging, asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# ─── Global Neo4j utilities ────────────────────────────────────
from app.configs.neo4j_setup import get_driver                   # singleton pool
from app.ai.utils.neo4j_utils import retry_on_connection_error   # auto-refresh

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Enhanced data structures ──────────────────────────────────
@dataclass
class DocstringRequest:
    node_id: str
    text: str
    node_type: str  # NEW: 'function', 'file', 'folder', 'branch'
    context: Dict = None  # NEW: Additional context for hierarchical processing

@dataclass
class DocstringData:
    node_id: str
    docstring: str
    tags: List[str]
    node_type: str  # NEW: Track node type
    summary: str = ""  # NEW: Brief summary for aggregation
    complexity_score: int = 0  # NEW: Complexity indicator

@dataclass
class DocstringResponse:
    docstrings: List[DocstringData]

@dataclass
class FileAnalysis:
    """NEW: Structure to analyze file characteristics"""
    size_category: str  # 'small', 'medium', 'large', 'huge'
    line_count: int
    char_count: int
    chunk_strategy: str  # 'direct', 'semantic_chunks', 'hierarchical_summary'
    estimated_tokens: int

# ─── Enhanced Main class ───────────────────────────────────────
class HierarchicalSemanticEnrichment:
    """
    Enhanced semantic enrichment supporting hierarchical docstring generation
    for functions, files, folders, and branches with intelligent size handling.
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        neo_db: str | None = None,
        per_node_limit: int | None = None,
    ) -> None:
        try:
            # Embedding + OpenAI
            self.embedding_model   = SentenceTransformer("all-mpnet-base-v2")
            self.gpt_model         = os.getenv("OPENAI_MODEL")
            self.openai_client     = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

            self.parallel_requests = int(os.getenv("PARALLEL_REQUESTS", "10"))
            self.per_node_limit    = per_node_limit or int(os.getenv("PER_NODE_LIMIT", "10"))

            # NEW: File size thresholds (in characters)
            self.size_thresholds = {
                'small': 2000000,      # < 2K chars - direct processing
                'medium': 4000000,     # 2K-8K chars - semantic chunking
                'large': 8000000,     # 8K-32K chars - hierarchical summary
                'huge': 10000000      # > 32K chars - multi-level chunking
            }
            
            # NEW: Token estimation (rough: 1 token ≈ 4 characters)
            self.max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", "16000"))

            # Neo4j
            self.driver           = get_driver()
            self.neo4j_database   = neo_db or os.getenv("NEO4J_DEFAULT_DB", "neo4j")

            if not self.verify_connection():
                raise ConnectionError("Failed to connect to Neo4j")

            self._setup_vector_indexes()

        except Exception as exc:
            logger.error(f"[HierarchicalSemanticEnrich] init failed: {exc}")
            raise

    # ── Connectivity helpers ──────────────────────────────────
    def verify_connection(self) -> bool:
        try:
            with self.driver.session(database=self.neo4j_database) as s:
                return s.run("RETURN 1 as ok").single().get("ok") == 1
        except Exception as exc:
            logger.error(f"[HierarchicalSemanticEnrich] connectivity check failed: {exc}")
            return False

    @retry_on_connection_error()
    def _setup_vector_indexes(self) -> None:
        # ENHANCED: Added indexes for all node types
        vec_labels = [
            ("class_vector_index",     "CLASS"),
            ("function_vector_index",  "FUNCTION"),
            ("method_vector_index",    "METHOD"),
            ("interface_vector_index", "INTERFACE"),
            ("file_vector_index",      "FILE"),        # NEW
            ("folder_vector_index",    "FOLDER"),      # NEW
            ("branch_vector_index",    "BRANCH"),      # NEW
        ]
        full_text_stmt = """
        CREATE FULLTEXT INDEX code_fulltext_index IF NOT EXISTS
        FOR (n:CLASS|FUNCTION|METHOD|INTERFACE|FILE|FOLDER|BRANCH)
        ON EACH [n.name, n.content, n.text, n.docstring, n.summary]
        """

        with self.driver.session(database=self.neo4j_database) as s:
            existing = {r["name"]: r["type"] for r in s.run("SHOW INDEXES YIELD name, type")}
            for name, label in vec_labels:
                if name not in existing:
                    logger.info(f"[Neo4j] creating vector index {name}")
                    s.run(f"""
                        CREATE VECTOR INDEX {name} IF NOT EXISTS
                        FOR (n:{label}) ON (n.embedding)
                        OPTIONS {{
                          indexConfig: {{
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                          }}
                        }}""")
            if "code_fulltext_index" not in existing:
                logger.info("[Neo4j] creating full-text index code_fulltext_index")
                s.run(full_text_stmt)

  
