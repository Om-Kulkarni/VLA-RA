"""
Database Management Interface

This module manages a local SQLite database for reading and writing processed paper data.
Any database failure must trigger an explicit crash/halt, not caught silently.
"""

import os
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_papers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id        TEXT    NOT NULL UNIQUE,
    title           TEXT,
    summary         TEXT,
    pdf_url         TEXT,
    status          TEXT    DEFAULT 'pending_review',
    priority_score  REAL,
    github_link     TEXT,
    project_website TEXT,
    metadata        TEXT    -- JSON-serialised dict of rubric criteria
);
"""


class DatabaseClient:
    """
    Client wrapper for interacting with the local SQLite database.
    The database file is created automatically if it does not exist.
    """

    def __init__(self, db_path: str = "data/vla_ra.db"):
        """
        Initialises the SQLite connection and ensures the schema exists.

        Args:
            db_path (str): Path to the SQLite database file.
                           Parent directories are created automatically.
        """
        if not db_path:
            raise ValueError("db_path must be a non-empty string.")

        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        # check_same_thread=False is safe here because LangGraph runs nodes sequentially
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row  # enables dict-like row access
        self._conn.execute("PRAGMA journal_mode=WAL;")  # better crash safety
        self._init_schema()
        logger.info(f"DatabaseClient initialised. DB path: {db_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Creates tables if they do not already exist."""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Converts a sqlite3.Row to a plain dict, deserialising JSON metadata."""
        d = dict(row)
        if d.get("metadata") and isinstance(d["metadata"], str):
            try:
                d["metadata"] = json.loads(d["metadata"])
            except json.JSONDecodeError:
                pass
        return d

    # ------------------------------------------------------------------
    # Public API (mirrors the original Supabase interface)
    # ------------------------------------------------------------------

    def check_exists(self, external_id: str) -> bool:
        """
        Checks if a paper already exists in the database.

        Args:
            external_id (str): The unique ArXiv ID to check.

        Returns:
            bool: True if it exists, False otherwise.
        """
        cursor = self._conn.execute(
            "SELECT id FROM research_papers WHERE arxiv_id = ?", (external_id,)
        )
        return cursor.fetchone() is not None

    def insert_record(self, data: dict):
        """
        Inserts a new paper record into the database. Fails loudly on error.

        Args:
            data (dict): The paper payload. Expected keys: arxiv_id, title,
                         summary, pdf_url, status.
        """
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)

        self._conn.execute(
            """
            INSERT INTO research_papers (arxiv_id, title, summary, pdf_url, status, metadata)
            VALUES (:arxiv_id, :title, :summary, :pdf_url, :status, :metadata)
            """,
            {
                "arxiv_id": data.get("arxiv_id"),
                "title": data.get("title"),
                "summary": data.get("summary"),
                "pdf_url": data.get("pdf_url"),
                "status": data.get("status", "pending_review"),
                "metadata": metadata,
            },
        )
        self._conn.commit()

    def get_papers_by_status(self, status: str) -> list[dict]:
        """
        Fetches all papers that match the given status.

        Args:
            status (str): The status to filter by (e.g., 'pending_review').

        Returns:
            list[dict]: A list of paper records matching the status.
        """
        cursor = self._conn.execute(
            "SELECT * FROM research_papers WHERE status = ?", (status,)
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def update_status(self, external_id: str, new_status: str):
        """
        Updates the status of a specific paper. Fails loudly on error.

        Args:
            external_id (str): The ArXiv ID.
            new_status (str): The new status ('pending_review', 'scored', 'rejected').
        """
        self._conn.execute(
            "UPDATE research_papers SET status = ? WHERE arxiv_id = ?",
            (new_status, external_id),
        )
        self._conn.commit()

    def update_paper_score(self, external_id: str, score: float | None, metadata: dict):
        """
        Updates the score and/or metadata of a specific paper. Fails loudly on error.

        When called by the Critic, pass a float score to set priority_score and
        transition status to 'scored'.

        When called by the Analyst, pass score=None to append metadata (e.g. the
        analysis summary) without overwriting the Critic's existing priority_score.
        Status must be updated separately via update_status() in that case.

        Args:
            external_id (str): The ArXiv ID.
            score (float | None): The final priority score, or None to preserve existing value.
            metadata (dict): The dict of extracted heuristic values or analysis summary.
        """
        serialised_metadata = json.dumps(metadata)

        if score is not None:
            # Full Critic update — set score, status, and all metadata columns
            self._conn.execute(
                """
                UPDATE research_papers
                SET priority_score  = :priority_score,
                    status          = :status,
                    metadata        = :metadata,
                    github_link     = :github_link,
                    project_website = :project_website
                WHERE arxiv_id = :arxiv_id
                """,
                {
                    "priority_score": score,
                    "status": "scored",
                    "metadata": serialised_metadata,
                    "github_link": metadata.get("github_link"),
                    "project_website": metadata.get("project_website"),
                    "arxiv_id": external_id,
                },
            )
        else:
            # Analyst update — append metadata only, preserve existing priority_score
            self._conn.execute(
                """
                UPDATE research_papers
                SET metadata = :metadata
                WHERE arxiv_id = :arxiv_id
                """,
                {
                    "metadata": serialised_metadata,
                    "arxiv_id": external_id,
                },
            )

        self._conn.commit()

    def close(self):
        """Closes the database connection explicitly."""
        self._conn.close()

