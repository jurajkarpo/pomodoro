"""SQLite project database.

A project preserves:
  - source image references
  - page ordering
  - preprocessing parameters
  - OCR output
  - user corrections
  - PDF metadata
  - processing status

The database lives inside a project directory alongside the
copied original/processed images.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Optional

from ..core.types import (
    DocumentMetadata,
    PageStatus,
    ProcessingSettings,
)
from ..core.config import PROJECTS_DIR


_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    index INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    original_path TEXT,
    processed_path TEXT,
    ocr_json TEXT,
    status TEXT DEFAULT 'pending',
    detected_json TEXT,
    error TEXT,
    warnings TEXT,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS corrections (
    page_index INTEGER,
    word_index INTEGER,
    original_text TEXT,
    corrected_text TEXT,
    verified INTEGER DEFAULT 0,
    ignored INTEGER DEFAULT 0,
    PRIMARY KEY (page_index, word_index)
);
"""


class Project:
    """A BookScan SK project backed by a SQLite database."""

    def __init__(self, path: str) -> None:
        if path.endswith(".bsk"):
            self.dir = os.path.dirname(os.path.abspath(path))
            self.db_path = path
        else:
            self.dir = os.path.abspath(path)
            self.db_path = os.path.join(self.dir, "project.bsk")
        os.makedirs(self.dir, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO project_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM project_meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    @property
    def name(self) -> str:
        return self.get_meta("name", os.path.basename(self.dir))

    @name.setter
    def name(self, value: str) -> None:
        self.set_meta("name", value)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def save_settings(self, settings: ProcessingSettings) -> None:
        from ..pipeline.runner import _settings_to_dict

        self.set_meta("settings", json.dumps(_settings_to_dict(settings)))

    def load_settings(self) -> Optional[ProcessingSettings]:
        from ..pipeline.runner import _settings_from_dict

        raw = self.get_meta("settings")
        if not raw:
            return None
        return _settings_from_dict(json.loads(raw))

    # ------------------------------------------------------------------
    # Document metadata
    # ------------------------------------------------------------------

    def save_document_metadata(self, md: DocumentMetadata) -> None:
        self.set_meta("doc_title", md.title)
        self.set_meta("doc_author", md.author)
        self.set_meta("doc_subject", md.subject)
        self.set_meta("doc_creator", md.creator)
        self.set_meta("doc_ocr_language", md.ocr_language)
        self.set_meta("doc_creation_date", md.creation_date)
        self.set_meta("doc_keywords", md.keywords)
        self.set_meta("doc_first_page_number", str(md.first_page_number))
        self.set_meta("doc_page_offset", str(md.page_offset))

    def load_document_metadata(self) -> DocumentMetadata:
        return DocumentMetadata(
            title=self.get_meta("doc_title"),
            author=self.get_meta("doc_author"),
            subject=self.get_meta("doc_subject"),
            creator=self.get_meta("doc_creator", "BookScan SK"),
            ocr_language=self.get_meta("doc_ocr_language", "sk"),
            creation_date=self.get_meta("doc_creation_date"),
            keywords=self.get_meta("doc_keywords"),
            first_page_number=int(self.get_meta("doc_first_page_number", "1")),
            page_offset=int(self.get_meta("doc_page_offset", "0")),
        )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    def upsert_page(
        self,
        index: int,
        source_path: str,
        filename: str,
        original_path: str = "",
        processed_path: str = "",
        ocr_json: str = "",
        status: str = PageStatus.PENDING.value,
        detected_json: str = "",
        error: str = "",
        warnings: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO pages(index, source_path, filename, original_path, "
            "processed_path, ocr_json, status, detected_json, error, warnings, "
            "updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(index) DO UPDATE SET "
            "source_path=excluded.source_path, filename=excluded.filename, "
            "original_path=excluded.original_path, processed_path=excluded.processed_path, "
            "ocr_json=excluded.ocr_json, status=excluded.status, "
            "detected_json=excluded.detected_json, error=excluded.error, "
            "warnings=excluded.warnings, updated_at=excluded.updated_at",
            (
                index, source_path, filename, original_path, processed_path,
                ocr_json, status, detected_json, error, warnings, time.time(),
            ),
        )
        self._conn.commit()

    def get_page(self, index: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM pages WHERE index=?", (index,)
        ).fetchone()
        return dict(row) if row else None

    def list_pages(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM pages ORDER BY index").fetchall()
        return [dict(r) for r in rows]

    def page_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM pages").fetchone()
        return int(row["c"])

    def pages_by_status(self, status: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM pages WHERE status=? ORDER BY index", (status,)
        ).fetchall()
        return [dict(r) for r in rows]

    def set_page_status(self, index: int, status: str) -> None:
        self._conn.execute(
            "UPDATE pages SET status=?, updated_at=? WHERE index=?",
            (status, time.time(), index),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Corrections
    # ------------------------------------------------------------------

    def save_correction(
        self,
        page_index: int,
        word_index: int,
        original_text: str,
        corrected_text: str,
        verified: bool = False,
        ignored: bool = False,
    ) -> None:
        self._conn.execute(
            "INSERT INTO corrections(page_index, word_index, original_text, "
            "corrected_text, verified, ignored) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(page_index, word_index) DO UPDATE SET "
            "corrected_text=excluded.corrected_text, verified=excluded.verified, "
            "ignored=excluded.ignored",
            (page_index, word_index, original_text, corrected_text,
             int(verified), int(ignored)),
        )
        self._conn.commit()

    def get_corrections(self, page_index: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM corrections WHERE page_index=?", (page_index,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Project":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def create_project(name: str, base_dir: Optional[str] = None) -> Project:
    """Create a new project directory and database."""
    base = base_dir or str(PROJECTS_DIR)
    os.makedirs(base, exist_ok=True)
    slug = _slug(name)
    path = os.path.join(base, slug)
    counter = 1
    while os.path.exists(path):
        path = os.path.join(base, f"{slug}_{counter}")
        counter += 1
    proj = Project(path)
    proj.name = name
    proj.set_meta("created_at", str(time.time()))
    return proj


def _slug(name: str) -> str:
    keep = "-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    s = "".join(c if c in keep else "_" for c in name)
    return s[:60] or "project"


def open_project(path: str) -> Project:
    """Open an existing project."""
    return Project(path)


def list_recent_projects(limit: int = 10) -> list[dict]:
    """List recent projects from the projects directory."""
    out: list[dict] = []
    if not os.path.isdir(str(PROJECTS_DIR)):
        return out
    for name in sorted(os.listdir(str(PROJECTS_DIR)), reverse=True):
        p = os.path.join(str(PROJECTS_DIR), name)
        db = os.path.join(p, "project.bsk")
        if os.path.isfile(db):
            try:
                proj = Project(p)
                out.append({
                    "name": proj.name,
                    "path": p,
                    "pages": proj.page_count(),
                })
                proj.close()
            except Exception:
                pass
        if len(out) >= limit:
            break
    return out
