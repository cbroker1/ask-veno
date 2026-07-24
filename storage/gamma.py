"""Canonical GAMMA 0.9.5 SQLite storage: schema + read-only runtime access.

The database lives beside (never inside) the YouTube ingestion registry —
default .state/gamma_095.sqlite — and is produced atomically by
scripts/build_gamma_095.py. This module owns the DDL (single source of truth
for builder and tests) and the GammaDB reader used at query time.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.config import get_gamma_db_path

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE gamma_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE gamma_items (
    item_id             TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    name_key            TEXT,
    display_variant     TEXT,
    category            TEXT NOT NULL,
    description         TEXT,
    price               INTEGER,
    unobtainable        INTEGER NOT NULL DEFAULT 0,
    in_starting_loadout INTEGER NOT NULL DEFAULT 0,
    has_npc_drop        INTEGER NOT NULL DEFAULT 0,
    has_stash_drop      INTEGER NOT NULL DEFAULT 0,
    name_norm           TEXT NOT NULL,
    source_files        TEXT NOT NULL,
    raw_json            TEXT NOT NULL,
    content_hash        TEXT NOT NULL
);
CREATE INDEX idx_items_name_norm ON gamma_items(name_norm);
CREATE INDEX idx_items_category  ON gamma_items(category);

CREATE TABLE gamma_aliases (
    item_id       TEXT NOT NULL REFERENCES gamma_items(item_id),
    alias         TEXT NOT NULL,
    alias_norm    TEXT NOT NULL,
    alias_compact TEXT NOT NULL,
    source        TEXT NOT NULL,
    UNIQUE(item_id, alias_norm, source)
);
CREATE INDEX idx_aliases_norm    ON gamma_aliases(alias_norm);
CREATE INDEX idx_aliases_compact ON gamma_aliases(alias_compact);

CREATE TABLE gamma_properties (
    item_id     TEXT NOT NULL REFERENCES gamma_items(item_id),
    prop        TEXT NOT NULL,
    label       TEXT,
    value_text  TEXT NOT NULL,
    value_num   REAL,
    is_pct      INTEGER NOT NULL DEFAULT 0,
    unit        TEXT,
    source_file TEXT NOT NULL,
    UNIQUE(item_id, prop, source_file)
);
CREATE INDEX idx_props_item ON gamma_properties(item_id);
CREATE INDEX idx_props_prop ON gamma_properties(prop);

CREATE TABLE gamma_relations (
    src_item_id TEXT NOT NULL REFERENCES gamma_items(item_id),
    rel_type    TEXT NOT NULL,
    dst_item_id TEXT REFERENCES gamma_items(item_id),
    dst_text    TEXT,
    detail      TEXT,
    source_file TEXT NOT NULL
);
CREATE INDEX idx_rel_src ON gamma_relations(src_item_id, rel_type);
CREATE INDEX idx_rel_dst ON gamma_relations(dst_item_id);

CREATE TABLE gamma_facets (
    facet_id     TEXT PRIMARY KEY,
    item_id      TEXT NOT NULL REFERENCES gamma_items(item_id),
    facet_type   TEXT NOT NULL,
    text         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_files TEXT NOT NULL
);
CREATE INDEX idx_facets_item ON gamma_facets(item_id);

CREATE VIRTUAL TABLE gamma_fts USING fts5(
    text,
    facet_id UNINDEXED,
    item_id  UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class GammaDB:
    """Read-only access to the active canonical GAMMA database.

    Missing database files raise GammaUnavailable at open time so callers can
    degrade gracefully (transcript-only operation keeps working).
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else get_gamma_db_path()
        if not self.db_path.exists():
            raise GammaUnavailable(f"GAMMA database not found: {self.db_path}")
        self._conn = connect(self.db_path, read_only=True)

    def close(self) -> None:
        self._conn.close()

    # ── meta ────────────────────────────────────────────────────────────
    def meta(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM gamma_meta").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def active_collection(self) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM gamma_meta WHERE key='active_chroma_collection'"
        ).fetchone()
        return row["value"] if row and row["value"] else None

    # ── items ───────────────────────────────────────────────────────────
    def item(self, item_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM gamma_items WHERE item_id=?", (item_id,)).fetchone()
        return dict(row) if row else None

    def item_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) c FROM gamma_items").fetchone()["c"]

    # ── alias lookup ────────────────────────────────────────────────────
    def lookup_exact(self, variants: list[tuple[str, str]], limit: int = 8) -> list[dict]:
        """Exact alias resolution. `variants` is a list of
        (normalized, compact) strings derived from the query; returns matched
        items ordered deterministically (obtainable first, shortest matched
        alias first, then item_id)."""
        if not variants:
            return []
        norms = sorted({n for n, _ in variants if n})
        compacts = sorted({c for _, c in variants if c})
        if not norms and not compacts:
            return []
        sql = f"""
            SELECT DISTINCT i.*, a.alias AS matched_alias, a.source AS alias_source,
                   length(a.alias) AS alias_len
            FROM gamma_aliases a JOIN gamma_items i ON i.item_id = a.item_id
            WHERE a.alias_norm IN ({",".join("?" * len(norms)) or "''"})
               OR a.alias_compact IN ({",".join("?" * len(compacts)) or "''"})
            ORDER BY i.unobtainable ASC, alias_len ASC, i.item_id ASC
            LIMIT ?
        """
        rows = self._conn.execute(sql, [*norms, *compacts, limit]).fetchall()
        return [dict(r) for r in rows]

    def lookup_prefix(self, compact: str, limit: int = 8) -> list[dict]:
        """Compact-form prefix lookup ('ak74' -> AK-74N …). Only sensible for
        reasonably specific fragments; callers gate on length."""
        if not compact or len(compact) < 3:
            return []
        like = compact.replace("%", "").replace("_", "") + "%"
        rows = self._conn.execute(
            """
            SELECT DISTINCT i.*, a.alias AS matched_alias, a.source AS alias_source,
                   length(a.alias) AS alias_len
            FROM gamma_aliases a JOIN gamma_items i ON i.item_id = a.item_id
            WHERE a.alias_compact LIKE ?
            ORDER BY i.unobtainable ASC, alias_len ASC, i.item_id ASC
            LIMIT ?
            """, (like, limit)).fetchall()
        return [dict(r) for r in rows]

    # ── structured facts ────────────────────────────────────────────────
    def properties(self, item_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT prop, label, value_text, value_num, is_pct, unit, source_file"
            " FROM gamma_properties WHERE item_id=? ORDER BY source_file, prop",
            (item_id,)).fetchall()
        return [dict(r) for r in rows]

    def relations(self, item_id: str, rel_type: str | None = None) -> list[dict]:
        sql = ("SELECT src_item_id, rel_type, dst_item_id, dst_text, detail, source_file"
               " FROM gamma_relations WHERE src_item_id=?")
        args: list = [item_id]
        if rel_type:
            sql += " AND rel_type=?"
            args.append(rel_type)
        sql += " ORDER BY rel_type, dst_item_id, dst_text"
        out = []
        for r in self._conn.execute(sql, args).fetchall():
            d = dict(r)
            d["detail"] = json.loads(d["detail"]) if d["detail"] else None
            out.append(d)
        return out

    def relations_to(self, item_id: str, rel_type: str | None = None) -> list[dict]:
        """Relations pointing AT this item (e.g. weapons using this ammo)."""
        sql = ("SELECT src_item_id, rel_type, dst_item_id, dst_text, detail, source_file"
               " FROM gamma_relations WHERE dst_item_id=?")
        args: list = [item_id]
        if rel_type:
            sql += " AND rel_type=?"
            args.append(rel_type)
        sql += " ORDER BY rel_type, src_item_id"
        out = []
        for r in self._conn.execute(sql, args).fetchall():
            d = dict(r)
            d["detail"] = json.loads(d["detail"]) if d["detail"] else None
            out.append(d)
        return out

    # ── facets ──────────────────────────────────────────────────────────
    def facets(self, item_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT facet_id, item_id, facet_type, text, source_files"
            " FROM gamma_facets WHERE item_id=? ORDER BY facet_id", (item_id,)).fetchall()
        return [dict(r) for r in rows]

    def facet(self, facet_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT facet_id, item_id, facet_type, text, source_files"
            " FROM gamma_facets WHERE facet_id=?", (facet_id,)).fetchone()
        return dict(row) if row else None

    def facet_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) c FROM gamma_facets").fetchone()["c"]

    # ── lexical (BM25) ──────────────────────────────────────────────────
    def fts_search(self, match_query: str, limit: int = 20) -> list[dict]:
        """BM25 search over facet text. `match_query` must already be a valid
        FTS5 MATCH expression (see rag.fusion.build_match_query). Failures
        return [] so dense retrieval keeps working."""
        if not match_query:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT f.facet_id, f.item_id, f.facet_type, f.text, f.source_files,
                       bm25(gamma_fts) AS bm25_score
                FROM gamma_fts JOIN gamma_facets f ON f.facet_id = gamma_fts.facet_id
                WHERE gamma_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """, (match_query, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]


class GammaUnavailable(Exception):
    """The canonical GAMMA database is not present/usable."""
