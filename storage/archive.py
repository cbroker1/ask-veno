"""SQLite archive reads: video statistics and video records.

Query semantics and returned dictionary shapes are unchanged from the
original web_app.py. The status/duration/date helpers live here (not in the
presentation layer) because they are part of the video-row mapping that both
the HTML archive table and the /api/videos endpoint return.
"""

from __future__ import annotations

import sqlite3

from app.config import get_db_path


def dur_fmt(s):
    if not s or s < 60: return f"{s or 0}s"
    m, sec = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {sec}s"

def date_fmt(raw):
    if not raw: return "N/A"
    if len(raw) == 8 and raw.isdigit(): return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw

def st(k):
    return {"complete":"COMPLETE","queued":"QUEUED","processing":"PROCESSING","failed":"FAILED",
            "not_started":"NOT STARTED","downloaded":"DOWNLOAD▰","transcribed":"TRANSCRIBE▰",
            "embedded":"EMBED▰","cleaned":"CLEAN▰"}.get(k, k.upper())

def stt(k):
    return {"complete":"ok","queued":"pending","processing":"progress","failed":"alert",
            "not_started":"neutral"}.get(k, "neutral")

def get_video_stats():
    db_path = get_db_path()
    if not db_path.exists():
        return {"total": 0, "completed": 0, "in_progress": 0, "failed": 0, "total_chunks": 0}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT COUNT(*) as total,
                SUM(CASE WHEN ingest_status = 'complete' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN ingest_status IN ('queued', 'processing') THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN ingest_status = 'failed' THEN 1 ELSE 0 END) as failed,
                COALESCE(SUM(chunk_count), 0) as total_chunks
            FROM videos
        """).fetchone()
    return {"total": row["total"] or 0, "completed": row["completed"] or 0,
            "in_progress": row["in_progress"] or 0, "failed": row["failed"] or 0,
            "total_chunks": row["total_chunks"] or 0}

def get_videos(limit=20):
    db = get_db_path()
    if not db.exists(): return []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT video_id, title, url, upload_date, duration, ingest_status,
                   audio_status, whisper_status, embedding_status, chunk_count
            FROM videos ORDER BY discovered_at DESC LIMIT ?
        """, (limit,)).fetchall()
    out = []
    for r in rows:
        out.append({**r,
            "ingest_st": st(r["ingest_status"]),
            "ingest_tt": stt(r["ingest_status"]),
            "dur": dur_fmt(r["duration"]),
            "dt": date_fmt(r["upload_date"])})
    return out
