#!/usr/bin/env python3
"""
Stalker Gamma YouTube RAG Dashboard — Stalker-themed web UI.
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"
DEFAULT_CHROMA_PATH = "data/chroma"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "youtube_chunks")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large")

app = FastAPI(title="⌈ GAMMA PDA — YouTube RAG ⌋")

embedding_model: SentenceTransformer | None = None


def get_db_path():
    return Path(os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH))

def get_chroma_path():
    return Path(os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH))

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
    return embedding_model

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

def dur_fmt(s):
    if not s or s < 60: return f"{s or 0}s"
    m, sec = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h}h {m}m" if h else f"{m}m {sec}s"

def date_fmt(raw):
    if not raw: return "N/A"
    if len(raw) == 8 and raw.isdigit(): return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw

def _dash_offset(count, total=257, circumference=138.23):
    """Compute SVG stroke-dashoffset for a gauge."""
    if total <= 0: return f"{circumference}"
    pct = count / total
    return f"{circumference * (1.0 - pct):.2f}"

def st(k):
    return {"complete":"COMPLETE","queued":"QUEUED","processing":"PROCESSING","failed":"FAILED",
            "not_started":"NOT STARTED","downloaded":"DOWNLOAD▰","transcribed":"TRANSCRIBE▰",
            "embedded":"EMBED▰","cleaned":"CLEAN▰"}.get(k, k.upper())

def stt(k):
    return {"complete":"ok","queued":"pending","processing":"progress","failed":"alert",
            "not_started":"neutral"}.get(k, "neutral")

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

def search_qb(query, top_k=5):
    cb = get_chroma_path()
    if not cb.exists(): raise HTTPException(500, "ChromaDB not found")
    model = get_embedding_model()
    emb = model.encode([f"query: {query}"], convert_to_numpy=True)[0].tolist()
    client = chromadb.PersistentClient(path=str(cb))
    try: col = client.get_collection(CHROMA_COLLECTION)
    except Exception as e: raise HTTPException(500, f"Collection not found: {e}")
    res = col.query(query_embeddings=[emb], n_results=top_k,
                     include=["metadatas", "documents", "distances"])
    raw_chunks = []
    if res["ids"] and res["ids"][0]:
        docs = res.get("documents")
        if docs and docs[0]:
            for i in range(len(res["ids"][0])):
                md = res["metadatas"][0][i]
                dist = res.get("distances", [[None]])[0][i]
                sim = max(0, 1 - dist) if dist else 0
                raw_chunks.append({"id": res["ids"][0][i], "distance": dist,
                    "sim_pct": round(sim*100,1),
                    "vt": md.get("video_title","Unknown")[:100],
                    "vid": md.get("video_id","Unknown"),
                    "s": md.get("start_hms","N/A"),
                    "e": md.get("end_hms","N/A"),
                    "url": md.get("youtube_time_url","N/A"),
                    "txt": docs[0][i][:1200]})
    summary = ""
    if raw_chunks and _OLLAMA_AVAILABLE:
        try:
            summary = _llm_answer(query, raw_chunks)
        except Exception:
            summary = ""
    return {"chunks": raw_chunks, "summary": summary}


_OLLAMA_AVAILABLE = False
try:
    import requests as _req
    _req.get("http://127.0.0.1:11434/api/tags", timeout=2)
    _OLLAMA_AVAILABLE = True
    del _req
except Exception:
    pass

def _llm_answer(query: str, chunks: list) -> str:
    """Ask ollama to answer from the retrieved chunks."""
    try:
        import requests as _req
        resp = _req.get("http://127.0.0.1:11434/api/tags", timeout=2)
        models = resp.json().get("models", [])
        if not any("qwen3:0.6b" in m["name"] for m in models):
            return ""
        available_name = "qwen3:0.6b"
    except Exception:
        return ""
    ctx_lines = []
    for c in chunks:
        ctx_lines.append(f"[{c['s']}-{c['e']}] {c['vt']} ({c['vid']}): {c['txt']}")
    context = "\n\n---\n\n".join(ctx_lines)
    system = (
        "You are a knowledge base assistant answering questions from YouTube video transcripts. "
        "Answer directly and concisely in 3-5 sentences. Use only the provided chunks. "
        "DO NOT just list videos — summarize the answer to the question."
    )
    prompt = f"Question: {query}\n\nSource chunks:\n{context}\n\nAnswer:"
    try:
        import requests as _req2
        resp = _req2.post("http://127.0.0.1:11434/api/generate", json={
            "model": available_name,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": 1024, "temperature": 0.2}
        }, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        return ""
    return ""

# ── presentation layer (UI only — everything below renders HTML around the
#    unchanged retrieval/LLM code above) ──
import html as _html
import time as _time

RING_C = 138.23  # circumference of the 22px-radius telemetry rings


def esc(s: Any) -> str:
    """HTML-escape any value for safe interpolation into markup/attributes."""
    return _html.escape("" if s is None else str(s), quote=True)


def fmt_n(n: Any) -> str:
    try:
        return f"{int(n or 0):,}"
    except (TypeError, ValueError):
        return "0"


def hms_to_seconds(hms: str | None) -> int:
    """'1:53:14' → 6794. Returns 0 for missing/unparseable values."""
    if not hms or hms == "N/A":
        return 0
    try:
        parts = [int(p or 0) for p in str(hms).split(":")]
    except ValueError:
        return 0
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _ring(count, total) -> str:
    """SVG stroke-dashoffset for a telemetry ring filled count/total."""
    total = max(int(total or 0), 1)
    frac = min(max((count or 0) / total, 0.0), 1.0)
    return f"{RING_C * (1.0 - frac):.2f}"


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _clean_summary(text: str) -> str:
    """Display-side cleanup of the local LLM output: drop stray think-tags,
    escape, and preserve line breaks."""
    text = _THINK_RE.sub("", text or "").strip()
    return esc(text).replace("\n", "<br>")


def _clean_vid(vid: str | None) -> str:
    vid = (vid or "").strip()
    if vid.startswith("youtube_id:"):
        vid = vid[len("youtube_id:"):]
    return vid or "N/A"


def _sig_class(sim_pct: float) -> str:
    if sim_pct >= 70:
        return "sig-hi"
    if sim_pct >= 60:
        return "sig-md"
    return "sig-lo"


# ── HTML section builders ──

def _rows(videos):
    if not videos:
        return ('<tr><td class="no-data" colspan="6">ARCHIVE EMPTY — no stream records ingested yet. '
                'Run the discovery pipeline, stalker.</td></tr>')
    h = ""
    for v in videos:
        badge = f'<span class="badge b-{esc(v["ingest_tt"])}">{esc(v["ingest_st"])}</span>'
        if v.get("url"):
            url = esc(v["url"])
            title_cell = f'<a class="row-link" href="{url}" target="_blank" rel="noopener">{esc(v["title"])}</a>'
            lnk = f'<a class="yt-link" href="{url}" target="_blank" rel="noopener" title="Open stream on YouTube">▶</a>'
        else:
            title_cell = esc(v["title"])
            lnk = '<span class="faint">—</span>'
        h += (f'<tr class="video-row" data-title="{esc((v["title"] or "").lower())}">'
              f'<td class="v-title">{title_cell}</td>'
              f'<td>{badge}</td>'
              f'<td class="mono num">{fmt_n(v["chunk_count"])}</td>'
              f'<td class="mono">{esc(v["dur"])}</td>'
              f'<td class="mono dim">{esc(v["dt"])}</td>'
              f'<td class="act">{lnk}</td></tr>')
    return h


_YT_ICON = ('<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
            '<path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0'
            '-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0'
            ' 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24'
            ' 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>')


def _results(chunks, searched: bool) -> str:
    """Stream-evidence section: one card per retrieved transcript segment."""
    if not searched:
        return ""
    if not chunks:
        return ('<div class="no-signal panel">⌁ NO SIGNAL — nothing in the archive matches this query.'
                '<span>Adjust the transmission: try weapon names, locations, mutants, quests, or game mechanics.</span></div>')
    h = ('<section class="evidence"><h2 class="sec-title">▦ STREAM EVIDENCE '
         f'<span class="sec-sub">{len(chunks)} MATCHED SEGMENT{"S" if len(chunks) != 1 else ""} — click a card to play the clip at its timestamp</span></h2>'
         '<div class="results-list">')
    for i, c in enumerate(chunks, 1):
        text = esc(c["txt"]).replace("\n", "<br>")
        sp = c.get("sim_pct") or 0
        vid = _clean_vid(c.get("vid"))
        seconds = hms_to_seconds(c.get("s"))
        yt = c.get("url")
        open_link = (f'<a class="ev-open" href="{esc(yt)}" target="_blank" rel="noopener" '
                     f'title="Open on YouTube at {esc(c["s"])}">OPEN ↗</a>') if yt and yt != "N/A" else ""
        h += f'''<article class="result-card" id="ev-{i}" data-vid="{esc(vid)}" data-start="{seconds}" tabindex="0">
<div class="ev-head"><span class="ev-idx">[{i:02d}]</span><span class="sig {_sig_class(sp)}" style="--w:{min(sp, 100)}%"><span></span></span><span class="sig-text">{sp}% SIGNAL</span><span class="sep">│</span><span class="mono ev-time">{esc(c["s"])} → {esc(c["e"])}</span>{open_link}</div>
<div class="result-title">{esc(c["vt"])}</div>
<div class="result-id">REC {esc(vid)}</div>
<div class="result-text">{text}</div>
<div class="video-expand-hint">{_YT_ICON} PLAY SEGMENT AT {esc(c["s"])}</div>
<div class="video-embed"><div class="embed-frame"><iframe title="Stream segment player" allow="autoplay; encrypted-media" allowfullscreen loading="lazy"></iframe></div></div>
</article>'''
    h += "</div></section>"
    return h


def _echo_html(query: str, chunks, scan_s: float | None) -> str:
    stats_bits = []
    if chunks:
        stats_bits.append(f"{len(chunks)} SEGMENTS MATCHED")
        stats_bits.append(f"TOP SIGNAL {chunks[0]['sim_pct']}%")
    else:
        stats_bits.append("0 SEGMENTS MATCHED")
    if scan_s is not None:
        stats_bits.append(f"SCAN {scan_s:.1f}s")
    meta = " · ".join(stats_bits)
    return (f'<div class="echo-bar"><div class="echo-q">» QUERY: <span class="echo-text">“{esc(query)}”</span></div>'
            f'<div class="echo-meta">{meta}</div>'
            f'<a class="echo-new" href="/">◄ NEW SCAN</a></div>')


def _report_html(chunks, summary: str, error: str) -> str:
    if error:
        return ('<div class="fault-card panel"><div class="fault-label">✚ SCAN FAULT</div>'
                f'<div class="fault-text">Archive query failed: {esc(error)}</div>'
                '<div class="fault-hint">Check that ChromaDB exists and the collection is built, then retry.</div></div>')
    if summary:
        cleaned = _clean_summary(summary)
        if cleaned:
            n = len(chunks)
            return ('<div class="summary-card panel"><div class="summary-head">'
                    '<span class="summary-label">⌁ FIELD REPORT</span>'
                    '<span class="summary-src">SYNTH: qwen3:0.6b · LOCAL</span></div>'
                    f'<div class="summary-response">{cleaned}</div>'
                    f'<div class="summary-note">Synthesized from {n} matched segment{"s" if n != 1 else ""} by a local LLM — verify against the stream evidence below:</div>'
                    f'{_sources_html(chunks)}'
                    '</div>')
    if chunks:
        return ('<div class="offline-card panel"><span class="offline-dot"></span>'
                'SYNTH MODULE OFFLINE — local LLM unavailable or returned nothing. Raw stream evidence below.</div>')
    return ""


def _sources_html(chunks) -> str:
    """Single-column source list embedded in the field report card."""
    if not chunks:
        return ""
    h = '<div class="sources-col">'
    for i, c in enumerate(chunks, 1):
        title = c["vt"] or "Unknown"
        h += (f'<button type="button" class="src-chip" data-target="ev-{i}" '
              f'title="Jump to evidence [{i:02d}] — {esc(title)}">'
              f'<span class="src-n">{i:02d}</span> {esc(c["s"])} · {esc(title)}</button>')
    h += "</div>"
    return h


def _render_page(query: str = "", chunks=None, summary: str = "", error: str = "",
                 scan_s: float | None = None, searched: bool = False) -> str:
    """Single render path for both the dashboard and search-result views."""
    chunks = chunks or []
    stats = get_video_stats()
    videos = get_videos(1000)
    # display order: most recent stream first (upload_date is YYYYMMDD; blanks last,
    # discovered_at order preserved as tiebreak by the stable sort)
    videos.sort(key=lambda v: v.get("upload_date") or "", reverse=True)
    total = stats["total"]

    ctx = dict(
        # topbar / status
        led_db_cls="on" if get_db_path().exists() else "off",
        led_db_txt="ARCHIVE ONLINE" if get_db_path().exists() else "ARCHIVE MISSING",
        led_synth_cls="on" if _OLLAMA_AVAILABLE else "off",
        led_synth_txt="SYNTH ONLINE" if _OLLAMA_AVAILABLE else "SYNTH OFFLINE",
        # telemetry
        total=fmt_n(total),
        completed=fmt_n(stats["completed"]),
        in_progress=fmt_n(stats["in_progress"]),
        failed=fmt_n(stats["failed"]),
        chunks_fmt=fmt_n(stats["total_chunks"]),
        ring_done=_ring(stats["completed"], total),
        ring_prog=_ring(stats["in_progress"], total),
        ring_fail=_ring(stats["failed"], total),
        # search state
        q_attr=esc(query),
        echo_html=_echo_html(query, chunks, scan_s) if searched else "",
        report_html=_report_html(chunks, summary, error) if searched else "",
        results_html=_results(chunks, searched),
        # archive
        video_rows=_rows(videos),
        archive_shown=fmt_n(len(videos)),
    )
    return tmpl(PAGE, **ctx)


# ── template engine ──
def tmpl(template: str, **ctx) -> str:
    """Simple template renderer with two features:
    1. {{key}} — replaced with ctx[key] (or "" if missing)
    2. {{key_if}} … {{/key}} — conditionally keep/discard based on ctx[key]
    """
    result = template

    # Step 1: handle conditional blocks first
    for key in list(ctx.keys()):
        if not key.endswith("_if"):
            continue
        val = ctx[key]
        open_tag = "{{" + key + "}}"
        close_tag = "{{/" + key[:-3] + "}}"
        # Find the block
        start = result.find(open_tag)
        if start < 0:
            continue
        end = result.find(close_tag, start + len(open_tag))
        if end < 0:
            continue
        inner = result[start + len(open_tag):end]
        if val:
            # Keep content (remove the markers)
            result = result[:start] + inner + result[end + len(close_tag):]
        else:
            # Discard content entirely
            result = result[:start] + result[end + len(close_tag):]

    # Step 2: simple variable replacement
    for key, value in ctx.items():
        if key.endswith("_if"):
            continue  # skip conditional keys
        placeholder = "{{" + key + "}}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))

    return result


# ── routes ──
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return HTMLResponse(content=_render_page())


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    query = (query or "").strip()
    if not query:
        return HTMLResponse(content=_render_page())
    error_msg = ""
    t0 = _time.perf_counter()
    try:
        search_res = search_qb(query, 5)
    except Exception as e:
        search_res = {"chunks": [], "summary": ""}
        error_msg = str(e)
        print(f"Search error: {e}")
    scan_s = _time.perf_counter() - t0
    return HTMLResponse(content=_render_page(
        query=query,
        chunks=search_res.get("chunks", []),
        summary=search_res.get("summary", ""),
        error=error_msg,
        scan_s=scan_s,
        searched=True,
    ))


@app.get("/api/stats")
async def api_stats(): return get_video_stats()
@app.get("/api/videos")
async def api_videos(limit: int = 20): return get_videos(limit)
@app.post("/api/search")
async def api_search(query: str = Form(...), top_k: int = Form(5)):
    return search_qb(query, top_k=top_k)


# ── HTML page ──

_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="dark">
<meta name="theme-color" content="#0a0a08">
<title>ASK VENO ⌁ Zone Archive Terminal</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%230a0a08'/%3E%3Crect x='1.5' y='1.5' width='29' height='29' rx='6' fill='none' stroke='%23d4a829' stroke-width='2'/%3E%3Ctext x='16' y='23.5' font-size='19' text-anchor='middle' fill='%23e8d060'%3E%E2%8C%81%3C/text%3E%3C/svg%3E">
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
:root{
  --bg:#0a0a08; --panel:#13110a; --panel2:#0d0b06; --line:#2a2510; --line2:#1a1700;
  --amber:#d4a829; --amber-hi:#f0dc78; --amber-glow:rgba(232,208,96,.35);
  --text:#c8b446; --body:#a3964f; --dim:#6a6230; --faint:#4a4420;
  --green:#40e040; --green-dim:#1a3e1a; --red:#e04040; --red-dim:#3e1a1a;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px;scroll-behavior:smooth}
body{font-family:'Share Tech Mono','Courier New',monospace;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;position:relative}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 3px);pointer-events:none;z-index:10000}
body::after{content:'';position:fixed;inset:0;background:radial-gradient(1100px 380px at 50% -80px,rgba(212,168,41,.06),transparent 70%),radial-gradient(ellipse at center,transparent 58%,rgba(0,0,0,.5) 100%);pointer-events:none;z-index:9999}
a{color:var(--amber);text-decoration:none}
a:hover{color:var(--amber-hi);text-shadow:0 0 8px var(--amber)}
::selection{background:rgba(212,168,41,.35);color:#fff8d8}
:focus-visible{outline:1px solid var(--amber);outline-offset:2px}
body::-webkit-scrollbar{width:9px}
body::-webkit-scrollbar-track{background:var(--bg)}
body::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px;border:2px solid var(--bg)}
.mono{font-family:'Share Tech Mono',monospace}
.dim{color:var(--dim)}
.faint{color:var(--faint)}

/* ── topbar ── */
.topbar{display:flex;align-items:center;gap:18px;padding:8px 18px;border-bottom:1px solid var(--line2);font-size:.68rem;letter-spacing:.14em;color:var(--dim);position:relative;z-index:2;flex-wrap:wrap}
.topbar .brand{font-family:'Orbitron',sans-serif;font-weight:700;color:var(--amber);letter-spacing:.2em}
.topbar .spacer{flex:1}
.led{display:inline-flex;align-items:center;gap:7px;text-transform:uppercase}
.led::before{content:'';width:7px;height:7px;border-radius:50%;background:var(--faint);box-shadow:none}
.led.on{color:#8fb968}
.led.on::before{background:var(--green);box-shadow:0 0 7px rgba(64,224,64,.7);animation:led-pulse 2.6s ease-in-out infinite}
.led.off{color:#8a5a3a}
.led.off::before{background:var(--red);box-shadow:0 0 6px rgba(224,64,64,.55)}
@keyframes led-pulse{0%,100%{opacity:1}50%{opacity:.55}}
#zone-clock{color:var(--dim);letter-spacing:.18em}

/* ── layout / header ── */
.page-wrap{max-width:1240px;margin:0 auto;padding:22px 20px 60px;position:relative;z-index:1}
.header{text-align:center;padding:26px 20px 20px;position:relative}
.h1g{font-family:'Orbitron',sans-serif;font-weight:900;font-size:2rem;letter-spacing:.18em;color:#e8d060;text-shadow:0 0 20px var(--amber-glow),0 0 60px rgba(232,208,96,.1)}
.sub{font-size:.72rem;color:#7a7040;letter-spacing:.28em;text-transform:uppercase;margin-top:8px}
.hs{height:5px;width:100%;margin:6px 0 26px;background:repeating-linear-gradient(-45deg,var(--amber),var(--amber) 8px,var(--line2) 8px,var(--line2) 16px);opacity:.4}
@keyframes pg{0%,100%{text-shadow:0 0 6px rgba(232,208,96,.2)}50%{text-shadow:0 0 18px rgba(232,208,96,.45)}}
.pulse{animation:pg 3.2s ease-in-out infinite}

/* ── panels (PDA corner brackets) ── */
.panel{position:relative;background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:4px}
.corners::before,.corners::after{content:'';position:absolute;width:11px;height:11px;pointer-events:none}
.corners::before{top:-1px;left:-1px;border-top:2px solid var(--amber);border-left:2px solid var(--amber)}
.corners::after{bottom:-1px;right:-1px;border-bottom:2px solid var(--amber);border-right:2px solid var(--amber)}

/* ── console (search) ── */
.console{padding:22px 24px;margin-bottom:22px}
.console-head{display:flex;align-items:baseline;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.console h2{font-family:'Orbitron',sans-serif;font-size:.95rem;letter-spacing:.14em;color:#b0a548;text-transform:uppercase}
.console-hint{font-size:.68rem;color:var(--dim);letter-spacing:.1em;text-transform:uppercase}
.sf{display:flex;gap:8px;margin-top:12px}
.si-wrap{flex:1;display:flex;align-items:center;background:var(--bg);border:1px solid var(--line);border-radius:3px;transition:border-color .2s,box-shadow .2s}
.si-wrap:focus-within{border-color:var(--amber);box-shadow:0 0 14px rgba(212,168,41,.14)}
.si-prompt{color:var(--green);padding:0 4px 0 13px;font-size:1rem;text-shadow:0 0 6px rgba(64,224,64,.4)}
.si{flex:1;font-family:'Share Tech Mono',monospace;font-size:.95rem;background:transparent;border:none;color:#e0cf68;padding:12px 14px 12px 6px;outline:none;min-width:0}
.si::placeholder{color:var(--dim);font-style:italic}
.sbtn{font-family:'Orbitron',sans-serif;font-size:.78rem;font-weight:700;letter-spacing:.12em;background:linear-gradient(180deg,#1e1a0c,#151206);border:1px solid #3a3214;border-radius:3px;color:#e8d060;padding:12px 26px;cursor:pointer;transition:all .2s;text-transform:uppercase}
.sbtn:hover{background:linear-gradient(180deg,#2a2410,#1e1a0a);border-color:var(--amber);box-shadow:0 0 16px rgba(232,208,96,.14)}
.sbtn:disabled{opacity:.5;cursor:wait}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:13px;align-items:center}
.chips-label{font-size:.62rem;color:var(--faint);letter-spacing:.18em;text-transform:uppercase;margin-right:2px}
.chip{font-family:'Share Tech Mono',monospace;font-size:.72rem;color:#b0a548;background:#0c0a05;border:1px solid var(--line2);border-radius:2px;padding:5px 11px;cursor:pointer;transition:all .15s;letter-spacing:.03em;max-width:100%}
.chip:hover{border-color:var(--amber);color:var(--amber-hi);box-shadow:0 0 10px rgba(212,168,41,.12)}
.chip.recent{color:var(--dim);border-style:dashed}
.chip.chip-x{border-color:transparent;color:var(--faint);padding:5px 6px}
.chip.chip-x:hover{color:var(--red);box-shadow:none;border-color:transparent}
#recent-row{display:none}
#recent-row.has-items{display:flex}

/* ── echo bar ── */
.echo-bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:10px 16px;margin-bottom:18px;border:1px solid var(--line2);border-left:3px solid var(--amber);border-radius:3px;background:#0d0b06;font-size:.74rem}
.echo-q{color:var(--dim);letter-spacing:.08em}
.echo-text{color:#e8d060}
.echo-meta{color:var(--dim);letter-spacing:.1em;margin-left:auto}
.echo-new{font-size:.68rem;letter-spacing:.12em;border:1px solid var(--line);border-radius:2px;padding:4px 10px;color:#b0a548}
.echo-new:hover{border-color:var(--amber);text-shadow:none}

/* ── field report / fault / offline ── */
.summary-card{border-left:3px solid var(--green);padding:18px 20px;margin-bottom:22px}
.summary-card.corners::before{border-color:var(--green)}
.summary-card.corners::after{border-color:var(--green)}
.summary-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:10px}
.summary-label{font-family:'Orbitron',sans-serif;font-size:.72rem;letter-spacing:.14em;color:var(--green);text-transform:uppercase;text-shadow:0 0 10px rgba(64,224,64,.3)}
.summary-src{font-size:.62rem;color:var(--dim);letter-spacing:.14em}
.summary-response{font-size:.9rem;color:#cdbd5e;line-height:1.75;border-top:1px solid var(--line2);padding-top:12px}
.summary-note{margin-top:12px;font-size:.66rem;color:var(--dim);letter-spacing:.06em;text-transform:uppercase}
.fault-card{border-left:3px solid var(--red);padding:16px 20px;margin-bottom:18px}
.fault-label{font-family:'Orbitron',sans-serif;font-size:.72rem;letter-spacing:.14em;color:var(--red);margin-bottom:8px}
.fault-text{font-size:.82rem;color:#c88}
.fault-hint{margin-top:8px;font-size:.68rem;color:var(--dim)}
.offline-card{display:flex;align-items:center;gap:10px;border-left:3px solid var(--amber);padding:13px 18px;margin-bottom:18px;font-size:.78rem;color:#b0a548;letter-spacing:.05em}
.offline-dot{width:8px;height:8px;border-radius:50%;background:var(--amber);box-shadow:0 0 8px rgba(212,168,41,.6);flex:none;animation:led-pulse 1.8s ease-in-out infinite}

/* ── report sources (single column inside the field report) ── */
.sources-col{display:flex;flex-direction:column;gap:6px;margin-top:12px}
.src-chip{font-family:'Share Tech Mono',monospace;font-size:.72rem;color:#b0a548;background:#0c0a05;border:1px solid var(--line2);border-radius:2px;padding:6px 11px;cursor:pointer;transition:all .15s;width:100%;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.src-chip:hover{border-color:var(--green);color:#cde0a0;box-shadow:0 0 10px rgba(64,224,64,.1)}
.src-n{color:var(--green);margin-right:4px}

/* ── evidence cards ── */
.sec-title{font-family:'Orbitron',sans-serif;font-size:.9rem;letter-spacing:.12em;color:#b0a548;text-transform:uppercase;margin:4px 0 14px}
.sec-sub{font-family:'Share Tech Mono',monospace;font-size:.64rem;color:var(--faint);letter-spacing:.12em;margin-left:10px}
.results-list{display:flex;flex-direction:column;gap:12px;margin-bottom:26px}
.result-card{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);border-left:3px solid var(--amber);border-radius:3px;padding:16px 18px;cursor:pointer;transition:border-color .2s,background .2s}
.result-card:hover{border-left-color:var(--amber-hi);border-color:#3a3518;background:linear-gradient(160deg,#1a1708,#12100a)}
.result-card.expanded{border-left-color:var(--green);background:linear-gradient(160deg,#100e06,#0d0b04)}
.result-card.flash{animation:card-flash 1.2s ease-out}
@keyframes card-flash{0%{box-shadow:0 0 0 1px var(--green),0 0 24px rgba(64,224,64,.35)}100%{box-shadow:none}}
.ev-head{display:flex;align-items:center;gap:11px;font-size:.7rem;margin-bottom:9px;flex-wrap:wrap}
.ev-idx{color:var(--green);font-size:.72rem;letter-spacing:.05em}
.sig{width:58px;height:5px;border-radius:2px;background:var(--line2);overflow:hidden;position:relative;flex:none}
.sig>span{position:absolute;left:0;top:0;bottom:0;width:var(--w,0%);border-radius:2px;background:repeating-linear-gradient(90deg,currentColor,currentColor 3px,transparent 3px,transparent 5px)}
.sig-hi{color:var(--green)}
.sig-md{color:var(--amber)}
.sig-lo{color:#8a7e3a}
.sig-text{font-size:.64rem;color:var(--dim);letter-spacing:.06em}
.sep{color:var(--line)}
.ev-time{font-size:.7rem;color:#b0a548}
.ev-open{margin-left:auto;font-size:.62rem;letter-spacing:.14em;color:var(--dim);border:1px solid var(--line2);border-radius:2px;padding:3px 8px}
.ev-open:hover{color:var(--amber-hi);border-color:var(--amber);text-shadow:none}
.result-title{font-size:.86rem;color:var(--amber);margin-bottom:3px}
.result-id{font-size:.6rem;color:var(--faint);margin-bottom:9px;letter-spacing:.06em}
.result-text{font-size:.82rem;color:var(--body);line-height:1.65;max-height:132px;overflow-y:auto;border-top:1px dashed var(--line2);padding-top:9px}
.result-card.expanded .result-text{max-height:none;overflow:visible}
.result-text::-webkit-scrollbar{width:4px}
.result-text::-webkit-scrollbar-track{background:var(--panel)}
.result-text::-webkit-scrollbar-thumb{background:var(--line);border-radius:2px}
.video-expand-hint{display:flex;align-items:center;gap:6px;font-family:'Orbitron',sans-serif;font-size:.6rem;letter-spacing:.12em;color:var(--amber);margin-top:10px;opacity:.65;transition:opacity .2s}
.result-card:hover .video-expand-hint{opacity:1;color:var(--amber-hi)}
.result-card.expanded .video-expand-hint{display:none}
.video-embed{display:none;margin-top:14px;pointer-events:none}
.result-card.expanded .video-embed{display:block;pointer-events:auto}
.embed-frame{width:100%;aspect-ratio:16/9;max-height:560px;position:relative;background:var(--bg);border:1px solid var(--line);border-radius:3px}
.embed-frame iframe{position:absolute;inset:0;width:100%;height:100%;border:none;background:var(--bg)}
.no-signal{text-align:center;padding:34px 20px;margin-bottom:26px;color:#8a7e3a;font-size:.88rem;letter-spacing:.06em}
.no-signal span{display:block;margin-top:9px;font-size:.72rem;color:var(--faint);font-style:italic}

/* ── archive telemetry ── */
.telemetry{margin-top:10px}
.gauges{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:26px}
.gauge{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:4px;padding:16px 12px;text-align:center;position:relative}
.gauge::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(212,168,41,.25),transparent)}
.gauge svg{display:block;margin:0 auto 6px}
.gv{font-family:'Orbitron',sans-serif;font-size:1.55rem;font-weight:700;color:#e8d060;text-shadow:0 0 12px rgba(232,208,96,.28);line-height:1.15}
.gl{font-size:.6rem;color:var(--dim);text-transform:uppercase;letter-spacing:.16em;margin-top:5px}
.gauge.wide .gv{font-size:1.9rem;padding:14px 0 2px}

/* ── archive table ── */
.vs{margin-top:6px}
.vs-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.vs-head h2{font-family:'Orbitron',sans-serif;font-size:.9rem;letter-spacing:.12em;color:#b0a548;text-transform:uppercase}
.vs-count{font-size:.66rem;color:var(--faint);letter-spacing:.12em}
.filter-wrap{margin-left:auto;display:flex;align-items:center;gap:8px}
.filter-wrap label{font-size:.62rem;color:var(--faint);letter-spacing:.16em}
#archive-filter{font-family:'Share Tech Mono',monospace;font-size:.76rem;background:var(--bg);border:1px solid var(--line);border-radius:2px;color:#e0cf68;padding:6px 10px;outline:none;width:210px}
#archive-filter:focus{border-color:var(--amber)}
.tw{background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:4px;overflow:auto;max-height:540px}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th{font-family:'Orbitron',sans-serif;font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#5a5230;padding:12px 16px;text-align:left;border-bottom:1px solid var(--line);background:var(--panel2);position:sticky;top:0;z-index:2}
tr{border-bottom:1px solid var(--line2);transition:background .15s}
tbody tr:hover{background:#1a1708}
td{padding:10px 16px;color:#b0a548}
.v-title{color:var(--text);max-width:520px}
.row-link{color:var(--text)}
.row-link:hover{color:var(--amber-hi);text-shadow:none}
.num{text-align:left}
.act{white-space:nowrap;text-align:center}
.yt-link{font-size:1.05rem;display:inline-block;transition:transform .15s}
.yt-link:hover{transform:scale(1.2)}
.badge{font-size:.66rem;padding:3px 9px;border-radius:2px;letter-spacing:.06em;display:inline-block;white-space:nowrap}
.b-ok{background:#0a1e0a;color:var(--green);border:1px solid var(--green-dim);text-shadow:0 0 6px rgba(64,224,64,.2)}
.b-pending{background:#1e1a06;color:var(--amber);border:1px solid #3a3214}
.b-progress{background:#1e1a06;color:var(--amber-hi);border:1px solid #3a3214;animation:led-pulse 1.6s ease-in-out infinite}
.b-alert{background:#1e0a0a;color:var(--red);border:1px solid var(--red-dim)}
.b-neutral{background:var(--line2);color:#7a7040;border:1px solid var(--line)}
.no-data{text-align:center;padding:34px 20px;color:#3a3618;font-style:italic;font-size:.85rem}
.no-data::before{content:'⌁ ';color:var(--line)}

/* ── footer ── */
.ft{text-align:center;padding:32px 20px 6px;border-top:1px solid var(--line2);margin-top:44px;font-size:.64rem;color:#3a3618;letter-spacing:.18em;text-transform:uppercase;line-height:2}
.ft .keys{color:var(--faint)}
.ft .keys b{color:var(--dim);font-weight:400}

/* ── scan loader overlay ── */
.scan-loader{position:fixed;inset:0;z-index:10001;display:none;justify-content:center;align-items:center;background:rgba(8,8,6,.86);backdrop-filter:blur(1px)}
.scan-loader.active{display:flex}
.scan-box{position:relative;text-align:left;background:linear-gradient(160deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:4px;padding:26px 30px;min-width:min(480px,92vw)}
.scan-box::before,.scan-box::after{content:'';position:absolute;width:12px;height:12px}
.scan-box::before{top:-1px;left:-1px;border-top:2px solid var(--amber);border-left:2px solid var(--amber)}
.scan-box::after{bottom:-1px;right:-1px;border-bottom:2px solid var(--amber);border-right:2px solid var(--amber)}
.scan-top{display:flex;align-items:center;gap:16px;margin-bottom:16px}
.scan-sym{width:56px;height:56px;flex:none;display:flex;align-items:center;justify-content:center;font-size:2.6rem;line-height:1;color:#e8d060;text-shadow:0 0 22px rgba(232,208,96,.5);animation:spin 1.6s linear infinite}
.scan-title{font-family:'Orbitron',sans-serif;font-size:.82rem;letter-spacing:.16em;color:var(--amber);text-transform:uppercase}
.scan-elapsed{font-size:.68rem;color:var(--dim);letter-spacing:.14em;margin-top:3px}
.scan-log{font-size:.74rem;color:#b0a548;line-height:2;min-height:6em}
.scan-log .cur::after{content:'▮';margin-left:4px;color:var(--green);animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
@keyframes spin{from{transform:rotate(0)}to{transform:rotate(360deg)}}

@media(max-width:768px){
  html{font-size:14px}
  .h1g{font-size:1.4rem;letter-spacing:.1em}
  .sub{letter-spacing:.16em;font-size:.66rem}
  .topbar{gap:8px 14px;padding:7px 12px}
  .topbar .spacer{display:none}
  .topbar .dim{display:none}
  #zone-clock{display:none}
  .gauges{grid-template-columns:repeat(2,1fr)}
  .sf{flex-direction:column}
  .sbtn{width:100%}
  .echo-meta{margin-left:0}
  .filter-wrap{margin-left:0;width:100%}
  #archive-filter{flex:1;width:auto}
  td,th{padding:8px 9px}
  .v-title{max-width:52vw}
}
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation:none!important;transition:none!important}
  html{scroll-behavior:auto}
}
</style>
</head>
<body>

<div class="topbar">
<span class="brand">◈ ASK-VENO PDA</span>
<span class="dim">ZONE ARCHIVE TERMINAL v2.0</span>
<span class="spacer"></span>
<span class="led {{led_db_cls}}">{{led_db_txt}}</span>
<span class="led {{led_synth_cls}}">{{led_synth_txt}}</span>
<span id="zone-clock" class="mono">--:--:--</span>
</div>

<div class="page-wrap">

<div class="header">
<h1 class="h1g pulse">⌈ ASK VENO ⌋</h1>
<div class="sub">Venoxium Stream Archive — S.T.A.L.K.E.R. GAMMA · One Life Ironman</div>
</div>
<div class="hs"></div>

<section class="console panel corners">
<div class="console-head">
<h2>⌁ Transmit Query to Archive</h2>
<span class="console-hint">semantic scan · {{total}} streams · {{chunks_fmt}} transcript segments · citations with timestamps</span>
</div>
<form class="sf" method="post" action="/search" id="search-form">
<div class="si-wrap">
<span class="si-prompt">&gt;</span>
<input class="si" type="text" name="query" id="search-input" value="{{q_attr}}"
       placeholder="ask the archive — e.g. what does Veno use for early-game armor?"
       autocomplete="off" spellcheck="false" required maxlength="300">
</div>
<button class="sbtn" type="submit" id="scan-btn">⌈ SCAN ▰</button>
</form>
<div class="chips" id="example-row">
<span class="chips-label">TRY ▸</span>
<button type="button" class="chip">best weapon for killing mutants</button>
<button type="button" class="chip">how to make money early game</button>
<button type="button" class="chip">surviving emissions and psi storms</button>
<button type="button" class="chip">artifact hunting and detectors</button>
<button type="button" class="chip">weapon repair and upgrade kits</button>
<button type="button" class="chip">which faction to join</button>
</div>
<div class="chips" id="recent-row">
<span class="chips-label">RECENT TRANSMISSIONS ▸</span>
</div>
</section>

{{echo_html}}
{{report_html}}
{{results_html}}

<section class="telemetry">
<h2 class="sec-title">▦ Archive Telemetry</h2>
<div class="gauges">
<div class="gauge" title="Streams discovered and registered in the archive"><svg width="58" height="58" viewBox="0 0 52 52"><defs><linearGradient id="g2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#40e040"/><stop offset="70%" stop-color="#d4a829"/><stop offset="100%" stop-color="#e04040"/></linearGradient></defs><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="url(#g2)" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="138.23" stroke-dashoffset="0" transform="rotate(-90 26 26)"/></svg><div class="gv">{{total}}</div><div class="gl">Streams on Record</div></div>
<div class="gauge" title="Streams fully transcribed and embedded"><svg width="58" height="58" viewBox="0 0 52 52"><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="#40e040" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="138.23" stroke-dashoffset="{{ring_done}}" transform="rotate(-90 26 26)"/></svg><div class="gv" style="color:#40e040">{{completed}}</div><div class="gl">Fully Processed</div></div>
<div class="gauge" title="Streams queued or mid-pipeline"><svg width="58" height="58" viewBox="0 0 52 52"><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="#d4a829" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="138.23" stroke-dashoffset="{{ring_prog}}" transform="rotate(-90 26 26)"/></svg><div class="gv">{{in_progress}}</div><div class="gl">In Pipeline</div></div>
<div class="gauge wide" title="Transcript chunks embedded in ChromaDB"><div class="gv">{{chunks_fmt}}</div><div class="gl">Segments Indexed</div></div>
<div class="gauge" title="Failed ingests"><svg width="58" height="58" viewBox="0 0 52 52"><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="#e04040" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="138.23" stroke-dashoffset="{{ring_fail}}" transform="rotate(-90 26 26)"/></svg><div class="gv" style="color:#e04040">{{failed}}</div><div class="gl">Anomalies</div></div>
</div>
</section>

<section class="vs">
<div class="vs-head">
<h2>▦ Stream Records</h2>
<span class="vs-count"><span id="archive-count">{{archive_shown}}</span> / {{archive_shown}} SHOWN</span>
<div class="filter-wrap">
<label for="archive-filter">FILTER ▸</label>
<input id="archive-filter" type="text" placeholder="title contains…" autocomplete="off" spellcheck="false">
</div>
</div>
<div class="tw"><table><thead><tr><th>Title</th><th>Status</th><th>Segments</th><th>Runtime</th><th>Streamed</th><th>▶</th></tr></thead><tbody id="archive-body">{{video_rows}}</tbody></table></div>
</section>

<div class="ft">
◈ ASK-VENO // ZONE ARCHIVE TERMINAL ◈ FULLY LOCAL — CHROMADB · E5-LARGE · QWEN3:0.6B VIA OLLAMA ◈<br>
<span class="keys"><b>[/]</b> FOCUS QUERY &nbsp;·&nbsp; <b>[ENTER]</b> SCAN &nbsp;·&nbsp; <b>[ESC]</b> COLLAPSE PLAYBACK</span>
</div>

</div><!-- /page-wrap -->

<!-- scan loader overlay -->
<div class="scan-loader" id="scan-loader" role="status" aria-live="polite">
<div class="scan-box">
<div class="scan-top">
<span class="scan-sym">☢</span>
<div>
<div class="scan-title">Scanning the Archive</div>
<div class="scan-elapsed" id="scan-elapsed">T+0.0s</div>
</div>
</div>
<div class="scan-log" id="scan-log"></div>
</div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    var form   = document.getElementById('search-form');
    var input  = document.getElementById('search-input');
    var btn    = document.getElementById('scan-btn');
    var loader = document.getElementById('scan-loader');

    // ── zone clock ──
    var clock = document.getElementById('zone-clock');
    function tick() {
        var d = new Date();
        function p(n){ return (n < 10 ? '0' : '') + n; }
        if (clock) clock.textContent = p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds()) + ' LOCAL';
    }
    tick(); setInterval(tick, 1000);

    // ── recent transmissions (localStorage, client-side only) ──
    var LS_KEY = 'askveno_recent';
    function getRecent() {
        try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]'); } catch (e) { return []; }
    }
    function saveRecent(q) {
        try {
            var list = getRecent().filter(function(x){ return x !== q; });
            list.unshift(q);
            localStorage.setItem(LS_KEY, JSON.stringify(list.slice(0, 5)));
        } catch (e) {}
    }
    function submitQuery(q) {
        if (!input || !form) return;
        input.value = q;
        form.requestSubmit();
    }
    function renderRecent() {
        var row = document.getElementById('recent-row');
        if (!row) return;
        row.querySelectorAll('.chip').forEach(function(c){ c.remove(); });
        var list = getRecent();
        list.forEach(function(q) {
            var b = document.createElement('button');
            b.type = 'button';
            b.className = 'chip recent';
            b.textContent = q.length > 48 ? q.slice(0, 48) + '…' : q;
            b.title = q;
            b.addEventListener('click', function(){ submitQuery(q); });
            row.appendChild(b);
        });
        if (list.length) {
            var x = document.createElement('button');
            x.type = 'button';
            x.className = 'chip chip-x';
            x.textContent = '✕ CLEAR';
            x.title = 'Forget recent transmissions';
            x.addEventListener('click', function(){
                try { localStorage.removeItem(LS_KEY); } catch (e) {}
                renderRecent();
            });
            row.appendChild(x);
        }
        row.classList.toggle('has-items', list.length > 0);
    }
    renderRecent();

    // ── example chips ──
    document.querySelectorAll('#example-row .chip').forEach(function(c) {
        c.addEventListener('click', function(){ submitQuery(c.textContent); });
    });

    // ── scan loader: staged, honest pipeline log + elapsed timer ──
    var scanTimers = [];
    function logLine(text) {
        var log = document.getElementById('scan-log');
        if (!log) return;
        log.querySelectorAll('.cur').forEach(function(el){ el.classList.remove('cur'); });
        var div = document.createElement('div');
        div.className = 'cur';
        div.textContent = text;
        log.appendChild(div);
    }
    function startScanFx() {
        if (!loader) return;
        loader.classList.add('active');
        if (btn) btn.disabled = true;
        var t0 = Date.now();
        var el = document.getElementById('scan-elapsed');
        scanTimers.push(setInterval(function() {
            if (el) el.textContent = 'T+' + ((Date.now() - t0) / 1000).toFixed(1) + 's';
        }, 100));
        logLine('> encoding query vector  [e5-large // cpu]');
        scanTimers.push(setTimeout(function(){ logLine('> sweeping {{chunks_fmt}} archive segments'); }, 2800));
        scanTimers.push(setTimeout(function(){ logLine('> synthesizing field report  [qwen3:0.6b // local]'); }, 5600));
        scanTimers.push(setTimeout(function(){ logLine('> extended scan — cold codec warm-up, hold position stalker'); }, 20000));
    }
    if (form) {
        form.addEventListener('submit', function() {
            var q = input ? input.value.trim() : '';
            if (q) saveRecent(q);
            startScanFx();
        });
    }

    // ── evidence card playback (one open at a time) ──
    function collapseCard(card) {
        card.classList.remove('expanded');
        var f = card.querySelector('iframe');
        if (f && f.src && f.dataset.filled === 'true') {
            setTimeout(function(){ f.src = ''; f.dataset.filled = ''; }, 300);
        }
    }
    function expandCard(card) {
        document.querySelectorAll('.result-card.expanded').forEach(function(o) {
            if (o !== card) collapseCard(o);
        });
        card.classList.add('expanded');
        var vid = card.dataset.vid;
        var start = card.dataset.start || 0;
        var f = card.querySelector('iframe');
        if (vid && vid !== 'N/A' && f && !f.dataset.filled) {
            f.src = 'https://www.youtube.com/embed/' + vid + '?autoplay=1&start=' + start +
                    '&rel=0&origin=' + encodeURIComponent(window.location.origin);
            f.dataset.filled = 'true';
        }
    }
    document.addEventListener('click', function(e) {
        var chip = e.target.closest('.src-chip');
        if (chip) {
            var target = document.getElementById(chip.dataset.target);
            if (target) {
                expandCard(target);
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                target.classList.remove('flash');
                void target.offsetWidth;
                target.classList.add('flash');
            }
            return;
        }
        if (e.target.closest('a')) return;               // let links behave normally
        var card = e.target.closest('.result-card');
        if (card) {
            var sel = window.getSelection();
            if (sel && sel.toString()) return;            // don't hijack text selection
            if (card.classList.contains('expanded')) collapseCard(card);
            else expandCard(card);
        }
    });
    document.addEventListener('keydown', function(e) {
        if ((e.key === 'Enter' || e.key === ' ') && e.target.classList && e.target.classList.contains('result-card')) {
            e.preventDefault();
            e.target.click();
        }
        if (e.key === 'Escape') {
            document.querySelectorAll('.result-card.expanded').forEach(collapseCard);
            if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
        }
        if (e.key === '/' && input) {
            var tag = (e.target.tagName || '').toLowerCase();
            if (tag !== 'input' && tag !== 'textarea') {
                e.preventDefault();
                input.focus();
                input.select();
            }
        }
    });

    // ── archive table filter (client-side, over loaded rows) ──
    var filter = document.getElementById('archive-filter');
    var counter = document.getElementById('archive-count');
    if (filter) {
        filter.addEventListener('input', function() {
            var q = filter.value.trim().toLowerCase();
            var shown = 0;
            document.querySelectorAll('#archive-body tr[data-title]').forEach(function(tr) {
                var hit = !q || tr.dataset.title.indexOf(q) !== -1;
                tr.style.display = hit ? '' : 'none';
                if (hit) shown++;
            });
            if (counter) counter.textContent = shown;
        });
    }

    // ── focus query on a fresh terminal ──
    if (input && !input.value) {
        try { input.focus({ preventScroll: true }); } catch (e) { input.focus(); }
    }
});
</script>
</body>
</html>
"""

SEARCH_PAGE = _PAGE
PAGE = _PAGE


if __name__ == "__main__":
    import uvicorn
    print("=" * 62)
    print("  ⌈ ASK VENO ⌋ — Zone Archive Terminal (S.T.A.L.K.E.R. GAMMA)")
    print("=" * 62)
    print(f"  DB       : {get_db_path()}")
    print(f"  ChromaDB : {get_chroma_path()}  [{CHROMA_COLLECTION}]")
    print(f"  Embedder : {EMBED_MODEL_NAME}")
    print(f"  Synth    : {'ONLINE' if _OLLAMA_AVAILABLE else 'OFFLINE'} (local Ollama)")
    print(f"  Terminal : http://localhost:9876")
    print("=" * 62)
    uvicorn.run(app, host="0.0.0.0", port=9876, log_level="info")
