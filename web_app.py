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

def search_qb(query, top_k=8):
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
        "Give a clear, direct answer first, then briefly cite your sources below. "
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
            "options": {"num_predict": 256, "temperature": 0.2}
        }, timeout=120)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        return ""
    return ""

# ── HTML builders ──
def _rows(videos):
    if not videos: return '<tr><td class="no-data" colspan="6">No videos in database. Run discovery first.</td></tr>'
    h = ""
    for v in videos:
        badge = f'<span class="badge badge-{v["ingest_tt"]}">{v["ingest_st"]}</span>'
        lnk = '<a class="yt-link" href="'+v["url"]+'" target="_blank" rel="noopener">▶</a>' if v.get("url") else "—"
        t = v["title"][:85] + ('…' if len(v["title"])>85 else '')
        h += f'<tr class="video-row"><td class="video-title">{t}</td><td>{badge}</td><td class="mono">{v["chunk_count"]}</td><td class="mono">{v["dur"]}</td><td class="mono dim">{v["dt"]}</td><td class="actions">{lnk}</td></tr>'
    return h

def _results(results):
    if not results: return '<div class="no-data">⌁ No matches found.</div>'
    h = '<div class="results"><h3>⌁ '+str(len(results))+' matches found</h3>'
    for r in results:
        text = r["txt"].replace("\n","<br>").replace("<","&lt;").replace(">","&gt;")
        sp = r["sim_pct"]
        cl = f"sim-{min(10,int(sp/10))}" if sp>0 else "sim-0"
        start_hms = r.get("s", "00:00:00")
        seconds = 0
        if start_hms and start_hms != "N/A":
            try:
                parts = start_hms.split(":")
                hours = int(parts[0]) if parts[0] else 0
                minutes = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                secs = int(parts[2]) if len(parts) > 2 and parts[2] else 0
                seconds = hours * 3600 + minutes * 60 + secs
            except (ValueError, IndexError):
                seconds = 0
        vid = r.get("vid") or "N/A"
        video_id = vid
        if video_id and video_id.startswith("youtube_id:"):
            video_id = video_id.replace("youtube_id:", "")
        h += f'''<div class="result-card" data-vid="{video_id}" data-start="{seconds}">
<div class="result-meta"><span class="sim-meter {cl}"></span><span class="sim-text">{sp}% match</span><span class="sep">│</span><span class="mono">{r["s"]}–{r["e"]}</span></div>
<div class="result-title">{r["vt"]}</div>
<div class="result-id mono small">{r["vid"]}</div>
<div class="result-text">{text}</div>
<div class="video-expand-hint">▶ CLICK TO WATCH</div>
<div class="video-embed"><div class="embed-frame"><iframe allow="autoplay; encrypted-media" allowfullscreen loading="lazy"></iframe></div></div>
</div>'''
    h += '</div>'
    return h

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
    stats = get_video_stats()
    videos = get_videos(20)
    tables_html = _rows(videos)
    ctx = dict(
        total=stats["total"],
        completed=stats["completed"],
        in_progress=stats["in_progress"],
        failed=stats["failed"],
        total_chunks=stats["total_chunks"],
        total_pct=_dash_offset(stats["total"]),
        done_pct=_dash_offset(stats["completed"]),
        in_prog_pct=_dash_offset(stats["in_progress"]),
        v=tables_html,
        r_if=False,
        summary_html="",
    )
    return HTMLResponse(content=tmpl(PAGE, **ctx))

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    stats = get_video_stats()
    videos = get_videos(20)
    try:
        search_res = search_qb(query, 8)
    except Exception as e:
        search_res = {"chunks": [], "summary": ""}; print(f"Search error: {e}")
    chunks = search_res.get("chunks", [])
    summary = search_res.get("summary", "")
    tables_html = _rows(videos)
    results_html = _results(chunks)
    rc = str(len(chunks)) if chunks else '0'

    summary_html = ""
    has_summary = False
    if summary:
        has_summary = True
        summary_html = (
            '<div class="summary-card">'
            '<div class="summary-label">⌁ Gamma Analysis</div>'
            f'<div class="summary-response">{summary}</div></div>'
        )

    ctx = dict(
        total=stats["total"],
        completed=stats["completed"],
        in_progress=stats["in_progress"],
        failed=stats["failed"],
        total_chunks=stats["total_chunks"],
        query=query,
        q=query,
        results_html=results_html,
        r=results_html,
        rc=rc,
        summary_html=summary_html,
        v=tables_html,
        has_summary=has_summary,
        r_if=bool(chunks),
    )
    return HTMLResponse(content=tmpl(SEARCH_PAGE, **ctx))

@app.get("/api/stats")
async def api_stats(): return get_video_stats()
@app.get("/api/videos")
async def api_videos(limit: int = 20): return get_videos(limit)
@app.post("/api/search")
async def api_search(query: str = Form(...), top_k: int = Form(8)):
    return search_qb(query, top_k=top_k)


# ── HTML page ──

_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⌈ GAMMA PDA — YouTube RAG ⌋</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px}
body{font-family:'Share Tech Mono','Courier New',monospace;background:#0a0a08;color:#c8b446;min-height:100vh;overflow-x:hidden;position:relative}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 1px,rgba(0,0,0,.08) 1px,rgba(0,0,0,.08) 2px);pointer-events:none;z-index:10000}
body::after{content:'';position:fixed;inset:0;background:radial-gradient(ellipse at center,transparent 55%,rgba(0,0,0,.55) 100%);pointer-events:none;z-index:9999}
a{color:#d4a829;text-decoration:none}
a:hover{color:#f5d060;text-shadow:0 0 8px #d4a829}
.page-wrap{max-width:1280px;margin:0 auto;padding:24px 20px 60px;position:relative;z-index:1}
.header{text-align:center;padding:40px 20px 28px;border-bottom:1px solid #2a2510;margin-bottom:32px;position:relative}
.header::after{content:'';position:absolute;bottom:-1px;left:10%;right:10%;height:1px;background:linear-gradient(90deg,transparent,#d4a829,transparent)}
.h1g{font-family:'Orbitron',sans-serif;font-weight:900;font-size:2.2rem;letter-spacing:.15em;color:#e8d060;text-shadow:0 0 20px rgba(232,208,96,.35),0 0 60px rgba(232,208,96,.1);margin-bottom:6px}
.sub{font-size:.8rem;color:#7a7040;letter-spacing:.3em;text-transform:uppercase}
.hs{height:6px;width:100%;margin-bottom:32px;background:repeating-linear-gradient(-45deg,#d4a829,#d4a829 8px,#1a1700 8px,#1a1700 16px);opacity:.55}
.gauges{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px;margin-bottom:36px}
.gauge{background:linear-gradient(160deg,#141208,#0e0c06);border:1px solid #2a2510;border-radius:4px;padding:22px 20px;text-align:center;position:relative;overflow:hidden}
.gauge::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#d4a82940,transparent)}
.gv{font-family:'Orbitron',sans-serif;font-size:2.4rem;font-weight:700;color:#e8d060;text-shadow:0 0 14px rgba(232,208,96,.3);line-height:1.1}
.gl{font-size:.7rem;color:#6a6230;text-transform:uppercase;letter-spacing:.18em;margin-top:6px}
.sb{background:linear-gradient(160deg,#141208,#0e0c06);border:1px solid #2a2510;border-radius:4px;padding:24px;margin-bottom:28px;position:relative}
.sb h2{font-family:'Orbitron',sans-serif;font-size:1rem;letter-spacing:.12em;color:#a09640;margin-bottom:16px;text-transform:uppercase}
.sf{display:flex;gap:8px}
.si{flex:1;font-family:'Share Tech Mono',monospace;font-size:.95rem;background:#0a0a08;border:1px solid #2a2510;border-radius:3px;color:#d4a829;padding:12px 16px;outline:none;transition:border-color .2s,box-shadow .2s}
.si:focus{border-color:#d4a829;box-shadow:0 0 12px rgba(212,168,41,.15)}
.si::placeholder{color:#6a6230;font-style:italic}
.sbtn{font-family:'Orbitron',sans-serif;font-size:.8rem;font-weight:700;letter-spacing:.12em;background:linear-gradient(180deg,#1e1a0c,#151206);border:1px solid #3a3214;border-radius:3px;color:#e8d060;padding:12px 28px;cursor:pointer;transition:all .2s;text-transform:uppercase}
.sbtn:hover{background:linear-gradient(180deg,#2a2410,#1e1a0a);border-color:#d4a829;box-shadow:0 0 16px rgba(232,208,96,.12)}
.summary-card{background:linear-gradient(160deg,#141208,#0f0d06);border:1px solid #2a2510;border-left:3px solid #40e040;border-radius:3px;padding:20px;margin-bottom:20px}
.summary-label{font-family:'Orbitron',sans-serif;font-size:.7rem;letter-spacing:.12em;color:#40e040;margin-bottom:10px;text-transform:uppercase}
.summary-response{font-size:.88rem;color:#c8b446;line-height:1.7;border-top:1px solid #1e1a0c;padding-top:12px;white-space:pre-wrap}
.results-box{margin-top:20px}
.results-title{font-family:'Orbitron',sans-serif;font-size:.85rem;letter-spacing:.1em;color:#a09640;margin-bottom:14px;text-transform:uppercase}
.results-list{display:flex;flex-direction:column;gap:12px}
.no-data{text-align:center;padding:32px 20px;color:#3a3618;font-style:italic;font-size:.85rem}
.no-data::before{content:'⌁ ';color:#2a2510}
.result-card{background:linear-gradient(160deg,#141208,#0f0d06);border:1px solid #2a2510;border-left:3px solid #d4a829;border-radius:3px;padding:18px}
.result-card{cursor:pointer;transition:border-left-color .2s,border-color .2s,background .2s,margin .3s ease}
.result-card:hover{border-left-color:#e8d060;border-color:#3a3518;background:linear-gradient(160deg,#1a1708,#12100a)}
.result-card.expanded{border-left-color:#40e040;border-color:#2a2510;background:linear-gradient(160deg,#120e06,#0e0c04);margin-bottom:12px}
.result-text{font-size:.82rem;color:#8a7e3a;line-height:1.65;max-height:120px;overflow-y:auto;white-space:pre-wrap;transition:max-height .35s ease}.result-card.expanded .result-text{max-height:none}

.video-expand-hint{font-family:'Orbitron',sans-serif;font-size:.6rem;letter-spacing:.1em;color:#d4a829;margin-top:8px;opacity:.7;transition:opacity .2s}
.result-card:hover .video-expand-hint{opacity:1;color:#f5d060;text-shadow:0 0 6px rgba(245,208,96,.3)}
.result-card.expanded .video-expand-hint{display:none}
.video-embed{display:none;margin-top:14px;pointer-events:none}
.result-card.expanded .video-embed{display:block;margin-top:14px;pointer-events:auto}
.embed-frame{width:100%;height:500px;position:relative;background:#0a0a08;border:1px solid #2a2510;border-radius:3px;overflow:visible}
.embed-frame iframe{position:absolute;inset:0;width:100%;height:100%;border:none;background:#0a0a08;min-height:370px}
.result-meta{display:flex;align-items:center;gap:12px;font-size:.7rem;margin-bottom:10px}
.sim-meter{display:inline-block;width:56px;height:4px;border-radius:2px;background:#1a1700;overflow:hidden;position:relative}
.sim-meter::after{content:'';position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:repeating-linear-gradient(90deg,#40e040,#40e040 2px,transparent 2px,transparent 4px)}
.sim-10::after,.sim-9::after,.sim-8::after,.sim-7::after{width:100%}.sim-6::after,.sim-5::after{width:80%}.sim-4::after,.sim-3::after{width:60%}.sim-2::after{width:40%}.sim-1::after{width:20%}.sim-0::after{width:0}
.sim-text{font-size:.65rem;color:#6a6230}
.result-title{font-size:.85rem;color:#d4a829;margin-bottom:4px}
.result-id{font-size:.6rem;color:#4a4420;margin-bottom:8px;font-family:'Share Tech Mono',monospace;word-break:break-all}

.result-text::-webkit-scrollbar{width:4px}.result-text::-webkit-scrollbar-track{background:#141208}.result-text::-webkit-scrollbar-thumb{background:#2a2510;border-radius:2px}
.result-links{margin-top:8px}
.result-links a{font-size:.65rem;color:#a09640}
.result-links a:hover{color:#d4a829}
.results{margin-bottom:4px}
.results h3{font-family:'Orbitron',sans-serif;font-size:.85rem;letter-spacing:.1em;color:#a09640;margin-bottom:14px;text-transform:uppercase}
.vs{margin-top:32px}
.vs h2{font-family:'Orbitron',sans-serif;font-size:1rem;letter-spacing:.12em;color:#a09640;margin-bottom:16px;text-transform:uppercase}
.tw{background:linear-gradient(160deg,#141208,#0e0c06);border:1px solid #2a2510;border-radius:4px;overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{font-family:'Orbitron',sans-serif;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:#5a5230;padding:14px 18px;text-align:left;border-bottom:1px solid #2a2510;background:#0e0c06}
tr{border-bottom:1px solid #1a1700;transition:background .15s}
tr:hover{background:#1a1708}
td{padding:12px 18px;color:#a09640}
.vt2{color:#c8b446}.mo{font-family:'Share Tech Mono',monospace;font-size:.8rem}.dm{color:#4a4420}.ac{white-space:nowrap}
.badge{font-family:'Share Tech Mono',monospace;font-size:.7rem;padding:3px 10px;border-radius:2px;letter-spacing:.06em;display:inline-block}
.bo{background:#0a1e0a;color:#40e040;border:1px solid #1a3e1a;text-shadow:0 0 6px rgba(64,224,64,.2)}
.bp{background:#1e1a06;color:#d4a829;border:1px solid #3a3214}
.bn{background:#1a1700;color:#7a7040;border:1px solid #2a2510}
.ba{background:#1e0a0a;color:#e04040;border:1px solid #3e1a1a}
.bu{background:#141208;color:#6a6230;border:1px solid #1a1700}
.yt{color:#d4a829;font-size:1.2rem;transition:all .15s;display:inline-block}
.yt:hover{color:#f5d060;text-shadow:0 0 10px rgba(245,208,96,.4);transform:scale(1.1)}
@keyframes pg{0%,100%{text-shadow:0 0 6px rgba(232,208,96,.2)}50%{text-shadow:0 0 18px rgba(232,208,96,.45)}}
.pulse{animation:pg 3s ease-in-out infinite}
.nd{text-align:center;padding:48px 20px;color:#3a3618;font-style:italic;font-size:.85rem}
.nd::before{content:'⌁ ';color:#2a2510}
.ft{text-align:center;padding:36px 20px;border-top:1px solid #1e1a0c;margin-top:48px;font-size:.68rem;color:#3a3618;letter-spacing:.2em;text-transform:uppercase}

/* ── nuke loading overlay ── */
.nuke-loader{position:fixed;inset:0;z-index:10001;display:none;justify-content:center;align-items:center;background:rgba(10,10,8,.75)}
.nuke-loader.active{display:flex}
.nuke-box{text-align:center}
.nuke-icon{font-size:4rem;color:#e8d060;animation:spin 1s linear infinite;display:block;text-shadow:0 0 30px rgba(232,208,96,.5)}
.nuke-text{font-family:'Orbitron',sans-serif;font-size:.85rem;letter-spacing:.15em;color:#d4a829;margin-top:14px;display:block;text-shadow:0 0 10px rgba(212,168,41,.3)}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}

@media(max-width:768px){body{font-size:14px}.h1g{font-size:1.5rem;letter-spacing:.08em}.gauges{grid-template-columns:repeat(2,1fr)}.sf{flex-direction:column}.gv{font-size:1.8rem}td,th{padding:8px 10px}}
</style>
</head>
<body>
<div class="page-wrap">
<div class="header">
<h1 class="h1g pulse">⌈ GAMMA PDA ⌋</h1>
<div class="sub">YouTube RAG Knowledge Base — Stalker Gamma</div>
</div>
<div class="hs"></div>
<div class="gauges">
<div class="gauge"><svg width="72" height="72" viewBox="0 0 52 52"><defs><linearGradient id="g2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#40e040"/><stop offset="70%" stop-color="#d4a829"/><stop offset="100%" stop-color="#e04040"/></linearGradient></defs><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="url(#g2)" stroke-width="3.5" stroke-linecap="round" stroke-dasharray="138.23" stroke-dashoffset="{{total_pct}}" stroke-linecap="round" transform="rotate(-90 26 26)"/></svg><div class="gv">{{total}}</div><div class="gl">Videos Scanned</div></div>
<div class="gauge"><svg width="72" height="72" viewBox="0 0 52 52"><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="#40e040" stroke-width="3.5" stroke-dasharray="138.23" stroke-dashoffset="{{done_pct}}" transform="rotate(-90 26 26)"/></svg><div class="gv" style="color:#40e040">{{completed}}</div><div class="gl">Fully Processed</div></div>
<div class="gauge"><svg width="72" height="72" viewBox="0 0 52 52"><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="#d4a829" stroke-width="3.5" stroke-dasharray="138.23" stroke-dashoffset="{{in_prog_pct}}" transform="rotate(-90 26 26)"/></svg><div class="gv">{{in_progress}}</div><div class="gl">In Progress</div></div>
<div class="gauge"><svg width="72" height="72" viewBox="0 0 52 52"><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="#d4a829" stroke-width="3.5" stroke-dasharray="138.23" stroke-dashoffset="0" transform="rotate(-90 26 26)"/></svg><div class="gv" style="color:#d4a829">{{total_chunks}}</div><div class="gl">Chunks Embedded</div></div>
<div class="gauge"><svg width="72" height="72" viewBox="0 0 52 52"><circle cx="26" cy="26" r="22" fill="none" stroke="#1a1700" stroke-width="3.5"/><circle cx="26" cy="26" r="22" fill="none" stroke="#e04040" stroke-width="3.5" stroke-dasharray="138.23" stroke-dashoffset="138.23" transform="rotate(-90 26 26)"/></svg><div class="gv" style="color:#e04040">{{failed}}</div><div class="gl">Anomalies</div></div>
</div>
<div class="sb">
<h2>⌁ Query the Knowledge Base</h2>
<form class="sf" method="post" action="/search" id="search-form">
<input class="si" type="text" name="query" id="search-input" placeholder="type your question here stalker" autocomplete="off">
<button class="sbtn" type="submit">⌈ SCAN ▰</button>
</form>
</div>
{{summary_html}}
{{r_if}}
<div class="results-box">
<h2 class="results-title">▦ Results ({{rc}} matches)</h2>
<div class="results-list">{{r}}</div>
</div>
{{/r}}
<div class="vs">
<h2>▦ Video Archive</h2>
<div class="tw"><table><thead><tr><th>Title</th><th>Status</th><th>Chunks</th><th>Duration</th><th>Uploaded</th><th>▶</th></tr></thead><tbody>{{v}}</tbody></table></div>
</div>
<div class="ft">◈ GAMMA PDA — Powered by Stalker Gamma YouTube RAG v1.0 ◈</div>
</div>

<!-- nuke loading overlay -->
<div class="nuke-loader" id="nuke-loader">
<div class="nuke-box">
<div class="nuke-icon">☢</div>
<div class="nuke-text">⌁ SCANNING KNOWLEDGE BASE ▰</div>
</div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
    var input = document.getElementById('search-input');
    var form = document.getElementById('search-form');
    var loader = document.getElementById('nuke-loader');
    if (input) {
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (form) form.requestSubmit();
            }
        });
    }
    // ── Expandable result cards ──
    document.addEventListener('click', function(e) {
        var card = e.target.closest('.result-card');
        if (!card) return;
        
        // Collapse any other open cards first
        document.querySelectorAll('.result-card.expanded').forEach(function(other) {
            if (other !== card) {
                other.classList.remove('expanded');
                var iframe = other.querySelector('iframe');
                if (iframe && iframe.src && iframe.dataset.filled === 'true') {
                    setTimeout(function() { iframe.src = ''; iframe.dataset.filled = ''; }, 300);
                }
            }
        });
        
        card.classList.toggle('expanded');
        
        if (card.classList.contains('expanded')) {
            var vid = card.dataset.vid;
            var start = card.dataset.start || 0;
            var iframe = card.querySelector('iframe');
            if (vid && vid !== 'N/A' && iframe && !iframe.dataset.filled) {
                var src = 'https://www.youtube.com/embed/' + vid + '?autoplay=1&start=' + start + '&rel=0&origin=' + encodeURIComponent(window.location.origin);
                iframe.src = src;
                iframe.dataset.filled = 'true';
            }
        }
    });
    
    // Nuke loader on form submit
    var searchForm = document.getElementById('search-form');
    if (searchForm) {
        searchForm.addEventListener('submit', function() {
            if (loader) loader.classList.add('active');
        });
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
    print("="*60)
    print("  ⌈ GAMMA PDA — YouTube RAG Dashboard ⌋")
    print("="*60)
    print(f"  DB      : {get_db_path()}")
    print(f"  ChromaDB: {get_chroma_path()}")
    print(f"  Model   : {EMBED_MODEL_NAME}")
    print(f"  Server  : http://localhost:9876")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=9876, log_level="info")
