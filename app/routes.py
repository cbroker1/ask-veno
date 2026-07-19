"""FastAPI endpoints for the Ask Veno dashboard.

Handlers stay thin: validate the form input, call the RAG pipeline or the
archive, time the scan, and hand results to the renderer. Paths, form-field
names, defaults, response types, and error behavior are unchanged from the
original web_app.py.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from presentation.renderer import render_page
from rag import generator, pipeline
from storage.archive import get_video_stats, get_videos

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return HTMLResponse(content=render_page(synth_online=generator.OLLAMA_AVAILABLE))


@router.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    query = (query or "").strip()
    if not query:
        return HTMLResponse(content=render_page(synth_online=generator.OLLAMA_AVAILABLE))
    error_msg = ""
    t0 = time.perf_counter()
    try:
        search_res = pipeline.search(query, 5)
    except Exception as e:
        search_res = {"chunks": [], "summary": ""}
        error_msg = str(e)
        print(f"Search error: {e}")
    scan_s = time.perf_counter() - t0
    return HTMLResponse(content=render_page(
        query=query,
        chunks=search_res.get("chunks", []),
        summary=search_res.get("summary", ""),
        error=error_msg,
        scan_s=scan_s,
        searched=True,
        synth_online=generator.OLLAMA_AVAILABLE,
    ))


@router.get("/api/stats")
async def api_stats(): return get_video_stats()


@router.get("/api/videos")
async def api_videos(limit: int = 20): return get_videos(limit)


@router.post("/api/search")
async def api_search(query: str = Form(...), top_k: int = Form(5)):
    return pipeline.search(query, top_k=top_k)
