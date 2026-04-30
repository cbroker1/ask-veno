# Hermes Self-Healing YouTube RAG Agent

A local-first YouTube ingestion and RAG pipeline designed to eventually run under Hermes Agent.

The goal is to demonstrate a self-healing agent workflow:

1. Discover videos from a YouTube channel.
2. Filter videos by title substring.
3. Persist discovered videos in SQLite.
4. Process only new, queued, or explicitly retried videos.
5. Safely handle common yt-dlp failures such as stale cookies, auth errors, missing subtitles, and transient network issues.
6. Extract transcripts.
7. Chunk and embed transcripts into ChromaDB.
8. Let Hermes Agent diagnose failures, learn repair procedures, and reuse them in future runs.

## MVP Scope

The first version focuses on building a deterministic ingestion pipeline before adding Hermes.

Core components:

- `yt-dlp` for YouTube discovery and transcript extraction
- SQLite for ingestion state
- ChromaDB for vector storage
- Python scripts for discovery, processing, and embedding
- Local Ollama/Qwen helper for architecture and code review
- Hermes Agent integration later

## Safety Notes

This repo should never contain:

- `.env`
- `cookies.txt`
- browser cookie exports
- Chrome profile files
- API keys
- tokens
- private transcripts
- raw downloaded media
- local vector database files

## Planned Workflow

```text
Channel URL
  → lightweight yt-dlp discovery
  → title substring filter
  → SQLite video registry
  → process only new/queued videos
  → transcript extraction
  → chunking
  → ChromaDB embedding
  → run report
  → Hermes learns repair procedures
