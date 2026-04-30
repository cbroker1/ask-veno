# Project Instructions: Hermes Self-Healing YouTube RAG Agent

This repo is for building a local-first, self-healing YouTube RAG ingestion workflow.

The long-term goal is to wrap the workflow with Hermes Agent so it can diagnose failures, learn fixes, and reuse those fixes as skills.

## Current Priority

Build the deterministic pipeline first.

Do not start with Hermes integration until the basic workflow works:

1. Discover videos from a channel.
2. Filter videos by title substring.
3. Save discovered video metadata into SQLite.
4. Process only videos that are new, queued, or explicitly retried.
5. Extract transcripts using yt-dlp.
6. Chunk transcripts.
7. Embed chunks into ChromaDB.
8. Write a run report.

## Preferred Stack

Use:

- Python
- Ubuntu/Linux-compatible scripts
- yt-dlp
- SQLite for workflow state
- ChromaDB for vector storage
- Chrome as the browser cookie source when cookies are needed
- local Ollama/Qwen as an optional advisory model
- small smoke tests before broad channel scans

Avoid:

- repeatedly scanning/downloading the full channel
- hammering YouTube with unnecessary calls
- hardcoding private local paths
- committing generated private data
- logging secrets or cookies

## Safety Rules

Never read, print, summarize, or commit:

- .env
- .env.*
- cookies.txt
- *.cookies
- browser profile files
- API keys
- auth tokens
- data/raw/
- data/private/
- data/chroma/
- .state/

Treat cookies as secrets.

Do not place cookies, tokens, local browser profile paths, private transcripts, or raw video/audio files in README files, logs, reports, examples, commits, or generated artifacts.

## Design Principle

The pipeline should expose clear machine-readable failure states so Hermes can later learn from them.

Good statuses include:

- discovered
- queued
- downloading
- transcript_ready
- embedded
- complete
- failed_auth
- failed_network
- failed_no_transcript
- failed_ytdlp
- skipped

## Local Qwen Helper

This repo will include a project-local Claude Code slash command:

```text
/qwen [question]
