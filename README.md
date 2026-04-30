# Hermes Self-Healing YouTube RAG Agent

A local-first YouTube ingestion and RAG pipeline designed to become a self-healing agent workflow with Hermes Agent.

This project started as a set of Jupyter notebooks and was refactored into a stateful, queue-driven Python pipeline.

## Current MVP

The current pipeline can:

1. Discover YouTube videos from a channel using title filters.
2. Store candidate videos in SQLite.
3. Download audio only with `yt-dlp`.
4. Transcribe audio with Whisper on CUDA.
5. Clean Whisper transcripts into simplified timestamped segments.
6. Chunk and embed transcripts into ChromaDB.
7. Query ChromaDB and send retrieved context to Ollama.
8. Run one pipeline pass with a single command.

## Pipeline

```text
YouTube channel
  → discover matching videos
  → SQLite registry
  → download queued audio
  → Whisper transcription
  → transcript cleanup
  → ChromaDB embedding
  → Ollama inference
```

## Main scripts

```text
scripts/init_db.py
scripts/migrate_db.py
scripts/discover_audio_candidates.py
scripts/process_audio_queue.py
scripts/process_whisper_queue.py
scripts/process_transcript_queue.py
scripts/process_chromadb_queue.py
scripts/query_chromadb.py
scripts/run_pipeline_once.py
```

## Safety notes

Never commit:

- `.env`
- cookies
- browser cookie exports
- Chrome profile files
- raw audio
- raw transcripts
- ChromaDB files
- SQLite state DB
- API keys
- auth tokens

Generated data is intentionally ignored by Git.

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate hermes-youtube-rag
```

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
YOUTUBE_CHANNEL_URL=https://www.youtube.com/@SomeChannel/streams
TITLE_FILTERS=ONE LIFE,1 LIFE
MAX_DISCOVERY_VIDEOS=50
MAX_NEW_VIDEOS=1
```

Initialize local state:

```bash
python scripts/init_db.py
python scripts/migrate_db.py
```

## Run one pipeline pass

```bash
python scripts/run_pipeline_once.py \
  --max-discovery-videos 25 \
  --max-new-videos 1 \
  --max-transcribe-videos 1 \
  --max-clean-videos 1 \
  --max-embed-videos 1
```

## Query the indexed collection

Retrieval only:

```bash
python scripts/query_chromadb.py "What is the miracle machine?" --no-ollama
```

With Ollama inference:

```bash
python scripts/query_chromadb.py "What is the miracle machine?"
```

Use a specific Ollama model:

```bash
python scripts/query_chromadb.py \
  "What is the miracle machine?" \
  --ollama-model llama3.2:latest
```

## Inspect pipeline state

```bash
sqlite3 .state/youtube_ingest.sqlite \
"select video_id, ingest_status, audio_status, whisper_status, clean_transcript_status, embedding_status, chunk_count, last_error_type from videos;"
```

## Self-healing agent direction

The deterministic pipeline now works. The next layer is Hermes Agent supervision.

Example future repair loop:

```text
If yt-dlp fails with auth/cookie errors:
  1. classify failure as failed_auth
  2. retry with --cookies-from-browser chrome
  3. if successful, save this as a reusable repair skill
  4. if not successful, ask for manual browser login/cookie refresh
```

Potential Hermes skills:

```text
skills/
  yt_dlp_auth_repair.md
  pipeline_status_triage.md
  failed_video_retry.md
  chromadb_health_check.md
```

## MVP status

Working end-to-end for at least one video:

```text
discover
→ audio download
→ Whisper transcription
→ transcript cleanup
→ ChromaDB embedding
→ retrieval
→ Ollama inference
```
