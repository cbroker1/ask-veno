# ⌈ Ask Veno ⌋

## Stalker Gamma PDA — Local YouTube RAG Dashboard

A fully offline, single-file FastAPI dashboard that turns any YouTube channel into a searchable knowledge base — styled as a S.T.A.L.K.E.R. Gamma PDA.

**No cloud APIs. No paid services. Just ChromaDB + Ollama + FastAPI running on your own hardware.**

---

### ✦ What it does

```
YouTube channel
  → whisper transcriptions
  → chunked embeddings (intfloat/multilingual-e5-large)
  → ChromaDB vector store (21,959 chunks across 257 videos)
  → local Ollama query with qwen3.6:latest
  → direct answer + source-anchored results
```

Ask any question about the channel's content. The system retrieves the most relevant transcript chunks, generates a direct summary using a local LLM, and returns source-anchored video clips with timestamps so you can jump straight to the relevant moment.

---

### ✦ Screenshots

#### Home Dashboard

![Home Dashboard](https://raw.githubusercontent.com/cbroker1/ask-veno/main/docs/screenshots/hero-home-dashboard.png)

Full dashboard with the Stalker Gamma PDA aesthetic — amber phosphor glow, CRT scanlines, circular progress stats, interactive search bar, and the video archive table.

#### Search Results

![Search Results](https://raw.githubusercontent.com/cbroker1/ask-veno/main/docs/screenshots/search-results-page.png)

After submitting a query: the Gamma Analysis panel renders directly below the search bar with the LLM-generated summary, followed by source-anchored results with expandable video players.

---

### ✦ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│                 │     │              │     │              │     │              │
│  FastAPI UI     │────▶│ ChromaDB     │────▶│ multilingual- │────▶│  Ollama      │
│  (Stalker PDA  │     │ youtube_     │     │ e5-large     │     │  qwen3.6:    │
│   "Gamma"      │     │ chunks       │     │ embeddings   │     │  latest      │
│   Theme)       │     │ (257 videos, │     │              │     │              │
│                 │     │  21,959       │     │              │     │              │
└─────────────────┘     │  chunks)      │     └──────────────┘     └──────────────┘
                         └──────────────┘
```

| Layer | Technology |
|-------|------|
| Dashboard | FastAPI + Jinja (single file) |
| Vector DB | ChromaDB (persisted) |
| Embeddings | intfloat/multilingual-e5-large |
| LLM | ollama qwen3.6:latest / llama3.2:latest |
| Audio | yt-dlp + Whisper (CUDA) |
| State | SQLite registry |
| Deployment | conda env + uvicorn |

---

### ✦ Setup

```bash
# Clone & enter
git clone https://github.com/cbroker1/ask-veno.git
cd ask-veno

# Environment
conda env create -f environment.yml
conda activate veno-rag
pip install -r requirements.txt

# Ollama models needed
ollama pull qwen3.6:latest
ollama pull llama3.2:latest

# Start the dashboard
python -m uvicorn web_app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

---

### ✦ Quick Start

1. **Browse** — The video archive table shows all ingested videos with status, chunk count, duration, and upload date.
2. **Search** — Type any question in the search bar and hit Enter (or click SCAN).
3. **Read** — The Gamma Analysis panel above the results gives you a direct, locally-generated answer.
4. **Deep-link** — Click any result card to expand inline video playback at the exact timestamp where that snippet came from.

---

### ✦ Key UI Details

- **Phosphor glow** — Each text element has a subtle `text-shadow` glow mimicking CRT phosphor excitation
- **Scanlines** — CSS `linear-gradient` on a pseudo-element overlay creates horizontal scanline effect
- **Progress rings** — Circular SVG/progress elements show pipeline completion at a glance
- **Interactive cards** — Hover triggers amber glow transition; click expands inline video player
- **Loading state** — Spinning nuclear ☢ loader during Ollama inference with fade-out on completion
- **Responsive layout** — Grid-based stat panels collapse gracefully on narrower viewports
- **No external fonts loaded at runtime** — Orbitron loaded once, then system fallbacks

---

### ✦ Pipeline Stats (current run)

| Metric | Value |
|-------|--------|
| Videos ingested | 257 |
| Total chunks | 21,959 |
| Processing status | 100% complete |
| Anomalies | 0 |
| Embedding model | `intfloat/multilingual-e5-large` |
| Inference model | `qwen3.6:latest` (local Ollama) |

---

### ✦ License

Private / personal project. All rights reserved.

---

*Designed with the Gamma PDA — because the best RAG interface is one that feels like it survived the Zone.*
