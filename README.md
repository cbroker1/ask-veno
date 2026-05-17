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

### ✦ S.T.A.L.K.E.R. Gamma PDA Theme

The entire interface is themed after the iconic PDA device from S.T.A.L.K.E.R. 2: Hope. Every element — from the phosphor glow to the amber terminal fonts to the scanline CRT effect — was designed to feel like a real piece of Stalker tech.

**Features:**

- Dark amber/gold phosphor monitor aesthetic with CRT scanlines
- Circular progress indicators for pipeline stats
- Interactive result cards with hover effects
- Click-to-expand video playback with timestamp deep-links
- Spinning nuclear loader during LLM generation
- Custom "Gamma Analysis" answer panel that appears between search and results

---

### ✦ Screenshots

#### Home Dashboard
The main dashboard showing ingestion stats, search interface, and the full video archive table — all in the Gamma PDA theme.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                                 │
│                                              ⌈ GAMMA PDA ⌋                                                      │
│                                  YOUTUBE RAG KNOWLEDGE BASE — STALKER GAMMA                                     │
│                                                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                 │
│   [● gradient ring]              [● green ring]              [○ empty]              [● amber ring]              │
│      257                          257                         0                       21959                     │
│   VIDEOS SCANNED            FULLY PROCESSED                  IN PROGRESS          CHUNKS EMBEDDED             │
│                                                                                                                 │
│                                                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                 │
│                                         ⌁ QUERY THE KNOWLEDGE BASE                                              │
│                                                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │ type your question here stalker                                       ⌈ SCAN ▰                    │  │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                 │
│                                        ▦ VIDEO ARCHIVE                                                          │
│                                                                                                                 │
│   ┌──────────────────────────────────────┬──────────┬────────┬──────────┬──────────────┬───┐                 │
│   │ TITLE                                │ STATUS   │ CHUNKS │ DURATION │ UPLOADED     │ ▶ │                 │
│   ├──────────────────────────────────────┼──────────┼────────┼──────────┼──────────────┼───┤                 │
│   │ NEW UPDATE TOMORROW - GAMMA ONE...  │ COMPLETE │   93   │ 2h 12m   │ 2026-05-07   │ ▶ │                 │
│   │ I'M BACK AGAIN...Also ESPRESSO...  │ COMPLETE │   89   │ 2h 04m   │ 2026-05-06   │ ▶ │                 │
│   │ Trying Out NEW Texture Mod - G...  │ COMPLETE │   85   │ 2h 08m   │ 2026-05-02   │ ▶ │                 │
│   │ ARTIFACT SEARCH TIME - GAMMA 1...  │ COMPLETE │   30   │ 49m 55s  │ 2025-03-11   │ ▶ │                 │
│   │ Time for UPGRADES - GAMMA ONE...  │ COMPLETE │   82   │ 2h 27m   │ 2025-03-09   │ ▶ │                 │
│   │ Will we get EXPERT TOOLS? - G...  │ COMPLETE │   83   │ 2h 20m   │ 2025-03-09   │ ▶ │                 │
│   │ Let's Get EXPERT TOOLS - GAM...  │ COMPLETE │   74   │ 2h 16m   │ 2025-03-07   │ ▶ │                 │
│   │ Will Red Forest be my END? - G... │ COMPLETE │  103   │ 2h 55m   │ 2025-03-06   │ ▶ │                 │
│   └──────────────────────────────────────┴──────────┴────────┴──────────┴──────────────┴───┘                 │
│                                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### Search Results with Gamma Analysis
After submitting a query — the Gamma Analysis panel renders directly below the search bar with the LLM-generated summary, followed by source-anchored results with expandable video players.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ type your question here stalker                                      ⌈ SCAN ▰                     │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ ⌁ Gamma Analysis                                                                                                         │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ [Summary text from local Ollama model — direct answer to your question with inline source references]                  │ │
│ │                                                                                                                       │ │
│ │ Source: None found.                                                                                                   │ │
│ └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                             │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ ▦ Results (8 matches)                                                                                                       │
│ ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│ │ ⌁ 8 matches found                                                                                                           │ │
│ │                                                                                                                             │ │
│ │ ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ CLICK TO EXPAND VIDEO                                                                                                      │ │ │
│ │ │ [Video card with hover color shift, expand animation                                                                    │ │ │
│ │ │  timestamped deep-link into source video                                                                                │ │ │
│ │ │  embedded YouTube player appears on click                                                                               │ │ │
│ │ │  full bottom HUD/controls visible                                                                                       │ │ │
│ │ └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │ │
│ │                                                                                                                             │ │
│ │ ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ CLICK TO EXPAND VIDEO                                                                                                      │ │ │
│ │ │ [Second result card — hover effects, expand, inline player                                                              │ │ │
│ └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                                             │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

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

**Tech Stack:**
| Layer | Technology |
|-------|-----------|
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
- **Scanlines** — CSS `linear-gradient` on a `::after` overlay creates horizontal scanline effect
- **Progress rings** — Circular SVG/progress elements show pipeline completion at a glance
- **Interactive cards** — Hover triggers amber glow transition; click expands inline video player
- **Loading state** — Spinning nuclear ☢ loader during Ollama inference with fade-out on completion
- **Responsive layout** — Grid-based stat panels collapse gracefully on narrower viewports
- **No external fonts loaded at runtime** — Orbitron loaded via Google Fonts once, then system fallbacks

---

### ✦ Pipeline Stats (current run)

| Metric | Value |
|--------|-------|
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
