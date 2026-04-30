#!/usr/bin/env python3
"""
Embed cleaned transcript chunks into ChromaDB.

This script is based on the existing Chroma notebook workflow:

- Load transcript_clean.json
- Chunk by tokenizer token count
- Preserve overlap
- Create youtube_time_url at each chunk start time
- Embed with intfloat/multilingual-e5-large
- Store chunks in a persistent ChromaDB collection

Unlike the notebook, this script is SQLite-state driven:

  transcript_clean_ready + cleaned
    -> ChromaDB chunks
    -> embedding_status = embedded
    -> ingest_status = complete
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


console = Console()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"
DEFAULT_CHROMA_PATH = "data/chroma"


@dataclass
class Config:
    db_path: Path
    chroma_path: Path
    collection_name: str
    embed_model_name: str
    max_embed_videos: int
    chunk_max_tokens: int
    chunk_overlap: float
    embed_batch_size: int
    dry_run: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def build_youtube_url(base_url: str, start_seconds: float) -> str:
    joiner = "&" if "?" in base_url else "?"
    return f"{base_url}{joiner}t={int(start_seconds)}"


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv()

    max_embed_videos = args.max_embed_videos
    if max_embed_videos is None:
        max_embed_videos = int(os.getenv("MAX_EMBED_VIDEOS", "1"))

    return Config(
        db_path=Path(args.db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)),
        chroma_path=Path(args.chroma_path or os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)),
        collection_name=args.collection_name or os.getenv("CHROMA_COLLECTION", "youtube_chunks"),
        embed_model_name=args.embed_model_name or os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large"),
        max_embed_videos=max_embed_videos,
        chunk_max_tokens=int(args.chunk_max_tokens or os.getenv("CHUNK_MAX_TOKENS", "512")),
        chunk_overlap=float(args.chunk_overlap or os.getenv("CHUNK_OVERLAP", "0.25")),
        embed_batch_size=int(args.embed_batch_size or os.getenv("EMBED_BATCH_SIZE", "32")),
        dry_run=args.dry_run,
    )


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite DB not found at {db_path}. Run python scripts/init_db.py first."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_embed_ready_videos(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM videos
        WHERE ingest_status = 'transcript_clean_ready'
          AND clean_transcript_status = 'cleaned'
          AND clean_transcript_path IS NOT NULL
          AND embedding_status != 'embedded'
        ORDER BY last_success_at ASC, discovered_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return list(rows)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_video_metadata_path(clean_transcript_path: Path, video_id: str) -> Path | None:
    folder = clean_transcript_path.parent
    candidates = list(folder.glob(f"video_metadata_{video_id}.json"))
    if candidates:
        return candidates[0]

    candidates = list(folder.glob("video_metadata_*.json"))
    if candidates:
        return candidates[0]

    return None


def find_youtube_url_path(clean_transcript_path: Path, video_id: str) -> Path | None:
    folder = clean_transcript_path.parent
    candidates = list(folder.glob(f"youtube_url_{video_id}.txt"))
    if candidates:
        return candidates[0]

    candidates = list(folder.glob("youtube_url_*.txt"))
    if candidates:
        return candidates[0]

    return None


def get_youtube_url(row: sqlite3.Row, clean_transcript_path: Path) -> str:
    video_id = row["video_id"]

    metadata_path = find_video_metadata_path(clean_transcript_path, video_id)
    if metadata_path and metadata_path.exists():
        metadata = load_json(metadata_path)
        for key in ["url", "webpage_url"]:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    url_path = find_youtube_url_path(clean_transcript_path, video_id)
    if url_path and url_path.exists():
        value = url_path.read_text(encoding="utf-8").strip()
        if value:
            return value

    value = row["url"]
    if value:
        return str(value)

    return f"https://www.youtube.com/watch?v={video_id}"


def get_video_title(row: sqlite3.Row, clean_transcript_path: Path) -> str:
    video_id = row["video_id"]

    metadata_path = find_video_metadata_path(clean_transcript_path, video_id)
    if metadata_path and metadata_path.exists():
        metadata = load_json(metadata_path)
        value = metadata.get("title")
        if isinstance(value, str) and value.strip():
            return value.strip()

    value = row["title"]
    if value:
        return str(value)

    return clean_transcript_path.parent.name


def chunk_transcript(
    transcript: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int,
    overlap: float,
) -> list[dict[str, Any]]:
    valid_segments = [
        seg
        for seg in transcript
        if isinstance(seg, dict)
        and "text" in seg
        and "start" in seg
        and str(seg["text"]).strip()
    ]

    all_texts = [str(seg["text"]).strip() for seg in valid_segments]
    all_starts = [float(seg["start"]) for seg in valid_segments]
    all_ends = [float(seg.get("end", seg["start"])) for seg in valid_segments]

    chunks: list[dict[str, Any]] = []
    idx = 0

    while idx < len(all_texts):
        chunk_texts: list[str] = []
        chunk_starts: list[float] = []
        chunk_ends: list[float] = []
        token_count = 0
        i = idx

        while i < len(all_texts):
            text = all_texts[i]
            tokens = tokenizer.encode(text, add_special_tokens=False)

            if token_count + len(tokens) > max_tokens:
                break

            chunk_texts.append(text)
            chunk_starts.append(all_starts[i])
            chunk_ends.append(all_ends[i])
            token_count += len(tokens)
            i += 1

        if chunk_texts:
            chunks.append(
                {
                    "chunk_text": " ".join(chunk_texts),
                    "start_time": min(chunk_starts),
                    "end_time": max(chunk_ends),
                    "token_count": token_count,
                }
            )

        chunk_size = i - idx
        step = max(1, int(chunk_size * (1 - overlap))) if chunk_size > 0 else 1
        idx += step

    return chunks


def get_existing_ids(collection: Any) -> set[str]:
    try:
        existing = collection.get(include=[])
        if existing and "ids" in existing:
            return set(existing["ids"])
    except Exception:
        return set()
    return set()


def mark_embedded(
    conn: sqlite3.Connection,
    video_id: str,
    collection_name: str,
    chunk_count: int,
) -> None:
    now = utc_now()
    conn.execute(
        """
        UPDATE videos
        SET
            ingest_status = 'complete',
            embedding_status = 'embedded',
            chroma_collection = ?,
            chunk_count = ?,
            embedded_at = ?,
            completed_at = ?,
            last_success_at = ?,
            last_error_type = NULL,
            last_error_message = NULL
        WHERE video_id = ?
        """,
        (
            collection_name,
            chunk_count,
            now,
            now,
            now,
            video_id,
        ),
    )
    conn.commit()


def mark_failed(
    conn: sqlite3.Connection,
    video_id: str,
    error_message: str,
) -> None:
    conn.execute(
        """
        UPDATE videos
        SET
            ingest_status = 'failed_embedding',
            embedding_status = 'failed',
            attempt_count = attempt_count + 1,
            last_attempt_at = ?,
            last_error_type = 'chromadb_embedding',
            last_error_message = ?
        WHERE video_id = ?
        """,
        (
            utc_now(),
            error_message[:2000],
            video_id,
        ),
    )
    conn.commit()


def print_queue(rows: list[sqlite3.Row]) -> None:
    table = Table(title=f"Clean transcripts to embed: {len(rows)}")
    table.add_column("Video ID", no_wrap=True)
    table.add_column("Ingest")
    table.add_column("Clean")
    table.add_column("Embedding")
    table.add_column("Clean Transcript Path")

    for row in rows:
        table.add_row(
            row["video_id"],
            row["ingest_status"],
            row["clean_transcript_status"],
            row["embedding_status"],
            str(row["clean_transcript_path"])[:100],
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path")
    parser.add_argument("--chroma-path")
    parser.add_argument("--collection-name")
    parser.add_argument("--embed-model-name")
    parser.add_argument("--max-embed-videos", type=int)
    parser.add_argument("--chunk-max-tokens", type=int)
    parser.add_argument("--chunk-overlap", type=float)
    parser.add_argument("--embed-batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        config = load_config(parse_args())
        conn = connect_db(config.db_path)
    except Exception as exc:
        console.print(f"[red]Startup failed:[/red] {exc}")
        return 2

    console.print("[bold]ChromaDB queue processor config[/bold]")
    console.print(f"DB path: {config.db_path}")
    console.print(f"Chroma path: {config.chroma_path}")
    console.print(f"Collection: {config.collection_name}")
    console.print(f"Embedding model: {config.embed_model_name}")
    console.print(f"Max embed videos: {config.max_embed_videos}")
    console.print(f"Chunk max tokens: {config.chunk_max_tokens}")
    console.print(f"Chunk overlap: {config.chunk_overlap}")
    console.print(f"Embed batch size: {config.embed_batch_size}")
    console.print(f"Dry run: {config.dry_run}")
    console.print()

    rows = get_embed_ready_videos(conn, config.max_embed_videos)
    print_queue(rows)

    if not rows:
        console.print("[green]No transcript_clean_ready videos to embed.[/green]")
        return 0

    if config.dry_run:
        console.print("[yellow]Dry run enabled. No embeddings or Chroma writes performed.[/yellow]")
        return 0

    console.print("[bold]Loading tokenizer and embedding model...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(config.embed_model_name)
    embedding_model = SentenceTransformer(config.embed_model_name)
    console.print("[green]Embedding model loaded.[/green]")

    config.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chroma_path))
    collection = client.get_or_create_collection(name=config.collection_name)
    existing_ids = get_existing_ids(collection)

    console.print(f"Existing chunk IDs in collection: {len(existing_ids)}")

    for row in rows:
        video_id = row["video_id"]
        clean_path = Path(row["clean_transcript_path"])

        console.rule(f"Embedding {video_id}")

        try:
            if not clean_path.exists():
                raise FileNotFoundError(f"Clean transcript does not exist: {clean_path}")

            transcript = load_json(clean_path)
            if not isinstance(transcript, list):
                raise RuntimeError("Clean transcript JSON must be a list of segments.")

            chunks = chunk_transcript(
                transcript=transcript,
                tokenizer=tokenizer,
                max_tokens=config.chunk_max_tokens,
                overlap=config.chunk_overlap,
            )

            if not chunks:
                raise RuntimeError("No chunks produced from clean transcript.")

            youtube_url = get_youtube_url(row, clean_path)
            video_title = get_video_title(row, clean_path)
            source_folder = clean_path.parent.name

            documents = [chunk["chunk_text"] for chunk in chunks]

            metadatas = []
            ids = []

            for idx, chunk in enumerate(chunks):
                start_seconds = int(chunk["start_time"])
                end_seconds = int(chunk["end_time"])

                metadatas.append(
                    {
                        "youtube_time_url": build_youtube_url(youtube_url, chunk["start_time"]),
                        "youtube_url": youtube_url,
                        "video_title": video_title,
                        "video_id": video_id,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "start_hms": seconds_to_hms(chunk["start_time"]),
                        "end_hms": seconds_to_hms(chunk["end_time"]),
                        "source_folder": source_folder,
                        "source_type": "youtube_transcript",
                        "chunk_index": idx,
                        "token_count": int(chunk["token_count"]),
                    }
                )
                ids.append(f"{video_id}_chunk_{idx:05d}")

            new_documents: list[str] = []
            new_metadatas: list[dict[str, Any]] = []
            new_ids: list[str] = []

            skipped = 0
            for doc, meta, chunk_id in zip(documents, metadatas, ids):
                if chunk_id in existing_ids:
                    skipped += 1
                    continue
                new_documents.append(doc)
                new_metadatas.append(meta)
                new_ids.append(chunk_id)

            if new_ids:
                # E5-style embedding. Store raw document text, but embed passage-prefixed text.
                texts_for_embedding = [f"passage: {doc}" for doc in new_documents]

                embeddings = embedding_model.encode(
                    texts_for_embedding,
                    batch_size=config.embed_batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )

                collection.add(
                    documents=new_documents,
                    embeddings=embeddings.tolist(),
                    metadatas=new_metadatas,
                    ids=new_ids,
                )

                for chunk_id in new_ids:
                    existing_ids.add(chunk_id)

            mark_embedded(
                conn=conn,
                video_id=video_id,
                collection_name=config.collection_name,
                chunk_count=len(chunks),
            )

            console.print(f"[bold green]Embedded video:[/bold green] {video_title}")
            console.print(f"Chunks total: {len(chunks)}")
            console.print(f"Chunks uploaded: {len(new_ids)}")
            console.print(f"Chunks skipped existing: {skipped}")

        except Exception as exc:
            message = str(exc)
            mark_failed(conn, video_id, message)
            console.print(f"[red]Failed {video_id}:[/red] {message}")

    console.print("[bold green]ChromaDB processing complete.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
