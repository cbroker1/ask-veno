#!/usr/bin/env python3
"""
query_chromadb.py -- Query the local ChromaDB YouTube transcript collection
and optionally send retrieved context to Ollama for an answer.

This script is based on the original inference notebook. It embeds a user
query, retrieves the top-k chunks from ChromaDB, filters out likely
sponsor/promo/outro segments, and optionally sends the remaining context
to a local Ollama instance for answer generation.

USAGE
-----
    python scripts/query_chromadb.py "your question here"

OPTIONS
    --chroma-path PATH          Path to the ChromaDB persistent directory
                                (default: data/chroma)
    --collection-name NAME      ChromaDB collection name (default: youtube_chunks)
    --embed-model-name NAME     Sentence-transformers model to use for
                                embedding (default: intfloat/multilingual-e5-large)
    --top-k N                   Number of chunks to retrieve (default: 5)
    --ollama-server HOST        Ollama server hostname (default: localhost)
    --ollama-model NAME         Ollama model name (default: qwen3:0.6b)
    --no-ollama                 Retrieve and print sources only; skip
                                Ollama answer generation
    --show-context              Print the full retrieved transcript excerpts

EXAMPLES
    # Ask a question and get an answer from Ollama
    python scripts/query_chromadb.py "What are the best late-game artifacts?"

    # Retrieve sources only (no Ollama)
    python scripts/query_chromadb.py --no-ollama "What is Venoxium?"

    # Show retrieved context alongside sources
    python scripts/query_chromadb.py --no-ollama --show-context "How does the pipeline work?"

    # Use a custom embedding model and top-k
    python scripts/query_chromadb.py --top-k 16 --embed-model-name "intfloat/multilingual-e5-large" "Question?"

WHAT IT DOES
------------
1. Embeds the query using the configured sentence-transformers model (with
   a "query: " prefix for E5-style models).
2. Queries ChromaDB for the top-k most similar chunks.
3. Filters out chunks likely to be sponsor/promo/outro content (scored by
   keywords like "subscribe", "sponsor", "patreon", "merch", etc.).
4. Displays a source reference table with video title, timestamp, and link.
5. If --show-context is set, prints the raw transcript excerpts.
6. If --no-ollama is not set, sends the filtered context to Ollama for
   answer generation and prints the result.

ENVIRONMENT VARIABLES
    CHROMA_PATH               Path to ChromaDB data (default: data/chroma)
    CHROMA_COLLECTION         Collection name (default: youtube_chunks)
    EMBED_MODEL_NAME          Embedding model name (default: intfloat/multilingual-e5-large)
    RETRIEVAL_TOP_K           Number of chunks to retrieve (default: 8)
    OLLAMA_SERVER             Ollama host (default: localhost)
    OLLAMA_MODEL              Ollama model (default: qwen3:0.6b)
    OLLAMA_TIMEOUT_SECONDS    Ollama request timeout in seconds (default: 300)
    QWEN_MODEL                Fallback Ollama model if OLLAMA_MODEL is unset

EXIT CODES
    0  Query completed successfully.
    1  An error occurred.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sentence_transformers import SentenceTransformer


console = Console()

DEFAULT_CHROMA_PATH = "data/chroma"


@dataclass
class Config:
    chroma_path: Path
    collection_name: str
    embed_model_name: str
    retrieval_top_k: int
    ollama_server: str
    ollama_model: str
    ollama_timeout_seconds: int
    no_ollama: bool
    show_context: bool


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv(dotenv_path=Path(".env"))

    return Config(
        chroma_path=Path(args.chroma_path or os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)),
        collection_name=args.collection_name or os.getenv("CHROMA_COLLECTION", "youtube_chunks"),
        embed_model_name=args.embed_model_name or os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large"),
        retrieval_top_k=args.top_k or int(os.getenv("RETRIEVAL_TOP_K", "8")),
        ollama_server=args.ollama_server or os.getenv("OLLAMA_SERVER", "localhost"),
        ollama_model=args.ollama_model or os.getenv("OLLAMA_MODEL", os.getenv("QWEN_MODEL", "qwen3:0.6b")),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")),
        no_ollama=args.no_ollama,
        show_context=args.show_context,
    )


def get_embedding(model: SentenceTransformer, text: str) -> list[float]:
    return model.encode([f"query: {text}"], convert_to_numpy=True)[0].tolist()


def retrieve_documents(
    collection: Any,
    embedding_model: SentenceTransformer,
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    query_embedding = get_embedding(embedding_model, query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    retrieved_chunks: list[dict[str, Any]] = []
    ids = results.get("ids", [[]])

    if not ids or not ids[0]:
        return retrieved_chunks

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        document = results["documents"][0][i]
        distance = results.get("distances", [[None]])[0][i]

        retrieved_chunks.append(
            {
                "id": results["ids"][0][i],
                "distance": distance,
                "source_folder": metadata.get("source_folder", "Unknown"),
                "chunk_text": document,
                "start_seconds": metadata.get("start_seconds", -1),
                "start_hms": metadata.get("start_hms", "Unknown"),
                "end_hms": metadata.get("end_hms", "Unknown"),
                "youtube_time_url": metadata.get("youtube_time_url", "N/A"),
                "video_title": metadata.get("video_title", "Unknown Title"),
                "video_id": metadata.get("video_id", "Unknown"),
            }
        )

    return retrieved_chunks


def filter_chunks(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []

    promo_terms = [
        "subscribe",
        "sponsor",
        "promo code",
        "patreon",
        "merch",
        "discord server",
        "link in description",
        "thanks for watching",
    ]

    for item in retrieved_chunks:
        text = item["chunk_text"].lower()
        score = sum(1 for term in promo_terms if term in text)

        if score < 2:
            filtered.append(item)

    console.print(
        f"Filtered {len(filtered)} / {len(retrieved_chunks)} chunks retained after scoring filter."
    )
    return filtered


def print_sources(chunks: list[dict[str, Any]]) -> None:
    table = Table(title="Retrieved Source References")
    table.add_column("#", justify="right")
    table.add_column("Distance")
    table.add_column("Video")
    table.add_column("Time")
    table.add_column("URL")

    for idx, item in enumerate(chunks, 1):
        distance = item.get("distance")
        distance_text = f"{distance:.4f}" if isinstance(distance, float) else "n/a"

        table.add_row(
            str(idx),
            distance_text,
            str(item["video_title"])[:70],
            str(item["start_hms"]),
            str(item["youtube_time_url"]),
        )

    console.print(table)


def build_context(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        [
            f"Video: {item['video_title']}\n"
            f"Time: {item['start_hms']}\n"
            f"Link: {item['youtube_time_url']}\n"
            f"Transcript:\n{item['chunk_text']}"
            for item in chunks
        ]
    )


def process_with_ollama(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    server: str,
    model_name: str,
    timeout_seconds: int,
) -> str:
    if not retrieved_chunks:
        return (
            "I don't have specific information about this from the indexed streams. "
            "The retrieved transcripts did not cover this topic."
        )

    context = build_context(retrieved_chunks)

    prompt = f"""You are answering questions about Venoxium's YouTube streams.

Use ONLY the transcript excerpts below to answer the question.
If the answer is not clearly supported by the excerpts, say so plainly.
Be specific and concise.
Do not include a sources section.
Do not list links.
Answer only with the final answer.

Question:
{query}

Transcript excerpts:
{context}
"""

    url = f"http://{server}:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(url, json=payload, timeout=timeout_seconds)
    response.raise_for_status()

    return response.json()["response"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*", help="Question to ask the local RAG system")
    parser.add_argument("--chroma-path")
    parser.add_argument("--collection-name")
    parser.add_argument("--embed-model-name")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--ollama-server")
    parser.add_argument("--ollama-model")
    parser.add_argument("--no-ollama", action="store_true", help="Retrieve and print sources only")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved transcript excerpts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args)

    query = " ".join(args.query).strip()
    if not query:
        query = "What's the best artifacts for late game?"

    console.print("[bold]RAG query config[/bold]")
    console.print(f"Chroma path: {config.chroma_path}")
    console.print(f"Collection: {config.collection_name}")
    console.print(f"Embedding model: {config.embed_model_name}")
    console.print(f"Top K: {config.retrieval_top_k}")
    console.print(f"Ollama server: {config.ollama_server}")
    console.print(f"Ollama model: {config.ollama_model}")
    console.print(f"No Ollama: {config.no_ollama}")
    console.print()

    console.print(f"[bold]Query:[/bold] {query}")

    console.print("[bold]Loading embedding model...[/bold]")
    embedding_model = SentenceTransformer(config.embed_model_name)

    console.print("[bold]Connecting to ChromaDB...[/bold]")
    chroma_client = chromadb.PersistentClient(path=str(config.chroma_path))
    collection = chroma_client.get_collection(config.collection_name)

    console.print(f"Collection count: {collection.count()}")

    retrieved_chunks = retrieve_documents(
        collection=collection,
        embedding_model=embedding_model,
        query=query,
        top_k=config.retrieval_top_k,
    )

    console.print(f"Retrieved {len(retrieved_chunks)} chunks")

    filtered_chunks = filter_chunks(retrieved_chunks)
    print_sources(filtered_chunks)

    if config.show_context:
        console.rule("Retrieved Context")
        console.print(build_context(filtered_chunks))

    if config.no_ollama:
        console.print("[yellow]--no-ollama enabled. Skipping generation.[/yellow]")
        return 0

    console.print("[bold]Sending context to Ollama...[/bold]")
    final_result = process_with_ollama(
        query=query,
        retrieved_chunks=filtered_chunks,
        server=config.ollama_server,
        model_name=config.ollama_model,
        timeout_seconds=config.ollama_timeout_seconds,
    )

    console.rule("Final Result from Ollama")
    console.print(final_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
