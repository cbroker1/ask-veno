"""Common evidence representation shared by both corpora.

Transcript chunks keep their legacy dictionary shape for backwards
compatibility ("chunks" in API responses); Evidence is the internal
common currency for fusion, reranking, routing diagnostics, and rendering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Evidence:
    id: str                      # stable source id (chunk id / facet id)
    corpus: str                  # "transcripts" | "gamma"
    source_type: str             # "youtube_transcript" | "gamma_facet" | "gamma_exact"
    title: str                   # video title / item name
    text: str
    score: float | None = None   # retrieval score (higher = better)
    url: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def from_chunk(chunk: dict) -> Evidence:
    """Wrap a legacy transcript chunk dict without altering it."""
    return Evidence(
        id=chunk.get("id", ""),
        corpus="transcripts",
        source_type="youtube_transcript",
        title=chunk.get("vt", "Unknown"),
        text=chunk.get("txt", ""),
        score=(chunk.get("sim_pct") or 0) / 100.0,
        url=chunk.get("url") if chunk.get("url") not in (None, "N/A") else None,
        metadata={
            "video_id": chunk.get("vid"),
            "start_hms": chunk.get("s"),
            "end_hms": chunk.get("e"),
            "distance": chunk.get("distance"),
            "sim_pct": chunk.get("sim_pct"),
            "chunk": chunk,          # lossless legacy mapping
        },
    )


def to_chunk(ev: Evidence) -> dict | None:
    """Recover the legacy chunk dict from transcript evidence."""
    if ev.corpus != "transcripts":
        return None
    return ev.metadata.get("chunk")


def from_gamma_facet(facet: dict, item: dict | None = None,
                     score: float | None = None,
                     source_type: str = "gamma_facet") -> Evidence:
    """Build evidence from a gamma_facets row (+ optional gamma_items row)."""
    item = item or {}
    return Evidence(
        id=facet["facet_id"],
        corpus="gamma",
        source_type=source_type,
        title=item.get("name") or facet.get("item_id", ""),
        text=facet.get("text", ""),
        score=score,
        url=None,   # no verified upstream per-item URL scheme (see docs)
        metadata={
            "item_id": facet.get("item_id"),
            "item_name": item.get("name"),
            "category": item.get("category"),
            "facet_type": facet.get("facet_type"),
            "game_version": "0.9.5",
            "source_files": facet.get("source_files"),
        },
    )
