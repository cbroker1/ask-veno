"""Pure formatting and sanitization helpers used when rendering the page.

Everything here is deterministic and side-effect free.
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

RING_C = 138.23  # circumference of the 22px-radius telemetry rings


def esc(s: Any) -> str:
    """HTML-escape any value for safe interpolation into markup/attributes."""
    return _html.escape("" if s is None else str(s), quote=True)


def fmt_n(n: Any) -> str:
    try:
        return f"{int(n or 0):,}"
    except (TypeError, ValueError):
        return "0"


def hms_to_seconds(hms: str | None) -> int:
    """'1:53:14' → 6794. Returns 0 for missing/unparseable values."""
    if not hms or hms == "N/A":
        return 0
    try:
        parts = [int(p or 0) for p in str(hms).split(":")]
    except ValueError:
        return 0
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _dash_offset(count, total=257, circumference=138.23):
    """Compute SVG stroke-dashoffset for a gauge."""
    if total <= 0: return f"{circumference}"
    pct = count / total
    return f"{circumference * (1.0 - pct):.2f}"


def _ring(count, total) -> str:
    """SVG stroke-dashoffset for a telemetry ring filled count/total."""
    total = max(int(total or 0), 1)
    frac = min(max((count or 0) / total, 0.0), 1.0)
    return f"{RING_C * (1.0 - frac):.2f}"


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def _clean_summary(text: str) -> str:
    """Display-side cleanup of the local LLM output: drop stray think-tags,
    escape, and preserve line breaks."""
    text = _THINK_RE.sub("", text or "").strip()
    return esc(text).replace("\n", "<br>")


def _clean_vid(vid: str | None) -> str:
    vid = (vid or "").strip()
    if vid.startswith("youtube_id:"):
        vid = vid[len("youtube_id:"):]
    return vid or "N/A"


def _sig_class(sim_pct: float) -> str:
    if sim_pct >= 70:
        return "sig-hi"
    if sim_pct >= 60:
        return "sig-md"
    return "sig-lo"
