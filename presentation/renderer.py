"""Server-side page construction: archive rows, evidence cards, echo bar,
field report, source list, and the full-page render context.

The renderer reads dashboard data from storage.archive and never touches
Chroma or Ollama — search results and the synth-online flag are passed in by
the routes.
"""

from __future__ import annotations

from app.config import get_db_path
from presentation.formatters import (
    _clean_summary,
    _clean_vid,
    _ring,
    _sig_class,
    esc,
    fmt_n,
    hms_to_seconds,
)
from presentation.template_engine import PAGE, tmpl
from storage.archive import get_video_stats, get_videos


def _rows(videos):
    if not videos:
        return ('<tr><td class="no-data" colspan="6">ARCHIVE EMPTY — no stream records ingested yet. '
                'Run the discovery pipeline, stalker.</td></tr>')
    h = ""
    for v in videos:
        badge = f'<span class="badge b-{esc(v["ingest_tt"])}">{esc(v["ingest_st"])}</span>'
        if v.get("url"):
            url = esc(v["url"])
            title_cell = f'<a class="row-link" href="{url}" target="_blank" rel="noopener">{esc(v["title"])}</a>'
            lnk = f'<a class="yt-link" href="{url}" target="_blank" rel="noopener" title="Open stream on YouTube">▶</a>'
        else:
            title_cell = esc(v["title"])
            lnk = '<span class="faint">—</span>'
        h += (f'<tr class="video-row" data-title="{esc((v["title"] or "").lower())}">'
              f'<td class="v-title">{title_cell}</td>'
              f'<td>{badge}</td>'
              f'<td class="mono num">{fmt_n(v["chunk_count"])}</td>'
              f'<td class="mono">{esc(v["dur"])}</td>'
              f'<td class="mono dim">{esc(v["dt"])}</td>'
              f'<td class="act">{lnk}</td></tr>')
    return h


_YT_ICON = ('<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
            '<path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0'
            '-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0'
            ' 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24'
            ' 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/></svg>')


def _results(chunks, searched: bool) -> str:
    """Stream-evidence section: one card per retrieved transcript segment."""
    if not searched:
        return ""
    if not chunks:
        return ('<div class="no-signal panel">⌁ NO SIGNAL — nothing in the archive matches this query.'
                '<span>Adjust the transmission: try weapon names, locations, mutants, quests, or game mechanics.</span></div>')
    h = ('<section class="evidence"><h2 class="sec-title">▦ STREAM EVIDENCE '
         f'<span class="sec-sub">{len(chunks)} MATCHED SEGMENT{"S" if len(chunks) != 1 else ""} — click a card to play the clip at its timestamp</span></h2>'
         '<div class="results-list">')
    for i, c in enumerate(chunks, 1):
        text = esc(c["txt"]).replace("\n", "<br>")
        sp = c.get("sim_pct") or 0
        vid = _clean_vid(c.get("vid"))
        seconds = hms_to_seconds(c.get("s"))
        yt = c.get("url")
        open_link = (f'<a class="ev-open" href="{esc(yt)}" target="_blank" rel="noopener" '
                     f'title="Open on YouTube at {esc(c["s"])}">OPEN ↗</a>') if yt and yt != "N/A" else ""
        h += f'''<article class="result-card" id="ev-{i}" data-vid="{esc(vid)}" data-start="{seconds}" tabindex="0">
<div class="ev-head"><span class="ev-idx">[{i:02d}]</span><span class="sig {_sig_class(sp)}" style="--w:{min(sp, 100)}%"><span></span></span><span class="sig-text">{sp}% SIGNAL</span><span class="sep">│</span><span class="mono ev-time">{esc(c["s"])} → {esc(c["e"])}</span>{open_link}</div>
<div class="result-title">{esc(c["vt"])}</div>
<div class="result-id">REC {esc(vid)}</div>
<div class="result-text">{text}</div>
<div class="video-expand-hint">{_YT_ICON} PLAY SEGMENT AT {esc(c["s"])}</div>
<div class="video-embed"><div class="embed-frame"><iframe title="Stream segment player" allow="autoplay; encrypted-media" allowfullscreen loading="lazy"></iframe></div></div>
</article>'''
    h += "</div></section>"
    return h


def _echo_html(query: str, chunks, scan_s: float | None) -> str:
    stats_bits = []
    if chunks:
        stats_bits.append(f"{len(chunks)} SEGMENTS MATCHED")
        stats_bits.append(f"TOP SIGNAL {chunks[0]['sim_pct']}%")
    else:
        stats_bits.append("0 SEGMENTS MATCHED")
    if scan_s is not None:
        stats_bits.append(f"SCAN {scan_s:.1f}s")
    meta = " · ".join(stats_bits)
    return (f'<div class="echo-bar"><div class="echo-q">» QUERY: <span class="echo-text">“{esc(query)}”</span></div>'
            f'<div class="echo-meta">{meta}</div>'
            f'<a class="echo-new" href="/">◄ NEW SCAN</a></div>')


def _report_html(chunks, summary: str, error: str) -> str:
    if error:
        return ('<div class="fault-card panel"><div class="fault-label">✚ SCAN FAULT</div>'
                f'<div class="fault-text">Archive query failed: {esc(error)}</div>'
                '<div class="fault-hint">Check that ChromaDB exists and the collection is built, then retry.</div></div>')
    if summary:
        cleaned = _clean_summary(summary)
        if cleaned:
            n = len(chunks)
            return ('<div class="summary-card panel"><div class="summary-head">'
                    '<span class="summary-label">⌁ FIELD REPORT</span>'
                    '<span class="summary-src">SYNTH: qwen3:0.6b · LOCAL</span></div>'
                    f'<div class="summary-response">{cleaned}</div>'
                    f'<div class="summary-note">Synthesized from {n} matched segment{"s" if n != 1 else ""} by a local LLM — verify against the stream evidence below:</div>'
                    f'{_sources_html(chunks)}'
                    '</div>')
    if chunks:
        return ('<div class="offline-card panel"><span class="offline-dot"></span>'
                'SYNTH MODULE OFFLINE — local LLM unavailable or returned nothing. Raw stream evidence below.</div>')
    return ""


def _sources_html(chunks) -> str:
    """Single-column source list embedded in the field report card."""
    if not chunks:
        return ""
    h = '<div class="sources-col">'
    for i, c in enumerate(chunks, 1):
        title = c["vt"] or "Unknown"
        h += (f'<button type="button" class="src-chip" data-target="ev-{i}" '
              f'title="Jump to evidence [{i:02d}] — {esc(title)}">'
              f'<span class="src-n">{i:02d}</span> {esc(c["s"])} · {esc(title)}</button>')
    h += "</div>"
    return h


def render_page(query: str = "", chunks=None, summary: str = "", error: str = "",
                scan_s: float | None = None, searched: bool = False,
                synth_online: bool = False) -> str:
    """Single render path for both the dashboard and search-result views."""
    chunks = chunks or []
    stats = get_video_stats()
    videos = get_videos(1000)
    # display order: most recent stream first (upload_date is YYYYMMDD; blanks last,
    # discovered_at order preserved as tiebreak by the stable sort)
    videos.sort(key=lambda v: v.get("upload_date") or "", reverse=True)
    total = stats["total"]

    ctx = dict(
        # topbar / status
        led_db_cls="on" if get_db_path().exists() else "off",
        led_db_txt="ARCHIVE ONLINE" if get_db_path().exists() else "ARCHIVE MISSING",
        led_synth_cls="on" if synth_online else "off",
        led_synth_txt="SYNTH ONLINE" if synth_online else "SYNTH OFFLINE",
        # telemetry
        total=fmt_n(total),
        completed=fmt_n(stats["completed"]),
        in_progress=fmt_n(stats["in_progress"]),
        failed=fmt_n(stats["failed"]),
        chunks_fmt=fmt_n(stats["total_chunks"]),
        ring_done=_ring(stats["completed"], total),
        ring_prog=_ring(stats["in_progress"], total),
        ring_fail=_ring(stats["failed"], total),
        # search state
        q_attr=esc(query),
        echo_html=_echo_html(query, chunks, scan_s) if searched else "",
        report_html=_report_html(chunks, summary, error) if searched else "",
        results_html=_results(chunks, searched),
        # archive
        video_rows=_rows(videos),
        archive_shown=fmt_n(len(videos)),
    )
    return tmpl(PAGE, **ctx)
