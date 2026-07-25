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

# Verified upstream database URL (no per-item deep-link scheme exists upstream;
# see docs/gamma-095-data-model.md).
GAMMA_DB_URL = "https://stalker-gamma-db.com/db/gamma-0.9.5/inventory?lang=en"

CORPUS_MODES = [
    ("auto", "AUTO"),
    ("veno", "VENO"),
    ("gamma", "GAMMA 0.9.5"),
    ("combined", "COMBINED"),
]


def _corpus_selector(corpus: str) -> str:
    """Radio-chip corpus selector; native inputs keep keyboard behavior."""
    current = (corpus or "auto").lower()
    if current not in {m for m, _ in CORPUS_MODES}:
        current = "auto"
    h = ('<div class="corpus-row" role="radiogroup" aria-label="Knowledge source">'
         '<span class="chips-label">SOURCE ▸</span>')
    for value, label in CORPUS_MODES:
        checked = " checked" if value == current else ""
        h += (f'<label class="corpus-chip"><input type="radio" name="corpus" '
              f'value="{value}"{checked}><span>{label}</span></label>')
    h += '</div>'
    return h


def _route_badge(route: str) -> str:
    labels = {"transcripts": "VENO ARCHIVE", "gamma": "GAMMA 0.9.5 DB",
              "combined": "COMBINED"}
    if route not in labels:
        return ""
    return f'<span class="route-badge">ROUTE: {labels[route]}</span>'


def _gamma_results(gamma_evidence: list, combined: bool) -> str:
    """GAMMA database evidence cards, PDA-styled but visibly distinct from
    stream evidence."""
    if not gamma_evidence:
        return ""
    sub = ("authoritative game-data records — GAMMA 0.9.5"
           if not combined else
           "authoritative game-data records — distinct from stream commentary")
    h = ('<section class="evidence gamma-evidence"><h2 class="sec-title">▦ GAMMA DATABASE '
         f'<span class="sec-sub">{len(gamma_evidence)} RECORD{"S" if len(gamma_evidence) != 1 else ""} — {sub}</span></h2>'
         '<div class="results-list">')
    for i, e in enumerate(gamma_evidence, 1):
        md = e.get("metadata") or {}
        name = esc(e.get("title") or md.get("item_id") or "Unknown item")
        category = esc(md.get("category") or "")
        facet = esc(md.get("facet_type") or "")
        item_id = esc(md.get("item_id") or "")
        exact = e.get("source_type") == "gamma_exact"
        exact_badge = '<span class="g-exact">EXACT MATCH</span>' if exact else ""
        sim = md.get("sim_pct")
        sim_txt = f'<span class="sig-text">{sim}% SIGNAL</span><span class="sep">│</span>' if sim else ""
        text = esc(e.get("text") or "").replace("\n", "<br>")
        srcs = esc(str(md.get("source_files") or "")).replace("[", "").replace("]", "").replace('&quot;', "")
        h += f'''<article class="result-card gamma-card" id="gev-{i}" tabindex="0">
<div class="ev-head"><span class="ev-idx">[G{i:02d}]</span><span class="g-tag">GAMMA 0.9.5</span>{exact_badge}{sim_txt}<span class="g-facet">{facet.upper()}</span>
<a class="ev-open" href="{GAMMA_DB_URL}" target="_blank" rel="noopener" title="Open the GAMMA 0.9.5 database">DATABASE ↗</a></div>
<div class="result-title">{name}</div>
<div class="result-id">ITEM {item_id} · {category}</div>
<div class="result-text">{text}</div>
<div class="g-prov">SOURCE: gamma-0.9.5 · {srcs}</div>
</article>'''
    h += "</div></section>"
    return h


def _results(chunks, searched: bool, combined: bool = False,
             has_any_evidence: bool = False) -> str:
    """Stream-evidence section: one card per retrieved transcript segment."""
    if not searched:
        return ""
    if not chunks:
        if has_any_evidence:
            return ""   # gamma-only result: no empty-transcript fault card
        return ('<div class="no-signal panel">⌁ NO SIGNAL — nothing in the archive matches this query.'
                '<span>Adjust the transmission: try weapon names, locations, mutants, quests, or game mechanics.</span></div>')
    title = "▦ VENO ARCHIVE — STREAM EVIDENCE" if combined else "▦ STREAM EVIDENCE"
    h = (f'<section class="evidence"><h2 class="sec-title">{title} '
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


def _echo_html(query: str, chunks, scan_s: float | None, route: str = "",
               gamma_count: int = 0) -> str:
    stats_bits = []
    if chunks:
        stats_bits.append(f"{len(chunks)} SEGMENTS MATCHED")
        stats_bits.append(f"TOP SIGNAL {chunks[0]['sim_pct']}%")
    elif not gamma_count:
        stats_bits.append("0 SEGMENTS MATCHED")
    if gamma_count:
        stats_bits.append(f"{gamma_count} GAMMA RECORD{'S' if gamma_count != 1 else ''}")
    if scan_s is not None:
        stats_bits.append(f"SCAN {scan_s:.1f}s")
    meta = " · ".join(stats_bits)
    badge = _route_badge(route)
    return (f'<div class="echo-bar"><div class="echo-q">» QUERY: <span class="echo-text">“{esc(query)}”</span></div>'
            f'{badge}<div class="echo-meta">{meta}</div>'
            f'<a class="echo-new" href="/">◄ NEW SCAN</a></div>')


def _report_html(chunks, summary: str, error: str, gamma_count: int = 0) -> str:
    if error:
        return ('<div class="fault-card panel"><div class="fault-label">✚ SCAN FAULT</div>'
                f'<div class="fault-text">Archive query failed: {esc(error)}</div>'
                '<div class="fault-hint">Check that ChromaDB exists and the collection is built, then retry.</div></div>')
    if summary:
        cleaned = _clean_summary(summary)
        if cleaned:
            n = len(chunks)
            if gamma_count and n:
                src_note = (f"Synthesized from {gamma_count} GAMMA 0.9.5 database "
                            f"record{'s' if gamma_count != 1 else ''} and {n} stream "
                            f"segment{'s' if n != 1 else ''}")
            elif gamma_count:
                src_note = (f"Synthesized from {gamma_count} GAMMA 0.9.5 database "
                            f"record{'s' if gamma_count != 1 else ''}")
            else:
                src_note = f"Synthesized from {n} matched segment{'s' if n != 1 else ''}"
            return ('<div class="summary-card panel"><div class="summary-head">'
                    '<span class="summary-label">⌁ FIELD REPORT</span>'
                    '<span class="summary-src">SYNTH: qwen3:0.6b · LOCAL</span></div>'
                    f'<div class="summary-response">{cleaned}</div>'
                    f'<div class="summary-note">{src_note} by a local LLM — verify against the evidence below:</div>'
                    f'{_sources_html(chunks)}'
                    '</div>')
    if chunks or gamma_count:
        return ('<div class="offline-card panel"><span class="offline-dot"></span>'
                'SYNTH MODULE OFFLINE — local LLM unavailable or returned nothing. Raw evidence below.</div>')
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
                synth_online: bool = False, corpus: str = "auto",
                route: str = "", route_reason: str = "",
                evidence=None) -> str:
    """Single render path for both the dashboard and search-result views."""
    chunks = chunks or []
    evidence = evidence or []
    gamma_evidence = [e for e in evidence
                      if isinstance(e, dict) and e.get("corpus") == "gamma"]
    combined = route == "combined"
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
        corpus_selector=_corpus_selector(corpus),
        echo_html=_echo_html(query, chunks, scan_s, route,
                             len(gamma_evidence)) if searched else "",
        report_html=_report_html(chunks, summary, error,
                                 len(gamma_evidence)) if searched else "",
        gamma_results_html=_gamma_results(gamma_evidence, combined) if searched else "",
        results_html=_results(chunks, searched, combined,
                              has_any_evidence=bool(gamma_evidence)),
        # archive
        video_rows=_rows(videos),
        archive_shown=fmt_n(len(videos)),
    )
    return tmpl(PAGE, **ctx)
