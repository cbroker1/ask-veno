"""Custom placeholder template engine and page-template loading.

The full PDA page (HTML/CSS/JS) lives verbatim in templates/page.html and is
loaded once at import time, resolved relative to this module so the working
directory never matters.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


def tmpl(template: str, **ctx) -> str:
    """Simple template renderer with two features:
    1. {{key}} — replaced with ctx[key] (or "" if missing)
    2. {{key_if}} … {{/key}} — conditionally keep/discard based on ctx[key]
    """
    result = template

    # Step 1: handle conditional blocks first
    for key in list(ctx.keys()):
        if not key.endswith("_if"):
            continue
        val = ctx[key]
        open_tag = "{{" + key + "}}"
        close_tag = "{{/" + key[:-3] + "}}"
        # Find the block
        start = result.find(open_tag)
        if start < 0:
            continue
        end = result.find(close_tag, start + len(open_tag))
        if end < 0:
            continue
        inner = result[start + len(open_tag):end]
        if val:
            # Keep content (remove the markers)
            result = result[:start] + inner + result[end + len(close_tag):]
        else:
            # Discard content entirely
            result = result[:start] + result[end + len(close_tag):]

    # Step 2: simple variable replacement
    for key, value in ctx.items():
        if key.endswith("_if"):
            continue  # skip conditional keys
        placeholder = "{{" + key + "}}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))

    return result


PAGE = load_template("page.html")
SEARCH_PAGE = PAGE  # kept for parity with the original web_app.py aliases
