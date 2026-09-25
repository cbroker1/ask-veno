"""Deterministic normalization helpers for the GAMMA 0.9.5 build.

Pure functions only — no I/O, no models. Normalization is intentionally
conservative: capitalization, whitespace, and simple punctuation variants are
folded, but caliber/model information (digits, dots, "x") is never destroyed.
"""

from __future__ import annotations

import hashlib
import json
import re

_WS_RE = re.compile(r"\s+")
# Characters removed for the "compact" matching form. Dots stay: they separate
# calibers (5.45x39 vs 545…), and removing them would create collisions.
_COMPACT_RE = re.compile(r"[\s\-_'\"]+")


def norm_name(s: str) -> str:
    """Lookup-normalized form: lowercase, trimmed, single-spaced, unquoted."""
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return _WS_RE.sub(" ", s.lower()).strip()


def compact_name(s: str) -> str:
    """Aggressive-but-safe matching form: normalized, then spaces, dashes,
    underscores and quotes removed ("AK-74N" -> "ak74n"). Dots preserved."""
    return _COMPACT_RE.sub("", norm_name(s))


def parse_stat(raw) -> tuple[str, float | None, bool]:
    """Parse an upstream display-formatted stat value.

    Returns (value_text, value_num, is_pct). Numbers are extracted only from
    values that are entirely numeric (optionally %-suffixed); anything else
    keeps just its raw text.
    """
    text = str(raw).strip()
    is_pct = text.endswith("%")
    body = text[:-1].strip() if is_pct else text
    num: float | None = None
    if re.fullmatch(r"-?\d+(\.\d+)?", body):
        num = float(body)
    elif isinstance(raw, bool):
        num = 1.0 if raw else 0.0
    elif isinstance(raw, (int, float)):
        num = float(raw)
    return text, num, is_pct


def parse_amount(raw) -> tuple[str, float | None]:
    """'x3' -> ('x3', 3.0); non-matching values keep text only."""
    text = str(raw).strip()
    m = re.fullmatch(r"[xX]\s*(\d+(\.\d+)?)", text)
    return text, float(m.group(1)) if m else None


def content_hash(obj) -> str:
    """Stable SHA-256 of any JSON-serializable object."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def clean_game_text(text: str | None) -> str | None:
    """Normalize X-Ray engine string markup in translated text.

    Upstream translation values carry LITERAL backslash-n sequences (the
    engine's line-break directive, preserved verbatim by the JSON export —
    1,559 EN strings at the inspected commit). Convert them to real newlines
    so facets, prompts, and the UI never show a raw "\\n". No other engine
    markup (e.g. %c color codes) exists in the 0.9.5 export.
    """
    if text is None:
        return None
    return text.replace("\\n", "\n")


class Translator:
    """Deterministic translation-key resolution with safe fallback.

    resolve() returns the English text when the key is translated, otherwise
    the raw key itself (matching upstream display behavior for the handful of
    untranslated keys). missing() reports which keys failed to resolve.
    """

    def __init__(self, en: dict[str, str], ru: dict[str, str] | None = None):
        self._en = en or {}
        self._ru = ru or {}
        self.missing_keys: set[str] = set()

    def resolve(self, key: str | None) -> str | None:
        if not key:
            return None
        text = self._en.get(key)
        if text is not None:
            return clean_game_text(text)
        self.missing_keys.add(key)
        return key

    def resolve_or_none(self, key: str | None) -> str | None:
        """Like resolve(), but returns None (and records the miss) when the
        key has no translation — used for descriptions, where echoing a raw
        key would read as garbage."""
        if not key:
            return None
        text = self._en.get(key)
        if text is None:
            self.missing_keys.add(key)
        return clean_game_text(text)

    def resolve_ru(self, key: str | None) -> str | None:
        if not key:
            return None
        return clean_game_text(self._ru.get(key))

    def has(self, key: str) -> bool:
        return key in self._en


def resolve_display_variant(display_name_key: str | None, tr: Translator) -> str | None:
    """index.json displayName may carry a variant suffix: 'st_wpn_ak74 [74]'.
    The key part is translated, the suffix kept verbatim."""
    if not display_name_key:
        return None
    m = re.fullmatch(r"(.*?)\s+(\[[^\]]+\])", display_name_key)
    if m:
        base = tr.resolve(m.group(1))
        return f"{base} {m.group(2)}"
    return tr.resolve(display_name_key)


def ammo_token_to_id(token: str, ammo_ids: set[str],
                     pda_to_id: dict[str, str]) -> str | None:
    """Map a ';'-list token from ui_ammo_types ('ammo-9x39-pab9') to an ammo
    item id. Verified resolution order: the ammo table's pda name, then the
    dash->underscore transform of the id. Returns None when unresolvable."""
    token = token.strip()
    if not token:
        return None
    if token in pda_to_id:
        return pda_to_id[token]
    candidate = token.replace("-", "_")
    if candidate in ammo_ids:
        return candidate
    return None
