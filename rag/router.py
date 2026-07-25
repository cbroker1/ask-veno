"""Deterministic corpus routing for AUTO mode.

Priority (manual corpus selection is handled by the pipeline and always
wins before this module is consulted):

  1. Venoxium intent + GAMMA mechanic/stat keywords  -> combined
  2. Venoxium intent                                 -> transcripts
  3. recommendation phrasing + GAMMA signals         -> combined
  4. exact GAMMA item/alias resolution               -> gamma
  5. GAMMA mechanic/stat/category keywords           -> gamma
  6. safe default                                    -> transcripts

Exact item resolution consults the canonical SQLite DB when present and
degrades to keyword-only routing when it is not. No LLM involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from storage.gamma import GammaDB, GammaUnavailable

# Explicit references to the streamer / stream context.
VENO_TERMS = {
    "veno", "venoxium", "invictus", "stream", "streams", "streamer",
    "video", "videos", "episode", "playthrough", "ironman",
    "said", "say", "says", "think", "thinks", "thought", "opinion",
    "prefer", "prefers", "favorite", "favourite", "reacted", "reaction",
    "happened", "watch", "watched",
}
# Strong personal markers that only count together with a weaker one would be
# overkill here: the list above is deliberately explicit-only ("uses" alone is
# NOT a Veno signal — "what ammo does the AK-74 use" is a database question).

# Game-mechanic / item-data vocabulary (stats, systems, and item categories).
GAMMA_TERMS = {
    "stats", "stat", "weight", "ammo", "ammunition", "caliber", "calibre",
    "magazine", "magazines", "capacity", "damage", "penetration", "accuracy",
    "recoil", "handling", "reliability", "durability", "repair", "craft",
    "crafting", "recipe", "recipes", "disassemble", "disassembly", "upgrade",
    "upgrades", "attachment", "attachments", "scope", "scopes", "silencer",
    "silencers", "suppressor", "drop", "drops", "loadout", "loadouts",
    "item", "items", "price", "cost", "trader", "traders", "sell", "sells",
    "buy", "stash", "stashes", "protection", "artefact", "artefacts",
    "artifact", "artifacts", "rifle", "rifles", "pistol", "pistols",
    "shotgun", "shotguns", "smg", "smgs", "sniper", "snipers", "outfit",
    "outfits", "helmet", "helmets", "exoskeleton", "detector", "detectors",
    "medkit", "bandage", "rounds",
}
RECOMMEND_TERMS = {"best", "better", "worth", "should", "good", "recommend",
                   "recommended", "vs", "versus", "compare", "comparison"}

_WORD_RE = re.compile(r"[0-9a-zA-Z]+(?:[-.'][0-9a-zA-Z]+)*")


@dataclass
class RouteDecision:
    corpus: str                 # transcripts | gamma | combined
    reason: str
    veno_terms: list[str] = field(default_factory=list)
    gamma_terms: list[str] = field(default_factory=list)
    recommend_terms: list[str] = field(default_factory=list)
    exact_items: list[dict] = field(default_factory=list)


def _tokens(query: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(query or "")]


def _exact_items(query: str, db_path=None) -> list[dict]:
    try:
        db = GammaDB(db_path)
    except GammaUnavailable:
        return []
    try:
        from rag.gamma_retriever import resolve_exact_items
        return resolve_exact_items(db, query)
    finally:
        db.close()


def route(query: str, db_path=None) -> RouteDecision:
    toks = _tokens(query)
    # split hyphenated tokens too so "AK-74" contributes "ak" and "74"
    veno = sorted({t for t in toks if t in VENO_TERMS})
    gamma = sorted({t for t in toks if t in GAMMA_TERMS})
    recommend = sorted({t for t in toks if t in RECOMMEND_TERMS})

    if veno and gamma:
        exact = _exact_items(query, db_path)
        return RouteDecision("combined",
                             "Venoxium intent plus GAMMA mechanic keywords",
                             veno, gamma, recommend, exact)
    if veno:
        return RouteDecision("transcripts", "explicit Venoxium intent",
                             veno, gamma, recommend, [])

    exact = _exact_items(query, db_path)
    if recommend and (gamma or exact):
        return RouteDecision("combined",
                             "recommendation phrasing over game items — both corpora",
                             veno, gamma, recommend, exact)
    if exact:
        return RouteDecision("gamma", "query names a GAMMA item "
                             f"({exact[0]['name']})", veno, gamma, recommend, exact)
    if gamma:
        return RouteDecision("gamma", "GAMMA mechanic/stat keywords",
                             veno, gamma, recommend, [])
    return RouteDecision("transcripts", "no strong signals — default archive",
                         veno, gamma, recommend, [])
