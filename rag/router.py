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
# The list is deliberately explicit-only ("uses" alone is NOT a Veno signal —
# "what ammo does the AK-74 use" is a database question).

# Mechanic/stat/system vocabulary: questions the database answers directly.
GAMMA_MECHANIC_TERMS = {
    "stats", "stat", "weight", "ammo", "ammunition", "cartridge",
    "cartridges", "caliber", "calibre", "magazine", "magazines", "capacity",
    "damage", "penetration", "accuracy", "recoil", "handling", "reliability",
    "durability", "repair", "craft", "crafting", "recipe", "recipes",
    "disassemble", "disassembly", "upgrade", "upgrades", "attachment",
    "attachments", "scope", "scopes", "silencer", "silencers", "suppressor",
    "drop", "drops", "loadout", "loadouts", "price", "cost", "trader",
    "traders", "sell", "sells", "buy", "stash", "stashes", "protection",
    "rounds", "gamma",
}
# Item-category nouns: game-data flavored, but common in stream talk too —
# they only pull toward GAMMA without an explicit Veno signal.
GAMMA_CATEGORY_TERMS = {
    "item", "items", "weapon", "weapons", "rifle", "rifles", "pistol",
    "pistols", "shotgun", "shotguns", "smg", "smgs", "sniper", "snipers",
    "outfit", "outfits", "helmet", "helmets", "exoskeleton", "detector",
    "detectors", "artefact", "artefacts", "artifact", "artifacts", "medkit",
    "bandage",
}
RECOMMEND_TERMS = {"best", "better", "worth", "should", "good", "recommend",
                   "recommended", "vs", "versus", "compare", "comparison"}

_WORD_RE = re.compile(r"[0-9a-zA-Z]+(?:[-.][0-9a-zA-Z]+)*")


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
    veno = sorted({t for t in toks if t in VENO_TERMS})
    mech = sorted({t for t in toks if t in GAMMA_MECHANIC_TERMS})
    cat = sorted({t for t in toks if t in GAMMA_CATEGORY_TERMS})
    recommend = sorted({t for t in toks if t in RECOMMEND_TERMS})
    gamma_terms = sorted(set(mech) | set(cat))

    if veno:
        if mech:
            return RouteDecision("combined",
                                 "Venoxium intent plus GAMMA mechanic keywords",
                                 veno, gamma_terms, recommend,
                                 _exact_items(query, db_path))
        if recommend and cat:
            return RouteDecision("combined",
                                 "Venoxium intent plus item recommendation",
                                 veno, gamma_terms, recommend,
                                 _exact_items(query, db_path))
        return RouteDecision("transcripts", "explicit Venoxium intent",
                             veno, gamma_terms, recommend, [])

    exact = _exact_items(query, db_path)
    if exact:
        if recommend:
            return RouteDecision("combined",
                                 "recommendation phrasing over a named item — both corpora",
                                 veno, gamma_terms, recommend, exact)
        return RouteDecision("gamma", "query names a GAMMA item "
                             f"({exact[0]['name']})", veno, gamma_terms,
                             recommend, exact)
    if mech:
        return RouteDecision("gamma", "GAMMA mechanic/stat keywords",
                             veno, gamma_terms, recommend, [])
    if recommend and cat:
        return RouteDecision("combined",
                             "recommendation over an item category — both corpora",
                             veno, gamma_terms, recommend, [])
    if cat:
        return RouteDecision("gamma", "GAMMA item-category keywords",
                             veno, gamma_terms, recommend, [])
    return RouteDecision("transcripts", "no strong signals — default archive",
                         veno, gamma_terms, recommend, [])
