"""Deterministic natural-language facet generation for GAMMA 0.9.5 items.

Facets are the retrieval documents: compact, self-describing text blocks built
purely from canonical data (no LLM, no invented facts). Every facet names the
item, its internal id, and "GAMMA 0.9.5" so retrieved passages stay
self-identifying. Facet ids are stable: gamma095:<item_id>:<facet_type>.

Unit note: weight values are labeled kg because upstream's own upgrade
effects declare `st_prop_weight` deltas with unit "kg"; no other units are
attached to table stats (percent signs come through in the raw values).
"""

from __future__ import annotations

import json as _json
from collections import defaultdict
from dataclasses import dataclass

FACET_ID_PREFIX = "gamma095"
MAX_LIST = 14          # names listed before "… and N more"
MAX_UPGRADES = 24
MAX_STASHES = 8


@dataclass
class Facet:
    facet_id: str
    item_id: str
    facet_type: str
    text: str
    source_files: list[str]


def _fid(item_id: str, facet_type: str) -> str:
    return f"{FACET_ID_PREFIX}:{item_id}:{facet_type}"


def _clip(names: list[str], limit: int = MAX_LIST) -> str:
    names = [n for n in names if n]
    if len(names) > limit:
        return ", ".join(names[:limit]) + f", … and {len(names) - limit} more"
    return ", ".join(names)


def _prettify(prop: str) -> str:
    return prop.replace("st_data_export_", "").replace("st_prop_", "") \
               .replace("ui_inv_", "").replace("ui_", "").replace("_", " ").strip()


def generate_all(res, trader_labels: dict[str, str] | None = None,
                 repair_labels: dict[str, str] | None = None) -> list[Facet]:
    """Generate every facet for every canonical item in `res` (a BuildResult
    or any object with .items / .properties / .relations)."""
    trader_labels = trader_labels or {}
    repair_labels = repair_labels or {}

    name_of = {i.item_id: i.name for i in res.items}
    items_by_id = {i.item_id: i for i in res.items}

    props: dict[str, dict[str, tuple]] = defaultdict(dict)
    for item_id, prop, label, text, num, is_pct, unit, source in res.properties:
        props[item_id].setdefault(prop, (label, text, num, is_pct, unit, source))

    rels: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rels_to: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for src, rtype, dst_id, dst_text, detail_json, source in res.relations:
        detail = _json.loads(detail_json) if detail_json else {}
        row = (src, dst_id, dst_text, detail, source)
        rels[src][rtype].append(row)
        if dst_id:
            rels_to[dst_id][rtype].append(row)

    facets: list[Facet] = []
    for item in res.items:
        facets.extend(_item_facets(item, props[item.item_id],
                                   rels[item.item_id], rels_to[item.item_id],
                                   name_of, items_by_id, repair_labels))
    return facets


def _ref(item_id: str | None, dst_text: str | None, name_of: dict) -> str:
    if item_id and item_id in name_of:
        return f"{name_of[item_id]} ({item_id})"
    return dst_text or item_id or "?"


# Properties surfaced in the overview facet instead of the stats facet.
_OVERVIEW_PROPS = {"st_prop_weight", "st_upgr_cost", "ui_mm_repair"}
# Bookkeeping properties that never belong in retrieval text.
_HIDDEN_PROPS = {"hasDisassemble", "hasWeaponParts", "hasUpgrades", "shared",
                 "noDrop", "nimble", "tacticalKit", "st_ui_dbg_spawn_type",
                 "mag_capacity_provider"}


def _item_facets(item, props, rel, rel_to, name_of, items_by_id, repair_labels):
    out: list[Facet] = []
    head = f"{item.name} ({item.item_id}) — {item.category} — GAMMA 0.9.5"

    # ── overview ────────────────────────────────────────────────────────
    lines = [f"{head} database item."]
    if item.display_variant and item.display_variant != item.name:
        lines.append(f"Variant label: {item.display_variant}.")
    if item.description:
        lines.append(item.description)
    bits = []
    if "st_prop_weight" in props:
        bits.append(f"Weight: {props['st_prop_weight'][1]} kg")
    if item.price is not None:
        bits.append(f"Base price: {item.price}")
    if "ui_mm_repair" in props:
        letter = props["ui_mm_repair"][1]
        label = repair_labels.get(letter, letter)
        bits.append(f"Repair class: {letter}" + (f" ({label})" if label != letter else ""))
    if "caliber" in props:
        bits.append(f"Caliber: {props['caliber'][1]}")
    if bits:
        lines.append(". ".join(bits) + ".")
    obtain_bits = []
    if item.unobtainable:
        obtain_bits.append("not obtainable in normal play")
    if item.has_npc_drop:
        obtain_bits.append("drops from NPCs")
    if item.has_stash_drop:
        obtain_bits.append("found in stashes")
    if item.in_starting_loadout:
        obtain_bits.append("available in starting loadouts")
    if obtain_bits:
        lines.append("Obtainability: " + "; ".join(obtain_bits) + ".")
    out.append(Facet(_fid(item.item_id, "overview"), item.item_id, "overview",
                     "\n".join(lines), item.source_files))

    # ── stats ───────────────────────────────────────────────────────────
    stat_lines = []
    for prop, (label, text, num, is_pct, unit, source) in props.items():
        if prop in _OVERVIEW_PROPS or prop in _HIDDEN_PROPS or prop == "caliber":
            continue
        if num is None and not is_pct:
            # keep short text-valued stats (e.g. repair class handled above,
            # single-handed Y/N, engine class); skip long lists
            if len(text) > 40:
                continue
        shown = label or _prettify(prop)
        unit_sfx = f" {unit}" if unit else ""
        stat_lines.append(f"{shown}: {text}{unit_sfx}")
    if stat_lines:
        text = (f"{head} statistics.\n" + ". ".join(stat_lines) + ".")
        out.append(Facet(_fid(item.item_id, "stats"), item.item_id, "stats",
                         text, item.source_files))

    # ── compatibility ───────────────────────────────────────────────────
    comp_lines: list[str] = []
    comp_sources: set[str] = set()

    ammo_main = [r for r in rel.get("uses_ammo", []) if not r[3].get("alt")]
    ammo_alt = [r for r in rel.get("uses_ammo", []) if r[3].get("alt")]
    if ammo_main:
        comp_lines.append("Compatible ammunition: "
                          + _clip([_ref(r[1], r[2], name_of) for r in ammo_main]))
    if ammo_alt:
        comp_lines.append("Alternate ammunition: "
                          + _clip([_ref(r[1], r[2], name_of) for r in ammo_alt]))
    mags = rel.get("uses_magazine", [])
    if mags:
        comp_lines.append("Compatible magazines: "
                          + _clip([_ref(r[1], r[2], name_of) for r in mags]))
    addons = rel.get("compatible_addon", [])
    if addons:
        by_slot: dict[str, list[str]] = defaultdict(list)
        for r in addons:
            by_slot[r[3].get("slot", "addon")].append(_ref(r[1], r[2], name_of))
        for slot in ("scope", "silencer", "launcher", "kit"):
            if by_slot.get(slot):
                comp_lines.append(f"Compatible {slot}s: " + _clip(by_slot[slot]))
    for slot in ("silencer", "scope", "launcher"):
        p = props.get(f"{slot}_status")
        if p:
            comp_lines.append(f"{slot.capitalize()} mount: {p[1]}")
    used_by = rel.get("ammo_used_by", [])
    if used_by:
        comp_lines.append("Used by weapons: "
                          + _clip([_ref(r[1], r[2], name_of) for r in used_by]))
    addon_for = rel.get("addon_for", [])
    if addon_for:
        comp_lines.append("Fits weapons: "
                          + _clip([_ref(r[1], r[2], name_of) for r in addon_for]))
    mag_for = rel_to.get("uses_magazine", [])
    if mag_for:
        comp_lines.append("Magazine for: "
                          + _clip(sorted({_ref(r[0], None, name_of) for r in mag_for})))
    kits = rel.get("kit_converts_to", [])
    if kits:
        comp_lines.append("Kit converts weapon into: "
                          + _clip([_ref(r[1], r[2], name_of) for r in kits]))
    for size in ("small", "medium", "large"):
        p = props.get(f"mag_capacity_{size}")
        if p:
            comp_lines.append(f"Holds {size} magazines: {p[1]}")
    if comp_lines:
        for group in ("uses_ammo", "uses_magazine", "compatible_addon",
                      "ammo_used_by", "addon_for", "kit_converts_to"):
            comp_sources.update(r[4] for r in rel.get(group, []))
        comp_sources.update(r[4] for r in mag_for)
        text = f"{head} compatibility.\n" + "\n".join(comp_lines)
        out.append(Facet(_fid(item.item_id, "compatibility"), item.item_id,
                         "compatibility", text,
                         sorted(comp_sources) or item.source_files))

    # ── crafting ────────────────────────────────────────────────────────
    craft_lines = []
    craft_sources: set[str] = set()
    requires = rel.get("craft_requires", [])
    if requires:
        by_recipe: dict[tuple, list] = defaultdict(list)
        for r in requires:
            d = r[3]
            key = (d.get("tool_category") or "", d.get("tool_tier"), d.get("recipe") or "")
            by_recipe[key].append(r)
        for (tool, tier, recipe), rows in sorted(by_recipe.items()):
            ing = _clip([f"{_ref(r[1], r[2], name_of)} {r[3].get('amount', '')}".strip()
                         for r in rows])
            prefix = f"Craftable at {tool}" if tool else "Craftable"
            if tier is not None:
                prefix += f" (tool tier {tier})"
            if recipe:
                prefix += f" with recipe {recipe}"
            craft_lines.append(f"{prefix}. Ingredients: {ing}.")
            craft_sources.update(r[4] for r in rows)
    used_in = rel_to.get("craft_requires", [])
    if used_in:
        craft_lines.append("Used as crafting ingredient for: "
                           + _clip(sorted({_ref(r[0], None, name_of) for r in used_in})))
        craft_sources.update(r[4] for r in used_in)
    if craft_lines:
        text = f"{head} crafting.\n" + "\n".join(craft_lines)
        out.append(Facet(_fid(item.item_id, "crafting"), item.item_id, "crafting",
                         text, sorted(craft_sources)))

    # ── disassembly ─────────────────────────────────────────────────────
    dis_lines = []
    dis_sources: set[str] = set()
    dis = rel.get("disassembles_to", [])
    if dis:
        dis_lines.append("Disassembles into: "
                         + _clip([f"{_ref(r[1], r[2], name_of)} {r[3].get('amount', '')}".strip()
                                  for r in dis]))
        dis_sources.update(r[4] for r in dis)
    dis_from = rel_to.get("disassembles_to", [])
    if dis_from:
        dis_lines.append("Obtained by disassembling: "
                         + _clip(sorted({_ref(r[0], None, name_of) for r in dis_from})))
        dis_sources.update(r[4] for r in dis_from)
    if dis_lines:
        text = f"{head} disassembly.\n" + "\n".join(dis_lines)
        out.append(Facet(_fid(item.item_id, "disassembly"), item.item_id,
                         "disassembly", text, sorted(dis_sources)))

    # ── parts ───────────────────────────────────────────────────────────
    part_lines = []
    part_sources: set[str] = set()
    parts = rel.get("has_part", [])
    if parts:
        kind = parts[0][3].get("kind")
        label = {"weapon": "Weapon parts", "outfit": "Outfit parts"}.get(kind, "Parts")
        part_lines.append(f"{label} (condition system): "
                          + _clip([_ref(r[1], r[2], name_of) for r in parts]))
        part_sources.update(r[4] for r in parts)
    part_of = rel_to.get("has_part", [])
    if part_of:
        part_lines.append("Part of: "
                          + _clip(sorted({_ref(r[0], None, name_of) for r in part_of})))
        part_sources.update(r[4] for r in part_of)
    if part_lines:
        text = f"{head} parts.\n" + "\n".join(part_lines)
        out.append(Facet(_fid(item.item_id, "parts"), item.item_id, "parts",
                         text, sorted(part_sources)))

    # ── obtaining ───────────────────────────────────────────────────────
    obt_lines = []
    obt_sources: set[str] = set()
    sold = rel.get("sold_by", [])
    if sold:
        entries = []
        for r in sorted(sold, key=lambda r: (str(r[3].get("tier")), r[2] or "")):
            d = r[3]
            e = d.get("trader_label") or r[2]
            if d.get("tier") is not None:
                e += f" (supply tier {d['tier']}"
                e += ", conditional)" if d.get("cond") else ")"
            entries.append(e)
        obt_lines.append("Sold by: " + _clip(entries))
        obt_sources.update(r[4] for r in sold)
    dropped = rel.get("dropped_by_faction", [])
    if dropped:
        factions = []
        for r in dropped:
            ranks = r[3].get("ranks") or []
            f = r[2]
            if ranks:
                f += f" ({ranks[0]}+)" if len(ranks) > 1 else f" ({ranks[0]})"
            factions.append(f)
        obt_lines.append("Dropped by NPCs of factions: " + _clip(factions))
        obt_sources.update(r[4] for r in dropped)
    stashes = rel.get("stash_location", [])
    if stashes:
        top = sorted(stashes, key=lambda r: (-(r[3].get("chance") or 0), r[2] or ""))[:MAX_STASHES]
        entries = [f"{r[2]} ({r[3].get('chance')}% {r[3].get('rarity', '')})".replace("% )", "%)").strip()
                   for r in top]
        more = len(stashes) - len(top)
        line = "Found in stashes: " + ", ".join(entries)
        if more > 0:
            line += f", … and {more} more locations"
        obt_lines.append(line)
        obt_sources.update(r[4] for r in stashes)
    loadouts = rel.get("faction_loadout", [])
    if loadouts:
        entries = [f"{r[2]} ({r[3].get('cost')} pts)" for r in
                   sorted(loadouts, key=lambda r: r[2] or "")]
        obt_lines.append("Faction loadout availability: " + _clip(entries))
        obt_sources.update(r[4] for r in loadouts)
    starting = rel.get("starting_item", [])
    if starting:
        entries = sorted({r[2] for r in starting})
        obt_lines.append("Starting-loadout option for: " + _clip(entries))
        obt_sources.update(r[4] for r in starting)
    exchanges = rel.get("exchanges_to", [])
    if exchanges:
        entries = [f"{_ref(r[1], r[2], name_of)} ({r[3].get('target_faction')})"
                   for r in exchanges]
        obt_lines.append("Faction outfit exchange: " + _clip(entries))
        obt_sources.update(r[4] for r in exchanges)
    if obt_lines:
        if item.unobtainable:
            obt_lines.insert(0, "Flagged not obtainable in normal play.")
        text = f"{head} obtaining.\n" + "\n".join(obt_lines)
        out.append(Facet(_fid(item.item_id, "obtaining"), item.item_id,
                         "obtaining", text, sorted(obt_sources)))

    # ── upgrades ────────────────────────────────────────────────────────
    ups = rel.get("has_upgrade", [])
    if ups:
        entries = []
        for r in ups[:MAX_UPGRADES]:
            d = r[3]
            e = r[2] or "?"
            eff = d.get("prop_label")
            if eff and d.get("value") is not None:
                e += f" ({eff} {d['value']})"
            if d.get("cost") is not None:
                e += f" [cost {d['cost']}]"
            entries.append(e)
        more = len(ups) - min(len(ups), MAX_UPGRADES)
        text = (f"{head} upgrades.\n"
                f"{len(ups)} upgrades available: " + "; ".join(entries)
                + (f"; … and {more} more" if more > 0 else ""))
        out.append(Facet(_fid(item.item_id, "upgrades"), item.item_id, "upgrades",
                         text, sorted({r[4] for r in ups})))

    return out
