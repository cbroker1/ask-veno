"""Deterministic canonicalization of the gamma-0.9.5 dataset.

Pure Python, no LLM, no network. Consumes a loaded GammaDataset and produces
canonical items, aliases, properties, relations, facets, and a completeness
audit; write_database() persists everything into a fresh SQLite file (the
caller stages + atomically activates it).
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.config import GAMMA_PACK_ID, GAMMA_UPSTREAM_URL
from indexing.gamma import facets as facets_mod
from indexing.gamma.loader import (
    CATEGORY_TABLE_FILES,
    RELATION_FILES,
    SKIPPED_FILES,
    GammaDataset,
)
from indexing.gamma.normalize import (
    Translator,
    ammo_token_to_id,
    compact_name,
    content_hash,
    norm_name,
    parse_amount,
    parse_stat,
    resolve_display_variant,
)
from storage.gamma import SCHEMA_VERSION, create_schema

# Minimum acceptable canonical coverage of index.json before the build fails.
MIN_COVERAGE = 0.95

# Table-row keys that are folded into the item record / relations instead of
# becoming generic properties.
_NON_PROPERTY_KEYS = {
    "id", "pda_encyclopedia_name", "st_data_export_description", "displayName",
    "ui_ammo_types", "st_data_export_ammo_types_alt", "ingredients",
    "hasNpcWeaponDrop", "hasStashDrop", "inStartingLoadout", "unobtainable",
}


class BuildError(Exception):
    """A canonicalization invariant failed; the build must not activate."""


@dataclass
class CanonicalItem:
    item_id: str
    name: str
    name_key: str | None
    display_variant: str | None
    category: str
    description: str | None
    price: int | None
    unobtainable: bool
    in_starting_loadout: bool
    has_npc_drop: bool
    has_stash_drop: bool
    source_files: list[str]
    raw: dict


@dataclass
class BuildResult:
    items: list[CanonicalItem] = field(default_factory=list)
    aliases: list[tuple] = field(default_factory=list)       # (item_id, alias, norm, compact, source)
    properties: list[tuple] = field(default_factory=list)    # (item_id, prop, label, text, num, is_pct, unit, source_file)
    relations: list[tuple] = field(default_factory=list)     # (src, rel, dst_id, dst_text, detail_json, source_file)
    facets: list[facets_mod.Facet] = field(default_factory=list)
    audit: dict = field(default_factory=dict)


class _RefResolver:
    """Resolve upstream references (item ids or translation name keys) to
    canonical item ids, deterministically, with ambiguity/miss accounting."""

    def __init__(self, ds: GammaDataset):
        self.index_ids = set(ds.index_by_id)
        unobtainable = {e["id"] for e in ds.index if e.get("unobtainable")}
        by_name_key: dict[str, list[str]] = defaultdict(list)
        for e in ds.index:
            if e.get("name"):
                by_name_key[e["name"]].append(e["id"])
        # items-common maps many more ids to name keys; use it as a fallback
        # bridge name_key -> indexed id when index.name misses the key.
        common = ds.file("items-common.json") or {}
        for iid, info in common.items():
            nk = isinstance(info, dict) and info.get("name")
            if nk and iid in self.index_ids:
                by_name_key[nk].append(iid)
        self.by_name_key = {
            k: sorted(set(v), key=lambda i: (i in unobtainable, len(i), i))
            for k, v in by_name_key.items()
        }
        self.ambiguous: Counter = Counter()
        self.unresolved: dict[str, Counter] = defaultdict(Counter)

    def note_unresolved(self, source_file: str, token: str) -> None:
        self.unresolved[source_file][token] += 1

    def resolve(self, token: str, source_file: str) -> str | None:
        if token in self.index_ids:
            return token
        ids = self.by_name_key.get(token)
        if ids:
            if len(ids) > 1:
                self.ambiguous[token] += 1
            return ids[0]
        self.note_unresolved(source_file, token)
        return None


def build(ds: GammaDataset) -> BuildResult:
    tr = Translator(ds.translations_en, ds.translations_ru)
    res = BuildResult()
    resolver = _RefResolver(ds)

    # id -> [(file, row)] over every category table
    table_rows: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for fname in CATEGORY_TABLE_FILES:
        table = ds.file(fname)
        if not table:
            continue
        for row in table["items"]:
            table_rows[row["id"]].append((fname, row))

    common = ds.file("items-common.json") or {}
    mag_capacity = ds.file("mag-capacity.json") or {}
    addon_status = ds.file("weapon-addon-status.json") or {}

    ammo_table = ds.file("ammo.json") or {"items": []}
    ammo_ids = {a["id"] for a in ammo_table["items"]}
    ammo_pda_to_id = {a["pda_encyclopedia_name"]: a["id"]
                      for a in ammo_table["items"] if a.get("pda_encyclopedia_name")}

    trader_labels = {t["id"]: tr.resolve(t.get("label")) or t["id"]
                     for t in (ds.file("traders-meta.json") or [])}
    repair_labels = {}
    dl = ds.file("display-labels.json") or {}
    for letter, info in (dl.get("ui_mm_repair") or {}).items():
        label = info.get("label", letter)
        repair_labels[letter] = tr.resolve(label) if label else letter

    # ── canonical items, aliases, properties ───────────────────────────
    def flag(entry: dict, rows: list[tuple[str, dict]], key: str) -> bool:
        if key in entry:
            return bool(entry[key])
        return any(bool(r.get(key)) for _, r in rows)

    for entry in ds.index:
        item_id = entry["id"]
        name_key = entry.get("name")
        rows = table_rows.get(item_id, [])
        # primary row: the table whose category matches the index category
        primary = next(((f, r) for f, r in rows
                        if (ds.file(f) or {}).get("category") == entry["category"]),
                       rows[0] if rows else None)

        name = tr.resolve(name_key) or item_id
        display_variant = resolve_display_variant(entry.get("displayName"), tr)
        description = None
        for f, r in ([primary] if primary else []) + [x for x in rows if x != primary]:
            dk = r.get("st_data_export_description")
            if dk:
                description = tr.resolve_or_none(dk)
                if description:
                    break

        info = common.get(item_id)
        price = info.get("price") if isinstance(info, dict) else None

        raw: dict = {"index": entry, "tables": {f: r for f, r in rows}}
        if isinstance(info, dict):
            raw["items-common.json"] = info
        if item_id in mag_capacity:
            raw["mag-capacity.json"] = mag_capacity[item_id]
        if item_id in addon_status:
            raw["weapon-addon-status.json"] = addon_status[item_id]

        source_files = sorted({"index.json", *(f for f, _ in rows),
                               *(k for k in raw if k.endswith(".json"))})

        item = CanonicalItem(
            item_id=item_id,
            name=name,
            name_key=name_key,
            display_variant=display_variant,
            category=entry["category"],
            description=description,
            price=price,
            unobtainable=flag(entry, rows, "unobtainable"),
            in_starting_loadout=flag(entry, rows, "inStartingLoadout"),
            has_npc_drop=flag(entry, rows, "hasNpcWeaponDrop"),
            has_stash_drop=flag(entry, rows, "hasStashDrop"),
            source_files=source_files,
            raw=raw,
        )
        res.items.append(item)

        # aliases
        def add_alias(alias: str | None, source: str):
            if not alias:
                return
            n, c = norm_name(alias), compact_name(alias)
            if n:
                res.aliases.append((item_id, alias, n, c, source))

        add_alias(name, "display_name")
        add_alias(item_id, "item_id")
        if name_key and name_key != name:
            add_alias(name_key, "name_key")
        if display_variant and display_variant != name:
            add_alias(display_variant, "display_variant")
        ru = tr.resolve_ru(name_key)
        if ru and ru != name:
            add_alias(ru, "name_ru")
        for f, r in rows:
            pda = r.get("pda_encyclopedia_name")
            if pda and pda != name_key:
                resolved = tr.resolve(pda)
                if resolved and resolved != name:
                    add_alias(resolved, "alt_name")

        # properties from table rows
        for f, r in rows:
            for key, val in r.items():
                if key in _NON_PROPERTY_KEYS or val is None or val == "":
                    continue
                if isinstance(val, list):
                    text = ";".join(str(v) for v in val)
                    res.properties.append((item_id, key, tr.resolve_or_none(key),
                                           text, None, 0, None, f))
                    continue
                text, num, is_pct = parse_stat(val)
                if isinstance(val, bool):
                    text = "true" if val else "false"
                label = tr.resolve_or_none(key)
                if key == "ui_mm_repair" and text in repair_labels:
                    label = "Repair class"
                res.properties.append((item_id, key, label, text, num, int(is_pct), None, f))

        if item_id in mag_capacity:
            mc = mag_capacity[item_id]
            for size in ("small", "medium", "large"):
                if size in mc:
                    res.properties.append((item_id, f"mag_capacity_{size}",
                                           f"{size.capitalize()} magazines held",
                                           str(mc[size]), float(mc[size]), 0, None,
                                           "mag-capacity.json"))
            if mc.get("provider"):
                res.properties.append((item_id, "mag_capacity_provider", "Capacity provider",
                                       str(mc["provider"]), None, 0, None, "mag-capacity.json"))
        if item_id in addon_status:
            # X-Ray addon status enum (verified upstream): 0 disabled,
            # 1 permanent/integral, 2 attachable.
            for slot, code in addon_status[item_id].items():
                res.properties.append((item_id, f"{slot}_status",
                                       f"{slot.capitalize()} mount",
                                       {0: "not supported", 1: "integral", 2: "attachable"}.get(code, str(code)),
                                       float(code), 0, None, "weapon-addon-status.json"))

    canonical_ids = {i.item_id for i in res.items}

    # ── relations ───────────────────────────────────────────────────────
    def rel(src: str, rtype: str, dst_id: str | None, dst_text: str | None,
            detail: dict | None, source: str):
        if src not in canonical_ids:
            resolver.note_unresolved(source, src)
            return
        if dst_id is not None and dst_id not in canonical_ids:
            dst_text = dst_text or dst_id
            dst_id = None
        res.relations.append((src, rtype, dst_id, dst_text,
                              json.dumps(detail, sort_keys=True) if detail else None,
                              source))

    # weapon tables: uses_ammo
    for fname in ("rifles.json", "pistols.json", "shotguns.json", "smgs.json",
                  "snipers.json", "launchers.json"):
        table = ds.file(fname)
        if not table:
            continue
        for row in table["items"]:
            for fld, alt in (("ui_ammo_types", False), ("st_data_export_ammo_types_alt", True)):
                for token in (row.get(fld) or "").split(";"):
                    token = token.strip()
                    if not token:
                        continue
                    ammo_id = ammo_token_to_id(token, ammo_ids, ammo_pda_to_id)
                    if ammo_id:
                        rel(row["id"], "uses_ammo", ammo_id, None, {"alt": alt}, fname)
                    else:
                        resolver.unresolved[fname].append(token)

    for ammo_id, weapons in sorted((ds.file("ammo-weapons.json") or {}).items()):
        for w in weapons:
            detail = {"alt": bool(w.get("isAlt"))}
            if w.get("noDrop"):
                detail["no_drop"] = True
            rel(ammo_id, "ammo_used_by", w.get("id"), None, detail, "ammo-weapons.json")

    for wid, mags in sorted((ds.file("weapon-magazines.json") or {}).items()):
        for mag in mags:
            rel(wid, "uses_magazine", mag, None, None, "weapon-magazines.json")

    for wid, slots in sorted((ds.file("weapon-addons.json") or {}).items()):
        for slot_key, slot in (("scopes", "scope"), ("silencers", "silencer"),
                               ("launchers", "launcher"), ("kits", "kit")):
            for addon in slots.get(slot_key) or []:
                dst = resolver.resolve(addon, "weapon-addons.json")
                rel(wid, "compatible_addon", dst, addon if not dst else None,
                    {"slot": slot}, "weapon-addons.json")

    for addon, weapons in sorted((ds.file("addon-weapons.json") or {}).items()):
        src = resolver.resolve(addon, "addon-weapons.json")
        if not src:
            continue
        for wid in weapons:
            rel(src, "addon_for", resolver.resolve(wid, "addon-weapons.json"),
                None if wid in canonical_ids else wid, None, "addon-weapons.json")

    for iid, spec in sorted((ds.file("item-parts.json") or {}).items()):
        src = resolver.resolve(iid, "item-parts.json")
        if not src:
            continue
        for part in spec.get("parts") or []:
            dst = resolver.resolve(part, "item-parts.json")
            rel(src, "has_part", dst, part if not dst else None,
                {"kind": spec.get("kind")}, "item-parts.json")

    for iid, outputs in sorted((ds.file("disassemble.json") or {}).items()):
        src = resolver.resolve(iid, "disassemble.json")
        if not src:
            continue
        for out in outputs:
            token = out.get("name") or ""
            dst = resolver.resolve(token, "disassemble.json")
            amount, amount_num = parse_amount(out.get("amount", ""))
            rel(src, "disassembles_to", dst,
                tr.resolve(token) if not dst else None,
                {"amount": amount, "amount_num": amount_num}, "disassemble.json")

    for cat_key, group in sorted((ds.file("craft-recipes.json") or {}).items()):
        tool_label = group.get("label") or cat_key
        for row in group.get("items") or []:
            src = resolver.resolve(row.get("id") or "", "craft-recipes.json")
            if not src:
                continue
            recipe_name = tr.resolve_or_none(row.get("recipeReqName")) or row.get("recipeReq")
            for ing in row.get("ingredients") or []:
                token = ing.get("name") or ""
                dst = resolver.resolve(token, "craft-recipes.json")
                amount, amount_num = parse_amount(ing.get("amount", ""))
                rel(src, "craft_requires", dst,
                    tr.resolve(token) if not dst else None,
                    {"amount": amount, "amount_num": amount_num,
                     "tool_category": tool_label, "tool_tier": row.get("toolTier"),
                     "recipe": recipe_name}, "craft-recipes.json")

    recipes_table = ds.file("recipes.json")
    if recipes_table:
        for row in recipes_table["items"]:
            src = resolver.resolve(row.get("id") or "", "recipes.json")
            if not src:
                continue
            for ing in row.get("ingredients") or []:
                token = ing.get("name") or ""
                dst = resolver.resolve(token, "recipes.json")
                amount, amount_num = parse_amount(ing.get("amount", ""))
                rel(src, "craft_requires", dst,
                    tr.resolve(token) if not dst else None,
                    {"amount": amount, "amount_num": amount_num,
                     "tool_category": "Artefact recipe"}, "recipes.json")

    for iid, entries in sorted((ds.file("sold-by.json") or {}).items()):
        src = resolver.resolve(iid, "sold-by.json")
        if not src:
            continue
        for e in entries:
            trader = e.get("trader") or ""
            rel(src, "sold_by", None, trader,
                {"tier": e.get("tier"), "cond": e.get("cond"),
                 "trader_label": trader_labels.get(trader, trader)}, "sold-by.json")

    for iid, factions in sorted((ds.file("drops.json") or {}).items()):
        src = resolver.resolve(iid, "drops.json")
        if not src:
            continue
        for faction, ranks in sorted(factions.items()):
            rel(src, "dropped_by_faction", None, faction,
                {"ranks": [tr.resolve(r) for r in ranks]}, "drops.json")

    for iid, stash_types in sorted((ds.file("item-drops.json") or {}).items()):
        src = resolver.resolve(iid, "item-drops.json")
        if not src:
            continue
        for stash_key, spots in sorted(stash_types.items()):
            rarity = tr.resolve(stash_key)
            for spot in spots:
                rel(src, "stash_location", None, spot.get("map"),
                    {"chance": spot.get("chance"), "rarity": rarity,
                     "ecos": spot.get("ecos")}, "item-drops.json")

    for iid, entries in sorted((ds.file("item-loadouts.json") or {}).items()):
        src = resolver.resolve(iid, "item-loadouts.json")
        if not src:
            continue
        for e in entries:
            rel(src, "faction_loadout", None, e.get("faction"),
                {"cost": e.get("cost"), "selectable": e.get("selectable")},
                "item-loadouts.json")

    sl = ds.file("starting-loadouts.json") or {}
    for fac in sl.get("factions") or []:
        for e in fac.get("items") or []:
            src = resolver.resolve(e.get("id") or "", "starting-loadouts.json")
            if src:
                rel(src, "starting_item", None, fac.get("id"),
                    {"quantity": e.get("quantity"), "cost": e.get("cost"),
                     "selectable": e.get("selectable"),
                     "difficulty_lock": e.get("difficultyLock")},
                    "starting-loadouts.json")
    for e in sl.get("shared") or []:
        src = resolver.resolve(e.get("id") or "", "starting-loadouts.json")
        if src:
            rel(src, "starting_item", None, "shared",
                {"quantity": e.get("quantity"), "cost": e.get("cost"),
                 "selectable": e.get("selectable"),
                 "difficulty_lock": e.get("difficultyLock")},
                "starting-loadouts.json")

    for kit, weapons in sorted((ds.file("kit-weapons.json") or {}).items()):
        src = resolver.resolve(kit, "kit-weapons.json")
        if not src:
            continue
        for wid in weapons:
            dst = resolver.resolve(wid, "kit-weapons.json")
            rel(src, "kit_converts_to", dst, wid if not dst else None,
                None, "kit-weapons.json")

    oe = ds.file("outfit-exchange.json") or {}
    for ex in oe.get("exchanges") or []:
        src = resolver.resolve(ex.get("name") or "", "outfit-exchange.json")
        if not src:
            continue
        for faction, target_key in sorted((ex.get("results") or {}).items()):
            dst = resolver.resolve(target_key, "outfit-exchange.json")
            rel(src, "exchanges_to", dst, target_key if not dst else None,
                {"source_faction": ex.get("sourceFaction"),
                 "target_faction": faction}, "outfit-exchange.json")

    for iid, nodes in sorted((ds.file("upgrades.json") or {}).items()):
        src = resolver.resolve(iid, "upgrades.json")
        if not src:
            continue
        for node in nodes:
            up_name = tr.resolve_or_none(node.get("name")) or node.get("name")
            rel(src, "has_upgrade", None, up_name,
                {"cost": node.get("cost"), "prop": node.get("prop"),
                 "prop_label": tr.resolve_or_none(node.get("prop")),
                 "value": node.get("val"), "section": node.get("section"),
                 "desc": tr.resolve_or_none(node.get("desc")),
                 "effects": node.get("effects"), "stats": node.get("stats")},
                "upgrades.json")

    # caliber grouping as an ammo property
    for caliber, spec in sorted((ds.file("calibers.json") or {}).items()):
        for variant in spec.get("variants") or []:
            vid = variant.get("id")
            if vid in canonical_ids:
                res.properties.append((vid, "caliber", "Caliber",
                                       spec.get("name") or caliber, None, 0, None,
                                       "calibers.json"))

    # ── facets ─────────────────────────────────────────────────────────
    res.facets = facets_mod.generate_all(res, trader_labels=trader_labels,
                                         repair_labels=repair_labels)

    # ── audit ──────────────────────────────────────────────────────────
    res.audit = _build_audit(ds, res, resolver, tr)
    return res


def _build_audit(ds: GammaDataset, res: BuildResult, resolver: _RefResolver,
                 tr: Translator) -> dict:
    rel_counts = Counter(r[1] for r in res.relations)
    facet_counts = Counter(f.facet_type for f in res.facets)
    cat_counts = Counter(i.category for i in res.items)
    index_ids = set(ds.index_by_id)
    canonical_ids = {i.item_id for i in res.items}
    parsed = sorted(set(CATEGORY_TABLE_FILES + RELATION_FILES) & set(ds.files))
    declared = set(ds.manifest)
    unknown = sorted(declared - set(parsed) - set(SKIPPED_FILES))

    audit = {
        "pack_id": GAMMA_PACK_ID,
        "manifest_sha256": ds.manifest_sha256,
        "index_item_count": len(ds.index),
        "canonical_item_count": len(res.items),
        "missing_index_items": sorted(index_ids - canonical_ids),
        "duplicate_item_ids": [],
        "alias_count": len(res.aliases),
        "property_count": len(res.properties),
        "relation_count": len(res.relations),
        "relations_by_type": dict(sorted(rel_counts.items())),
        "facet_count": len(res.facets),
        "facets_by_type": dict(sorted(facet_counts.items())),
        "categories": dict(sorted(cat_counts.items())),
        "manifest_file_count": len(ds.manifest),
        "files_parsed": parsed,
        "files_skipped": {k: v for k, v in sorted(SKIPPED_FILES.items()) if k in declared},
        "manifest_files_unaccounted": unknown,
        "extra_non_manifest_inputs": ["traders/ (29 per-trader files; summarized by sold-by.json)"],
        "missing_translation_count": len(tr.missing_keys),
        "missing_translation_samples": sorted(tr.missing_keys)[:25],
        "ambiguous_name_key_refs": dict(resolver.ambiguous.most_common(25)),
        "unresolved_references": {
            k: {"count": sum(v.values()), "distinct": len(v),
                "samples": [t for t, _ in v.most_common(10)]}
            for k, v in sorted(resolver.unresolved.items()) if v
        },
    }
    return audit


def check_audit(audit: dict) -> list[str]:
    """Hard-gate validation; returns list of fatal problems (empty = pass)."""
    problems = []
    if audit["canonical_item_count"] < audit["index_item_count"] * MIN_COVERAGE:
        problems.append(
            f"catastrophic coverage loss: {audit['canonical_item_count']}/"
            f"{audit['index_item_count']} items canonicalized")
    if audit["missing_index_items"]:
        problems.append(f"index items without canonical record: "
                        f"{audit['missing_index_items'][:10]}")
    if audit["duplicate_item_ids"]:
        problems.append(f"duplicate item ids: {audit['duplicate_item_ids'][:10]}")
    if audit["manifest_files_unaccounted"]:
        problems.append("manifest files neither parsed nor explicitly skipped: "
                        f"{audit['manifest_files_unaccounted']}")
    return problems


def write_database(db_path: Path, ds: GammaDataset, res: BuildResult,
                   upstream_commit: str, active_collection: str | None) -> None:
    """Create a fresh canonical database at db_path (must not exist)."""
    if Path(db_path).exists():
        raise BuildError(f"refusing to overwrite existing database: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        create_schema(conn)
        with conn:
            for it in res.items:
                conn.execute(
                    """INSERT INTO gamma_items
                       (item_id, name, name_key, display_variant, category,
                        description, price, unobtainable, in_starting_loadout,
                        has_npc_drop, has_stash_drop, name_norm, source_files,
                        raw_json, content_hash)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (it.item_id, it.name, it.name_key, it.display_variant,
                     it.category, it.description, it.price,
                     int(it.unobtainable), int(it.in_starting_loadout),
                     int(it.has_npc_drop), int(it.has_stash_drop),
                     norm_name(it.name), json.dumps(it.source_files),
                     json.dumps(it.raw, sort_keys=True, ensure_ascii=False),
                     content_hash(it.raw)))
            conn.executemany(
                "INSERT OR IGNORE INTO gamma_aliases"
                " (item_id, alias, alias_norm, alias_compact, source)"
                " VALUES (?,?,?,?,?)", res.aliases)
            conn.executemany(
                "INSERT OR IGNORE INTO gamma_properties"
                " (item_id, prop, label, value_text, value_num, is_pct, unit, source_file)"
                " VALUES (?,?,?,?,?,?,?,?)", res.properties)
            conn.executemany(
                "INSERT INTO gamma_relations"
                " (src_item_id, rel_type, dst_item_id, dst_text, detail, source_file)"
                " VALUES (?,?,?,?,?,?)", res.relations)
            for f in res.facets:
                conn.execute(
                    "INSERT INTO gamma_facets"
                    " (facet_id, item_id, facet_type, text, content_hash, source_files)"
                    " VALUES (?,?,?,?,?,?)",
                    (f.facet_id, f.item_id, f.facet_type, f.text,
                     content_hash(f.text), json.dumps(f.source_files)))
                conn.execute(
                    "INSERT INTO gamma_fts (text, facet_id, item_id) VALUES (?,?,?)",
                    (f.text, f.facet_id, f.item_id))
            meta = {
                "pack_id": GAMMA_PACK_ID,
                "upstream_repository": GAMMA_UPSTREAM_URL,
                "upstream_commit": upstream_commit,
                "manifest_sha256": ds.manifest_sha256,
                "manifest_json": json.dumps(ds.manifest, sort_keys=True),
                "schema_version": str(SCHEMA_VERSION),
                "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "item_count": str(len(res.items)),
                "facet_count": str(len(res.facets)),
                "active_chroma_collection": active_collection or "",
            }
            conn.executemany("INSERT INTO gamma_meta (key, value) VALUES (?,?)",
                             sorted(meta.items()))
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise BuildError(f"foreign-key violations after build: {violations[:5]}")
    finally:
        conn.close()
