# GAMMA 0.9.5 Upstream Data Model

Discovery notes for the generated dataset consumed by Ask Veno's GAMMA corpus.
Everything below was verified by direct inspection of the synchronized files —
not assumed from documentation.

## Source

| | |
|---|---|
| Upstream project | [stalker-anomaly-gamma-db](https://github.com/simonwdev/stalker-anomaly-gamma-db) (AGPL-3.0) |
| Pack (hard pin) | `gamma-0.9.5` |
| Dataset path | `site/public/data/gamma-0.9.5/` in the upstream repo |
| Local checkout | sparse sibling clone at `../stalker-anomaly-gamma-db-upstream` (see `scripts/sync_gamma_095.py`) |
| Inspected commit | `109f8b3e8e675352f85e0d9c41776fa8c9a2a706` (2026-07-17 "More items; fix loadouts") |
| manifest.json SHA-256 | `c9426225998accb24e487260149d97168b9985ccfab0160bd63bbf7190e69373` |
| Manifest entries | 61 files |

`manifest.json` is a flat `{"<file>.json": "<8-hex content hash>"}` map. It is
the change-detection authority: a changed manifest hash triggers a rebuild,
an unchanged one skips it. The manifest does not list itself or the `traders/`
subdirectory (29 per-trader files).

## Master item list — `index.json`

Array of **1,786** entries: `{id, name, category, displayName, …flags}`.

- `id` — internal item id (`wpn_ak74`). **Unique** (0 duplicates at the
  inspected commit). This is our canonical `item_id`.
- `name` — translation key for the display name (`st_wpn_ak74`).
- `displayName` — same key, possibly suffixed for variant disambiguation:
  `"st_wpn_ak74 [74]"` vs `"st_wpn_ak74 [s74_new]"`.
- Optional flags: `hasNpcWeaponDrop`, `hasStashDrop`, `inStartingLoadout`,
  `unobtainable`, `tacticalKit`.
- 25 categories (`Rifles` 154, `Misc` 363, `Magazines` 214, `Outfits` 161, …).

Every category-table row id is present in `index.json` and vice versa
(1,786 ids across 26 table files; 62 ids appear in two table files —
artefacts also listed in `recipes.json`, materials shared with other
categories). An item's canonical category comes from `index.json`; extra
table rows contribute additional properties.

## Category tables (26 files)

`ammo, artefacts, belt-attachments, explosives, food, grenade-launchers,
helmets, launchers, magazines, materials, medicine, melee, misc,
mutant-parts, outfit-parts, outfits, pistols, recipes, rifles, scopes,
shotguns, silencers, smgs, snipers, tactical-kits, weapon-parts`

Shared shape: `{"category": "<label>", "headers": [...], "items": [...]}`.
Each item is an object keyed by header names **plus** extra fields not listed
in `headers` (`displayName`, `hasDisassemble`, `hasWeaponParts`,
`hasUpgrades`, `nimble`, `factions`, `shared`, `noDrop`, and for
`recipes.json` an embedded `ingredients` array).

Important header semantics (labels resolved from `translations.json`):

- `pda_encyclopedia_name` — name translation key.
- `st_data_export_description` — description translation key.
- Stats are mostly **strings**: `st_prop_weight: "1.80"`, `ui_inv_accuracy:
  "82%"`, `ui_inv_rate_of_fire: "700"`. Percent signs are meaningful; no
  explicit units are given for table stats (weights are kg by game
  convention, but the export does not say so — we preserve raw values and
  only attach units where upstream states them).
- Weapons: `ui_ammo_types` / `st_data_export_ammo_types_alt` are
  `;`-separated **dashed ammo name keys** (`ammo-9x39-pab9`), which map to
  ammo item ids by `-` → `_` on the id (`ammo_9x39_pab9`) or via the ammo
  table's own `pda_encyclopedia_name`. Verified: all 1,136 tokens across the
  6 weapon files resolve this way.
- `ui_mm_repair` — repair-class letter; `display-labels.json` maps letters to
  label keys (`L → app_repair_light`, …). A/B/C/D entries carry only colors.
- `base_mag_size` vs `ui_ammo_count`: base magazine vs shown mag size
  (`ui_ammo_count` label = "Mag. Size"; `base_mag_size` has no translation —
  kept as raw property).
- `misc.json` adds `st_class` / `st_ui_dbg_spawn_type` (engine class info).
- `materials.json` / `recipes.json` / `mutant-parts.json` are thin
  (id + name keys, plus per-file extras like trader-name keys).

## Translations — `translations.json`

`{"locales": ["en","ru","fr"], "en": {...5205}, "ru": {...4610}, "fr": {...91}}`.

- Key → text. Some keys are quoted literals that map to themselves
  (`'"tch_upgr"' → '"tch_upgr"'`).
- EN coverage of index name keys: 1,781/1,786. The 5 misses (`MDR`,
  `MDR_black`, `st_addon_scope_specterDR`) are already readable raw keys —
  deterministic fallback is the raw key itself.
- Descriptions: sampled 326 desc keys → 4 missing translations (fallback:
  no description).
- `fr` holds only ~91 UI labels, no item names — not used for aliases.
- `ru` holds full item names — used as `name_ru` aliases.

## Item ↔ item / world relationship files

| File | Shape | Meaning (verified) |
|---|---|---|
| `items-common.json` | id → `{name, price}` (6,473 ids) | prices + name keys for *all* engine items; superset of index. Only indexed ids are canonicalized; the rest are not part of the site's item list. |
| `ammo-weapons.json` | ammo id → `[{id, name, displayName, category, isAlt, noDrop?}]` | weapons chambering the ammo (`isAlt` = alternate ammo slot). |
| `weapon-magazines.json` | weapon id → `[mag ids]` (302 weapons) | magazine compatibility. |
| `weapon-addons.json` | weapon id → `{scopes, silencers, launchers, kits}` (addon ids) | attachable addons per slot. |
| `addon-weapons.json` | addon id → `[weapon ids]` | reverse of the above. |
| `weapon-addon-status.json` | weapon id → `{silencer, scope, launcher}` ∈ {0,1,2} | X-Ray addon status enum, verified in upstream `site/js/damage-calc.js`: **0 disabled, 1 permanent/integral, 2 attachable**. |
| `disassemble.json` | item id → `[{name: <part name key>, amount: "x1"}]` (168) | disassembly outputs; parts referenced by **translation key**, resolved back to items via name-key → item map. |
| `craft-recipes.json` | tool-category → `{label, items: [{id, toolTier, recipeReq, recipeReqName, ingredients: [{name, amount}]}]}` (9 categories) | workshop crafting; ingredients by name key. |
| `recipes.json` | category table + embedded `ingredients` | artefact-recipe outputs (subset of artefacts). |
| `drops.json` | item id → faction → `[rank keys]` (165) | NPC drops by faction and rank (`st_rank_experienced` → "Experienced"). |
| `item-drops.json` | item id → stash-type key → `[{map, chance, ecos}]` (549) | stash spawn locations with % chance (`st_data_export_stash_type_rare` → "Rare"). |
| `sold-by.json` | item id → `[{trader, tier, cond}]` (601) | trader availability (supply tier 1–5, goodwill condition string or null). |
| `traders-meta.json` | array of 29 `{id, label, labelKey, color, primaryOrigin}` | trader display names (label = translation key). |
| `item-loadouts.json` | item id → `[{faction, cost, selectable}]` (166) | faction starting-loadout availability and point cost. |
| `starting-loadouts.json` | `{points, factions: [{id, money, items}], shared, ammoPerWeapon, ammoCount}` | new-game loadout system. |
| `kit-weapons.json` | kit id → `[weapon ids]` (30) | tactical-kit weapon conversions. |
| `item-parts.json` | item id → `{kind, parts: [part ids]}` (548) | condition-system parts of weapons/outfits. |
| `item-part-defs.json` | part id → `{name, descr, cost, weight}` (120) | part definitions (name/descr are translation keys). |
| `calibers.json` | caliber → `{name, variants: [ammo rows]}` (68) | caliber grouping of ammo. |
| `mag-capacity.json` | item id → `{provider, small, medium, large}` (168) | how many small/medium/large mags a pouch/rig holds. |
| `outfit-exchange.json` | `{factions, exchanges: [{name, sourceFaction, results}]}` | faction-equivalent outfit exchange (by name key). |
| `categories.json` | array of 27 labels | site nav categories (25 item categories + Outfit Exchange + Starting Loadouts views). |
| `display-labels.json` | `{ui_mm_repair: {letter → {label, color}}}` | repair-class letter labels. |

## Files intentionally NOT ingested (and why)

| File | Reason |
|---|---|
| `map-entities.json` | world-map entity/anomaly placement for the site's map view; not item facts. |
| `mutant-profiles.json` (391) | mutant combat hit-zone profiles; not inventory items. |
| `npc-armor-profiles.json` (4,437) | NPC armor simulation profiles; not inventory items. |
| `adb-constants.json`, `adb-plate-mitigation.json`, `gbo-constants.json`, `pba-constants.json`, `ballistic-ranges.json` | ballistics-engine constants for the site's damage simulator; per-item stats already live in the category tables. |
| `toolkit-rates.json` | per-map toolkit spawn-rate matrix (map view); not per-item retrieval facts. |
| `starting-loadouts-drunks.json` | alternate "drunk start" game-mode variant of starting loadouts; base file covers the standard system. |
| `traders/*.json` (29 files) | full per-trader goodwill condition lists; `sold-by.json` is upstream's own item-level summary and is what we ingest. Not manifest-declared. |
| `manifest.json` | build metadata; stored in `gamma_meta`, not item data. |

Every skipped file is also reported (with reason) by the build audit.

## Ambiguities / data-quality notes

- Table stat values are display-formatted strings; `%` retained as a flag,
  numeric part parsed alongside the raw string. No unit fabrication: units
  are stored only where upstream states them (upgrade `effects[].unit` —
  observed set: `kg, pct, rpm, num, microg, healpct, speed, mlmin`).
- `upgrades.json` (547 items) entries: `{row, col, cell, prop, val, name,
  desc, section, cost, stats, effects}` — names/descs are translation keys;
  group keys like `up_gr_firsta_ak74` have no translation (sections are
  internal ids).
- 3 index entries have quoted-literal name keys (`'"tch_upgr"'`) that
  translate to themselves — displayed as-is, as upstream does.
- Some relationship files reference targets by translation key rather than
  item id (disassemble, craft ingredients, outfit-exchange). Name keys can be
  shared by several items (e.g. `st_wpn_ak74` covers `wpn_ak74` and
  `wpn_aks74_new`); resolution prefers the non-`unobtainable` indexed item
  and records unresolved/ambiguous references in the audit rather than
  guessing.
- The upstream site has **no verified per-item deep-link URL** (checked the
  whole `site/src` tree: only the damage simulator writes query params), so
  Ask Veno links GAMMA evidence to the general database page instead of
  inventing an item URL scheme.
