"""Load and validate the upstream gamma-0.9.5 dataset from disk.

The loader is deterministic and read-only. It enforces the version pin (the
dataset directory must be the one named by GAMMA_PACK_ID), verifies the
manifest, and exposes the raw parsed files plus a few derived lookup maps.
Interpretation/joining happens in builder.py, not here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import GAMMA_DATA_SUBDIR, GAMMA_PACK_ID

# Category table files (all share the {category, headers, items} shape).
CATEGORY_TABLE_FILES = [
    "ammo.json", "artefacts.json", "belt-attachments.json", "explosives.json",
    "food.json", "grenade-launchers.json", "helmets.json", "launchers.json",
    "magazines.json", "materials.json", "medicine.json", "melee.json",
    "misc.json", "mutant-parts.json", "outfit-parts.json", "outfits.json",
    "pistols.json", "recipes.json", "rifles.json", "scopes.json",
    "shotguns.json", "silencers.json", "smgs.json", "snipers.json",
    "tactical-kits.json", "weapon-parts.json",
]

# Relationship / auxiliary files ingested by the builder.
RELATION_FILES = [
    "ammo-weapons.json", "addon-weapons.json", "weapon-addons.json",
    "weapon-addon-status.json", "weapon-magazines.json", "disassemble.json",
    "craft-recipes.json", "drops.json", "item-drops.json", "sold-by.json",
    "traders-meta.json", "item-loadouts.json", "starting-loadouts.json",
    "kit-weapons.json", "item-parts.json", "item-part-defs.json",
    "calibers.json", "mag-capacity.json", "outfit-exchange.json",
    "categories.json", "display-labels.json", "items-common.json",
    "translations.json", "index.json", "upgrades.json",
]

# Manifest-declared files intentionally not ingested, with the reason recorded
# in every build audit. See docs/gamma-095-data-model.md.
SKIPPED_FILES = {
    "map-entities.json": "world-map entity placement for the site's map view; not item facts",
    "mutant-profiles.json": "mutant combat hit-zone profiles; not inventory items",
    "npc-armor-profiles.json": "NPC armor simulation profiles; not inventory items",
    "adb-constants.json": "damage-simulator ballistics constants; per-item stats live in category tables",
    "adb-plate-mitigation.json": "damage-simulator ballistics constants; per-item stats live in category tables",
    "gbo-constants.json": "damage-simulator ballistics constants; per-item stats live in category tables",
    "pba-constants.json": "damage-simulator ballistics constants; per-item stats live in category tables",
    "ballistic-ranges.json": "damage-simulator ballistics constants; per-item stats live in category tables",
    "toolkit-rates.json": "per-map toolkit spawn-rate matrix (map view); not per-item facts",
    "starting-loadouts-drunks.json": "alternate 'drunk start' game-mode loadouts; base starting-loadouts.json covers the standard system",
}


class GammaLoadError(Exception):
    """Raised when the on-disk dataset fails validation."""


@dataclass
class GammaDataset:
    data_dir: Path
    manifest: dict[str, str]
    manifest_sha256: str
    files: dict[str, object] = field(default_factory=dict)

    # Derived maps (populated by load_dataset)
    index: list[dict] = field(default_factory=list)
    index_by_id: dict[str, dict] = field(default_factory=dict)
    translations_en: dict[str, str] = field(default_factory=dict)
    translations_ru: dict[str, str] = field(default_factory=dict)

    def file(self, name: str):
        return self.files.get(name)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GammaLoadError(f"Malformed JSON in {path.name}: {exc}") from exc


def load_dataset(repo_path: Path) -> GammaDataset:
    """Read the pinned dataset. Raises GammaLoadError on any validation
    failure — the caller must treat that as a failed build, never a partial
    one."""
    data_dir = Path(repo_path) / GAMMA_DATA_SUBDIR
    if data_dir.name != GAMMA_PACK_ID:
        raise GammaLoadError(
            f"Version pin violated: dataset dir {data_dir} is not {GAMMA_PACK_ID}")
    if not data_dir.is_dir():
        raise GammaLoadError(f"Dataset directory missing: {data_dir}")

    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise GammaLoadError(f"manifest.json missing in {data_dir}")
    manifest = _read_json(manifest_path)
    if not (isinstance(manifest, dict) and manifest
            and all(isinstance(k, str) and isinstance(v, str)
                    for k, v in manifest.items())):
        raise GammaLoadError(f"Malformed manifest.json in {data_dir}")

    ds = GammaDataset(
        data_dir=data_dir,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    wanted = [f for f in CATEGORY_TABLE_FILES + RELATION_FILES]
    missing = [f for f in wanted if f in manifest and not (data_dir / f).is_file()]
    if missing:
        raise GammaLoadError(f"Manifest-declared files missing on disk: {missing}")

    for name in sorted(set(wanted)):
        path = data_dir / name
        if path.is_file():
            ds.files[name] = _read_json(path)

    index = ds.files.get("index.json")
    if not isinstance(index, list) or not index:
        raise GammaLoadError("index.json missing, empty, or not an array")
    for entry in index:
        if not isinstance(entry, dict) or "id" not in entry or "category" not in entry:
            raise GammaLoadError(f"index.json entry lacks id/category: {entry!r:.120}")
    ds.index = index
    ds.index_by_id = {e["id"]: e for e in index}
    if len(ds.index_by_id) != len(index):
        seen: set[str] = set()
        dups = sorted({e["id"] for e in index if e["id"] in seen or seen.add(e["id"])})
        raise GammaLoadError(f"Duplicate item ids in index.json: {dups[:20]}")

    translations = ds.files.get("translations.json")
    if not isinstance(translations, dict) or "en" not in translations:
        raise GammaLoadError("translations.json missing an 'en' locale table")
    ds.translations_en = translations.get("en") or {}
    ds.translations_ru = translations.get("ru") or {}

    for name in CATEGORY_TABLE_FILES:
        table = ds.files.get(name)
        if table is None:
            continue  # not declared by this manifest — audit will report it
        if not (isinstance(table, dict)
                and isinstance(table.get("headers"), list)
                and isinstance(table.get("items"), list)):
            raise GammaLoadError(f"{name} is not a category table (headers/items)")

    return ds
