"""Shared fixture: a miniature gamma-0.9.5 dataset with the real upstream
shapes (verified against commit 109f8b3e), written to a temp directory in the
layout load_dataset() expects."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import GAMMA_DATA_SUBDIR


def fixture_files() -> dict[str, object]:
    rifles = {
        "category": "Rifles",
        "headers": ["id", "pda_encyclopedia_name", "st_data_export_description",
                    "st_prop_weight", "st_upgr_cost", "ui_mm_repair",
                    "ui_inv_accuracy", "ui_inv_rate_of_fire", "ui_ammo_count",
                    "base_mag_size", "ui_ammo_types", "st_data_export_ammo_types_alt"],
        "items": [{
            "id": "wpn_tak", "pda_encyclopedia_name": "st_wpn_tak",
            "st_data_export_description": "st_wpn_tak_descr",
            "st_prop_weight": "3.10", "st_upgr_cost": "26950",
            "ui_mm_repair": "D", "ui_inv_accuracy": "82%",
            "ui_inv_rate_of_fire": "650", "ui_ammo_count": "30",
            "base_mag_size": "30",
            "ui_ammo_types": "ammo-5.45x39-test",
            "st_data_export_ammo_types_alt": "ammo-5.45x39-alt",
            "displayName": "st_wpn_tak [74]",
            "hasNpcWeaponDrop": True, "hasStashDrop": True,
            "hasDisassemble": False, "hasWeaponParts": True,
            "hasUpgrades": True, "inStartingLoadout": False,
            "unobtainable": False, "tacticalKit": False, "nimble": False,
            "factions": ["wp"],
        }],
    }
    ammo = {
        "category": "Ammo",
        "headers": ["id", "pda_encyclopedia_name", "st_data_export_description",
                    "st_prop_weight", "ui_inv_damage", "st_data_export_ap"],
        "items": [
            {"id": "ammo_5.45x39_test", "pda_encyclopedia_name": "ammo-5.45x39-test",
             "st_data_export_description": "ammo-5.45x39-test_descr",
             "st_prop_weight": "0.30", "ui_inv_damage": "1.1",
             "st_data_export_ap": 20, "displayName": "ammo-5.45x39-test",
             "hasStashDrop": True, "factions": ["wp"], "shared": True},
            {"id": "ammo_5.45x39_alt", "pda_encyclopedia_name": "ammo-5.45x39-alt",
             "st_prop_weight": "0.28", "ui_inv_damage": "1.6",
             "st_data_export_ap": 40, "displayName": "ammo-5.45x39-alt",
             "hasStashDrop": False},
        ],
    }
    magazines = {
        "category": "Magazines",
        "headers": ["id", "pda_encyclopedia_name", "st_prop_weight",
                    "caliber", "magSize", "magRounds"],
        "items": [{
            "id": "mag_tak_default", "pda_encyclopedia_name": "st_mag_tak",
            "st_prop_weight": "0.23", "caliber": "5.45x39", "magSize": "30",
            "magRounds": "30", "displayName": "st_mag_tak",
        }],
    }
    weapon_parts = {
        "category": "Weapon Parts",
        "headers": ["id", "pda_encyclopedia_name", "st_upgr_cost", "st_prop_weight"],
        "items": [{
            "id": "prt_w_barrel_t", "pda_encyclopedia_name": "st_prt_w_barrel_t",
            "st_upgr_cost": "900", "st_prop_weight": "0.40",
            "displayName": "st_prt_w_barrel_t",
        }],
    }
    misc = {
        "category": "Misc",
        "headers": ["id", "pda_encyclopedia_name", "st_class"],
        "items": [{
            "id": "junk_item", "pda_encyclopedia_name": "st_junk",
            "st_class": "II_ATTCH", "displayName": "st_junk",
        }],
    }
    index = []
    for entry in rifles["items"]:
        index.append({"id": entry["id"], "name": entry["pda_encyclopedia_name"],
                      "category": "Rifles", "displayName": entry["displayName"],
                      "hasNpcWeaponDrop": True, "hasStashDrop": True,
                      "inStartingLoadout": False, "unobtainable": False,
                      "tacticalKit": False})
    for entry in ammo["items"]:
        index.append({"id": entry["id"], "name": entry["pda_encyclopedia_name"],
                      "category": "Ammo", "displayName": entry["displayName"]})
    index.append({"id": "mag_tak_default", "name": "st_mag_tak",
                  "category": "Magazines", "displayName": "st_mag_tak"})
    index.append({"id": "prt_w_barrel_t", "name": "st_prt_w_barrel_t",
                  "category": "Weapon Parts", "displayName": "st_prt_w_barrel_t"})
    index.append({"id": "junk_item", "name": "st_junk", "category": "Misc",
                  "displayName": "st_junk"})

    translations = {
        "locales": ["en", "ru"],
        "en": {
            "st_wpn_tak": "TAK-74",
            # literal backslash-n, exactly as the upstream export carries it
            "st_wpn_tak_descr": "A dependable test rifle.\\nTrusted in the Zone.",
            "ammo-5.45x39-test": "5.45x39mm Test FMJ",
            "ammo-5.45x39-test_descr": "Test cartridge.",
            "ammo-5.45x39-alt": "5.45x39mm Test AP",
            "st_mag_tak": "TAK Standard Magazine",
            "st_prt_w_barrel_t": "Test Barrel",
            "st_junk": "Junk Trinket",
            "st_prop_weight": "Weight", "ui_inv_accuracy": "Accuracy",
            "ui_inv_rate_of_fire": "Fire Rate", "ui_ammo_count": "Mag. Size",
            "st_upgr_cost": "Cost", "ui_inv_damage": "Damage",
            "st_data_export_ap": "AP", "ui_mm_repair": "Repair",
            "st_rank_veteran": "Veteran",
            "st_data_export_stash_type_common": "Common",
            "st_up_a1_name": "Lightened bolt",
            "st_up_a1_descr": "Reduces weight.",
            "st_prop_recoil": "Recoil",
            "app_repair_light": "Light weapons kit",
            "val_trader_sido": "Sidorovich",
        },
        "ru": {"st_wpn_tak": "ТАК-74"},
    }
    files = {
        "index.json": index,
        "rifles.json": rifles,
        "ammo.json": ammo,
        "magazines.json": magazines,
        "weapon-parts.json": weapon_parts,
        "misc.json": misc,
        "translations.json": translations,
        "items-common.json": {
            "wpn_tak": {"name": "st_wpn_tak", "price": 26950},
            "ammo_5.45x39_test": {"name": "ammo-5.45x39-test", "price": 700},
            "not_indexed_item": {"name": "st_hidden", "price": 1},
        },
        "categories.json": ["Rifles", "Ammo", "Magazines", "Weapon Parts", "Misc"],
        "ammo-weapons.json": {
            "ammo_5.45x39_test": [
                {"id": "wpn_tak", "name": "st_wpn_tak", "displayName": "st_wpn_tak [74]",
                 "category": "Rifles", "isAlt": False}],
        },
        "weapon-magazines.json": {"wpn_tak": ["mag_tak_default"]},
        "weapon-addons.json": {
            "wpn_tak": {"scopes": [], "silencers": ["sil_missing"], "launchers": [], "kits": []}},
        "addon-weapons.json": {},
        "weapon-addon-status.json": {"wpn_tak": {"silencer": 2, "scope": 0, "launcher": 0}},
        "disassemble.json": {
            "wpn_tak": [{"name": "st_prt_w_barrel_t", "amount": "x1"},
                        {"name": "st_unknown_part", "amount": "x2"}]},
        "craft-recipes.json": {
            "equipment": {"label": "Equipment", "items": [
                {"id": "mag_tak_default", "pda_encyclopedia_name": "st_mag_tak",
                 "toolTier": 1, "recipeReq": "recipe_basic_1",
                 "recipeReqName": "st_recipe_1",
                 "ingredients": [{"name": "st_prt_w_barrel_t", "amount": "x2"}]}]},
        },
        "drops.json": {"wpn_tak": {"army": ["st_rank_veteran"]}},
        "item-drops.json": {
            "wpn_tak": {"st_data_export_stash_type_common": [
                {"map": "jupiter", "chance": 3, "ecos": [1, 2]}]}},
        "sold-by.json": {
            "wpn_tak": [{"trader": "sido", "tier": "2", "cond": None}]},
        "traders-meta.json": [
            {"id": "sido", "labelKey": "app_sido", "label": "val_trader_sido",
             "color": "#fff", "primaryOrigin": "wp"}],
        "item-loadouts.json": {"wpn_tak": [{"faction": "stalker", "cost": 500,
                                            "selectable": True}]},
        "starting-loadouts.json": {
            "_info": "fixture", "points": [1000],
            "factions": [{"id": "stalker", "money": 5000,
                          "items": [{"id": "wpn_tak", "selectable": True,
                                     "quantity": 1, "cost": 500,
                                     "difficultyLock": None}]}],
            "shared": [], "ammoPerWeapon": {}, "ammoCount": {}},
        "kit-weapons.json": {},
        "item-parts.json": {"wpn_tak": {"kind": "weapon", "parts": ["prt_w_barrel_t"]}},
        "item-part-defs.json": {
            "prt_w_barrel_t": {"name": "st_prt_w_barrel_t",
                               "descr": "st_prt_w_barrel_t_descr",
                               "cost": 900, "weight": 0.4}},
        "calibers.json": {
            "5.45x39": {"name": "5.45x39", "variants": [
                {"id": "ammo_5.45x39_test", "name": "ammo-5.45x39-test"},
                {"id": "ammo_5.45x39_alt", "name": "ammo-5.45x39-alt"}]}},
        "mag-capacity.json": {},
        "outfit-exchange.json": {"factions": [], "exchanges": []},
        "display-labels.json": {
            "ui_mm_repair": {"D": {"label": "D", "color": "#b8863e"},
                             "L": {"label": "app_repair_light"}}},
        "upgrades.json": {
            "wpn_tak": [{"row": 1, "col": 1, "cell": 1, "prop": "st_prop_recoil",
                         "val": "-10", "name": "st_up_a1_name",
                         "desc": "st_up_a1_descr", "section": "up_sect_a_tak",
                         "cost": 3000,
                         "effects": [{"key": "st_prop_weight", "value": "0.25",
                                      "unit": "kg"}]}]},
    }
    return files


def write_fixture(root: Path, *, drop: set[str] | None = None,
                  mutate=None) -> Path:
    """Write the fixture dataset under root in the upstream layout and return
    the repo path. `drop` removes files from disk+manifest; `mutate(files)`
    can edit the dict before writing."""
    files = fixture_files()
    if mutate:
        mutate(files)
    drop = drop or set()
    data_dir = root / GAMMA_DATA_SUBDIR
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, content in files.items():
        if name in drop:
            continue
        (data_dir / name).write_text(json.dumps(content), encoding="utf-8")
        manifest[name] = "00000000"
    (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
