"""Geographic tile split from Wenger et al. Remote Sensing 2023 (Fig. 4).

Paper lists train tiles explicitly but omits T32ULV; including it yields the
published counts (train 3369 / val 1911 / test 610 = 5890 at 17-day S2 gap).
Rule: val/test fixed; every other MultiSenGE tile → train.
"""

from __future__ import annotations

_VAL = {"31UFP", "31UGP"}
_TEST = {"31UEQ"}
_TRAIN = {
    "32UMV",
    "32ULU",
    "32TLT",
    "31UGQ",
    "31TFN",
    "31UFQ",
    "31UFR",
    "32ULV",
}


def normalize_tile(tile: str) -> str:
    t = tile.strip().upper()
    if t.startswith("T") and len(t) > 1:
        t = t[1:]
    return t


def split_for_tile(tile: str) -> str | None:
    """Return 'train' | 'val' | 'test' | None if tile not used."""
    t = normalize_tile(tile)
    if t in _VAL:
        return "val"
    if t in _TEST:
        return "test"
    if t in _TRAIN:
        return "train"
    # Any leftover MultiSenGE tile (e.g. future adds) → train if not held out
    # Known held-out-only: none beyond val/test. Unknown tiles return None.
    return None


def tile_from_patch_id(patch_id: str) -> str:
    """31TFN_4626_514 → 31TFN"""
    return patch_id.split("_", 1)[0]
