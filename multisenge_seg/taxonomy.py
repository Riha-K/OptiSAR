"""Class merges for CNN validation (Remote Sensing 2023 MultiSenGE paper)."""

from __future__ import annotations

# Paper 6-class: keep urban 1-5, merge everything else → 6
CLASS6_MAP: dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 6,
    8: 6,
    9: 6,
    10: 6,
    11: 6,
    12: 6,
    13: 6,
    14: 6,
}

# Paper 10-class merge (from RS 2023 Table 3 narrative)
CLASS10_MAP: dict[int, int] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,  # vineyards
    8: 7,  # orchards → vineyards+orchards
    9: 8,  # grasslands
    10: 9,  # groves/hedges
    11: 9,  # forests
    12: 9,  # open mineral
    13: 10,  # wetlands
    14: 10,  # water
}

CLASS6_NAMES = {
    1: "Dense Built-Up",
    2: "Sparse Built-Up",
    3: "Specialized Built-Up Areas",
    4: "Specialized but Vegetative Areas",
    5: "Large Scale Networks",
    6: "Non-urban / other",
}

CLASS10_NAMES = {
    1: "Dense Built-Up",
    2: "Sparse Built-Up",
    3: "Specialized Built-Up Areas",
    4: "Specialized but Vegetative Areas",
    5: "Large Scale Networks",
    6: "Arable Lands",
    7: "Vineyards and Orchards",
    8: "Grasslands",
    9: "Forests and semi-natural areas",
    10: "Water Surfaces",
}


def remap_mask(mask, num_classes: int = 6):
    """Remap native 1–14 mask to 6 or 10 classes. 0 stays ignore."""
    import numpy as np

    mapping = CLASS6_MAP if num_classes == 6 else CLASS10_MAP
    out = np.zeros_like(mask, dtype=np.int64)
    for src, dst in mapping.items():
        out[mask == src] = dst
    # anything unknown (and 0) → 0 ignore
    known = set(mapping.keys())
    unknown = (mask != 0) & ~np.isin(mask, list(known))
    out[unknown] = 0
    return out


def num_output_classes(num_classes: int = 6) -> int:
    return 6 if num_classes == 6 else 10
