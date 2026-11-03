"""MultiSenGE multitemporal patch index + torch Dataset scaffold."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .splits import split_for_tile, tile_from_patch_id
from .taxonomy import remap_mask

_DATE_RE = re.compile(r"_(\d{8})_S[12]_")


@dataclass
class PatchRecord:
    patch_id: str
    tile: str
    split: str
    label_json: Path
    gr_path: Path
    s2_by_month: dict[int, Path]  # month -> path (one chosen date)
    s1_by_month: dict[int, Path]


def _parse_date(name: str) -> datetime | None:
    m = _DATE_RE.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d")


def _files_from_label_field(field: str) -> list[str]:
    if not field:
        return []
    return [x.strip() for x in str(field).split(";") if x.strip()]


def pick_four_dates(
    names: list[str],
    months: tuple[int, ...] = (7, 8, 9, 11),
    min_gap_days: int = 17,
) -> dict[int, str] | None:
    """Pick one filename per target month with ≥ min_gap_days between consecutive picks."""
    by_month: dict[int, list[tuple[datetime, str]]] = {m: [] for m in months}
    for n in names:
        d = _parse_date(n)
        if d is None or d.month not in by_month:
            continue
        by_month[d.month].append((d, n))
    for m in months:
        by_month[m].sort(key=lambda x: x[0])
        if not by_month[m]:
            return None

    chosen: dict[int, tuple[datetime, str]] = {}
    prev: datetime | None = None
    for m in months:
        picked = None
        for d, n in by_month[m]:
            if prev is None or (d - prev).days >= min_gap_days:
                picked = (d, n)
                break
        if picked is None:
            return None
        chosen[m] = picked
        prev = picked[0]
    return {m: chosen[m][1] for m in months}


def pick_s1_for_s2_months(
    s1_names: list[str],
    s2_pick: dict[int, str],
    max_day_delta: int = 45,
) -> dict[int, str] | None:
    """Pair each picked S2 date with nearest on-disk S1 (prefer same month).

    MultiSenGE-Tools `get_s1_dates` takes first S1 in the same month. We prefer that,
    then fall back to nearest S1 within ``max_day_delta`` (disk completeness).
    """
    parsed: list[tuple[datetime, str]] = []
    by_month: dict[int, list[tuple[datetime, str]]] = {}
    for n in s1_names:
        d = _parse_date(n)
        if d is None:
            continue
        parsed.append((d, n))
        by_month.setdefault(d.month, []).append((d, n))
    for m in by_month:
        by_month[m].sort(key=lambda x: x[0])
    if not parsed:
        return None

    out: dict[int, str] = {}
    for m, s2_name in s2_pick.items():
        s2_d = _parse_date(s2_name)
        if s2_d is None:
            return None
        cands = by_month.get(m) or []
        if cands:
            out[m] = min(cands, key=lambda dn: abs((dn[0] - s2_d).days))[1]
            continue
        best = min(parsed, key=lambda dn: abs((dn[0] - s2_d).days))
        if abs((best[0] - s2_d).days) > max_day_delta:
            return None
        out[m] = best[1]
    return out


def build_patch_index(
    data_root: Path,
    months: tuple[int, ...] = (7, 8, 9, 11),
    min_gap_days: int = 17,
    require_s1: bool = True,
    max_labels: int | None = None,
) -> list[PatchRecord]:
    """Scan labels/ and keep patches that pass paper-like date + tile split filters.

    ``max_labels``: stop after reading this many JSON files (quick smoke only).
    """
    data_root = Path(data_root)
    labels_dir = data_root / "labels"
    s1_dir = data_root / "s1"
    s2_dir = data_root / "s2"
    gr_dir = data_root / "ground_reference"

    records: list[PatchRecord] = []
    label_paths = sorted(labels_dir.glob("*.json"))
    if max_labels is not None:
        label_paths = label_paths[: max_labels]
    for jp in label_paths:
        patch_id = jp.stem
        tile = tile_from_patch_id(patch_id)
        split = split_for_tile(tile)
        if split is None:
            continue

        meta: dict[str, Any] = json.loads(jp.read_text(encoding="utf-8"))
        # Select among files that exist (JSON lists many dates; earliest JSON date may be absent).
        s2_names = [
            n
            for n in _files_from_label_field(meta.get("corresponding_s2", ""))
            if (s2_dir / n).is_file()
        ]
        s1_names = [
            n
            for n in _files_from_label_field(meta.get("corresponding_s1", ""))
            if (s1_dir / n).is_file()
        ]
        s2_pick = pick_four_dates(s2_names, months=months, min_gap_days=min_gap_days)
        if s2_pick is None:
            continue
        s1_pick = pick_s1_for_s2_months(s1_names, s2_pick) if require_s1 else None
        if require_s1 and s1_pick is None:
            continue

        parts = patch_id.split("_")
        if len(parts) >= 3:
            gr_path = gr_dir / f"{parts[0]}_GR_{parts[1]}_{parts[2]}.tif"
        else:
            gr_path = gr_dir / f"{tile}_GR_{patch_id.split('_', 1)[1]}.tif"
        if not gr_path.is_file():
            continue

        s2_paths: dict[int, Path] = {}
        ok = True
        for m, name in s2_pick.items():
            p = s2_dir / name
            if not p.is_file():
                ok = False
                break
            s2_paths[m] = p
        if not ok:
            continue

        s1_paths: dict[int, Path] = {}
        if s1_pick:
            for m, name in s1_pick.items():
                p = s1_dir / name
                if not p.is_file():
                    ok = False
                    break
                s1_paths[m] = p
            if not ok:
                continue

        records.append(
            PatchRecord(
                patch_id=patch_id,
                tile=tile,
                split=split,
                label_json=jp,
                gr_path=gr_path,
                s2_by_month=s2_paths,
                s1_by_month=s1_paths,
            )
        )
    return records


def summarize_splits(records: list[PatchRecord]) -> dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for r in records:
        out[r.split] = out.get(r.split, 0) + 1
    return out


class MultiSenGETemporalDataset:
    """Lazy raster reader. Requires torch + rasterio at runtime."""

    def __init__(
        self,
        records: list[PatchRecord],
        split: str,
        num_classes: int = 6,
        months: tuple[int, ...] = (7, 8, 9, 11),
        s2_bands: slice | None = None,
        augment: bool = False,
        s1_mean: np.ndarray | None = None,
        s1_std: np.ndarray | None = None,
        s2_mean: np.ndarray | None = None,
        s2_std: np.ndarray | None = None,
    ) -> None:
        import numpy as np

        self.records = [r for r in records if r.split == split]
        self.num_classes = num_classes
        self.months = months
        self.s2_bands = s2_bands
        self.augment = augment and split == "train"
        self.s1_mean = None if s1_mean is None else np.asarray(s1_mean, dtype=np.float32)
        self.s1_std = None if s1_std is None else np.asarray(s1_std, dtype=np.float32)
        self.s2_mean = None if s2_mean is None else np.asarray(s2_mean, dtype=np.float32)
        self.s2_std = None if s2_std is None else np.asarray(s2_std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def _normalize(self, x, mean, std):
        import numpy as np

        # x: T,C,H,W - paper: multitemporal channel mean/std
        if mean is not None and std is not None:
            m = mean.reshape(1, -1, 1, 1)
            s = std.reshape(1, -1, 1, 1)
            return (x - m) / (s + 1e-6)
        mean_x = x.mean(axis=(0, 2, 3), keepdims=True)
        std_x = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        return (x - mean_x) / std_x

    def _maybe_augment(self, s1, s2, target):
        import numpy as np

        if not self.augment:
            return s1, s2, target
        # Paper: geometric augmentations (~75% of samples)
        if np.random.rand() < 0.75:
            k = int(np.random.randint(0, 4))
            if k:
                s1 = np.rot90(s1, k, axes=(2, 3)).copy()
                s2 = np.rot90(s2, k, axes=(2, 3)).copy()
                target = np.rot90(target, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.5:
                s1 = np.flip(s1, axis=3).copy()
                s2 = np.flip(s2, axis=3).copy()
                target = np.flip(target, axis=1).copy()
            if np.random.rand() < 0.5:
                s1 = np.flip(s1, axis=2).copy()
                s2 = np.flip(s2, axis=2).copy()
                target = np.flip(target, axis=0).copy()
        return s1, s2, target

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import numpy as np
        import rasterio
        import torch

        r = self.records[idx]
        s2_stack = []
        s1_stack = []
        for m in self.months:
            with rasterio.open(r.s2_by_month[m]) as src:
                arr = src.read().astype(np.float32)  # C,H,W
            if self.s2_bands is not None:
                arr = arr[self.s2_bands]
            s2_stack.append(arr)
            with rasterio.open(r.s1_by_month[m]) as src:
                s1 = src.read().astype(np.float32)
            s1_stack.append(s1)

        s2 = np.stack(s2_stack, axis=0)  # T,C,H,W - paper: 10 S2 bands
        s1 = np.stack(s1_stack, axis=0)
        if self.s2_bands is None and s2.shape[1] > 10:
            s2 = s2[:, :10]
        if s1.shape[1] > 2:
            s1 = s1[:, :2]

        with rasterio.open(r.gr_path) as src:
            mask = src.read(1).astype(np.int64)
        mask = remap_mask(mask, num_classes=self.num_classes)
        target = np.full_like(mask, 255)
        for c in range(1, self.num_classes + 1):
            target[mask == c] = c - 1

        s1, s2, target = self._maybe_augment(s1, s2, target)
        s2 = self._normalize(s2, self.s2_mean, self.s2_std)
        s1 = self._normalize(s1, self.s1_mean, self.s1_std)

        return {
            "patch_id": r.patch_id,
            "s2": torch.from_numpy(np.ascontiguousarray(s2)),
            "s1": torch.from_numpy(np.ascontiguousarray(s1)),
            "mask": torch.from_numpy(np.ascontiguousarray(target)),
            "split": r.split,
        }
