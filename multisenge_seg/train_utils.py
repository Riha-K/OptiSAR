"""Training helpers used by the U-TAE loop."""

from __future__ import annotations

import random

import numpy as np
import torch

from multisenge_seg.dataset import MultiSenGETemporalDataset, PatchRecord
from multisenge_seg.taxonomy import remap_mask

def set_seed(seed: int) -> None:
    """Lock shuffle / init / numpy aug. CUDA can still differ slightly across GPUs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_class_boost(spec: str, n_cls: int) -> np.ndarray:
    """Paper class ids 1..C, e.g. '8:2.5,4:1.5,10:0.6'. Missing ids stay 1.0."""
    boost = np.ones(n_cls, dtype=np.float64)
    spec = (spec or "").strip()
    if not spec:
        return boost
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        cid_s, fac_s = part.split(":")
        cid = int(cid_s)
        fac = float(fac_s)
        if cid < 1 or cid > n_cls:
            raise ValueError(f"class-boost id {cid} not in 1..{n_cls}")
        boost[cid - 1] = fac
    return boost


def apply_class_boost(weights: np.ndarray, boost: np.ndarray) -> np.ndarray:
    out = weights.astype(np.float64) * boost
    return out / out.sum() * len(out)


@torch.no_grad()
def estimate_class_counts(loader, num_classes: int, max_batches: int = 50) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for i, batch in enumerate(loader):
        m = batch["mask"].numpy()
        for c in range(num_classes):
            counts[c] += int((m == c).sum())
        if i + 1 >= max_batches:
            break
    return counts


def estimate_class_counts_from_gr(
    records: list[PatchRecord],
    split: str,
    num_classes: int,
) -> np.ndarray:
    """Inverse-frequency weights from all train GR masks (paper: full train set).

    Reads ground_reference only - not S1/S2 - so a full train scan is cheap.
    """
    import rasterio

    counts = np.zeros(num_classes, dtype=np.int64)
    n = 0
    for r in records:
        if r.split != split:
            continue
        with rasterio.open(r.gr_path) as src:
            mask = src.read(1).astype(np.int64)
        mask = remap_mask(mask, num_classes=num_classes)
        for c in range(1, num_classes + 1):
            counts[c - 1] += int((mask == c).sum())
        n += 1
    print("class-count patches", n)
    return counts


def estimate_channel_stats(
    records: list[PatchRecord],
    num_classes: int,
    max_patches: int = 64,
) -> dict:
    """Multitemporal per-channel mean/std on a train subset (paper-style)."""
    ds = MultiSenGETemporalDataset(records, "train", num_classes=num_classes, augment=False)
    n = len(ds) if max_patches <= 0 else min(len(ds), max_patches)
    if n == 0:
        raise RuntimeError("no train patches for stats")
    s1_sum = s1_sq = s2_sum = s2_sq = None
    pixels = 0
    # Read raw by temporarily bypassing normalize: use per-sample identity via fake mean0 std1 then undo - easier: load via dataset internals
    import rasterio

    for i in range(n):
        r = ds.records[i]
        s2_stack, s1_stack = [], []
        for m in ds.months:
            with rasterio.open(r.s2_by_month[m]) as src:
                s2_stack.append(src.read().astype(np.float32)[:10])
            with rasterio.open(r.s1_by_month[m]) as src:
                s1_stack.append(src.read().astype(np.float32)[:2])
        s2 = np.stack(s2_stack, 0)
        s1 = np.stack(s1_stack, 0)
        if s1_sum is None:
            s1_sum = np.zeros(s1.shape[1], np.float64)
            s1_sq = np.zeros(s1.shape[1], np.float64)
            s2_sum = np.zeros(s2.shape[1], np.float64)
            s2_sq = np.zeros(s2.shape[1], np.float64)
        s1_sum += s1.sum(axis=(0, 2, 3))
        s1_sq += (s1.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
        s2_sum += s2.sum(axis=(0, 2, 3))
        s2_sq += (s2.astype(np.float64) ** 2).sum(axis=(0, 2, 3))
        pixels += s1.shape[0] * s1.shape[2] * s1.shape[3]
    s1_mean = s1_sum / pixels
    s2_mean = s2_sum / pixels
    s1_std = np.sqrt(np.maximum(s1_sq / pixels - s1_mean**2, 1e-12))
    s2_std = np.sqrt(np.maximum(s2_sq / pixels - s2_mean**2, 1e-12))
    return {
        "s1_mean": s1_mean.astype(np.float32).tolist(),
        "s1_std": s1_std.astype(np.float32).tolist(),
        "s2_mean": s2_mean.astype(np.float32).tolist(),
        "s2_std": s2_std.astype(np.float32).tolist(),
        "n_patches": n,
    }


