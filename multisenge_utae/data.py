"""Input stacking for U-TAE: concat S2+S1 per date -> B x T x C x H x W."""

from __future__ import annotations

import torch

MONTHS = (7, 8, 9, 11)
S2_DIM = 10
S1_DIM = 2


def stack_modalities(s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
  """s2: B,T,10,H,W and s1: B,T,2,H,W -> B,T,12,H,W (S2 then S1)."""
  return torch.cat([s2, s1], dim=2)


def modality_input_dim(modality: str) -> int:
  m = modality.lower().strip()
  if m == "both":
    return S2_DIM + S1_DIM
  if m == "s2":
    return S2_DIM
  if m == "s1":
    return S1_DIM
  raise ValueError(f"unknown modality={modality!r}; use both|s2|s1")


def select_modality(x: torch.Tensor, modality: str) -> torch.Tensor:
  """Slice stacked x (…,12,H,W) to S2-only / S1-only / both (paper-style ablation)."""
  m = modality.lower().strip()
  if m == "both":
    return x
  if m == "s2":
    return x[:, :, :S2_DIM]
  if m == "s1":
    return x[:, :, S2_DIM : S2_DIM + S1_DIM]
  raise ValueError(f"unknown modality={modality!r}; use both|s2|s1")


def batch_positions(batch_size: int, device: torch.device, months: tuple[int, ...] = MONTHS) -> torch.Tensor:
  pos = torch.tensor(months, dtype=torch.float32, device=device)
  return pos.unsqueeze(0).expand(batch_size, -1)


def collate_utae(batch: list[dict]) -> dict:
  s2 = torch.stack([b["s2"] for b in batch], dim=0)
  s1 = torch.stack([b["s1"] for b in batch], dim=0)
  return {
      "patch_id": [b["patch_id"] for b in batch],
      "s2": s2,
      "s1": s1,
      "x": stack_modalities(s2, s1),
      "mask": torch.stack([b["mask"] for b in batch], dim=0),
  }
