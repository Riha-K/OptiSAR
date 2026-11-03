"""Stage 2 land-cover losses. Default remains weighted CE.

Composite A: L = L_wCE + lambda * L_Dice  (mean soft Dice over C classes)
Composite B: L = L_wCE + lambda * L_GDL   (generalized Dice, 1/area^2)
Ignore label 255. Stage 1 InfoNCE is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegCompositeLoss(nn.Module):
  def __init__(
      self,
      weight: torch.Tensor | None,
      ignore_index: int = 255,
      mode: str = "ce",
      dice_lambda: float = 1.0,
      eps: float = 1.0,
  ):
    super().__init__()
    if mode not in ("ce", "ce_dice", "ce_gdl"):
      raise ValueError(f"unknown loss mode {mode}")
    self.ignore_index = int(ignore_index)
    self.mode = mode
    self.dice_lambda = float(dice_lambda)
    self.eps = float(eps)
    if weight is None:
      self.weight = None
    else:
      self.register_buffer("weight", weight)

  def _masked_prob_target(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_cls = logits.shape[1]
    valid = target != self.ignore_index
    clamped = target.clamp(min=0, max=n_cls - 1)
    one_hot = F.one_hot(clamped, n_cls).permute(0, 3, 1, 2).to(dtype=logits.dtype)
    valid_f = valid.unsqueeze(1).to(dtype=logits.dtype)
    probs = logits.softmax(dim=1) * valid_f
    one_hot = one_hot * valid_f
    return probs, one_hot, valid

  def _dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs, one_hot, valid = self._masked_prob_target(logits, target)
    if valid.sum() == 0:
      return logits.sum() * 0.0
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)
    p_sum = probs.sum(dim=dims)
    y_sum = one_hot.sum(dim=dims)
    dice = (2.0 * inter + self.eps) / (p_sum + y_sum + self.eps)
    return 1.0 - dice.mean()

  def _gdl(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs, one_hot, valid = self._masked_prob_target(logits, target)
    if valid.sum() == 0:
      return logits.sum() * 0.0
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)
    p_sum = probs.sum(dim=dims)
    y_sum = one_hot.sum(dim=dims)
    alpha = 1.0 / (y_sum * y_sum + self.eps)
    numer = 2.0 * (alpha * inter).sum()
    denom = (alpha * (p_sum + y_sum)).sum()
    return 1.0 - (numer + self.eps) / (denom + self.eps)

  def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        target,
        weight=self.weight,
        ignore_index=self.ignore_index,
    )
    if self.mode == "ce":
      return ce
    if self.mode == "ce_dice":
      overlap = self._dice(logits, target)
    else:
      overlap = self._gdl(logits, target)
    return ce + self.dice_lambda * overlap
