"""U-TAE backbone (adapted from utae-paps, MIT license)."""

from __future__ import annotations

import torch
import torch.nn as nn

from multisenge_utae.models.ltae import LTAE2d, nonlinearity


class UTAE(nn.Module):
  def __init__(
      self,
      input_dim: int,
      num_classes: int,
      encoder_widths=None,
      decoder_widths=None,
      str_conv_k: int = 4,
      str_conv_s: int = 2,
      str_conv_p: int = 1,
      agg_mode: str = "att_group",
      encoder_norm: str = "group",
      n_head: int = 16,
      d_model: int = 256,
      d_k: int = 4,
      pad_value: float = 0.0,
      padding_mode: str = "reflect",
      activation: str = "relu",
  ):
    super().__init__()
    if encoder_widths is None:
      encoder_widths = [64, 64, 64, 128]
    if decoder_widths is None:
      decoder_widths = [32, 32, 64, 128]
    self.n_stages = len(encoder_widths)
    self.encoder_widths = encoder_widths
    self.decoder_widths = decoder_widths
    self.pad_value = pad_value
    self.activation = activation

    assert len(encoder_widths) == len(decoder_widths)
    assert encoder_widths[-1] == decoder_widths[-1]

    self.in_conv = ConvBlock(
        nkernels=[input_dim, encoder_widths[0], encoder_widths[0]],
        pad_value=pad_value,
        norm=encoder_norm,
        padding_mode=padding_mode,
        activation=activation,
    )
    self.down_blocks = nn.ModuleList(
        DownConvBlock(
            d_in=encoder_widths[i],
            d_out=encoder_widths[i + 1],
            k=str_conv_k,
            s=str_conv_s,
            p=str_conv_p,
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
            activation=activation,
        )
        for i in range(self.n_stages - 1)
    )
    self.up_blocks = nn.ModuleList(
        UpConvBlock(
            d_in=decoder_widths[i],
            d_out=decoder_widths[i - 1],
            d_skip=encoder_widths[i - 1],
            k=str_conv_k,
            s=str_conv_s,
            p=str_conv_p,
            padding_mode=padding_mode,
            activation=activation,
        )
        for i in range(self.n_stages - 1, 0, -1)
    )
    self.temporal_encoder = LTAE2d(
        in_channels=encoder_widths[-1],
        d_model=d_model,
        n_head=n_head,
        mlp=[d_model, encoder_widths[-1]],
        return_att=True,
        d_k=d_k,
        activation=activation,
    )
    self.temporal_aggregator = TemporalAggregator(mode=agg_mode)
    self.out_conv = ConvBlock(
        nkernels=[decoder_widths[0], 32, num_classes],
        padding_mode=padding_mode,
        activation=activation,
    )

  def _pad_mask(self, input: torch.Tensor) -> torch.Tensor:
    return (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

  def encode_spatial_bottleneck(self, input: torch.Tensor) -> torch.Tensor:
    """Spatial CNN only (no L-TAE / decoder). For CMU teacher.

    input: B,T,C,H,W or B,C,H,W → bottleneck map B,T,C_b,h,w or B,C_b,h,w
    (encoder_widths[-1] channels, typically 128).
    """
    out = self.in_conv.smart_forward(input)
    for i in range(self.n_stages - 1):
      out = self.down_blocks[i].smart_forward(out)
    return out

  def encode_levels(self, input: torch.Tensor, batch_positions=None):
    """Return encoder maps L0-L2 (B,T,C,H,W) and L3 after temporal attention (B,C,H,W)."""
    pad_mask = self._pad_mask(input)
    out = self.in_conv.smart_forward(input)
    feature_maps = [out]
    for i in range(self.n_stages - 1):
      out = self.down_blocks[i].smart_forward(feature_maps[-1])
      feature_maps.append(out)
    l3, _att = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)
    return {"L0": feature_maps[0], "L1": feature_maps[1], "L2": feature_maps[2], "L3": l3}

  def forward(self, input: torch.Tensor, batch_positions=None, return_att: bool = False):
    pad_mask = self._pad_mask(input)
    out = self.in_conv.smart_forward(input)
    feature_maps = [out]
    for i in range(self.n_stages - 1):
      out = self.down_blocks[i].smart_forward(feature_maps[-1])
      feature_maps.append(out)
    out, att = self.temporal_encoder(feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask)
    for i in range(self.n_stages - 1):
      skip = self.temporal_aggregator(feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att)
      out = self.up_blocks[i](out, skip)
    out = self.out_conv(out)
    if return_att:
      return out, att
    return out

  def set_train_mode(self, mode: str) -> None:
    """mode: 'full' | 'head' (freeze encoder + temporal, train decoder/head)."""
    if mode == "full":
      for p in self.parameters():
        p.requires_grad = True
      return
    if mode != "head":
      raise ValueError(f"unknown train mode {mode}")
    for mod in [self.in_conv, *self.down_blocks, self.temporal_encoder, self.temporal_aggregator]:
      for p in mod.parameters():
        p.requires_grad = False
    for mod in [*self.up_blocks, self.out_conv]:
      for p in mod.parameters():
        p.requires_grad = True


class TemporallySharedBlock(nn.Module):
  def __init__(self, pad_value=None):
    super().__init__()
    self.out_shape = None
    self.pad_value = pad_value

  def smart_forward(self, input: torch.Tensor) -> torch.Tensor:
    if input.dim() == 4:
      return self.forward(input)
    b, t, c, h, w = input.shape
    if self.pad_value is not None:
      dummy = torch.zeros(input.shape, device=input.device).float()
      self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape
      out = input.view(b * t, c, h, w)
      pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
      if pad_mask.any():
        temp = torch.ones(self.out_shape, device=input.device, requires_grad=False) * self.pad_value
        temp[~pad_mask] = self.forward(out[~pad_mask])
        out = temp
      else:
        out = self.forward(out)
    else:
      out = self.forward(input.view(b * t, c, h, w))
    _, c2, h2, w2 = out.shape
    return out.view(b, t, c2, h2, w2)


class ConvLayer(nn.Module):
  def __init__(
      self,
      nkernels,
      norm="batch",
      k=3,
      s=1,
      p=1,
      n_groups=4,
      last_relu=True,
      padding_mode="reflect",
      activation: str = "relu",
  ):
    super().__init__()
    layers: list[nn.Module] = []
    if norm == "batch":
      nl = nn.BatchNorm2d
    elif norm == "instance":
      nl = nn.InstanceNorm2d
    elif norm == "group":
      nl = lambda num_feats: nn.GroupNorm(num_channels=num_feats, num_groups=n_groups)
    else:
      nl = None
    for i in range(len(nkernels) - 1):
      layers.append(
          nn.Conv2d(
              in_channels=nkernels[i],
              out_channels=nkernels[i + 1],
              kernel_size=k,
              padding=p,
              stride=s,
              padding_mode=padding_mode,
          )
      )
      if nl is not None:
        layers.append(nl(nkernels[i + 1]))
      if last_relu or i < len(nkernels) - 2:
        layers.append(nonlinearity(activation))
    self.conv = nn.Sequential(*layers)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    return self.conv(input)


class ConvBlock(TemporallySharedBlock):
  def __init__(
      self,
      nkernels,
      pad_value=None,
      norm="batch",
      last_relu=True,
      padding_mode="reflect",
      activation: str = "relu",
  ):
    super().__init__(pad_value=pad_value)
    self.conv = ConvLayer(
        nkernels=nkernels,
        norm=norm,
        last_relu=last_relu,
        padding_mode=padding_mode,
        activation=activation,
    )

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
  def __init__(
      self,
      d_in,
      d_out,
      k,
      s,
      p,
      pad_value=None,
      norm="batch",
      padding_mode="reflect",
      activation: str = "relu",
  ):
    super().__init__(pad_value=pad_value)
    self.down = ConvLayer(
        nkernels=[d_in, d_in], norm=norm, k=k, s=s, p=p, padding_mode=padding_mode, activation=activation
    )
    self.conv1 = ConvLayer(nkernels=[d_in, d_out], norm=norm, padding_mode=padding_mode, activation=activation)
    self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode, activation=activation)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = self.down(input)
    out = self.conv1(out)
    return out + self.conv2(out)


class UpConvBlock(nn.Module):
  def __init__(
      self,
      d_in,
      d_out,
      k,
      s,
      p,
      norm="batch",
      d_skip=None,
      padding_mode="reflect",
      activation: str = "relu",
  ):
    super().__init__()
    d = d_out if d_skip is None else d_skip
    self.skip_conv = nn.Sequential(nn.Conv2d(d, d, 1), nn.BatchNorm2d(d), nonlinearity(activation))
    self.up = nn.Sequential(
        nn.ConvTranspose2d(d_in, d_out, kernel_size=k, stride=s, padding=p),
        nn.BatchNorm2d(d_out),
        nonlinearity(activation),
    )
    self.conv1 = ConvLayer(nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode, activation=activation)
    self.conv2 = ConvLayer(nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode, activation=activation)

  def forward(self, input: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    out = self.up(input)
    if out.shape[-2:] != skip.shape[-2:]:
      out = nn.functional.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    out = torch.cat([out, self.skip_conv(skip)], dim=1)
    out = self.conv1(out)
    return out + self.conv2(out)


class TemporalAggregator(nn.Module):
  def __init__(self, mode="mean"):
    super().__init__()
    self.mode = mode

  def forward(self, x, pad_mask=None, attn_mask=None):
    if pad_mask is not None and pad_mask.any():
      if self.mode == "att_group":
        n_heads, b, t, h, w = attn_mask.shape
        attn = attn_mask.view(n_heads * b, t, h, w)
        if x.shape[-2] > w:
          attn = nn.functional.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
          attn = nn.functional.avg_pool2d(attn, kernel_size=w // x.shape[-2])
        attn = attn.view(n_heads, b, t, *x.shape[-2:])
        attn = attn * (~pad_mask).float()[None, :, :, None, None]
        out = torch.stack(x.chunk(n_heads, dim=2))
        out = attn[:, :, :, None, :, :] * out
        out = out.sum(dim=2)
        return torch.cat([group for group in out], dim=1)
      if self.mode == "att_mean":
        attn = attn_mask.mean(dim=0)
        attn = nn.functional.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attn = attn * (~pad_mask).float()[:, :, None, None]
        return (x * attn[:, :, None, :, :]).sum(dim=1)
      if self.mode == "mean":
        out = x * (~pad_mask).float()[:, :, None, None, None]
        return out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
    if self.mode == "att_group":
      n_heads, b, t, h, w = attn_mask.shape
      attn = attn_mask.view(n_heads * b, t, h, w)
      if x.shape[-2] > w:
        attn = nn.functional.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
      else:
        attn = nn.functional.avg_pool2d(attn, kernel_size=w // x.shape[-2])
      attn = attn.view(n_heads, b, t, *x.shape[-2:])
      out = torch.stack(x.chunk(n_heads, dim=2))
      out = attn[:, :, :, None, :, :] * out
      out = out.sum(dim=2)
      return torch.cat([group for group in out], dim=1)
    if self.mode == "att_mean":
      attn = attn_mask.mean(dim=0)
      attn = nn.functional.interpolate(attn, size=x.shape[-2:], mode="bilinear", align_corners=False)
      return (x * attn[:, :, None, :, :]).sum(dim=1)
    if self.mode == "mean":
      return x.mean(dim=1)
    raise ValueError(f"unsupported agg mode {self.mode}")
