"""L-TAE module (adapted from utae-paps, MIT license)."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from multisenge_utae.models.positional_encoding import PositionalEncoder


class StarReLU(nn.Module):
    """MetaFormer StarReLU: scale * ReLU(x)^2 + bias. Scale and bias are learned."""

    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.relu(x).pow(2) + self.bias


def nonlinearity(name: str, inplace: bool = False) -> nn.Module:
    key = (name or "relu").lower().strip()
    if key == "relu":
        return nn.ReLU(inplace=inplace)
    if key == "gelu":
        return nn.GELU()
    if key == "silu":
        return nn.SiLU(inplace=inplace)
    if key == "starrelu":
        return StarReLU()
    raise ValueError(f"activation must be relu, gelu, silu, or starrelu, got {name!r}")


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=4,
        mlp=None,
        dropout=0.2,
        d_model=256,
        T=1000,
        return_att=False,
        positional_encoding=True,
        activation: str = "relu",
    ):
        super().__init__()
        if mlp is None:
            mlp = [256, 128]
        self.in_channels = in_channels
        self.mlp_dims = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp_dims[0] == self.d_model

        self.positional_encoder = PositionalEncoder(self.d_model // n_head, T=T, repeat=n_head) if positional_encoding else None
        self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
        self.in_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.in_channels)
        self.out_norm = nn.GroupNorm(num_groups=n_head, num_channels=self.mlp_dims[-1])

        layers: list[nn.Module] = []
        for i in range(len(self.mlp_dims) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]),
                    nn.BatchNorm1d(self.mlp_dims[i + 1]),
                    nonlinearity(activation),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            pad_mask = pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None and batch_positions is not None:
            bp = batch_positions.unsqueeze(-1).repeat((1, 1, h)).unsqueeze(-1).repeat((1, 1, 1, w))
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)
        out = out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out)
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)
        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(0, 1, 4, 2, 3)
        if self.return_att:
            return out, attn
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))
        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()
        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)
        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        if pad_mask is not None:
            pad_mask = pad_mask.repeat((n_head, 1))
        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
        if return_comp:
            output, attn, comp = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
            return output, attn, comp
        output, attn = self.attention(q, k, v, pad_mask=pad_mask, return_comp=return_comp)
        attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(dim=2)
        output = output.view(n_head, sz_b, 1, d_in // n_head).squeeze(dim=2)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v)
            return output, attn, comp
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn
