"""
Transformer building blocks: RMSNorm, RoPE, SwiGLU MLP, DecoderLayer.
All layers are parametric — no hardcoded dimensions.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0,
                          device: torch.device = None) -> torch.Tensor:
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs.sin().unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    xq_out = xq * freqs_cos[:, :, :xq.size(2), :] + rotate_half(xq) * freqs_sin[:, :, :xq.size(2), :]
    xk_out = xk * freqs_cos[:, :, :xk.size(2), :] + rotate_half(xk) * freqs_sin[:, :, :xk.size(2), :]
    return xq_out, xk_out


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 1024,
                 theta: float = 10000.0, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        freqs = precompute_rope_freqs(self.head_dim, max_seq_len, theta)
        self.register_buffer("rope_freqs", freqs, persistent=False)

        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, self.rope_freqs[:T])

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_mask is None,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 max_seq_len: int = 1024, dropout: float = 0.0,
                 theta: float = 10000.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = Attention(d_model, n_heads, max_seq_len, theta, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), attn_mask, is_causal)
        x = x + self.mlp(self.ffn_norm(x))
        return x
