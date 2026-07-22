"""
Decoder-only transformer with weight tying, RMSNorm, RoPE, SwiGLU.
Fully parametric — no hardcoded dimensions.
Supports training (forward), generation, and activation checkpointing.
"""

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import RMSNorm, DecoderLayer


class TransformerConfig:
    def __init__(self, config_dict: dict):
        required = ["vocab_size", "d_model", "n_layers", "n_heads", "d_ff",
                     "max_seq_len", "dropout", "theta"]
        for key in required:
            if key not in config_dict:
                raise ValueError(f"Missing required config key: {key}")

        self.vocab_size = config_dict["vocab_size"]
        self.d_model = config_dict["d_model"]
        self.n_layers = config_dict["n_layers"]
        self.n_heads = config_dict["n_heads"]
        self.d_ff = config_dict["d_ff"]
        self.max_seq_len = config_dict["max_seq_len"]
        self.dropout = config_dict["dropout"]
        self.theta = config_dict.get("theta", 10000.0)
        self.pad_token_id = config_dict.get("pad_token_id", -100)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "theta": self.theta,
            "pad_token_id": self.pad_token_id,
        }


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                theta=config.theta,
            )
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._tie_weights()

        self.apply(self._init_weights)

    def _tie_weights(self):
        self.output.weight = self.tok_embeddings.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                use_checkpoint: bool = False) -> dict:
        B, T = input_ids.shape

        x = self.tok_embeddings(input_ids)

        for layer in self.layers:
            if use_checkpoint and self.training:
                x = checkpoint(layer, x, None, True, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)

        if labels is not None:
            logits = self.output(x)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=self.config.pad_token_id,
            )
            return {"loss": loss, "logits": logits}
        else:
            logits = self.output(x[:, [-1], :])
            return {"logits": logits}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            if seq_len > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            out = self.forward(input_ids)
            logits = out["logits"][:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.tok_embeddings.weight.numel()
        return total
