"""Small decoder-only Transformer used by the local ABHA AI prototype."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DecoderOnlyTransformer(nn.Module):
    """A compact causal language model for character-level next-token prediction."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int, dropout: float) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, indices: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = indices.shape
        if time > self.block_size:
            raise ValueError(f"Sequence length {time} exceeds block size {self.block_size}.")
        positions = torch.arange(time, device=indices.device)
        hidden = self.token_embedding(indices) + self.position_embedding(positions)
        mask = torch.triu(torch.ones(time, time, device=indices.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=mask, is_causal=True)
        logits = self.output(self.norm(hidden))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, indices: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        """Autoregressively sample tokens from the model."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = indices[:, -self.block_size :]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            indices = torch.cat((indices, next_token), dim=1)
        return indices
