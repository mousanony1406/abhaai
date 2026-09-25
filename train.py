"""Train and save ABHA AI V0.1 using the bundled local dialogue corpus."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from config import (BATCH_SIZE, BLOCK_SIZE, CHECKPOINT_PATH, DATA_PATH, DEVICE,
                     DROPOUT, EVAL_BATCHES, EVAL_INTERVAL, LEARNING_RATE, N_EMBD, N_HEAD,
                     N_LAYER, SEED, TRAIN_STEPS)
from model import DecoderOnlyTransformer
from tokenizer import CharTokenizer


def build_model(tokenizer: CharTokenizer) -> DecoderOnlyTransformer:
    return DecoderOnlyTransformer(tokenizer.vocab_size, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT).to(DEVICE)


def make_batch(data: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= BLOCK_SIZE:
        raise ValueError(f"Training data needs more than {BLOCK_SIZE} characters; received {len(data)}.")
    starts = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
    x = torch.stack([data[start : start + BLOCK_SIZE] for start in starts]).to(DEVICE)
    y = torch.stack([data[start + 1 : start + BLOCK_SIZE + 1] for start in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model: DecoderOnlyTransformer, data: torch.Tensor) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        _, loss = model(*make_batch(data, BATCH_SIZE))
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model: DecoderOnlyTransformer, tokenizer: CharTokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "model_config": {"block_size": BLOCK_SIZE, "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER, "dropout": DROPOUT},
        },
        path,
    )


def train(steps: int, checkpoint_path: Path) -> Path:
    random.seed(SEED)
    torch.manual_seed(SEED)
    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    model = build_model(tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for step in range(steps):
        x, y = make_batch(data, BATCH_SIZE)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % EVAL_INTERVAL == 0 or step + 1 == steps:
            print(f"step {step + 1}/{steps} | loss {estimate_loss(model, data):.4f}")
    save_checkpoint(model, tokenizer, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS, help="Number of optimization steps.")
    parser.add_argument("--output", type=Path, default=CHECKPOINT_PATH, help="Checkpoint file to create.")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    train(args.steps, args.output)


if __name__ == "__main__":
    main()
