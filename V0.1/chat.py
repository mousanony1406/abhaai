"""Run an interactive, fully local ABHA AI V0.1 chat session."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import CHECKPOINT_PATH, DEVICE
from model import DecoderOnlyTransformer
from tokenizer import CharTokenizer


def load_model(path: Path) -> tuple[DecoderOnlyTransformer, CharTokenizer]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train first with: python train.py")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = DecoderOnlyTransformer(tokenizer.vocab_size, **checkpoint["model_config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer


def respond(model: DecoderOnlyTransformer, tokenizer: CharTokenizer, message: str, max_new_tokens: int = 100) -> str:
    prompt = f"User: {message}\nABHA:"
    try:
        encoded = tokenizer.encode(prompt)
    except ValueError:
        # The small starter vocabulary cannot cover arbitrary Unicode input.
        allowed = set(tokenizer.chars)
        encoded = tokenizer.encode("".join(char if char in allowed else " " for char in prompt))
    context = torch.tensor([encoded[-model.block_size :]], dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    reply = tokenizer.decode(generated[len(context[0]) :]).split("\nUser:", 1)[0].strip()
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--prompt", help="Generate one reply and exit instead of opening the interactive shell.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    args = parser.parse_args()
    model, tokenizer = load_model(args.checkpoint)
    if args.prompt is not None:
        print(f"ABHA: {respond(model, tokenizer, args.prompt, args.max_new_tokens)}")
        return
    print("ABHA AI V0.1 local chat. Type 'quit' to exit.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if message.lower() in {"quit", "exit"}:
            print("ABHA: Goodbye!")
            break
        if message:
            print(f"ABHA: {respond(model, tokenizer, message, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
