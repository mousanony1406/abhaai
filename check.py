"""Fast structural and inference smoke checks for ABHA V0.1."""
from __future__ import annotations

import tempfile
from pathlib import Path

from chat import load_model, respond
from config import DATA_PATH
from tokenizer import CharTokenizer
from train import train


def main() -> None:
    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode("ABHA")) == "ABHA"
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "smoke.pt"
        train(steps=2, checkpoint_path=checkpoint)
        model, loaded_tokenizer = load_model(checkpoint)
        assert loaded_tokenizer.chars == tokenizer.chars
        reply = respond(model, loaded_tokenizer, "hello", max_new_tokens=8)
        assert isinstance(reply, str)
    print("All ABHA V0.1 checks passed.")


if __name__ == "__main__":
    main()
