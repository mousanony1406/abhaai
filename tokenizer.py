"""A dependency-free character tokenizer with explicit unknown-character handling."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharTokenizer:
    """Encode and decode text using a fixed character vocabulary."""

    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = tuple(sorted(set(text)))
        if not chars:
            raise ValueError("Cannot build a tokenizer from empty text.")
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {char: index for index, char in enumerate(self.chars)}

    @property
    def itos(self) -> dict[int, str]:
        return {index: char for index, char in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        mapping = self.stoi
        unknown = sorted(set(text) - set(mapping))
        if unknown:
            raise ValueError(f"Text contains characters outside the training vocabulary: {unknown!r}")
        return [mapping[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        mapping = self.itos
        try:
            return "".join(mapping[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token id: {exc.args[0]}") from exc

    def state_dict(self) -> dict[str, list[str]]:
        return {"chars": list(self.chars)}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[str]]) -> "CharTokenizer":
        return cls(tuple(state["chars"]))
