# ABHA AI V0.1

ABHA AI V0.1 is a small, fully local, character-level conversational language-model prototype. It uses a decoder-only PyTorch Transformer trained only on `data/dialogues.txt`; it calls no hosted or paid AI service.

## Requirements

- Python 3.10 or later (tested with the repository environment's Python 3.14)
- PyTorch (the only runtime dependency)

## Setup

From the repository root, create an optional virtual environment and install the declared dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r V0.1/requirements.txt
```

## Verify the project

```bash
cd V0.1
python3 check.py
python3 -m compileall -q .
```

`check.py` validates tokenizer round-tripping, creates a small temporary model checkpoint, reloads it, and performs inference.

## Train a local model

```bash
cd V0.1
python3 train.py
```

The model is saved to `V0.1/artifacts/abha_v0_1.pt`. For a faster experiment, use `python3 train.py --steps 50`.

## Chat locally

After training:

```bash
cd V0.1
python3 chat.py
```

Type a message and press Enter. Type `quit` or `exit` to close the chat. A non-interactive smoke/inference command is:

```bash
python3 chat.py --prompt "hello" --max-new-tokens 40
```

## Project layout and extension points

- `tokenizer.py`: local character tokenizer.
- `model.py`: decoder-only Transformer architecture.
- `train.py`: data reading, training, and checkpoint saving.
- `chat.py`: checkpoint loading and local chat UI.
- `config.py`: centralized file paths and model/training defaults.

Future memory, voice, vision, and PC-control features can be introduced as independent modules and wired into `chat.py`, leaving the model and training APIs unchanged.
