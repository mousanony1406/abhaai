"""Central configuration for the ABHA AI V0.1 local language model."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "dialogues.txt"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
CHECKPOINT_PATH = ARTIFACT_DIR / "abha_v0_1.pt"

# Intentionally small defaults so CPU training completes quickly.
BLOCK_SIZE = 64
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1
BATCH_SIZE = 16
LEARNING_RATE = 3e-3
TRAIN_STEPS = 300
EVAL_INTERVAL = 100
EVAL_BATCHES = 8
SEED = 7
DEVICE = "cpu"
