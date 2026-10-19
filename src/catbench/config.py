"""Single place for every protocol constant (see meetings/meeting1/protocol.md)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"                      # git-ignored: raw dataset + cached features
FEATURE_DIR = DATA_DIR / "features"
MANIFEST_PATH = ROOT / "manifests" / "split_manifest.csv"
RESULTS_DIR = ROOT / "results"

BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay",
    "Egyptian Mau", "Ragdoll", "Russian Blue", "Siamese",
]
MINORITY = "Egyptian Mau"

# Real images per class and role
N_TRAIN_MAJORITY = 70
N_TRAIN_MINORITY = 15
N_VAL = 20
N_TEST = 30

# Every "added data" condition adds exactly this many minority observations
ADDED_BUDGET = N_TRAIN_MAJORITY - N_TRAIN_MINORITY   # 55

SPLIT_SEED = 2026                 # used once, to create the manifest
RUN_SEEDS = [0, 1, 2, 3, 4]       # repeated runs of every condition

# Two images count as near-duplicates if their 64-bit dHash differs in <= this many bits
NEAR_DUP_BITS = 6

BACKBONES = ["resnet18", "efficientnet_b0"]
CONDITIONS = ["imbalanced", "oversampling", "augmentation"]

# Logistic-regression regularisation grid, chosen on validation macro-F1
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
