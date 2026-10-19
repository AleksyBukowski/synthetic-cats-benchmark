"""Automated overlap / leakage checks.

An image (or any transformed copy of it) may live in exactly one split. We check:
  * image IDs are unique and each has exactly one role;
  * class counts match the protocol (val/test balanced, imbalance only in train);
  * no near-duplicate photos (perceptual dHash, same breed) across train / val / test;
  * every added training observation (oversampled or augmented copy) comes from a
    *training* image of the minority class.
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import (BREEDS, MINORITY, N_TEST, N_TRAIN_MAJORITY, N_TRAIN_MINORITY,
                     N_VAL, NEAR_DUP_BITS)
from .data import load_image

USED_ROLES = ["train", "val", "test"]


def dhash(img, size: int = 8) -> int:
    """64-bit difference hash: robust to resizing / recompression, so it finds re-uploaded copies."""
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def compute_hashes(paths) -> list[str]:
    return [f"{dhash(load_image(p)):016x}" for p in tqdm(paths, desc="dHash")]


def hamming(hashes_a, hashes_b) -> np.ndarray:
    """Pairwise Hamming distances between two lists of hex hashes."""
    a = np.array([int(h, 16) for h in hashes_a], dtype=np.uint64)
    b = np.array([int(h, 16) for h in hashes_b], dtype=np.uint64)
    return np.bitwise_count(a[:, None] ^ b[None, :])


def near_duplicate_pairs(manifest: pd.DataFrame, max_bits: int = NEAR_DUP_BITS) -> pd.DataFrame:
    """Same-breed near-duplicate pairs whose two images sit in *different* used splits."""
    used = manifest[manifest.role.isin(USED_ROLES)].reset_index(drop=True)
    d = hamming(used.dhash, used.dhash)
    same_breed = used.breed.to_numpy()[:, None] == used.breed.to_numpy()[None, :]
    i, j = np.where(np.triu((d <= max_bits) & same_breed, k=1))
    pairs = pd.DataFrame({
        "id_a": used.image_id[i].values, "role_a": used.role[i].values,
        "id_b": used.image_id[j].values, "role_b": used.role[j].values,
        "bits": d[i, j],
    })
    return pairs[pairs.role_a != pairs.role_b]


def check_split(manifest: pd.DataFrame, raise_on_fail: bool = True) -> pd.DataFrame:
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    dup_ids = manifest.image_id[manifest.image_id.duplicated()]
    add("image IDs unique (one role per image)", dup_ids.empty, ", ".join(dup_ids[:5]))

    add("only known roles", manifest.role.isin(USED_ROLES + ["unused", "excluded_duplicate"]).all())

    counts = manifest.pivot_table(index="breed", columns="role", values="image_id",
                                  aggfunc="count", fill_value=0)
    expected_train = {b: N_TRAIN_MINORITY if b == MINORITY else N_TRAIN_MAJORITY for b in BREEDS}
    add(f"train counts ({N_TRAIN_MAJORITY} majority / {N_TRAIN_MINORITY} minority)",
        all(counts.loc[b, "train"] == n for b, n in expected_train.items()))
    add(f"validation balanced ({N_VAL} per class)", (counts.loc[BREEDS, "val"] == N_VAL).all())
    add(f"test balanced ({N_TEST} per class)", (counts.loc[BREEDS, "test"] == N_TEST).all())

    pairs = near_duplicate_pairs(manifest)
    add(f"no near-duplicates across splits (dHash <= {NEAR_DUP_BITS} bits)", pairs.empty,
        "; ".join(f"{r.id_a}({r.role_a})~{r.id_b}({r.role_b})" for r in pairs.head(5).itertuples()))

    report = pd.DataFrame(checks)
    if raise_on_fail and not report.passed.all():
        raise AssertionError("Leakage check failed:\n" + report[~report.passed].to_string())
    return report


def check_added_sources(source_ids, manifest: pd.DataFrame):
    """Every oversampled / augmented observation must be derived from a minority *training* image."""
    allowed = set(manifest.image_id[(manifest.role == "train") & (manifest.breed == MINORITY)])
    bad = sorted(set(source_ids) - allowed)
    if bad:
        raise AssertionError(f"Added data derived from non-training images: {bad[:5]}")
