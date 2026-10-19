"""Create the one-and-only real-image split and save it as a manifest."""
import numpy as np
import pandas as pd

from .config import (BREEDS, MANIFEST_PATH, MINORITY, N_TEST, N_TRAIN_MAJORITY,
                     N_TRAIN_MINORITY, N_VAL, NEAR_DUP_BITS, SPLIT_SEED)
from .data import list_cat_images
from .leakage import compute_hashes, hamming


def mark_near_duplicates(df: pd.DataFrame) -> pd.Series:
    """For each image, the ID of an earlier image it nearly duplicates (or "" if none).

    Duplicates are removed *before* splitting, so a photo and its copy can never
    end up in different splits. Only images of the same breed are compared: a real copy
    of a photo carries the same label, while cross-breed hash matches were false positives.
    """
    d = hamming(df.dhash, df.dhash)
    breed = df.breed.to_numpy()
    dup_of = [""] * len(df)
    kept = []
    for i in range(len(df)):
        match = [j for j in kept if breed[i] == breed[j] and d[i, j] <= NEAR_DUP_BITS]
        if match:
            dup_of[i] = df.image_id[match[0]]
        else:
            kept.append(i)
    return pd.Series(dup_of, index=df.index)


def make_manifest(seed: int = SPLIT_SEED) -> pd.DataFrame:
    df = list_cat_images()
    df["dhash"] = compute_hashes(df.path)
    df["duplicate_of"] = mark_near_duplicates(df)
    df["role"] = np.where(df.duplicate_of != "", "excluded_duplicate", "unused")

    rng = np.random.default_rng(seed)
    for breed in BREEDS:
        pool = df.index[(df.breed == breed) & (df.role == "unused")].to_numpy()
        pool = rng.permutation(pool)
        n_train = N_TRAIN_MINORITY if breed == MINORITY else N_TRAIN_MAJORITY
        df.loc[pool[:N_TEST], "role"] = "test"
        df.loc[pool[N_TEST:N_TEST + N_VAL], "role"] = "val"
        df.loc[pool[N_TEST + N_VAL:N_TEST + N_VAL + n_train], "role"] = "train"
        # the rest stays "unused" (for Egyptian Mau these are deliberately left out of training)

    return df[["image_id", "breed", "role", "official_split", "duplicate_of", "dhash", "path"]]


def get_manifest(create_if_missing: bool = True) -> pd.DataFrame:
    """Load the saved manifest. It is created only once; delete the file to re-split on purpose."""
    if not MANIFEST_PATH.exists():
        if not create_if_missing:
            raise FileNotFoundError(MANIFEST_PATH)
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        make_manifest().to_csv(MANIFEST_PATH, index=False)
    return pd.read_csv(MANIFEST_PATH, dtype=str, keep_default_na=False)


def class_count_table(manifest: pd.DataFrame) -> pd.DataFrame:
    roles = ["train", "val", "test", "unused", "excluded_duplicate"]
    table = (manifest.pivot_table(index="breed", columns="role", values="image_id",
                                  aggfunc="count", fill_value=0)
             .reindex(index=BREEDS, columns=roles, fill_value=0))
    table.loc["Total"] = table.sum()
    return table
