"""Train the lightweight classifier for each condition and evaluate it."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import BREEDS, C_GRID, MINORITY
from .features import augmented_features, minority_sources, real_features
from .leakage import check_added_sources


def split_xy(feats: dict, manifest, role: str):
    rows = manifest[manifest.role == role]
    return np.stack([feats[i] for i in rows.image_id]), rows.breed.to_numpy()


def training_set(condition: str, backbone: str, manifest, feats: dict, seed: int):
    """Real imbalanced training data + ADDED_BUDGET extra minority observations (if any)."""
    X, y = split_xy(feats, manifest, "train")
    if condition == "imbalanced":
        return X, y
    if condition == "oversampling":
        sources = minority_sources(manifest, seed)
        X_extra = np.stack([feats[s] for s in sources])          # exact repeats
    elif condition == "augmentation":
        sources, X_extra = augmented_features(backbone, manifest, seed)
    else:
        raise ValueError(condition)
    check_added_sources(sources, manifest)
    return np.vstack([X, X_extra]), np.concatenate([y, [MINORITY] * len(X_extra)])


def scores(y_true, y_pred) -> dict:
    mau = dict(labels=[MINORITY], average="macro", zero_division=0)
    return {
        "mau_recall": recall_score(y_true, y_pred, **mau),
        "mau_precision": precision_score(y_true, y_pred, **mau),
        "mau_f1": f1_score(y_true, y_pred, **mau),
        "macro_f1": f1_score(y_true, y_pred, labels=BREEDS, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def fit_with_validation(X, y, X_val, y_val):
    """Pick C from C_GRID by validation macro-F1 (ties -> stronger regularisation)."""
    best = None
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
        clf.fit(X, y)
        f1 = f1_score(y_val, clf.predict(X_val), average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return best[1], best[2]


def run_baselines(manifest, backbones, conditions, seeds):
    """Returns (runs, confusion) — one row per backbone/condition/seed/split, and summed test confusion matrices."""
    runs, confusion = [], {}
    for backbone in backbones:
        feats = real_features(backbone, manifest)
        X_val, y_val = split_xy(feats, manifest, "val")
        X_test, y_test = split_xy(feats, manifest, "test")
        for condition in conditions:
            cm = np.zeros((len(BREEDS), len(BREEDS)), dtype=int)
            for seed in seeds:
                X, y = training_set(condition, backbone, manifest, feats, seed)
                C, clf = fit_with_validation(X, y, X_val, y_val)
                for split, Xs, ys in [("val", X_val, y_val), ("test", X_test, y_test)]:
                    pred = clf.predict(Xs)
                    runs.append({"backbone": backbone, "condition": condition, "seed": seed,
                                 "split": split, "C": C, "n_train": len(y), **scores(ys, pred)})
                    if split == "test":
                        cm += confusion_matrix(ys, pred, labels=BREEDS)
            confusion[(backbone, condition)] = cm
    return pd.DataFrame(runs), confusion


def summarize(runs: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    metrics = ["mau_recall", "mau_precision", "mau_f1", "macro_f1", "accuracy"]
    g = runs[runs.split == split].groupby(["backbone", "condition"], sort=False)[metrics]
    mean, std = g.mean(), g.std(ddof=1).fillna(0)
    return mean.map("{:.3f}".format) + " ± " + std.map("{:.3f}".format)
