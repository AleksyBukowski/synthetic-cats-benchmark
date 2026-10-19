"""Frozen ImageNet backbones -> cached feature vectors, plus the minority augmentation."""
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms as T
from tqdm.auto import tqdm

from .config import ADDED_BUDGET, FEATURE_DIR, MINORITY
from .data import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Deterministic preprocessing for every real (and later synthetic) image.
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORMALIZE])

# "Ordinary augmentation": label-preserving geometric + photometric changes.
AUGMENT = T.Compose([
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
])
AUG_TF = T.Compose([T.ToTensor(), NORMALIZE])   # applied after AUGMENT


def get_backbone(name: str) -> nn.Module:
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()                                   # 512-d features
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = nn.Identity()                           # 1280-d features
    else:
        raise ValueError(name)
    return model.eval().to(DEVICE)


@torch.no_grad()
def embed(model: nn.Module, tensors: list[torch.Tensor], batch_size: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(tensors), batch_size):
        out.append(model(torch.stack(tensors[i:i + batch_size]).to(DEVICE)).cpu())
    return torch.cat(out).numpy()


def real_features(backbone: str, manifest) -> dict[str, np.ndarray]:
    """Features of every train/val/test image, cached in data/features/."""
    path = FEATURE_DIR / f"{backbone}_real.npz"
    used = manifest[manifest.role.isin(["train", "val", "test"])]
    if path.exists():
        cached = np.load(path)
        if list(cached["ids"]) == list(used.image_id):
            return dict(zip(cached["ids"], cached["feats"]))
    model = get_backbone(backbone)
    tensors = [EVAL_TF(load_image(p)) for p in tqdm(used.path, desc=f"{backbone} real")]
    feats = embed(model, tensors)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=used.image_id.to_numpy(dtype=str), feats=feats)
    return dict(zip(used.image_id, feats))


def minority_sources(manifest, seed: int) -> list[str]:
    """Which minority training image each of the ADDED_BUDGET extra observations comes from.

    Images are cycled in a seed-dependent order, so each one is used 3 or 4 times.
    Oversampling and augmentation share this allocation; they differ only in the transform.
    """
    ids = manifest.image_id[(manifest.role == "train") & (manifest.breed == MINORITY)].to_numpy()
    order = np.random.default_rng(seed).permutation(ids)
    return list(np.resize(order, ADDED_BUDGET))


def augmented_images(manifest, seed: int):
    """The ADDED_BUDGET augmented minority images for one seed, as (source_id, PIL image)."""
    paths = dict(zip(manifest.image_id, manifest.path))
    torch.manual_seed(seed)
    return [(src, AUGMENT(load_image(paths[src]))) for src in minority_sources(manifest, seed)]


def augmented_features(backbone: str, manifest, seed: int) -> tuple[list[str], np.ndarray]:
    path = FEATURE_DIR / f"{backbone}_aug_seed{seed}.npz"
    if path.exists():
        cached = np.load(path)
        return list(cached["sources"]), cached["feats"]
    pairs = augmented_images(manifest, seed)
    sources = [s for s, _ in pairs]
    feats = embed(get_backbone(backbone), [AUG_TF(img) for _, img in pairs])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sources=np.array(sources, dtype=str), feats=feats)
    return sources, feats
