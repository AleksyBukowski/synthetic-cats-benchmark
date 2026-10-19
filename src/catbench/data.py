"""Download Oxford-IIIT Pet and list the images of the eight cat breeds."""
import pandas as pd
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from .config import BREEDS, DATA_DIR

PET_DIR = DATA_DIR / "oxford-iiit-pet"


def download():
    """Download the dataset once (torchvision skips it if already present)."""
    OxfordIIITPet(root=DATA_DIR, split="trainval", download=True)


def list_cat_images() -> pd.DataFrame:
    """One row per cat image: image_id, breed, official_split, path."""
    download()
    rows = []
    for official in ["trainval", "test"]:
        for line in (PET_DIR / "annotations" / f"{official}.txt").read_text().splitlines():
            image_id = line.split()[0]                      # e.g. "Egyptian_Mau_12"
            breed = image_id.rsplit("_", 1)[0].replace("_", " ")
            if breed in BREEDS:
                rows.append({
                    "image_id": image_id,
                    "breed": breed,
                    "official_split": official,
                    "path": f"oxford-iiit-pet/images/{image_id}.jpg",  # relative to data/
                })
    return pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)


def load_image(rel_path: str) -> Image.Image:
    # A few Oxford-IIIT Pet files are grayscale/CMYK/RGBA; normalise everything to RGB.
    return Image.open(DATA_DIR / rel_path).convert("RGB")
