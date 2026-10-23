"""Meeting 1: the two classical ways to add Egyptian Mau training examples.

Both functions are yours to write; `uv run synth-train --method oversample` or
`--method augment` uses them.
"""

import os
import random

from PIL import Image
from torchvision import transforms


def oversample(
    minority_photos: list[str], budget: int, rng: random.Random
) -> list[str]:
    """Choose which Egyptian Mau training photos to repeat.

    Return a list of exactly `budget` names taken from `minority_photos`; names
    may repeat. Repeat the photos as evenly as possible and use `rng` for every
    random choice, so a run can be reproduced from its seed.
    """
    if not minority_photos:
        raise ValueError("there are no photos to repeat")
    # 55 = 3 * 15 + 10: every photo is repeated 3 times, and 10 photos chosen at
    # random get a 4th copy, so no photo is used more than once more than another.
    full_rounds, remainder = divmod(budget, len(minority_photos))
    chosen = list(minority_photos) * full_rounds + rng.sample(
        minority_photos, remainder
    )
    rng.shuffle(chosen)
    return chosen


# Every training image, added or not, later also goes through train.py's light
# TRAIN_TRANSFORM (crop of 75-100 % of the area, flip, small colour jitter), so
# augment() adds clearly stronger changes on top of it. The size stays the same.
#
# "moderate" (the default) keeps the cat and its coat clearly recognisable.
# "strong" is the variant tried once: tighter crops, larger rotations and colour
# changes, and sometimes greyscale. Choose it with SYNTH_AUGMENT=strong.
AUGMENTATIONS = {
    "moderate": {
        "crop_scale": (0.5, 1.0),
        "degrees": 15,
        "jitter": (0.3, 0.3, 0.3, 0.05),
        "grayscale": 0.0,
    },
    "strong": {
        "crop_scale": (0.25, 1.0),
        "degrees": 30,
        "jitter": (0.5, 0.5, 0.5, 0.15),
        "grayscale": 0.2,
    },
}
AUGMENTATION = os.environ.get("SYNTH_AUGMENT", "moderate")


def augment(image: Image.Image) -> Image.Image:
    """Return a randomly transformed copy of one Egyptian Mau training photo.

    The training script calls this every time an added copy is loaded, so each
    epoch sees a new variant. Typical choices are crops, flips, rotations, and
    colour changes from `torchvision.transforms`. Keep the cat recognisable.
    """
    settings = AUGMENTATIONS[AUGMENTATION]
    brightness, contrast, saturation, hue = settings["jitter"]
    transform = transforms.Compose(
        [
            transforms.RandomRotation(settings["degrees"]),
            # (height, width): the crop is resized back to the original size
            transforms.RandomResizedCrop(
                image.size[::-1], scale=settings["crop_scale"]
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness, contrast, saturation, hue),
            transforms.RandomGrayscale(settings["grayscale"]),
        ]
    )
    return transform(image)
