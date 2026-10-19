# Meeting 1 — Dataset protocol and real-data baselines

**Research question.** Can diffusion-generated images improve recognition of an underrepresented class
beyond oversampling and ordinary augmentation, without harming overall classification quality?

**Case study.** Eight visually similar cat breeds from Oxford-IIIT Pet: Abyssinian, Bengal, Birman, Bombay,
Egyptian Mau, Ragdoll, Russian Blue, Siamese. **Egyptian Mau is the minority class.**

Code: `src/catbench/` · Notebook: [`meeting1.ipynb`](meeting1.ipynb) · Results: `results/meeting1/`

---

## 1. Data and split

* **Source.** All 8 cat breeds from Oxford-IIIT Pet (`torchvision.datasets.OxfordIIITPet`), with the dataset's official
  `trainval` and `test` lists pooled into one pool of 1,571 images (184–200 per breed). We re-split because the official split is
  not imbalanced and has no validation set.
* **Near-duplicate removal (before splitting).** Every image gets a 64-bit perceptual difference hash (dHash).
  Two images of the **same breed** whose hashes differ in ≤ 6 bits are treated as copies, and only the first (by image ID)
  is kept. 18 images were removed (`role = excluded_duplicate`). We inspected all of them visually: they are the same
  photo, re-cropped or re-coloured (8 of them are Bombay). Comparing across breeds produced only false positives, which
  is why matching is restricted to the same breed.
* **One split, one seed.** Per breed, images are shuffled with `SPLIT_SEED = 2026` and assigned in this order:
  test → validation → train → unused. The result is saved once in
  [`manifests/split_manifest.csv`](../../manifests/split_manifest.csv), with one row per image: ID, breed, role, official split,
  dHash, and the duplicate it was removed for. The file is committed to Git and never regenerated, so every machine uses
  the same split.

| Data role | Majority breeds (each) | Egyptian Mau |
|---|---:|---:|
| Train | 70 | **15** |
| Validation | 20 | 20 |
| Test (locked) | 30 | 30 |

Class-count table (from the manifest):

| breed        | train | val | test | unused | excluded_duplicate |
|:-------------|------:|----:|-----:|-------:|-------------------:|
| Abyssinian   |    70 |  20 |   30 |     72 |                  6 |
| Bengal       |    70 |  20 |   30 |     80 |                  0 |
| Birman       |    70 |  20 |   30 |     79 |                  1 |
| Bombay       |    70 |  20 |   30 |     56 |                  8 |
| Egyptian Mau |    15 |  20 |   30 |    123 |                  2 |
| Ragdoll      |    70 |  20 |   30 |     79 |                  1 |
| Russian Blue |    70 |  20 |   30 |     80 |                  0 |
| Siamese      |    70 |  20 |   30 |     79 |                  0 |
| **Total**    |   505 | 160 |  240 |    648 |                 18 |

**Why these numbers.** They follow the suggested laptop-scale protocol: 70 vs 15 gives a clear 4.7 : 1 imbalance that
exists **only in training**. Validation and test are balanced, so the metrics are not driven by class frequency.
30 test images per class is the most every breed can afford: Bombay, the smallest breed, still has 56 unused images.
The extra Egyptian Mau images stay `unused` and are never trained on, so the imbalance is real.

**Test set is locked.** No choice is made on the test set. The only tuned setting in this meeting, the regularisation
strength `C`, is chosen on validation. Test metrics are computed and reported, but they never feed back into a decision.

## 2. Leakage check (automated)

`catbench.leakage.check_split` runs on every pipeline run and raises an error if any check fails:

1. Image IDs are unique, so every image has exactly one role.
2. Only known roles are present.
3. Train counts are exactly 70 / 15; validation and test are balanced (20 / 30 per class).
4. No same-breed near-duplicate pair (dHash ≤ 6 bits) is split across train / val / test.

A second check, `check_added_sources`, runs each time a training set is built. Every oversampled or augmented
observation must come from an Egyptian Mau **training** image, so no transformed copy of a validation or test image can
enter training.

Both checks were tested with planted leaks. Moving a duplicate's twin into train and the duplicate into test is caught
(`Abyssinian_122(train)~Abyssinian_172(test)`), and so is adding a test image as an augmentation source.

## 3. Added-data budget

**Budget = 55 added Egyptian Mau observations** (= 70 − 15), so Egyptian Mau ends at 70 training observations,
the same as every other breed, in every condition except the imbalanced control.

| Condition | Training data |
|---|---|
| Imbalanced real | 505 real images (15 Mau). Nothing added. |
| Oversampling | + 55 exact repeats of the 15 Mau training images |
| Augmentation | + 55 transformed copies of the 15 Mau training images |

As the assignment requires, this budget is fixed: any added-data condition, including synthetic data, adds exactly 55
Egyptian Mau observations.

The 15 source images are cycled in a seed-dependent order, so each is used 3 or 4 times. **Oversampling and augmentation
use the same allocation for a given seed**; the only difference between them is the transform.

Augmentation ("ordinary", label-preserving): `RandomResizedCrop(224, scale 0.5–1.0)`, horizontal flip,
rotation ±15°, `ColorJitter(brightness 0.3, contrast 0.3, saturation 0.3, hue 0.05)`.

## 4. Classifier and training procedure (identical for all conditions)

* **Representation:** frozen ImageNet-pretrained backbone, with the classification head removed.
  * **ResNet-18** (512-d), the main model.
  * **EfficientNet-B0** (1280-d), a second compact architecture for checking whether conclusions transfer.
  * Real images: resize 256 → centre-crop 224 → ImageNet normalisation. Features are cached in `data/features/`.
* **Classifier:** `StandardScaler` + multinomial `LogisticRegression` (scikit-learn, L2).
  No class weighting; the conditions differ **only** in the training data.
* **Hyper-parameter choice:** `C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}`, chosen by **validation macro-F1**
  separately for every condition and seed (ties → smaller C).
* **Why frozen features + logistic regression:** fast (the whole Meeting 1 run takes a few minutes, even on CPU),
  deterministic, and strong enough for fine-grained breeds.

## 5. Metrics

* **Primary:** Egyptian Mau **recall** on the test set.
* **Guardrail:** **macro-F1 over all 8 classes**. An intervention must not lower it noticeably.
* Also reported: Egyptian Mau precision and F1, accuracy, and row-normalised confusion matrices.

## 6. Random seeds

| What | Seed(s) |
|---|---|
| Split (once) | `SPLIT_SEED = 2026` |
| Repeated runs | `RUN_SEEDS = [0, 1, 2, 3, 4]` (oversampling allocation, augmentation draws via `torch.manual_seed`) |

Logistic regression (lbfgs) is deterministic, so the imbalanced condition has std = 0 across seeds: nothing in it is random.
All constants are in [`src/catbench/config.py`](../../src/catbench/config.py). Rerunning the notebook with the default install
reproduces the tables below exactly.

## 7. Results (5 seeds, mean ± std)

**Test set**

| Backbone | Condition | Mau recall | Mau precision | Mau F1 | Macro-F1 | Accuracy |
|---|---|---|---|---|---|---|
| ResNet-18 | Imbalanced | 0.367 ± 0.000 | 0.917 ± 0.000 | 0.524 ± 0.000 | 0.844 ± 0.000 | 0.854 ± 0.000 |
| ResNet-18 | Oversampling | **0.580 ± 0.045** | 0.945 ± 0.004 | 0.718 ± 0.036 | **0.877 ± 0.006** | 0.880 ± 0.005 |
| ResNet-18 | Augmentation | 0.473 ± 0.015 | 0.934 ± 0.002 | 0.628 ± 0.013 | 0.857 ± 0.004 | 0.863 ± 0.004 |
| EfficientNet-B0 | Imbalanced | 0.567 ± 0.000 | 0.944 ± 0.000 | 0.708 ± 0.000 | 0.862 ± 0.000 | 0.863 ± 0.000 |
| EfficientNet-B0 | Oversampling | **0.653 ± 0.030** | 0.876 ± 0.014 | 0.748 ± 0.016 | 0.863 ± 0.000 | 0.863 ± 0.000 |
| EfficientNet-B0 | Augmentation | 0.620 ± 0.018 | 0.912 ± 0.021 | 0.738 ± 0.018 | **0.866 ± 0.004** | 0.866 ± 0.003 |

**Validation set** (where the choices are made)

| Backbone | Condition | Mau recall | Macro-F1 |
|---|---|---|---|
| ResNet-18 | Imbalanced | 0.600 ± 0.000 | 0.868 ± 0.000 |
| ResNet-18 | Oversampling | 0.650 ± 0.000 | 0.869 ± 0.000 |
| ResNet-18 | Augmentation | 0.650 ± 0.000 | 0.880 ± 0.000 |
| EfficientNet-B0 | Imbalanced | 0.650 ± 0.000 | 0.904 ± 0.000 |
| EfficientNet-B0 | Oversampling | 0.800 ± 0.000 | 0.918 ± 0.001 |
| EfficientNet-B0 | Augmentation | 0.820 ± 0.045 | 0.932 ± 0.007 |

### Observations

**How precise are these numbers?** There are only 30 Egyptian Mau test images, so one image changes Mau recall by
3.3 points (5 points on validation, with 20 images). The "± std" above only measures variation between the 5 seeds, that
is, which images were repeated or how they were augmented. It does **not** include the uncertainty that comes from having
so few test images, which is much larger. No statistical tests were run, so a difference of a few test images is read
here as "no clear difference".

1. **The imbalanced baseline recognises Egyptian Mau poorly.** With 15 training images, ResNet-18 finds 11 of the 30 Mau
   test images (recall 0.367), while every other breed reaches 0.83–1.00. It predicts "Egyptian Mau" only about 12 times in
   the whole test set, and those predictions are almost always right (precision 0.917): the model is reluctant to choose
   the minority class, rather than confusing other breeds with it. EfficientNet-B0 does better (17 of 30) but shows the
   same gap. These runs cannot tell how much of this is caused by the small training count and how much by Mau simply
   being a hard breed, because there is no run with 70 real Mau images to compare against.
2. **Most mistakes on Egyptian Mau are "Bengal".** With the imbalanced ResNet-18, 15 of the 30 Mau test images are
   predicted as Bengal, the other spotted breed. EfficientNet-B0 makes the same mistake: 12 of its 13 missed Mau images
   are predicted as Bengal.
3. **Oversampling shows the clearest gain.** On ResNet-18, Mau recall rises from 0.367 to 0.580: 16–19 of 30 images
   found depending on the seed, instead of 11, so every seed is clearly above the baseline. The other gains are small:
   augmentation on ResNet-18 finds 3–4 more images, and on EfficientNet-B0 oversampling finds 1–3 more and augmentation
   1–2 more. These are within the few-image range described above, so for them we can only say "a possible improvement".
4. **Oversampling and augmentation cannot be ranked against each other.** On the test set oversampling is ahead on both backbones,
   while on validation augmentation is equal or ahead. The two disagree, and the differences are a few images either way.
5. **No sign that either method harms overall quality.** Macro-F1 is the same or slightly higher than the imbalanced
   baseline in every case (ResNet-18: 0.844 → 0.877 / 0.857; EfficientNet-B0: 0.862 → 0.863 / 0.866).

### Summary

* The real-image split is fixed in a manifest and passes the automated leakage check.
* All three baselines use the same split, budget (55), classifier and metrics, and differ only in the added training data.
* Egyptian Mau is the weakest class in the imbalanced baseline (recall 0.367 with ResNet-18, 0.567 with EfficientNet-B0),
  and its errors are mostly "Bengal".
* Oversampling clearly improves Mau recall with ResNet-18; the other improvements are a few images and not clearly
  different from the baseline.
* Macro-F1 did not drop in any condition.
