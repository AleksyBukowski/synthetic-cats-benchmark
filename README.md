# Synthetic Cats Benchmark

University project: *Synthetic Data Generation with Diffusion Models*. Can diffusion-generated images improve
recognition of an underrepresented cat breed (Egyptian Mau) beyond oversampling and ordinary augmentation, without
harming overall classification quality? It uses 8 cat breeds from Oxford-IIIT Pet.

## Setup (any computer: home PC, university lab, Windows or Linux)

The project uses [uv](https://docs.astral.sh/uv/), which creates a local `.venv/` with the exact package versions from
`uv.lock`. It downloads Python 3.12 by itself if the machine doesn't have it, and it needs no admin rights.

```bash
pip install uv
```

```bash
uv sync
```

That's all. PyTorch comes from the CUDA 12.8 build. It uses an NVIDIA GPU when one is present, and otherwise runs on
CPU (Meeting 1 takes a few minutes on CPU).

If `pip` isn't available, install uv with its standalone installer (see the uv website), or use
`python -m pip install --user uv` and then `python -m uv sync`.

## Running

Open a meeting notebook and select the **`.venv`** kernel (in VS Code: *Select Kernel → Python Environments → .venv*):

```bash
uv run jupyter lab
```

Or run a meeting's whole pipeline from the terminal:

```bash
uv run python -m catbench.meeting1
```

The first run downloads Oxford-IIIT Pet (~800 MB) into `data/` and caches backbone features there. `data/` is
git-ignored and re-created automatically. The **split itself is not regenerated**: it is read from the committed
`manifests/split_manifest.csv`, so every machine uses the same split.

## Layout

```
src/catbench/        all code (config, data, split, leakage, features, train_eval, plots, meetingN runners)
manifests/           split_manifest.csv: every image ID and its role (committed)
meetings/meeting1/   protocol.md + meeting1.ipynb (one subfolder per meeting)
results/meeting1/    class counts, leakage report, per-run metrics, summaries
data/                dataset + feature cache (git-ignored)
```

## Meetings

| # | Topic | Protocol |
|---|---|---|
| 1 | Dataset protocol and real-data baselines | [meetings/meeting1/protocol.md](meetings/meeting1/protocol.md) |
| 2 | Text-to-image generation and provenance | — |
| 3 | Embedding space, domain gap, and sample selection | — |
| 4 | Downstream evaluation and ablation | — |
| 5 | Statistical analysis and presentations | — |
