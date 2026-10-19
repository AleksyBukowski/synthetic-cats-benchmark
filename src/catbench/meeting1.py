"""Meeting 1 end-to-end: manifest -> leakage check -> three real-data baselines.

Run from the project root with:  uv run python -m catbench.meeting1
"""
import json

from .config import BACKBONES, CONDITIONS, RESULTS_DIR, RUN_SEEDS
from .leakage import check_split
from .split import class_count_table, get_manifest
from .train_eval import run_baselines, summarize

OUT = RESULTS_DIR / "meeting1"


def run(backbones=BACKBONES, seeds=RUN_SEEDS):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = get_manifest()

    counts = class_count_table(manifest)
    counts.to_csv(OUT / "class_counts.csv")

    leakage = check_split(manifest)          # raises if anything leaks
    leakage.to_csv(OUT / "leakage_report.csv", index=False)

    runs, confusion = run_baselines(manifest, backbones, CONDITIONS, seeds)
    runs.to_csv(OUT / "runs.csv", index=False)
    for split in ["val", "test"]:
        summarize(runs, split).to_csv(OUT / f"summary_{split}.csv")
    json.dump({f"{b}|{c}": cm.tolist() for (b, c), cm in confusion.items()},
              open(OUT / "confusion_test.json", "w"), indent=1)

    return {"manifest": manifest, "counts": counts, "leakage": leakage,
            "runs": runs, "confusion": confusion}


if __name__ == "__main__":
    res = run()
    print(res["counts"], res["leakage"], sep="\n\n")
    print("\nTEST (mean ± std over seeds)\n", summarize(res["runs"], "test").to_string())
