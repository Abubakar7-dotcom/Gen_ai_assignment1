"""Log the submitted checkpoints to W&B as model artifacts, attached to the training run that produced each one.

    python -m scripts.wandb_log_checkpoints            # needs `wandb login` and the files in checkpoints/

Runs trained before Tracker.checkpoint existed did not upload their checkpoints; this adds them afterwards
without retraining. The run id comes from last.pt (ae/cls/gan engines save it); T3 is found by its run name.
"""
import argparse

import wandb

from src.loaders import CKPT
from src.utils import ROOT, load_checkpoint, load_json

PROJECT = "genai-a1"


def find_run_id(key: str, path, api: wandb.Api, entity: str) -> tuple[str | None, str]:
    last = (ROOT / path).parent / "last.pt"
    run_name = load_json((ROOT / path).parent / "summary.json")["config"]["run_name"]
    ck = load_checkpoint(last) if last.exists() else None
    if ck and ck.get("run_id"):
        return ck["run_id"], run_name
    runs = [r for r in api.runs(f"{entity}/{PROJECT}", filters={"display_name": run_name}, order="-created_at")]
    return (runs[0].id if runs else None), run_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of keys, e.g. t1 t3")
    args = ap.parse_args()
    api = wandb.Api()
    entity = api.default_entity
    for key, path in CKPT.items():
        if args.only and key not in args.only:
            continue
        ck = load_checkpoint(ROOT / path)
        if ck is None:
            print(f"skip {key}: {path} missing")
            continue
        run_id, run_name = find_run_id(key, path, api, entity)
        run = wandb.init(project=PROJECT, id=run_id, resume="must" if run_id else None,
                         name=None if run_id else f"{run_name}-checkpoint", job_type=None if run_id else "upload")
        art = wandb.Artifact(run_name, type="model", metadata={"task": key, "epoch": ck.get("epoch"), "val": ck.get("val")})
        art.add_file(str(ROOT / path))
        if key == "t3":                                    # joint fine-tuning end state, kept for comparison
            art.add_file(str((ROOT / path).parent / "last.pt"))
        run.log_artifact(art, aliases=["best", "submitted"])
        run.finish()
        print(f"logged {key} -> run {run.id} artifact {run_name}")


if __name__ == "__main__":
    main()
