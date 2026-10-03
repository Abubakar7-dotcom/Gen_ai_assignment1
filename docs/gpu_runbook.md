# GPU PC runbook

Instructions for the Claude Code session running **on the GPU PC** (RTX 4050, 6 GB, Windows).
The student steers that session from the DEV PC; all GPU work happens locally on the GPU PC.
Work through the stages in order. Tick the matching box in `CLAUDE.md` after each stage and commit.

## State of the code when this was written

- All code was written on the DEV PC (no GPU) and has only run on **synthetic images** with `--smoke`.
- Nothing is trained. No dataset has been downloaded. No real image has passed through the pipeline.
- `scripts/setup_gpu_pc.ps1` has never been run on a real machine.
- `docker compose up` has never been run.

Expect bugs in every stage the first time it meets real data. Fix them in the code, re-run the smoke
test, commit, and push so the DEV PC stays in sync.

## Rules for this session

- Follow every rule in `CLAUDE.md` (spec values, one corruption module, commit conventions, no history rewrites).
- Never commit `data/raw/`, `data/cache/`, `checkpoints/`, `models_onnx/`, `wandb/`, `.env`.
- Do commit small outputs: `results/**`, `data/splits/*.json`, `data/manifests/*.json`,
  `optuna_studies/*.db`, `configs/*_best.yaml`.
- Before the first commit, check `git config user.name` / `user.email` in this repo are the student's
  (Abubakar). If they are not set, ask the student — do not guess.
- Ask the student before anything that cannot be undone or that the spec does not cover.
- At most 2 training processes at once; check `nvidia-smi` before starting the second.

## Stage 0 — machine setup (student, once)

In PowerShell **as Administrator**, from the repo root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force; .\scripts\setup_gpu_pc.ps1
```

It installs Python 3.11 and Git, disables sleep, creates `.venv`, installs PyTorch with CUDA and
`requirements.txt`, and writes `gpu_setup_report.txt` to the Desktop. The report must contain
`cuda available: True`. If it does not, stop and fix that first.

Every later command runs with the venv active:

```powershell
.venv\Scripts\Activate.ps1
```

## Stage 1 — verify the environment

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
pytest -q tests/
wandb login            # student pastes their API key from https://wandb.ai/authorize
```

All tests must pass before continuing.

## Stage 2 — data

**Oxford-IIIT Pet** (downloads automatically, about 800 MB):

```powershell
python -m scripts.prepare_data --pets
python -m scripts.prepare_data --sanity
```

Check: split sizes printed (80/20 of trainval, seed 42), `data/manifests/val.json` and `test.json`
exist, and `results/data/corruption_sanity_grid.png` shows clean + 3 severities of each corruption.
Open the grid and look at it; confirm the occlusion areas are about 10% / 20% / 35%.

**FS2K** (manual download, student): from https://github.com/DengPingFan/FS2K into `data/raw/FS2K/` so that
these exist:

```
data/raw/FS2K/photo/photo1, photo2, photo3
data/raw/FS2K/sketch/sketch1, sketch2, sketch3
data/raw/FS2K/anno_train.json
data/raw/FS2K/anno_test.json
```

The real archive layout has not been checked against this. If the files unpack differently, move
them to match rather than changing the loader, unless the loader is actually wrong.

```powershell
python -m scripts.prepare_data --fs2k
```

Check `results/data/fs2k_pairing_report.json`: `n_problems` must be 0 for train and test, and the
pair counts must add up to 2,104. Investigate any problem before training.

Commit the splits, manifests, pairing report and sanity grid.

## Stage 3 — one real-data dry run per task

Before the long runs, confirm each task trains for a moment on real data and fits in 6 GB:

```powershell
python -m train.train --task t4 --smoke
python -m train.train --task t1 --smoke
python -m train.train --task t2_cls --smoke
python -m train.train --task t2_spec --specialist blur --smoke
```

`--smoke` was built for synthetic CPU runs; confirm from the code whether it uses the real cache
when one exists, and report to the student what it actually exercised. `t3` needs the T2 checkpoints,
so its dry run comes later.

## Stage 4 — training

Start the GAN chain first (longest), then the restoration chain in a second terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\gpu_chain_gan.ps1
powershell -ExecutionPolicy Bypass -File scripts\gpu_chain_restoration.ps1
```

- Each chain is tune → train → evaluate and is resumable: after a crash, run the same command again.
- Unattended, one chain after the other (use this when someone else is also training on the GPU, or to
  schedule a run): `powershell -ExecutionPolicy Bypass -File scripts\gpu_run_all.ps1`. It uses the venv
  itself, waits for AC power, and logs to `checkpoints\logs\`.
- Run them as background tasks and check on them; do not sit blocked on them.
- Run time on this GPU is unknown. Time the first Optuna trial of each task and tell the student the
  projected total. If it will not fit the deadline, propose fewer trials/epochs and wait for the answer.
- On CUDA out-of-memory, lower the batch-size cap (AEs/classifier ≤ 128, MoE ≤ 64, cGAN ≤ 32).
- Watch for T3 routing collapse (one branch mean weight > 0.9 for every class).

After each task finishes, commit `results/<task>/`, `configs/<task>_best.yaml` and
`optuna_studies/<task>.db`, and push.

## Stage 5 — export and hand back

```powershell
python -m export.export_onnx
python -m scripts.make_samples
```

- `results/onnx_parity.csv` must show max abs diff < 1e-4 for every model. Commit it.
- Upload `models_onnx/*.onnx` to a GitHub Release named `models-v1`. This publishes files on a
  public repo — confirm with the student first. Then set the release URL in
  `scripts/download_models.sh` and `scripts/download_models.ps1`, commit and push.
  - Done by `scripts\publish_models.ps1` (the student runs it; it also uploads the submitted checkpoints
    and SHA256SUMS.txt). The download URL is already set.
- Checkpoints in W&B: runs trained from now on log `best.pt` as a model artifact (`Tracker.checkpoint`).
  For the runs trained before that, run `python -m scripts.wandb_log_checkpoints` once.

## What stays on the DEV PC

Frontend/backend changes, the fresh-clone `docker compose up --build` test, the IEEE report, the
Stitch screenshots and the demo video.
