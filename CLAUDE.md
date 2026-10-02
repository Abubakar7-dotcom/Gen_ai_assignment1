# CLAUDE.md — GenAI Assignment 1 (Image Restoration + Face-to-Sketch)

Individual university assignment (FAST-NUCES, Generative AI). Four related generative models, served through ONE browser app (React + FastAPI + ONNX) running via Docker Compose, plus an IEEE LaTeX report and a 5–7 min YouTube demo.

**Time budget: 2 days.** Prefer working, verifiable, simple solutions over clever ones. Every component must be explainable by the student in a viva (they may be asked to modify code live) — keep code readable, commented where non-obvious, no magic.

---

## Working rules for Claude Code

1. **Commit real work incrementally** with Conventional Commits (`feat(t1): ...`, `fix(data): ...`, `docs(report): ...`). One logical milestone per commit. **Never backdate commits or rewrite history.** Never commit datasets, checkpoints, `.onnx` files, `mlruns/`, `wandb/`, or `.env`.
2. **One corruption implementation** (`src/data/corruptions.py`, numpy/OpenCV). Training, manifests, evaluation AND the FastAPI backend all import it. Never reimplement corruption logic elsewhere.
3. **Spec values below are mandatory** — do not "improve" them (seed 42, 80/20, 128×128, corruption ranges, test severities, workspace names, endpoint set).
4. Every training script must: log to the tracker (W&B or MLflow), checkpoint every epoch, be **resumable** (Kaggle/Colab sessions die), and accept a `--config` YAML.
5. Every Optuna study: persist to SQLite (`optuna_studies/<task>.db`), use `MedianPruner`, save `best_params.json`, and export history + param-importance plots to `results/<task>/`.
6. Before marking any model done: ONNX export + PyTorch-vs-ONNXRuntime parity check (max abs diff < 1e-4) recorded in `results/onnx_parity.csv`.
7. Log design decisions + alternatives considered in `docs/decisions.md` as they happen (feeds the report's research component). Log AI-tool usage in `docs/ai_use.md` (required appendix).
8. After changes, run the relevant smoke test (`pytest -q tests/` or a 1-batch dry run with `--smoke`) before claiming it works.

---

## Repository layout (target)

```
configs/              # YAML per task (t1.yaml, t2_cls.yaml, t2_spec.yaml, t3.yaml, t4.yaml)
src/
  data/
    corruptions.py    # clean / salt_pepper / blur / occlusion — params in, image out (deterministic given seed)
    pets.py           # Oxford-IIIT Pet dataset, split, cached 128px tensors, runtime corruption
    sampler.py        # balanced batch sampler (exactly 1/4 per condition per batch)
    manifests.py      # build val/test manifests (JSON)
    fs2k.py           # FS2K paired loader, paired augmentation, stratified split
  models/
    autoencoder.py    # shared DAE (T1 universal, T2 specialists)
    classifier.py     # corruption classifier (T2, T3 gate init)
    moe.py            # soft MoE wrapper (T3)
    pix2pix.py        # UNet generator + PatchGAN discriminator with style embedding (T4)
  losses.py           # L1+SSIM, MoE loss, balance loss, GAN losses
  metrics.py          # PSNR, SSIM, classification metrics
  engine/             # ae.py (T1 + T2 specialists), cls.py, moe.py, gan.py — run(cfg, trial=None) -> best score
  loaders.py          # load best checkpoints; HardRouter (T2 inference)
train/train.py        # CLI: python -m train.train --task t1|t2_cls|t2_spec|t3|t4 [--specialist X] [--smoke]
optuna_studies/tune.py# CLI: python -m optuna_studies.tune --task ... (writes configs/<task>_best.yaml)
eval/evaluate.py      # CLI: python -m eval.evaluate --task t1|t2|t3|t4  -> results/<task>/
export/export_onnx.py # export all models + parity -> models_onnx/, results/onnx_parity.csv
backend/              # FastAPI app, Dockerfile
frontend/             # React + Vite + Tailwind, Dockerfile (nginx)
scripts/              # prepare_data.py, make_samples.py, download_models.{sh,ps1}, gpu_chain_*.ps1
results/              # metrics CSVs, figures (committed — small PNG/CSV only)
docs/                 # decisions.md, ai_use.md, stitch/ (design screenshots)
report/               # IEEE LaTeX (IEEEtran)
tests/
docker-compose.yml
README.md
```

---

## Datasets

### Oxford-IIIT Pet (Tasks 1–3)
- `torchvision.datasets.OxfordIIITPet(split="trainval")` = development; `split="test"` = final test **only** (never touch during training/tuning).
- Dev split: 80% train / 20% val, **random seed 42**. Save IDs to `data/splits/pets_split.json`. Same split for T1–T3.
- Convert to RGB, resize to **128×128**. Caching clean resized images is fine; **do not save corrupted copies** of the training set.
- Labels (breed) are unused. Clean image = target.

### Corruptions (exact spec)
Training: each loaded image gets one of 4 conditions with **equal probability**, resampled on every load:

| Condition | Training config |
|---|---|
| clean | unchanged |
| salt_pepper | p ~ U(0.02, 0.15); selected pixels set to black or white with prob 0.5 each |
| blur | kernel ∈ {3,5,7}; σ ~ U(0.5, 2.5) |
| occlusion | 1–3 black rectangles, random positions, **union** area 10%–35% of image (resample until satisfied) |

Label index order everywhere: `0=clean, 1=salt_pepper, 2=blur, 3=occlusion` (matches MoE weights w0..w3).

Val + test: **deterministic manifests** (`data/manifests/val.json`, `test.json`) storing per item: image id, type, severity, all params (p / kernel+σ / rect coords), seed.

Test severities (each test image × each corruption × 3 levels, plus clean):
- salt_pepper p: 0.03 / 0.08 / 0.15 (low / med / high)
- blur (k, σ): (3,0.7) / (5,1.5) / (7,2.5)
- occlusion: ~10% with 1 rect / ~20% with 2 rects / ~35% with 3 rects

### FS2K (Task 4)
- 2,104 paired photo/sketch, 3 sketch styles. Use **official train/test** definitions (`anno_train.json` / `anno_test.json`).
- 15% of official train → val, **seed 42, stratified by style**. Official test never used for training or tuning.
- Resize both to 128×128. Known issues: some photos are RGBA/grayscale → convert to RGB; verify every photo↔sketch pairing with a script before training.
- Spatial augmentation must be **identical for photo and sketch** (shared random params).

---

## Task 1 — Universal Denoising Autoencoder
- Conv encoder (spatial ↓, channels ↑) → **real compressed bottleneck** (latent dim tunable) → conv decoder → RGB 128×128 (sigmoid).
- No unrestricted skip connections. Baseline: no skips. If a limited skip is tried, it's an ablation that must be justified in `docs/decisions.md`.
- Loss: `L = α·L1 + (1−α)·(1−SSIM)`, start α=0.8 (use `pytorch_msssim`).
- Optuna: lr, batch size, bottleneck dim, encoder channels, dropout, α. Objective combines PSNR + SSIM on val (document the exact formula). Report search space, #trials, best trial, final config.
- Eval: metrics per condition (clean/sp/blur/occ) **and** per severity (low/med/high). Figures: ≥12 examples (clean | corrupted | output | abs error map), 4 failure cases.
- App workspace: **"Universal Restoration"** — upload corrupted image OR pick clean sample + apply corruption; show input, output, corruption settings, inference time.

## Task 2 — Classifier + Hard-Routed Specialists
- CNN classifier, 4 classes, cross-entropy, **balanced batches**. Optuna: lr, batch size, channel config, dropout, weight decay. Report accuracy, macro P/R/F1, per-class metrics, normalized 4×4 confusion matrix.
- 3 specialist AEs (salt_pepper, blur, occlusion), each trained **only** on its corruption, same clean target, independent weights. One shared Optuna search for architecture (lr, bottleneck, channels, batch size, L1/SSIM weight), then train all three with it.
- Inference: `r = argmax C(x̃)`; clean → **identity bypass**; else specialist.
- Evaluate in **oracle-routing** (manifest label) and **predicted-routing** mode. Identify + discuss misrouting-caused failures (expect clean vs mild blur confusion).
- Workspace: **"Hard-Routed Restoration"** — 4 probabilities, predicted corruption, selected expert, output, inference time. Export classifier + 3 specialists to ONNX.

## Task 3 — Soft Mixture-of-Experts
- Branches: identity (clean), A_salt, A_blur, A_occ. Gate `w = softmax(G(x̃)/τ)`; `x̂ = w0·x̃ + w1·A_salt + w2·A_blur + w3·A_occ`.
- Init gate from T2 classifier, experts from T2 specialists. **Warm-up**: experts frozen, train gate only. Then unfreeze all, smaller lr, joint fine-tune.
- Loss: `λ1·L1 + λs·(1−SSIM) + λc·CE(gate, label) + λb·L_balance`, start λ1=0.8, λs=0.2, λc=0.1, λb=0.01. `L_balance = Σ_k (mean_batch(w_k) − 1/4)²` (or a justified entropy alternative).
- Optuna: joint lr, τ, λc, λb, reconstruction weighting. Prune trials with routing collapse (one branch mean weight > ~0.9 across all classes).
- Analysis: mean expert weights per true corruption × severity (heatmap), examples of dominant vs distributed routing, check for dead/dominant experts.
- Workspace: **"Soft Mixture-of-Experts Restoration"** — all 4 weights, output, inference time, visual indicator of top contributors. Export the **whole MoE pipeline as one ONNX graph** (outputs: image + weights).

## Task 4 — Style-Conditioned Face-to-Sketch cGAN
- Generator: U-Net (pix2pix-style) `ŷ = G(x, s)`; style s = **learned nn.Embedding(3, d)** injected into generator (tiled channels and/or bottleneck) **and** discriminator.
- Discriminator: PatchGAN on (photo, sketch, style).
- Losses: BCE-with-logits adversarial; `L_G = L_adv + λ_L1·L1(y, G(x,s))`, start λ_L1=100.
- Optuna (short trials, fewer epochs): G lr, D lr, batch size, base channels, dropout, style-embedding dim, λ_L1. Select by val L1/SSIM, then retrain best for full schedule.
- Log separately: D real loss, D fake loss, G adv loss, G L1 loss, val metrics. Log sample grids from a **fixed** val photo set every N epochs.
- Workspace: **"Face-to-Sketch Generator"** — upload or webcam, choose Style 1/2/3, side-by-side photo + sketch, download button. Only generator exported to ONNX.

---

## Application
- Design first in **Google Stitch**; save screenshots to `docs/stitch/` (report evidence).
- Frontend: React + Tailwind, one app, 4 workspaces (exact names above), responsive.
- Backend: FastAPI + onnxruntime (CPU). Validates uploads (type/size), preprocesses (RGB, 128×128), runs ONNX, returns base64 image + timing + probs/weights.
- Endpoints (minimum): `GET /health`, `POST /corrupt`, `POST /restore/universal`, `POST /restore/hard`, `POST /restore/moe`, `POST /sketch`.
- Models are NOT in git: `scripts/download_models.sh` pulls `.onnx` from GitHub Releases / Hugging Face (avoid Git LFS bandwidth limits).
- `docker compose up --build` must start everything from a fresh clone; app reachable in browser. Test from a clean clone before submission.

## Tracking & deliverables
- Tracker: W&B (or MLflow) for every run: hyperparams, losses, metrics, checkpoints, sample images.
- README: setup, data download, training/eval/export commands, one-command Docker start, model links.
- Report (IEEE LaTeX, `report/`): intro, related work, dataset prep + full corruption config, architectures (diagrams), losses, training, Optuna design + results, results tables, curves, confusion matrix, routing heatmaps, image grids, error maps, failure cases, app architecture + screenshots + Stitch evidence, ONNX parity, limitations, conclusion, repo link, YouTube link, AI-use appendix. Every figure/table must be interpreted in text.
- Demo video (5–7 min, YouTube): startup, upload, runtime corruption, universal restoration, hard routing, soft weights, face-to-sketch, download, tracking dashboard.

---

## Compute notes — two machines
- **DEV machine (no GPU):** write code, CPU smoke tests (`--smoke`: 2 batches, 1 epoch), data pipeline, backend, frontend, Docker, report.
- **GPU machine:** runs Optuna studies, full training, evaluation, ONNX export. GPU: **RTX 4050 (6 GB VRAM, likely laptop)**. OS: **Windows** (roommate's PC). Claude Code runs **locally on that PC** in a clone of this repo and follows `docs/gpu_runbook.md`; the student steers that session from the DEV PC. No SSH/Tailscale.
  - Use AMP (`torch.autocast` + `GradScaler`), `cudnn.benchmark=True`.
  - Optuna batch-size caps: AEs/classifier ≤ 128, MoE ≤ 64, cGAN ≤ 32. Catch CUDA OOM inside objectives → `raise optuna.TrialPruned()`.
  - Data loading is the likely bottleneck (laptop CPU): cache clean 128px images as a uint8 `.npy`, `num_workers=2–4`, `pin_memory=True`, `persistent_workers=True`.
  - Max 2 concurrent training processes; check `nvidia-smi` memory before starting a second.
  - Laptop must be plugged in, sleep disabled, lid-close action = "Do nothing".
- **Sync = git only.** Code flows DEV → GitHub → GPU machine (`git pull`). Never edit code directly on the GPU machine without committing it back.
- **Large files never go through git:** checkpoints/ONNX go to `checkpoints/` + `models_onnx/` (gitignored) and are uploaded to Google Drive / Hugging Face Hub; `scripts/download_models.sh` fetches them.
- **Small results DO go through git:** `results/**` (CSV, PNG), `optuna_studies/*.db`, `best_params.json` — commit from the GPU machine after each run.
- **Commit identity on the GPU machine:** set repo-local `git config user.name/user.email` to the student's identity (individual assignment).
- **Tracking: use Weights & Biases** (cloud) so runs on the GPU machine are visible from the DEV machine.
- Long jobs on the GPU machine run as background tasks or in a persistent terminal; all scripts resumable from last checkpoint.
- Windows GPU machine: every training script needs `if __name__ == "__main__":` guard (DataLoader workers use spawn); start with `num_workers=2`.
- Every script takes `--device auto` (cuda if available else cpu) and `--data-root`; no hardcoded paths.
- Budget is tight: Optuna 10–15 trials, few epochs, subset of train during search, then full retrain of best config. State this trade-off in the report.
- Run T4 (longest) on the GPU as early as possible.

---

## Progress checklist (update as work completes)

**Day 1**
- [x] Repo scaffold, requirements, .gitignore
- [ ] Stitch design + screenshots (student)
- [x] Corruptions + balanced sampler + manifests code (tests pass)
- [ ] Pets split/cache/manifests GENERATED (needs annotations + images) + sanity grid
- [x] FS2K loader + pairing check code
- [ ] FS2K cache + split GENERATED (needs dataset)
- [x] All models, engines, Optuna, eval, ONNX export code (CPU smoke-tested)
- [ ] T4 Optuna → full train (background)
- [ ] T1 Optuna → full train
- [ ] T2 classifier Optuna → train; specialists shared Optuna → train ×3
- [ ] T3 warm-up + joint fine-tune + Optuna
- [x] FastAPI backend (all endpoints, tested with smoke ONNX models)
- [ ] ONNX export + parity for TRAINED models
- [ ] Eval scripts + all figures/tables in `results/`

**Day 2**
- [x] React frontend, 4 workspaces (tested against smoke backend)
- [x] Dockerfiles + compose + model download scripts
- [ ] Fresh-clone `docker compose up --build` test (not possible in dev sandbox: Docker Hub blocked)
- [ ] README
- [ ] IEEE report
- [ ] Demo video + link
- [ ] Final push + submit

## Handoff — which machine are you on?
- Check with `nvidia-smi`. **If it shows the RTX 4050, you are on the GPU PC: follow `docs/gpu_runbook.md` stage by stage.**
- Otherwise you are on the DEV PC (no GPU): app, Docker, report; never start real training here.
- Nothing is trained yet; the code has only run on synthetic images (`--smoke`). `scripts/setup_gpu_pc.ps1` and `docker compose up` have never run on a real machine.

## Open decisions / notes
- Smoke check of everything on CPU: `pytest -q tests/` and `python -m train.train --task <t> --smoke`.
- GPU order: start `scripts/gpu_chain_gan.ps1` first (longest), then `scripts/gpu_chain_restoration.ps1` in a 2nd terminal.
- After training: `python -m export.export_onnx`, `python -m scripts.make_samples`, upload `models_onnx/*.onnx`
  to a GitHub Release `models-v1`, set the URL in `scripts/download_models.*`.
