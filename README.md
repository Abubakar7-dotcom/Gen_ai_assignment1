# Restoration Lab: GenAI Assignment 1

[![docker-compose-e2e](https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/actions/workflows/docker.yml/badge.svg)](https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/actions/workflows/docker.yml)
[![report-pdf](https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/actions/workflows/report.yml/badge.svg)](https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/actions/workflows/report.yml)

Four generative models served through one web app:

| Workspace | Task | Model |
|---|---|---|
| Universal Restoration | 1 | One convolutional denoising autoencoder (8×8 spatial bottleneck) for clean, salt-and-pepper, blur and occlusion inputs |
| Hard-Routed Restoration | 2 | Corruption classifier → one of three specialist autoencoders; clean inputs bypass |
| Soft Mixture-of-Experts Restoration | 3 | Gate (initialised from the classifier) blends identity + the three experts; trained jointly |
| Face-to-Sketch Generator | 4 | pix2pix U-Net + PatchGAN with a learned 3-style embedding (FS2K) |

Stack: PyTorch · Optuna · Weights & Biases · ONNX Runtime · FastAPI · React + Tailwind · Docker Compose.

---

## 1. Run the app (evaluators)

Requirements: Git and Docker Desktop (or Docker Engine + Compose v2), and an internet connection for the first start.

```bash
git clone https://github.com/Abubakar7-dotcom/Gen_ai_assignment1.git restoration-lab
cd restoration-lab
docker compose up --build
```

Open **http://localhost:8080** once the log shows `Uvicorn running`. That is the only command needed:
on its first start the backend downloads the seven trained ONNX models (~330 MB) from the GitHub Release into
`./models_onnx`, checks their SHA-256 checksums, and reuses them on every later start. The status pill in the
app's top-right corner shows `API online · 7/7 models` when everything is loaded.
The API is reachable through the same port under `/api` (e.g. `http://localhost:8080/api/health`).
To stop: `Ctrl+C`, then `docker compose down`.

Troubleshooting:
* **Port 8080 is busy**: `APP_PORT=9090 docker compose up --build` (PowerShell: `$env:APP_PORT=9090; docker compose up --build`).
* **No internet on the evaluation machine**: fetch the models beforehand with `bash scripts/download_models.sh`
  (Windows: `powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1`); the containers then start offline.
* **Webcam** (Face-to-Sketch): browsers allow cameras only on `localhost` or HTTPS, so open the app as
  `http://localhost:8080`, not through the machine's IP address.

The same procedure (fresh clone → `docker compose up --build` → end-to-end test of all four workspaces → restart →
test again) runs automatically on GitHub Actions (`.github/workflows/docker.yml`, test script `tests/smoke_app.py`).

Trained models: [GitHub Release `models-v1`](https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/releases/tag/models-v1) (7 ONNX files for the app, plus the PyTorch checkpoints and `SHA256SUMS.txt`). They are not stored in git.

### Results at a glance (official test sets)

| Task | Result |
|---|---|
| 1 Universal DAE | PSNR 22.2 dB / SSIM 0.631; +6.5 dB on salt-and-pepper, +6.8 dB on occlusion; ≈23 dB ceiling from the 8×8 bottleneck |
| 2 Classifier + specialists | accuracy 0.996, macro-F1 0.993; predicted routing equals oracle routing within 0.05 dB on every corruption |
| 3 Soft MoE | gate routes 99.3% of images to the correct branch; no inactive or dominating expert; best occlusion SSIM (0.605) |
| 4 Style cGAN | L1 0.103, SSIM 0.481 (style 1/2/3: 0.52 / 0.40 / 0.61) |
| Ablation (T1, val) | one limited skip connection: PSNR 22.4 → 27.3 dB (not adopted, see `docs/decisions.md`) |

Full tables and figures: `results/`, the report in `report/`, experiment tracking in
[W&B project genai-a1](https://wandb.ai/abubaknaveed-national-university-of-computer-and-emergi/genai-a1).

---

## 2. Reproduce training

```bash
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or /cpu
pip install -r requirements.txt
wandb login
```

### Data
* **Oxford-IIIT Pet**: downloaded automatically by `python -m scripts.prepare_data --pets`.
  This writes `data/splits/pets_split.json` (80/20 split, seed 42), the deterministic manifests
  `data/manifests/{val,test}.json` and a 128×128 cache.
  If the Oxford server is slow, run `python -m scripts.pets_from_hf` first (same images from the Hugging Face
  mirror, needs `pyarrow`); `prepare_data --pets` then only downloads the small annotations archive.
* **FS2K**: download from https://github.com/DengPingFan/FS2K and extract to `data/raw/FS2K/`
  (`photo/`, `sketch/`, `anno_train.json`, `anno_test.json`). Then run `python -m scripts.prepare_data --fs2k`,
  which checks every photo↔sketch pairing and writes a stratified 15% val split.
* `python -m scripts.prepare_data --sanity` saves a grid of every corruption and severity level.

### Corruptions (exact spec, `src/data/corruptions.py`)
| Condition | Training (sampled every load) | Test severities (low / medium / high) |
|---|---|---|
| Salt-and-pepper | p ~ U(0.02, 0.15), black/white 50:50 | p = 0.03 / 0.08 / 0.15 |
| Gaussian blur | k ∈ {3,5,7}, σ ~ U(0.5, 2.5) | (3, 0.7) / (5, 1.5) / (7, 2.5) |
| Occlusion | 1–3 black rectangles, union area 10–35% | ~10% (1 rect) / ~20% (2) / ~35% (3) |

The same file is used by training, the manifests, evaluation and the FastAPI backend.

### Pipeline
```bash
pytest -q tests/                                         # unit tests
python -m train.train --task t1 --smoke                  # CPU smoke run of any task (synthetic data)

python -m optuna_studies.tune --task t1                  # writes configs/t1_best.yaml + results/t1/optuna/
python -m train.train --task t1                          # resumable; checkpoints in checkpoints/t1/
python -m optuna_studies.tune --task t2_cls && python -m train.train --task t2_cls
python -m optuna_studies.tune --task t2_spec             # shared architecture search (blur specialist)
python -m train.train --task t2_spec --specialist salt_pepper   # and blur, occlusion
python -m optuna_studies.tune --task t3 && python -m train.train --task t3
python -m optuna_studies.tune --task t4 && python -m train.train --task t4

python -m eval.evaluate --task t1    # t2, t3, t4 -> results/<task>/
python -m export.export_onnx         # models_onnx/*.onnx + results/onnx_parity.csv (PyTorch vs ONNX Runtime)
python -m scripts.make_samples       # bundle a few test images for the app
python -m scripts.wandb_log_checkpoints   # (runs trained before checkpoint logging existed) checkpoints -> W&B artifacts
```

### Report
```bash
python -m scripts.export_wandb_history   # per-epoch curves from the local W&B run files -> results/curves_history.json
python -m eval.plot_curves               # report/figures/curves_t*.pdf
python -m scripts.report_figures         # copies/re-tiles result figures into report/figures/
cd report && pdflatex main.tex && pdflatex main.tex   # or upload report/ to Overleaf
```
On Windows, `scripts\setup_gpu_pc.ps1` prepares the GPU machine (Python, venv, CUDA PyTorch), and
`scripts\gpu_chain_gan.ps1` and `scripts\gpu_chain_restoration.ps1` run these steps in order.
`docs/gpu_runbook.md` lists the full GPU run stage by stage.

---

## 3. Repository layout
```
src/data/        corruptions, Pets + FS2K datasets, balanced sampler, manifests
src/models/      autoencoder, classifier, soft MoE, style pix2pix
src/engine/      training loops (AMP, checkpoints, W&B, Optuna pruning hooks)
train/ optuna_studies/ eval/ export/   CLIs
backend/         FastAPI + ONNX Runtime        frontend/   React + Tailwind (nginx in Docker)
configs/         per-task YAML (+ *_best.yaml from Optuna)
results/         metrics tables and figures used in the report
docs/            design decisions, AI-use log, Google Stitch evidence
report/          IEEE LaTeX report
```
