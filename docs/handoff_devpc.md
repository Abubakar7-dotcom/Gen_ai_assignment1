# Handoff: GPU PC → DEV PC (2026-10-03)

Written by the Claude Code session on the GPU PC for the Claude Code session on the student's own PC.
Read this after `CLAUDE.md`. The assignment brief is `docs/GenAI_Assignment1.pdf` (9 pages); read it in full
before planning, because it is the grading reference.

## State: all training is finished

Nothing below needs a GPU. Do not train on the DEV PC. Retraining is only needed if the student decides to adopt
the T1 skip connection (see Open decisions). It would then run on the GPU PC (roommate's laptop, shared with their
own coursework, so ask the student first).

| Task | Result (official test set unless noted) | Where |
|---|---|---|
| T1 universal DAE | PSNR 22.2 dB / SSIM 0.631 (corrupted input: 27.4 / 0.672). Helps salt-and-pepper (+6.5 dB) and occlusion (+7 dB), hurts blur (27.9 → 23.0) and clean | `results/t1/` |
| T2 classifier | accuracy 0.996, macro-F1 0.993; misrouting < 1% (clean 0.8%, low-severity occlusion 3.2%) | `results/t2/` |
| T2 specialists | val objective 0.612 / 0.615 / 0.555 (salt-and-pepper / blur / occlusion); oracle vs predicted routing evaluated | `results/t2/` |
| T3 soft MoE | val objective 0.863, gate accuracy 99.3%, no inactive expert; best checkpoint = end of warm-up | `results/t3/` |
| T4 style cGAN | val objective 0.704; test L1 0.103, SSIM 0.481; SSIM per style 0.523 / 0.397 / 0.613 | `results/t4/` |
| T1 ablations (val) | latent 64: PSNR 22.4 → 23.1 dB. One limited skip: PSNR 22.4 → 27.3 dB, SSIM 0.639 → 0.844 | `results/ablations/` |

- Optuna studies: `optuna_studies/*.db`, plots and best params in `results/{t1,t2_cls,t2_spec,t3,t4}/optuna/`, chosen configs in `configs/*_best.yaml`.
- ONNX parity: 7 models, max abs diff 6.6e-6 (`results/onnx_parity.csv`).
- Models: GitHub Release `models-v1` (7 ONNX + 8 PyTorch checkpoints + SHA256SUMS). `scripts/download_models.ps1` / `.sh` fetch the ONNX files; a fresh download was verified.
  To re-run evaluation on the DEV PC, put the `ckpt_*.pt` files back as `checkpoints/<task>/best.pt` (T4: `best_G.pt`). The data must be prepared first (`scripts/prepare_data.py`; Pets via `scripts/pets_from_hf.py`).
- W&B: project `genai-a1` (entity `abubaknaveed-national-university-of-computer-and-emergi`). It holds every final run (losses, metrics, sample grids) and the checkpoints as model artifacts. The DEV PC needs `wandb login` with the student's key. Never write the key into the repo.
- `docs/decisions.md` records every design decision, measurement and result explanation; use it as the main source for the report.

## What was verified and what was not

- Verified on the GPU PC:
  - FastAPI backend (`backend/requirements.txt` pins) with the trained ONNX models, on a real uploaded JPEG.
  - All endpoints return correct routing, weights and timings; bad file 415, file > 10 MB 413, style 9 422.
  - Output colours checked visually.
- **Not verified yet:**
  - the React frontend against the trained models (only against the smoke backend),
  - `docker compose up --build` on any machine (Docker was never installed on the GPU PC).

## Remaining work vs the brief (suggested order)

1. **Docker fresh-clone test.** Clone into a new folder, run `scripts\download_models.ps1`, then `docker compose up --build`, then click through all 4 workspaces at http://localhost:8080. Include the webcam (Face-to-Sketch) and the download button. Fix anything that breaks.
2. **Google Stitch evidence (student).** The brief requires the interface to be designed first in Google Stitch, with screenshots in the report.
   - The prompt is in `docs/stitch/stitch_prompt.md`.
   - Screenshots go in `docs/stitch/`.
   - Align the frontend to the design if it differs.
3. **App screenshots** of each workspace for the report.
4. **Report figures still missing:**
   - training/validation curves (export from W&B via the API),
   - architecture diagrams (TikZ or matplotlib),
   - the app architecture diagram.
5. **IEEE LaTeX report** in `report/` (IEEEtran).
   - The brief requires LaTeX; Google Docs is not acceptable for the final report. The student can compile on Overleaf.
   - Sections are listed in `CLAUDE.md` → Tracking & deliverables.
   - Related work needs real, checkable citations: DAE, U-Net, pix2pix/PatchGAN, MoE, SSIM, Optuna/TPE, FS2K, Oxford-IIIT Pet.
   - Interpret every figure and table in the text.
   - The AI-use appendix comes from `docs/ai_use.md`.
6. **README.** It is mostly complete; check it against a fresh clone and add the YouTube link at the end.
7. **Demo video (student), 5–7 min.** Cover startup, upload, runtime corruption, all 4 workspaces, download and the W&B dashboard.
8. **Viva preparation (student).** The student may be asked to change code live; walk them through the code.

## Results that need explaining in the report (not bugs)

- **T1 outputs ≈ 23 dB for every input.** The 8×8 spatial bottleneck without skips loses detail. The latent-64 ablation barely helps, while the one-skip ablation helps a lot (`docs/decisions.md`, figure `results/ablations/t1_skip_examples.png`).
- **T3's best checkpoint is the end of warm-up.** Joint fine-tuning did not improve validation, so the experts equal the T2 specialists.
- **Overall PSNR for T2/T3 is inflated** by the identity branch on clean images (about 100 dB). Compare per-condition tables instead.
- **FS2K official test labels:** 179 `photo3` sketches are labelled style 0, but look like style 2. Headline numbers use the official labels; mention this as a limitation.

## Open decisions (ask the student; do not decide alone)

- Real submission deadline. The PDF says "March 16, 2024", which looks like a template date.
- Adopt the T1 limited skip or not. Adopting it means retraining T1, the T2 specialists and T3 on the GPU PC (about 1.5–2 h), then re-evaluating, re-exporting and re-releasing. Recommendation so far: keep the current models and report the skip as an investigated alternative.
- T3 submitted checkpoint: warm-up `best.pt` (current) or joint `last.pt` (also in the release as `ckpt_t3_last.pt`).
- FS2K: whether to also report a folder-corrected per-style breakdown.

## How the student likes to work

- Push every update to `main`: commit each milestone and push, so GitHub has a record. Fetch first; never push secrets or large files.
- Make long-running work efficient first (measure, then optimise) so time is not wasted.
- Ask before anything public or outward-facing (releases, publishing, sharing).
