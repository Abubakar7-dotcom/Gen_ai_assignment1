# Viva preparation notes

The brief says the evaluator may ask you to *justify the architecture, explain a decision, interpret a result, modify
the code, or run the app on unseen images*. These notes cover the likely questions. Read the report and
`docs/decisions.md` alongside them.

## 1. Where is what (open these files before the viva)

| Topic | File |
|---|---|
| The ONE corruption implementation (training, manifests, evaluation **and** the app) | `src/data/corruptions.py` (`sample_params` l.98 = training ranges, `TEST_SEVERITY` l.33 = test levels, `_salt_pepper` l.63, `_blur` l.74, `_occlude` l.79) |
| Balanced batches (exactly B/4 per condition) | `src/data/sampler.py` |
| Deterministic val/test manifests | `src/data/manifests.py`, output `data/manifests/*.json` |
| Autoencoder (T1, T2 specialists), `skip` flag for the ablation | `src/models/autoencoder.py` |
| Classifier / MoE gate | `src/models/classifier.py`, `src/models/moe.py` (`forward` l.23: softmax(G/τ), weighted sum) |
| U-Net generator + PatchGAN with style embedding | `src/models/pix2pix.py` |
| Losses (L1+SSIM, MoE loss with balance term, GAN losses) | `src/losses.py` |
| Optuna objective `0.5·min(PSNR,40)/40 + 0.5·SSIM` | `src/metrics.py` l.21 |
| Optuna search spaces | `optuna_studies/tune.py` (`space_ae`, `space_cls`, `space_moe`, `space_gan`) |
| Training loops (AMP, checkpoints, W&B, pruning) | `src/engine/ae.py`, `cls.py`, `moe.py`, `gan.py` |
| Selected hyperparameters | `configs/*_best.yaml` |
| Test evaluation + figures | `eval/evaluate.py` |
| ONNX export + parity | `export/export_onnx.py` |
| API (validation, hard routing with identity bypass, MoE weights, sketch) | `backend/app/main.py` (`/restore/hard` uses the classifier, then `if pred == "clean": out = x`) |
| Model auto-download on container start | `backend/app/fetch_models.py` |
| UI | `frontend/src/pages/RestorePage.jsx` (three restoration workspaces), `SketchPage.jsx` |

## 2. Numbers to know

* Pets: 3,680 trainval → 2,944 train / 736 val (seed 42); 3,669 test → 36,690 test items (clean + 3 corruptions × 3 severities).
* FS2K: 1,058 train → 899 train / 159 val (stratified by style, seed 42); 1,046 test.
* T1: latent 8 channels × 8×8 = 512 values (96× compression), base 48, α = 0.73, 40 epochs. Test PSNR 22.2 dB / SSIM 0.631;
  salt-and-pepper 16.4 → 22.9 dB, occlusion 13.6 → 20.4 dB, blur 27.9 → 23.0 dB (worse).
* T2: classifier accuracy 0.996, macro-F1 0.993; misrouting 0.8% of clean, 3.2% of low occlusion, 0 elsewhere.
* T3: τ = 0.65, gate accuracy 99.3%, mean weights ≈ 0.98–1.00 on the correct branch; best checkpoint = end of warm-up.
* T4: λ_L1 = 168.6, style embedding 32-d, batch 8, 150 epochs (best epoch 47); test L1 0.103, SSIM 0.481.
* ONNX parity: max |PyTorch − ONNX Runtime| = 6.6e-6.

## 3. "Why" questions

* **Why L1 + SSIM, not MSE?** MSE (L2) over-smooths and does not match perceived quality; L1 keeps edges better and SSIM
  rewards local structure and contrast (Zhao et al. 2017). α was tuned by Optuna (0.73).
* **Why is the bottleneck "genuine"?** No skip connections; the decoder only sees the 8×8×8 code, 96× smaller than the
  input, so the model cannot copy the image.
* **Why is T1 worse than the input on blur?** Every output is ≈23 dB because 512 numbers cannot hold fine texture; mild
  blur keeps more detail (28–32 dB) than the bottleneck can return. Ablation: 64 channels +0.8 dB only; one narrow skip
  +4.9 dB, so the 8×8 *spatial* size is the limit.
* **Why did you not adopt the skip?** The brief requires a meaningful bottleneck and allows limited skips only if
  investigated; we report it as an investigated alternative. Adopting it means retraining Tasks 1–3.
* **Why are the specialists not better than the universal model?** They have the same bottleneck. The real benefit of
  T2 is the identity bypass for clean images (SSIM 1.0 instead of 0.65).
* **Why do clean images get misrouted to occlusion?** They contain real black areas (letterbox bars, black background);
  the classifier learned "black rectangle = occlusion". See the misrouting figure.
* **Why is the T3 checkpoint from warm-up?** Validation objective peaked at the end of warm-up (0.863); joint fine-tuning
  stayed at 0.860–0.862. We kept the selection rule; the joint checkpoint is also released (`ckpt_t3_last.pt`).
* **What does the balance loss do?** Σ_k (mean weight_k − 1/4)². Batches are balanced, so the ideal routing also has mean
  weight 1/4 per branch; it penalises collapse onto one expert without fighting the CE term (as in Switch Transformers).
* **What does τ do?** softmax(G/τ): small τ → sharp (almost hard) routing, large τ → weights spread out.
* **Why PatchGAN?** It judges local 14×14 patches, which enforces stroke/texture realism; L1 handles global structure.
* **How is the style used?** `nn.Embedding(3, 32)`, tiled as extra input channels and again at the U-Net bottleneck, and
  tiled into the discriminator input, so D judges "is this a real style-s sketch of this photo".
* **Why is style 2 worst?** Dense dark hatching; even a plausible sketch puts strokes elsewhere, and pixel metrics punish
  that.
* **Why Optuna settings like 4 epochs on 50% data?** 2-day budget on one laptop GPU; the best config was retrained fully.
  Limitation: short trials hid the effect of the code size (importance < 0.01 in short trials).
* **Why ONNX Runtime on CPU in Docker?** Portable (no CUDA needed on the evaluator's machine), 30–200 ms per image.

## 4. Likely live modifications and how to do them

* **Change a corruption setting** (e.g. a new test severity): edit `TEST_SEVERITY` / `severity_params` in
  `src/data/corruptions.py`; the backend imports the same file, so restart: `docker compose up --build`.
* **Add a new API field** (e.g. return the entropy of the MoE weights): in `backend/app/main.py` `restore_moe`, compute
  `-sum(w*log w)` from `weights` and add it to the returned dict; show it in `RestorePage.jsx`.
* **Change τ at inference**: τ is baked into the exported ONNX graph. `load_moe()` in `src/loaders.py` reads it from the
  checkpoint; set `moe.tau = <new value>` there and re-export with `python -m export.export_onnx` (needs the
  checkpoints from the release in `checkpoints/`).
* **Run on an unseen image**: upload it in any workspace; choose *None (already corrupted)* if it is already damaged.
* **Restart the app**: `docker compose down` then `docker compose up --build` (models are reused).
* **Re-run an evaluation**: download `ckpt_*.pt` from the release into `checkpoints/<task>/best.pt` (T4: `best_G.pt`),
  prepare the data (`python -m scripts.prepare_data --pets --fs2k`), then `python -m eval.evaluate --task t1`.
