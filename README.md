# GenAI Assignment 1 — Image Restoration & Face-to-Sketch

Four generative models served through one web app:

1. **Universal Restoration**: a single denoising autoencoder for salt-and-pepper noise, Gaussian blur and occlusion.
2. **Hard-Routed Restoration**: a corruption classifier that sends each image to one of three specialist autoencoders.
3. **Soft Mixture-of-Experts Restoration**: a jointly trained gate that blends the outputs of all the experts.
4. **Face-to-Sketch Generator**: a style-conditioned pix2pix cGAN trained on FS2K.

Stack: PyTorch · Optuna · Weights & Biases · ONNX Runtime · FastAPI · React + Tailwind · Docker Compose.

> Status: work in progress. Full instructions will be added as each component lands.

## Quick start (app)

```bash
git clone <repo-url> && cd genai-a1
bash scripts/download_models.sh      # fetches ONNX models
docker compose up --build            # open http://localhost:8080
```

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or /cpu
pip install -r requirements.txt
```

## Repository layout

See `CLAUDE.md` for the full specification and layout.
