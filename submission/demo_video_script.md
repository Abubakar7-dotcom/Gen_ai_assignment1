# Demo video script (target 6 minutes, allowed 5–7)

The brief requires the video to show: **application startup, image uploading, runtime corruption, universal
restoration, hard routing, soft expert weights, face-to-sketch generation, result downloading, and
experiment-tracking records.** Each item below is marked ✅ where it is covered.

**No Docker on your PC? Use GitHub Codespaces (nothing to install, free monthly quota is enough):**
on the repository page click **Code → Codespaces → Create codespace on main**. In its terminal run
`docker compose up --build`; when port 8080 appears in the **Ports** tab, click the globe icon to open the app in a new
browser tab. Record that browser tab and the Codespaces terminal exactly as in the table below (the URL is HTTPS, so
the webcam works too). Stop the codespace afterwards (Codespaces page → ⋯ → Stop) to save quota.

**Before recording:** have one or two photos ready that are *not* from the dataset (e.g. your own pet or a pet photo
from the web, and a face photo or your webcam). Save one pet photo that you have already blurred or scribbled black
boxes on, to show the "already corrupted" upload. Close other apps; record the screen at 1080p (Windows: Xbox Game Bar
`Win+G`, or OBS). Speak while you click; do not cut the startup step.

| Time | Show | Say (short) |
|---|---|---|
| 0:00–0:20 | Title slide or the GitHub README | "GenAI Assignment 1: four generative models in one app: universal restoration, hard routing, soft mixture of experts, face-to-sketch." |
| 0:20–1:20 | Terminal: `git clone …`, `cd`, `docker compose up --build`; the backend log printing `downloading t1_universal.onnx … fetch_models: done`, then `Uvicorn running`. Open `http://localhost:8080`; point at the green **API online · 7/7 models** pill. | ✅ **startup**: "One command builds both containers; on first start the backend downloads the trained ONNX models from the GitHub release and checks their checksums." (If the build is slow, cut while it waits, but keep start and end.) |
| 1:20–2:20 | **Universal Restoration**: pick a clean sample, choose *Salt & pepper*, *High*, press Restore. Point at input / output / error map, the corruption settings (p, seed) and inference time. Change to *Occlusion*, *Medium*, Restore. Press **Download result** and show the downloaded file. | ✅ **runtime corruption**, ✅ **universal restoration**, ✅ **downloading**: "Corruptions are generated at runtime by the same code used in training. One autoencoder handles every condition." |
| 2:20–2:50 | Still Universal: drop your **already corrupted** photo into the upload box, choose *None (already corrupted)*, Restore. | ✅ **image uploading**: "An evaluator can also upload an image that is already damaged." |
| 2:50–3:50 | **Hard-Routed Restoration**: upload a new pet photo, apply *Gaussian blur, Medium*. Point at the 4 classifier probabilities, the predicted corruption and the selected expert. Then choose *None* with a clean photo: predicted *Clean* → *Identity bypass*. | ✅ **hard routing**: "The classifier picks one specialist; clean images bypass restoration entirely (99.6% test accuracy)." |
| 3:50–4:50 | **Soft Mixture-of-Experts**: same photo with *Occlusion, Low*. Point at the four routing weights, the expert-mix bar and the top contributor. Try *Salt & pepper, High* to show one expert at ~100%. | ✅ **soft expert weights**: "The gate gives every branch a weight; for small occlusions it hedges between experts, for clear cases one expert dominates." |
| 4:50–5:40 | **Face-to-Sketch Generator**: upload a face photo (or **Use webcam** → Capture). Generate *Style 1*, then *Style 2* and *Style 3* on the same photo. Press **Download sketch**. | ✅ **face-to-sketch**, ✅ **downloading**: "One generator, three styles through a learned style embedding." |
| 5:40–6:30 | Browser: W&B project `genai-a1`: runs table, open `t4-style-cgan` → the separate D real / D fake / G adv / G L1 charts and the sample grids over epochs; then the **Artifacts** tab (checkpoints). Optionally the GitHub Release page. | ✅ **experiment tracking**: "Every run logs hyperparameters, losses, validation metrics, sample images and checkpoints." |
| 6:30–6:45 | Back to the app | "Code, models and report are linked in the report. Thank you." |

Upload to YouTube as **Unlisted** (or Public), copy the link, and paste it into `report/main.tex` (`\youtubeurl`).
