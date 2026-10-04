# Submission folder: GenAI Assignment 1

## What gets submitted on Google Classroom

Only **one file**: the report PDF (IEEE format, written in LaTeX). The brief says everything else is reached through
links inside the report:

| Required by the brief | Where it is |
|---|---|
| Technical report (IEEE, LaTeX) | `GenAI_A1_Report.pdf` (this folder); source in `report/` and `report_overleaf.zip` |
| GitHub repository link | in the report: https://github.com/Abubakar7-dotcom/Gen_ai_assignment1 |
| Trained models (download link) | in the report and README: GitHub Release `models-v1` |
| Experiment tracking | in the report: W&B project `genai-a1` |
| YouTube demo video (5–7 min) | **link still to be added to the report** (only the link; do not upload the video to Classroom) |

## Status

Done and verified:
* All four models trained with Optuna, evaluated on the official test sets, exported to ONNX (parity < 1e-4).
* Application (React + Tailwind + FastAPI) with the four workspaces; tested end to end with the trained models.
* Docker: `git clone` → `docker compose up --build` → http://localhost:8080. The backend downloads the models by itself.
  Tested automatically on a clean Linux machine (GitHub Actions, workflow `docker-compose-e2e`).
* Report: complete draft, 15 pages, every figure and table interpreted, built with pdfLaTeX in CI (workflow `report-pdf`).

## What only you can do (in this order)

1. **Read the report** (`GenAI_A1_Report.pdf`) and change anything you would not say yourself. You will be asked
   about it in the viva. Red text marks the four placeholders below.
2. **Name and roll number**: in `report/main.tex`, lines 21–22: replace `\TODO{full name}` and `\TODO{roll number}`.
3. **Google Stitch design** (the brief requires the design to be made in Stitch and shown in the report):
   * Open https://stitch.withgoogle.com, choose *Web*, paste the prompt from `docs/stitch/stitch_prompt.md`.
   * Screenshot each of the four screens and save them as `docs/stitch/01_universal.png`, `02_hard.png`,
     `03_moe.png`, `04_sketch.png`.
   * Run `python -m scripts.report_figures` (copies them into `report/figures/`). The report then shows them
     automatically in Fig. 19 instead of the red box. (On Overleaf: upload them to `figures/` as
     `stitch_01_universal.png`, `stitch_02_hard.png`, `stitch_03_moe.png`, `stitch_04_sketch.png`.)
4. **Demo video** (5–7 min): follow `demo_video_script.md`, upload to YouTube (Unlisted is fine), then put the link in
   `report/main.tex` line 23: replace `\TODO{YouTube link of the demo video}` with `\url{https://youtu.be/...}`.
5. **Compile the final PDF**:
   * Overleaf (no install): New Project → Upload Project → `report_overleaf.zip` → Recompile → Download PDF.
     Make the same edits there (steps 2–4), or edit locally, push, and use the PDF from the `report-pdf` workflow
     run (Actions tab → latest run → artifact `report-pdf`).
   * Rename it e.g. `GenAI_A1_<RollNo>_<Name>.pdf` and submit it on Google Classroom before the deadline.
6. **Prepare for the viva** with `viva_notes.md`: the evaluator may ask you to explain or modify any part and to run
   the app on new images.

Optional: if you want the restoration models to be sharper, the limited-skip variant (val PSNR 22.4 → 27.3 dB) can be
adopted by retraining Tasks 1–3 on the GPU PC (about 2 h), then re-evaluating, re-exporting and updating the release
and the report. The current submission instead reports it as an investigated ablation, which the brief allows.
