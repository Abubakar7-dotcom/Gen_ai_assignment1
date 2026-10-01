# Google Stitch prompt (paste into https://stitch.withgoogle.com)

Use "Web" mode. After generating, take screenshots of every screen and save them here as
`docs/stitch/01_universal.png`, `02_hard.png`, `03_moe.png`, `04_sketch.png` (report evidence), and export the HTML/Tailwind
code to `docs/stitch/export/` if Stitch offers it.

---

Design a clean, modern single-page web app called "Restoration Lab": a research demo for image restoration and
face-to-sketch generation. Light theme with a dark-mode variant, neutral slate background, one indigo accent colour,
rounded-xl cards, subtle shadows, Inter font. Desktop-first but responsive down to phone width.

Layout:
- Top bar: app name "Restoration Lab", a small status pill "API online · 7/7 models" on the right.
- Below it, a tab/segmented control with four workspaces: "Universal Restoration", "Hard-Routed Restoration",
  "Soft Mixture-of-Experts Restoration", "Face-to-Sketch Generator".

Workspace 1 — Universal Restoration:
- Left panel card "Input": drag-and-drop upload area OR a row of small clean sample thumbnails to pick from.
  Under it a "Corruption" section: segmented control (None / Salt & pepper / Gaussian blur / Occlusion),
  severity chips (Low / Medium / High / Random), a seed number field, and a primary button "Restore".
- Right side: three equal image panels side by side labelled "Input", "Restored", "Error map", each square.
- A row of small stat tiles below: "Corruption settings" (e.g. kernel 5, σ 1.5), "Inference time 12.4 ms", "Model".

Workspace 2 — Hard-Routed Restoration:
- Same input card.
- Result area: image panels "Input" and "Restored", plus a card "Classifier" with four horizontal probability bars
  (Clean, Salt & pepper, Blur, Occlusion) with percentages, the winning bar highlighted, a badge
  "Predicted: Blur" and a badge "Selected expert: Blur expert", and an inference-time tile.

Workspace 3 — Soft Mixture-of-Experts Restoration:
- Same input card.
- Result area: "Input" and "Restored" panels, a card "Routing weights" with four bars (Identity, Salt & pepper
  expert, Blur expert, Occlusion expert), a stacked horizontal bar showing the mix, and a "Top contributor" badge.
  Inference-time tile.

Workspace 4 — Face-to-Sketch Generator:
- Input card with upload area, a "Use webcam" button with a live camera preview and a "Capture" button,
  and a 3-option style selector (Style 1 / Style 2 / Style 3) shown as cards.
- Result area: "Photo" and "Generated sketch" side by side, a "Download PNG" button, inference-time tile.
