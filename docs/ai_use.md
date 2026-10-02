# AI-use log (report appendix)

| Date | Tool | Used for | How output was verified / corrected |
|---|---|---|---|
| 2026-10-02 | Claude (Cowork) | Assignment breakdown, 2-day plan, repo scaffold, CLAUDE.md | Checked against assignment PDF requirement-by-requirement |
| 2026-10-02 | Claude (Cowork) | Corruption module + data pipeline code | Unit tests (`tests/test_corruptions.py`): param ranges, occlusion union-area bounds, determinism from seed; visual sanity grid inspected manually |
| 2026-10-02 | Claude Code | GPU PC setup script (`scripts/setup_gpu_pc.ps1`) and run instructions (`docs/gpu_runbook.md`) | Script syntax-checked with the PowerShell parser only; not yet run on the GPU PC |
| 2026-10-02 | Claude Code (on the GPU PC) | Dataset download + preparation: Pets images via Hugging Face mirror (`scripts/pets_from_hf.py`), FS2K cache/split | Mirror ids asserted equal to the official `trainval.txt`/`test.txt`; pairing report (0 problems, 2,104 pairs); corruption sanity grid and FS2K photo/sketch pairs inspected visually |
