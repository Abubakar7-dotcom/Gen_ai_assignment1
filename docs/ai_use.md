# AI-use log (report appendix)

| Date | Tool | Used for | How output was verified / corrected |
|---|---|---|---|
| 2026-10-02 | Claude (Cowork) | Assignment breakdown, 2-day plan, repo scaffold, CLAUDE.md | Checked against assignment PDF requirement-by-requirement |
| 2026-10-02 | Claude (Cowork) | Corruption module + data pipeline code | Unit tests (`tests/test_corruptions.py`): param ranges, occlusion union-area bounds, determinism from seed; visual sanity grid inspected manually |
| 2026-10-02 | Claude Code | GPU PC setup script (`scripts/setup_gpu_pc.ps1`) and run instructions (`docs/gpu_runbook.md`) | Script syntax-checked with the PowerShell parser only; not yet run on the GPU PC |
