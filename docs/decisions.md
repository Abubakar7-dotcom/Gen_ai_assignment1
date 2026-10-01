# Design decisions log

Each entry: decision, alternatives considered, evidence / reason, effect. Feeds the report's research discussion.

## Data pipeline

- **Single corruption module (`src/data/corruptions.py`, NumPy/OpenCV) shared by training, manifests, evaluation and the FastAPI backend.**
  Alternatives: torchvision transforms in training + separate backend code. Rejected: two implementations drift, so app behaviour would not match reported metrics.
- **Corruptions applied on uint8 HWC images before tensor conversion.** Salt-and-pepper must produce exact 0/255 values; doing it after normalisation would change the semantics.
- **Occlusion coverage measured as union area of rectangles (boolean mask), resampled until within range.** Summing rectangle areas over-counts overlaps and would violate the 10–35% spec.
- **Balanced batch sampler (exactly B/4 per condition per batch)** instead of i.i.d. uniform sampling: satisfies Task 2's "balanced batches" requirement deterministically and lowers gradient variance across conditions.
- **Clean 128×128 images cached once as a uint8 array.** Corruptions are still sampled at runtime (spec forbids saving corrupted copies). Removes JPEG decode + resize from the loop, which is the main CPU bottleneck on a laptop.
