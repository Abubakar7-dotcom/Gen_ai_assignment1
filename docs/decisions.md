# Design decisions log

Each entry: decision, alternatives considered, evidence / reason, effect. Feeds the report's research discussion.

## Data pipeline

- **Single corruption module (`src/data/corruptions.py`, NumPy/OpenCV) shared by training, manifests, evaluation and the FastAPI backend.**
  Alternatives: torchvision transforms in training + separate backend code. Rejected: two implementations drift, so app behaviour would not match reported metrics.
- **Corruptions applied on uint8 HWC images before tensor conversion.** Salt-and-pepper must produce exact 0/255 values; doing it after normalisation would change the semantics.
- **Occlusion coverage measured as union area of rectangles (boolean mask), resampled until within range.** Summing rectangle areas over-counts overlaps and would violate the 10–35% spec.
- **Balanced batch sampler (exactly B/4 per condition per batch)** instead of i.i.d. uniform sampling: satisfies Task 2's "balanced batches" requirement deterministically and lowers gradient variance across conditions.
- **Clean 128×128 images cached once as a uint8 array.** Corruptions are still sampled at runtime (spec forbids saving corrupted copies). Removes JPEG decode + resize from the loop, which is the main CPU bottleneck on a laptop.
- **Pet images taken from the Hugging Face mirror `timm/oxford-iiit-pet` (`scripts/pets_from_hf.py`), ids and split order from the official annotation files.**
  Alternative: the official `images.tar.gz` (what `prepare_data --pets` downloads by default). Rejected on the GPU PC only for speed: the Oxford server gave 30–60 KB/s (4+ h for 790 MB) against >1 MB/s from the mirror.
  Evidence the mirror is the same data: 3,680 trainval + 3,669 test images, the same counts as the official `trainval.txt` / `test.txt`; the files are the original bytes, including the four official `.jpg` files that are really PNGs (`Egyptian_Mau_14/156/186`, `Abyssinian_5`). The script asserts the mirror's ids equal the official id lists.
  Effect: none on the split (seed 42 over the official `trainval.txt` order) or on the 128×128 cache.

## Training setup

- **`cudnn.benchmark` off by default (config key `cudnn_benchmark`).** Alternative: on (the original plan). Measured on the RTX 4050, Task 1, 3 epochs on real data: on = 2 min 06 s wall (first epoch 67 s, later epochs 8.0 s); off = 56 s wall (first epoch 17 s, later epochs 7.2–7.7 s). The auto-tuner re-runs for every new layer shape, i.e. for every Optuna trial (channel widths and batch size change) and again for the validation batch size, so it costs about 70 s per run and gives no steady-state gain for networks this small.
- **`num_workers: 0` (data loading in the training process) on the Windows GPU PC.** Alternative: 2 workers per loader (the original plan). With workers, Windows starts every worker as a new process that re-imports the CUDA build of PyTorch and reserves 1.3–4.4 GB of commit memory; one chain (train + val + fixed-sample loaders) reached about 14 GB, and two chains on a 15.6 GB RAM machine hit `WinError 1455 (paging file too small)`, which killed workers and left both runs hanging. Cost of no workers, measured on Task 1: 11 s per epoch instead of 8 s; the images are a cached uint8 array and the corruptions are cheap, so loading is not the bottleneck.
