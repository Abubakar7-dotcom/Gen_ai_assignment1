"""Fallback source for the Oxford-IIIT Pet IMAGES when the Oxford server is too slow (30-60 KB/s measured).

    python -m scripts.pets_from_hf          # then: python -m scripts.prepare_data --pets

The Hugging Face mirror ``timm/oxford-iiit-pet`` stores the original JPEG bytes + file names in two parquet
files (train = official trainval, test = official test). This script writes them back to
``data/raw/oxford-iiit-pet/images/<image_id>.jpg`` - the layout ``src/data/pets.py`` expects - so nothing
else in the pipeline changes. The official ``annotations/trainval.txt`` and ``test.txt`` (small, still from
Oxford) stay the source of truth for ids and their order; this script checks every official id got an image.
Needs ``pyarrow`` (not in requirements.txt: only this optional script uses it).
"""
from __future__ import annotations

import urllib.request

from src.data import pets

HF_URL = "https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data/{split}-00000-of-00001.parquet"
HF_SPLITS = {"train": "trainval", "test": "test"}          # HF split name -> official annotation file


def main() -> None:
    import pyarrow.parquet as pq

    hf_dir, img_dir = pets.RAW_DIR / "hf", pets.RAW_DIR / "images"
    hf_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    for hf_split, official in HF_SPLITS.items():
        path = hf_dir / f"{hf_split}.parquet"
        if not path.exists():
            print(f"downloading {hf_split}.parquet")
            urllib.request.urlretrieve(HF_URL.format(split=hf_split), path)
        written, not_jpeg = set(), []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["image", "image_id"]):
            for row in batch.to_pylist():
                data = row["image"]["bytes"]
                if data[:2] != b"\xff\xd8":                 # a few official ".jpg" files are really PNG/GIF;
                    not_jpeg.append(row["image_id"])        # load_rgb_128 handles them (PIL fallback)
                (img_dir / f"{row['image_id']}.jpg").write_bytes(data)
                written.add(row["image_id"])
        print(f"{hf_split}: wrote {len(written)} images; not real JPEGs: {not_jpeg}")
        ids_file = pets.RAW_DIR / "annotations" / f"{official}.txt"
        if ids_file.exists():                               # mirror must cover the official list exactly
            ids = set(pets.read_official_ids(pets.RAW_DIR, official))
            assert ids == written, f"{official}: {len(ids - written)} missing, {len(written - ids)} extra"
            print(f"{official}: all {len(ids)} official ids present")
        else:
            print(f"{official}: annotations not downloaded yet, id check skipped")


if __name__ == "__main__":
    main()
