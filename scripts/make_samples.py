"""Copy a few official TEST images (Pets + FS2K photos) into backend/samples for the app's "pick a sample" option.

    python -m scripts.make_samples
Test images are only displayed/restored in the demo; they are never used for training or tuning.
"""
import cv2
import numpy as np

from src.utils import ROOT, load_json

OUT = ROOT / "backend" / "samples"


def main(n_pets: int = 8, n_faces: int = 4):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    pets = np.load(cache / "pets_test_128.npy", mmap_mode="r")
    ids = load_json(cache / "pets_test_ids.json")
    for i in np.linspace(0, len(pets) - 1, n_pets).astype(int):
        cv2.imwrite(str(OUT / f"pet_{ids[i]}.png"), cv2.cvtColor(np.asarray(pets[i]), cv2.COLOR_RGB2BGR))
    faces = np.load(cache / "fs2k_test_photo_128.npy", mmap_mode="r")
    for k, i in enumerate(np.linspace(0, len(faces) - 1, n_faces).astype(int)):
        cv2.imwrite(str(OUT / f"face_{k + 1}.png"), cv2.cvtColor(np.asarray(faces[i]), cv2.COLOR_RGB2BGR))
    print("samples written to", OUT)


if __name__ == "__main__":
    main()
