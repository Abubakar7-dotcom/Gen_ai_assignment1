"""Export every inference model to ONNX and verify PyTorch vs ONNX Runtime parity.

    python -m export.export_onnx            # all models -> models_onnx/*.onnx + results/onnx_parity.csv
    python -m export.export_onnx --smoke    # random small models, CPU (pipeline check only)

ONNX graphs (batch dimension dynamic, opset 17):
  t1_universal.onnx          input[N,3,128,128] in [0,1]         -> output[N,3,128,128]
  t2_classifier.onnx         input                              -> probs[N,4]   (softmax inside the graph)
  t2_spec_<name>.onnx        input                              -> output
  t3_moe.onnx                input                              -> output, weights[N,4]   (whole MoE in one graph)
  t4_generator.onnx          photo[N,3,128,128] in [-1,1], style[N] int64 -> sketch[N,3,128,128] in [-1,1]
Parity: max |torch - onnx| on 16 real validation inputs (random inputs in smoke mode). Pass if < 1e-4.
"""
from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn

from src.utils import ROOT

OUT = ROOT / "models_onnx"
TOL = 1e-4


class WithSoftmax(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.softmax(self.m(x), dim=1)


def export(model, args, path, in_names, out_names):
    model.eval()
    dyn = {n: {0: "batch"} for n in in_names + out_names}
    torch.onnx.export(model, args, str(path), input_names=in_names, output_names=out_names, dynamic_axes=dyn,
                      opset_version=17, do_constant_folding=True, dynamo=False)


def parity(model, args, path, in_names) -> float:
    with torch.no_grad():
        ref = model(*args)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {n: a.numpy() for n, a in zip(in_names, args)})
    return max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))


def build(smoke: bool):
    from src.loaders import load_ae, load_classifier, load_generator, load_moe, load_specialists
    from src.models.moe import ExportMoE
    if smoke:
        from src.models.autoencoder import DenoisingAE
        from src.models.classifier import CorruptionClassifier
        from src.models.moe import SoftMoE
        from src.models.pix2pix import StyleUNetGenerator
        ae = lambda: DenoisingAE(base=8, latent_ch=8)
        cls = CorruptionClassifier((8, 8, 8, 8))
        specs = {s: ae() for s in ("salt_pepper", "blur", "occlusion")}
        moe = SoftMoE(CorruptionClassifier((8, 8, 8, 8)), [ae() for _ in range(3)])
        t1, gen = ae(), StyleUNetGenerator(base=8, style_dim=4)
    else:
        t1, cls, specs, moe, gen = load_ae("t1"), load_classifier(), load_specialists(), load_moe(), load_generator()
    models = {"t1_universal": (t1, ["input"], ["output"]),
              "t2_classifier": (WithSoftmax(cls), ["input"], ["probs"])}
    models.update({f"t2_spec_{k}": (m, ["input"], ["output"]) for k, m in specs.items()})
    models["t3_moe"] = (ExportMoE(moe), ["input"], ["output", "weights"])
    models["t4_generator"] = (gen, ["photo", "style"], ["sketch"])
    return {k: (m.eval(), i, o) for k, (m, i, o) in models.items()}


def sample_inputs(smoke: bool):
    if smoke:
        x = torch.rand(4, 3, 128, 128)
        return x, (x * 2 - 1, torch.tensor([0, 1, 2, 0]))
    from src.data.fs2k import load_fs2k
    from src.data.pets import ManifestDataset
    from src.engine.common import pets_data
    data = pets_data({})
    ds = ManifestDataset(data["val"], data["val_items"][:16])
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    p, _, st = load_fs2k("val")
    photo = torch.from_numpy(p[:8]).permute(0, 3, 1, 2).float() / 127.5 - 1
    return x, (photo, torch.from_numpy(st[:8]).long())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = OUT / "smoke" if a.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    x, gen_args = sample_inputs(a.smoke)
    rows = []
    for name, (model, ins, outs) in build(a.smoke).items():
        args = gen_args if name == "t4_generator" else (x,)
        path = out / f"{name}.onnx"
        export(model, args, path, ins, outs)
        diff = parity(model, args, path, ins)
        rows.append({"model": name, "file": path.name, "size_mb": round(path.stat().st_size / 2**20, 2),
                     "max_abs_diff": diff, "pass": diff < TOL})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / ("onnx_parity_smoke.csv" if a.smoke else "onnx_parity.csv"), index=False)
    if not df["pass"].all():
        raise SystemExit("ONNX parity FAILED for: " + ", ".join(df[~df["pass"]]["model"]))


if __name__ == "__main__":
    main()
