"""Rebuild per-epoch training histories of the final runs from the local W&B run files (no API key needed).

    python -m scripts.export_wandb_history        # -> results/curves_history.json (used by eval/plot_curves.py)

W&B keeps each run in wandb/run-<time>-<id>/run-<id>.wandb, written in the LevelDB log format (7-byte header,
32 KiB blocks of chunks: crc32 4 B, length 2 B little-endian, type 1 B). Resumed runs have several files; their
history rows are merged by step. Epochs whose rows never reached W&B (the T2 classifier before the WinError 1455
restart, T4 epochs 12-23 before the power loss) are filled from the console logs in checkpoints/logs/.
"""
import ast
import glob
import json
import struct

from wandb.proto import wandb_internal_pb2 as pb

from src.utils import ROOT

RUNS = {"t1": "bd1eke2n", "t2_cls": "29wvrtww", "t2_spec_salt_pepper": "9bkfn5ol", "t2_spec_blur": "wn9hkvmn",
        "t2_spec_occlusion": "0eyw948h", "t3": "80jbrh4n", "t4": "b6ri27wg"}
LOG_FILLS = {"t2_cls": ("restoration_chain.prev.log", ">>> python -m train.train --task t2_cls"),
             "t4": ("gan_chain.prev.log", ">>> python -m train.train --task t4")}


def records(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b":W&B", path
    pos, buf, block = 7, b"", 32768
    while pos + 7 <= len(raw):
        left = block - pos % block
        if left < 7:                                     # block trailer padding
            pos += left
            continue
        _, n, kind = struct.unpack("<IHB", raw[pos:pos + 7])
        chunk = raw[pos + 7:pos + 7 + n]
        pos += 7 + n
        if kind == 0 and n == 0:
            break
        if kind == 1:
            yield chunk
        elif kind == 2:
            buf = chunk
        elif kind == 3:
            buf += chunk
        elif kind == 4:
            yield buf + chunk


def run_history(run_id):
    rows = {}
    for f in sorted(glob.glob(str(ROOT / "wandb" / f"run-*-{run_id}" / "*.wandb"))):
        for data in records(f):
            r = pb.Record()
            r.ParseFromString(data)
            if r.WhichOneof("record_type") != "history":
                continue
            d = {i.key or "/".join(i.nested_key): json.loads(i.value_json) for i in r.history.item}
            d = {k: v for k, v in d.items() if isinstance(v, (int, float)) and not k.startswith("samples/")}
            if "epoch" in d:
                rows[int(d["epoch"])] = {**rows.get(int(d["epoch"]), {}), **d}
    return rows


def log_epochs(log_name, header):
    rows, on = {}, False
    for ln in open(ROOT / "checkpoints" / "logs" / log_name, encoding="utf-16").read().splitlines():
        if ln.startswith(">>> "):
            on = ln.strip() == header
        elif on and ln.startswith("{'epoch'"):
            d = ast.literal_eval(ln.strip())
            rows[int(d["epoch"])] = {k: v for k, v in d.items() if isinstance(v, (int, float))}
    return rows


def main():
    out = {}
    for task, rid in RUNS.items():
        rows = run_history(rid)
        if task in LOG_FILLS:
            for e, r in log_epochs(*LOG_FILLS[task]).items():
                rows.setdefault(e, r)
        out[task] = [rows[e] for e in sorted(rows)]
        print(f"{task}: {len(rows)} epochs")
    with open(ROOT / "results" / "curves_history.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
