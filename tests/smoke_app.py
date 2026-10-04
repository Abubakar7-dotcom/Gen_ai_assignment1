"""End-to-end smoke test of the running application (no Docker or project imports needed, stdlib only).

    python tests/smoke_app.py http://localhost:8080        # through nginx, as the browser sees it
    python tests/smoke_app.py http://localhost:8000 --direct  # straight to FastAPI (no /api prefix)

Checks the page, every API operation of the four workspaces on bundled samples, and the upload validation.
Exits non-zero on the first failure. Used by .github/workflows/docker.yml.
"""
import json
import sys
import urllib.error
import urllib.request
import uuid


def request(url, fields=None, files=None):
    """GET when fields is None, else multipart POST. Returns (status, body bytes)."""
    data, headers = None, {}
    if fields is not None:
        b = uuid.uuid4().hex
        parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
        for k, (name, content, ctype) in (files or {}).items():
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                         f'Content-Type: {ctype}\r\n\r\n'.encode() + content + b"\r\n")
        data = b"".join(parts) + f"--{b}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={b}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        sys.exit(1)


def main():
    base = sys.argv[1].rstrip("/")
    api = base if "--direct" in sys.argv else base + "/api"
    if "--direct" not in sys.argv:
        s, body = request(base + "/")
        check("frontend page", s == 200 and b'<div id="root"' in body, f"status {s}")
    s, body = request(api + "/health")
    health = json.loads(body)
    missing = [m for m, v in health["models"].items() if not v["available"]]
    check("health + all 7 models present", s == 200 and not missing, f"missing={missing}")
    samples = json.loads(request(api + "/samples")[1])["samples"]
    pet = next(n for n in samples if n.startswith("pet_"))
    face = next(n for n in samples if n.startswith("face_"))
    check("samples", bool(pet and face), f"{len(samples)} samples")
    s, _ = request(f"{api}/samples/{pet}")
    check("sample image", s == 200)

    s, body = request(api + "/corrupt", {"sample": pet, "condition": "occlusion", "severity": "high", "seed": "7"})
    check("corrupt", s == 200 and json.loads(body)["corruption"]["params"]["n"] == 3)
    for cond in ["clean", "salt_pepper", "blur", "occlusion"]:
        f = {"sample": pet, "condition": cond, "severity": "high", "seed": "1"}
        s, body = request(api + "/restore/universal", f)
        j = json.loads(body)
        check(f"universal/{cond}", s == 200 and j["output"].startswith("data:image/png"), f"{j.get('inference_ms')} ms")
        s, body = request(api + "/restore/hard", f)
        j = json.loads(body)
        check(f"hard/{cond}", s == 200 and j["predicted"] == cond, f"pred={j.get('predicted')} expert={j.get('expert')}")
        s, body = request(api + "/restore/moe", f)
        j = json.loads(body)
        top = max(j["weights"], key=j["weights"].get) if s == 200 else None
        check(f"moe/{cond}", s == 200 and top == cond and abs(sum(j["weights"].values()) - 1) < 1e-3,
              f"top={top} weights={j.get('weights')}")
    for style in (1, 2, 3):
        s, body = request(api + "/sketch", {"sample": face, "style": str(style)})
        check(f"sketch/style{style}", s == 200 and json.loads(body)["sketch"].startswith("data:image/png"))

    s, _ = request(api + "/sketch", {"sample": face, "style": "9"})
    check("reject bad style (422)", s == 422)
    s, _ = request(api + "/restore/moe", {}, {"file": ("x.png", b"not an image", "image/png")})
    check("reject undecodable file (415)", s == 415)
    s, _ = request(api + "/restore/moe", {}, {"file": ("x.txt", b"hello", "text/plain")})
    check("reject non-image type (415)", s == 415)
    s, _ = request(api + "/restore/moe", {}, {"file": ("big.jpg", b"0" * (11 * 2**20), "image/jpeg")})
    check("reject > 10 MB (413)", s == 413)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
