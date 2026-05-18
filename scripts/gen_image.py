"""Image gen via Google AI Studio API.

Two backends:
  --model imagen     → imagen-4.0-generate-001 (:predict endpoint, higher quality)
  --model nano       → gemini-2.5-flash-image (:generateContent, "Nano Banana")

Usage:
    .venv/bin/python scripts/gen_image.py --model imagen --aspect 16:9 \
        "prompt..." out.png
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
master = Path.home() / "Documents/Sync/secrets/env.master"
if master.exists():
    for line in master.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY=") and "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip()
            break


def gen_imagen(prompt: str, out_path: Path, aspect: str) -> None:
    key = os.environ["GEMINI_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "imagen-4.0-generate-001:predict?key=" + key
    )
    body = json.dumps({
        "instances": [{"prompt": prompt}],
        "parameters": {"sampleCount": 1, "aspectRatio": aspect},
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    preds = data.get("predictions") or []
    if not preds or "bytesBase64Encoded" not in preds[0]:
        raise SystemExit(f"no image in response: {json.dumps(data)[:400]}")
    img = base64.b64decode(preds[0]["bytesBase64Encoded"])
    out_path.write_bytes(img)
    print(f"wrote {len(img):,} bytes -> {out_path}")


def gen_nano(prompt: str, out_path: Path) -> None:
    key = os.environ["GEMINI_API_KEY"]
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash-image:generateContent?key=" + key
    )
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    for p in data["candidates"][0]["content"]["parts"]:
        if "inlineData" in p:
            img = base64.b64decode(p["inlineData"]["data"])
            out_path.write_bytes(img)
            print(f"wrote {len(img):,} bytes -> {out_path}")
            return
    raise SystemExit(f"no image in response: {json.dumps(data)[:400]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("imagen", "nano"), default="imagen")
    ap.add_argument("--aspect", default="16:9", help="imagen only; 1:1, 16:9, 9:16, 3:4, 4:3")
    ap.add_argument("prompt")
    ap.add_argument("out")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.model == "imagen":
            gen_imagen(args.prompt, out, args.aspect)
        else:
            gen_nano(args.prompt, out)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:600]
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
