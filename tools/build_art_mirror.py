#!/usr/bin/env python3
"""Build the flyer mirror: one small WebP per game, cropped and sized for a card.

    python3 tools/build_art_mirror.py <mirror repo dir> [cache dir]

Reads shmup_deck/app/art.json. For each game it downloads the source scan
(once; kept in the cache dir), cuts it to the recorded crop box, scales it to
at most MAX_W wide (or MAX_H tall for flyers shown whole) and writes
<id>.webp into the mirror repo, plus a manifest.json of sizes and sources.
Sources are fetched a couple of seconds apart, with the Referer some hosts
need. Scans whose size no longer matches art.json are skipped and reported,
since the crop box would then be wrong.
"""

import hashlib
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MAX_W, MAX_H = 640, 900
QUALITY = 80
UA = "ShmupDeck-mirror/1.0 (+https://github.com/shmupfan/shmup-deck)"
ROOT = Path(__file__).resolve().parent.parent


def fetch(url, referer, cache):
    key = cache / (hashlib.sha1(url.encode()).hexdigest() + ".img")
    if key.exists():
        return key.read_bytes()
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        data = r.read()
    key.write_bytes(data)
    time.sleep(2)
    return data


def tone(im):
    """Even out a scan's levels so the wall looks like one set of flyers.

    Each step moves the scan only part of the way to a shared look, and is
    capped, so a dark or vivid flyer stays dark or vivid: black and white
    points stretched to the ends, a slight tint in the paper (yellowing)
    taken out, mid-tone brightness and saturation moved halfway to common
    levels. art.json's "tone" says which version of this pass a flyer gets
    (0 = none); a new version is a new number, so devices fetch it again.
    """
    a = np.asarray(im, dtype=np.float64) / 255

    def luma(a):
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114

    # levels: 0.5% / 99.5% luminance to black and white, same scale on every channel
    lo, hi = np.percentile(luma(a), [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 0.2), 0, 1)
    # paper: neutralise a mild tint in the brightest 2%; a strong one is the artwork's colour
    L = luma(a)
    m = a[L >= np.percentile(L, 98)].mean(0)
    if np.abs(m - m.mean()).max() < 0.12:
        a = np.clip(a * (1 + (m.mean() / np.maximum(m, 1e-3) - 1) * 0.8), 0, 1)
    # brightness: a gamma that takes the median luminance halfway to 0.42
    g = np.log(0.42) / np.log(np.clip(np.median(luma(a)), 0.05, 0.95))
    a = a ** (1 + (np.clip(g, 0.7, 1.4) - 1) * 0.5)
    # saturation: halfway to a mean of 0.48
    L = luma(a)[..., None]
    mx, mn = a.max(2), a.min(2)
    k = np.clip(0.48 / max(float(np.mean((mx - mn) / np.maximum(mx, 1e-3))), 0.05), 0.8, 1.25)
    a = np.clip(L + (a - L) * (1 + (k - 1) * 0.5), 0, 1)
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def main(out_dir, cache_dir):
    out, cache = Path(out_dir), Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    art = json.loads((ROOT / "shmup_deck" / "app" / "art.json").read_text())
    manifest, problems = {}, []
    for gid, a in art.items():
        if gid.startswith("_"):
            continue
        try:
            im = Image.open(io.BytesIO(fetch(a["url"], a.get("referer"), cache)))
        except Exception as e:
            problems.append(f"{gid}: download failed ({e})")
            continue
        if list(im.size) != a["size"]:
            problems.append(f"{gid}: scan is {im.size}, art.json says {a['size']}")
            continue
        l, t, w, h = a["crop"]
        im = im.convert("RGB").crop((l, t, l + w, t + h))
        if a.get("tone") == 1:
            im = tone(im)
        scale = min(MAX_W / im.width, MAX_H / im.height, 1)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        dest = out / f"{gid}.webp"
        im.save(dest, "WEBP", quality=QUALITY, method=6)
        manifest[gid] = {"size": list(im.size), "bytes": dest.stat().st_size, "source": a["url"]}
        print(f"{gid:10} {im.width}x{im.height} {dest.stat().st_size // 1024} KB")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    total = sum(m["bytes"] for m in manifest.values())
    print(f"{len(manifest)} flyers, {total / 1e6:.1f} MB")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ROOT.parent / "shmup-deck-artcache")
