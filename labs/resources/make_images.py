#!/usr/bin/env python3
"""
Generate the ForgeSight surface-inspection image set for C539 (Labs 12-16).

Creates 1200 greyscale 64x64 PNGs of machined part surfaces in three classes:

    ok       clean machined surface (tool marks + sensor noise only)
    scratch  a thin, bright, directional line - high contrast, easy-ish
    dent     a soft, dark, rounded depression - low contrast, genuinely hard

Layout (torchvision.datasets.ImageFolder reads this directly):

    data/defects/train/{dent,ok,scratch}/   300 images per class
    data/defects/val/{dent,ok,scratch}/     100 images per class

ImageFolder assigns class indices ALPHABETICALLY, so the mapping is
dent=0, ok=1, scratch=2 - not the order above. Lab 13 depends on you noticing.

Deterministic (seed 42): every learner gets byte-identical images, so the
accuracies quoted in the Learner Guide hold.

Run from your torch-vibe/ project root:   python make_images.py
"""
import numpy as np
from pathlib import Path
from PIL import Image

SEED = 42
SIZE = 64
N_TRAIN, N_VAL = 300, 100
CLASSES = ["ok", "scratch", "dent"]
OUT = Path("data/defects")

rng = np.random.default_rng(SEED)
yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


def base_surface():
    """A machined surface: directional tool marks, illumination gradient, noise."""
    angle = rng.uniform(0, np.pi)
    freq = rng.uniform(0.55, 1.25)
    marks = 11.0 * np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * freq
                          + rng.uniform(0, 2 * np.pi))
    # Uneven lighting across the part - the main nuisance variable.
    gx, gy = rng.uniform(-0.30, 0.30), rng.uniform(-0.30, 0.30)
    light = gx * (xx - SIZE / 2) + gy * (yy - SIZE / 2)
    grain = rng.normal(0, 5.5, (SIZE, SIZE))
    return 138.0 + marks + light + grain


def add_scratch(img):
    """A thin bright line: high contrast, strongly directional."""
    a = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(16, 48), rng.uniform(16, 48)
    half = rng.uniform(13, 27)
    width = rng.uniform(0.55, 1.15)
    # Distance from the (infinite) line, masked to a finite segment.
    d_perp = np.abs((xx - cx) * np.sin(a) - (yy - cy) * np.cos(a))
    d_along = np.abs((xx - cx) * np.cos(a) + (yy - cy) * np.sin(a))
    ridge = np.exp(-(d_perp ** 2) / (2 * width ** 2))
    ridge *= (d_along < half) * np.clip(1.0 - (d_along / half) ** 6, 0, 1)
    return img + rng.uniform(52, 86) * ridge


def add_dent(img):
    """A soft dark depression: low contrast, rounded, no strong direction."""
    cx, cy = rng.uniform(18, 46), rng.uniform(18, 46)
    r = rng.uniform(6.5, 12.5)
    ecc = rng.uniform(0.75, 1.35)
    a = rng.uniform(0, np.pi)
    dx = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    dy = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    blob = np.exp(-((dx / r) ** 2 + (dy / (r * ecc)) ** 2))
    # Depressions darken the centre and catch a faint highlight on one rim.
    img = img - rng.uniform(26, 44) * blob
    img = img + rng.uniform(8, 17) * np.roll(blob, int(r * 0.55), axis=0) * 0.6
    return img


MAKERS = {"ok": lambda im: im, "scratch": add_scratch, "dent": add_dent}


def build(split, count):
    for cls in CLASSES:
        d = OUT / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = MAKERS[cls](base_surface())
            arr = np.clip(img, 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(d / f"{cls}_{i:03d}.png")
        print(f"  {split}/{cls:<8} {count:>4} images")


if __name__ == "__main__":
    print(f"Generating defect images into {OUT.resolve()}")
    build("train", N_TRAIN)
    build("val", N_VAL)
    total = (N_TRAIN + N_VAL) * len(CLASSES)
    print(f"\nDone: {total} images, {SIZE}x{SIZE} greyscale, {len(CLASSES)} classes.")
    print("ImageFolder class_to_idx will be: {'dent': 0, 'ok': 1, 'scratch': 2}")
