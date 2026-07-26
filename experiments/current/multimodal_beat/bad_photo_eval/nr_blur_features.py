"""No-reference blur features computed from actual pixels (stronger than blur_score).

blur_score uses only 3 pre-computed columns. These read the image and add the classic
no-reference blur cues that target focus/softness directly:

  lap_var            variance of Laplacian (sharpness)
  reblur_ratio       lap_var(I) / lap_var(gaussian(I)) -- sharp images lose a LOT of
                     high-frequency when re-blurred; already-blurry images barely change
                     (this is the single most discriminative NR-blur cue)
  fft_highfreq_ratio fraction of FFT magnitude energy beyond 0.25*Nyquist (blur -> low)
  tenengrad          mean squared gradient
  grad_energy        mean gradient magnitude
  contrast           luminance std (to normalize the above)
  lap_over_contrast  lap_var / contrast^2  (texture-normalized sharpness)

Work on a copy scaled to longest side 768 (keeps enough detail for focus; caps cost).
Requires numpy, PIL, scipy.ndimage.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

EPS = 1e-8
_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _lum(path, longest=768):
    im = Image.open(path).convert("RGB")
    if max(im.size) > longest:
        im.thumbnail((longest, longest))
    a = np.asarray(im, dtype=np.float32) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _laplacian(g):
    from scipy.signal import convolve2d
    return convolve2d(g, _LAP, mode="valid")


def _fft_highfreq_ratio(g, cutoff=0.25):
    F = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    total = F.sum() + EPS
    return float(F[r > cutoff].sum() / total)


def extract_nr(path) -> dict:
    try:
        g = _lum(path)
    except Exception:
        return {k: np.nan for k in NR_KEYS} | {"nr_error": 1.0}
    lap = _laplacian(g)
    lap_var = float(lap.var())
    gb = gaussian_filter(g, sigma=1.5)
    lap_b = _laplacian(gb)
    reblur_ratio = float(lap_var / (lap_b.var() + EPS))
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    contrast = float(g.std())
    return {
        "nr_lap_var": lap_var,
        "nr_reblur_ratio": reblur_ratio,
        "nr_fft_highfreq_ratio": _fft_highfreq_ratio(g),
        "nr_tenengrad": float((gx ** 2 + gy ** 2).mean()),
        "nr_grad_energy": float(mag.mean()),
        "nr_contrast": contrast,
        "nr_lap_over_contrast": float(lap_var / (contrast ** 2 + EPS)),
        "nr_error": 0.0,
    }


NR_KEYS = ["nr_lap_var", "nr_reblur_ratio", "nr_fft_highfreq_ratio", "nr_tenengrad",
           "nr_grad_energy", "nr_contrast", "nr_lap_over_contrast"]


def extract_many(item_ids, id_index=None):
    """Resolve each item_id's first image and extract NR features. Returns DataFrame."""
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from embed_blocks import build_id_index, first_image
    idx = id_index or build_id_index()
    rows = []
    for i, iid in enumerate(item_ids):
        d = idx.get(str(iid))
        f = first_image(d) if d else None
        feat = extract_nr(f) if f else {k: np.nan for k in NR_KEYS} | {"nr_error": 1.0}
        rows.append({"item_id": str(iid), **feat})
        if (i + 1) % 200 == 0:
            print(f"  nr {i+1}/{len(item_ids)}", flush=True)
    return pd.DataFrame(rows)
