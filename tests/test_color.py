# test_color.py
import os
import tempfile
import numpy as np
from PIL import Image

from similarity_measures import compute_color_feature


def _save_solid(rgb, path, size=(64, 64), fmt="JPEG"):
    img = Image.new("RGB", size, rgb)
    img.save(path, format=fmt, quality=95)


def _bin_coord(argmax_idx, bins=8):
    """Convert flat argmax bin -> (L,A,B) (or H,S,V) 3D bin coordinates."""
    return np.array(np.unravel_index(int(argmax_idx), (bins, bins, bins)), dtype=np.float32)


def test_color_feature_shape_and_norm():
    with tempfile.TemporaryDirectory() as d:
        p_red = os.path.join(d, "red.jpg")
        _save_solid((255, 0, 0), p_red)

        v = compute_color_feature(p_red, bins=8)
        # shape & dtype
        assert v.ndim == 1
        assert v.shape[0] == 8 * 8 * 8
        assert v.dtype == np.float32
        # finite & normalized (L1)
        assert np.all(np.isfinite(v))
        assert np.isclose(float(np.sum(v)), 1.0, atol=1e-3)


def test_color_family_bin_proximity():
    """
    With hard-binned histograms, solid patches often land in a single bin.
    Kırmızı ve koyu kırmızı, kırmızı ve yeşile göre bin-uzayında daha yakın olmalı.
    """
    with tempfile.TemporaryDirectory() as d:
        p_red = os.path.join(d, "red.jpg")
        p_darkred = os.path.join(d, "darkred.jpg")
        p_green = os.path.join(d, "green.jpg")
        _save_solid((255, 0, 0), p_red)
        _save_solid((128, 0, 0), p_darkred)
        _save_solid((0, 255, 0), p_green)

        bins = 8
        v_r = compute_color_feature(p_red, bins=bins)
        v_dr = compute_color_feature(p_darkred, bins=bins)
        v_g = compute_color_feature(p_green, bins=bins)

        # En baskın binler (çoğunlukla tek bin dolu olur düz renklerde)
        ir = int(np.argmax(v_r))
        idr = int(np.argmax(v_dr))
        ig = int(np.argmax(v_g))

        # Bin koordinatlarına (L,A,B) gidip öklid mesafelerini karşılaştır
        cr = _bin_coord(ir, bins=bins)
        cdr = _bin_coord(idr, bins=bins)
        cg = _bin_coord(ig, bins=bins)

        dist_r_dr = float(np.linalg.norm(cr - cdr))
        dist_r_g  = float(np.linalg.norm(cr - cg))

        assert dist_r_dr <= dist_r_g, f"Expected red~darkred <= red~green in bin-space, got {dist_r_dr} vs {dist_r_g}"