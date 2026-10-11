# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Trimap generation from a (coarse) matte, e.g. an LTX alpha."""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def _morph(mask, radius, grow):
    """Binary erode/dilate with a round kernel of the given radius in px."""
    if radius <= 0:
        return mask
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        m = mask.astype(np.uint8)
        out = cv2.dilate(m, k) if grow else cv2.erode(m, k)
        return out.astype(bool)
    # numpy fallback: distance-free approximation with a square kernel
    from numpy.lib.stride_tricks import sliding_window_view

    pad = np.pad(mask, radius, mode="edge")
    win = sliding_window_view(pad, (2 * radius + 1, 2 * radius + 1))
    return win.any(axis=(-1, -2)) if grow else win.all(axis=(-1, -2))


def make_trimap(matte, threshold=0.5, core_erode=16, outer_dilate=16,
                core_mask=None, garbage_mask=None):
    """matte: HxW float [0,1] -> trimap HxW float {0, 0.5, 1}.

    core  = matte > threshold, eroded inwards  -> 1
    band  = matte > threshold, dilated outwards -> 0.5 (minus core)
    rest                                        -> 0
    core_mask forces 1, garbage_mask forces 0 (garbage wins).
    """
    hard = matte > threshold
    core = _morph(hard, int(core_erode), grow=False)
    outer = _morph(hard, int(outer_dilate), grow=True)
    tri = np.where(core, 1.0, np.where(outer, 0.5, 0.0)).astype(np.float32)
    if core_mask is not None:
        tri[core_mask > 0.5] = 1.0
    if garbage_mask is not None:
        tri[garbage_mask > 0.5] = 0.0
    return tri


def trimap_overlay(image, tri):
    """Preview: background darkened, unknown band tinted orange, core as is."""
    out = image.copy()
    bg = tri <= 0.25
    unk = (tri > 0.25) & (tri < 0.75)
    out[bg] *= 0.35
    tint = np.array([1.0, 0.45, 0.1], np.float32)
    out[unk] = out[unk] * 0.4 + tint * 0.6
    return np.clip(out, 0, 1)
