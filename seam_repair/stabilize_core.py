# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Core of Seam Stabilize and Overlap Blend: motion estimation (dense optical flow + robust homography),
trajectory smoothing with fixed anchor frames, local 'morph' smoothing, detail match and flow-based overlap
blending. numpy + OpenCV only, so it can be tested without ComfyUI. Frame I/O: core.Frame."""

import math
import os
import re
import shutil

import numpy as np

from ..image_io.readwrite import cv2

CV2_HINT = ("Seam Stabilize / Overlap Blend need OpenCV (cv2) in the Python environment of ComfyUI – the same package "
            "Save Media uses for PNG / TIFF. Install it with: <python> -m pip install opencv-python")

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DETAIL_SIGMAS = (1.0, 2.0, 4.0, 8.0)  # band edges in px: 0-1, 1-2, 2-4, 4-8


def need_cv2():
    if cv2 is None:
        raise RuntimeError(CV2_HINT)


# --------------------------------------------------------------------------
# Frame ranges
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"^(-?\d+)\s*(?:-\s*(-?\d+))?$")


def parse_seams(text, window, first, last):
    """'1668, 1872, 2060-2090' -> sorted list of windows {a, b, seams}. A single frame is a seam with +/- window
    around it (the seam is the first frame of the new section); 'a-b' sets the anchor frames directly.
    Overlapping windows are merged. a and b are the anchors: they stay exactly as they are."""
    wins = []
    text = re.sub(r"\s*-\s*(?=\d)", "-", (text or "").strip())
    for tok in re.split(r"[,;\s]+", text):
        tok = tok.strip()
        if not tok:
            continue
        m = _TOKEN_RE.match(tok.replace(" ", ""))
        if not m:
            raise ValueError(f"Cannot read '{tok}' in seams – use frame numbers like 1668 or ranges like 1650-1700.")
        if m.group(2) is None:
            s = int(m.group(1))
            a, b, seams = s - int(window), s + int(window), [s]
        else:
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            seams = []
        a, b = max(a, first), min(b, last)
        if b - a < 2:
            raise ValueError(f"'{tok}': the range {a}-{b} is too short or outside the sequence ({first}-{last}).")
        wins.append({"a": a, "b": b, "seams": seams})
    wins.sort(key=lambda w: w["a"])
    merged = []
    for w in wins:
        if merged and w["a"] <= merged[-1]["b"]:
            m = merged[-1]
            m["b"] = max(m["b"], w["b"])
            m["seams"] = sorted(set(m["seams"] + w["seams"]))
        else:
            merged.append(dict(w))
    return merged


def parse_frames(text):
    """'1668, 2484' -> [1668, 2484] (detail seams)."""
    out = []
    for tok in re.split(r"[,;\s]+", (text or "").strip()):
        if tok:
            if not re.match(r"^-?\d+$", tok):
                raise ValueError(f"Cannot read '{tok}' in detail_seams – use frame numbers like 1668.")
            out.append(int(tok))
    return sorted(set(out))


def taper(n, ramp=None):
    """Weights for the n frames of a window: 0 at both anchors, flat 1 in the middle, raised-cosine ramps."""
    if n < 3:
        return np.zeros(n, dtype=np.float64)
    ramp = max(1, int(round(ramp if ramp else n / 6.0)))
    d = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float64)
    w = np.where(d >= ramp, 1.0, 0.5 - 0.5 * np.cos(np.pi * d / ramp))
    w[0] = w[-1] = 0.0
    return w


def gauss1d(x, sigma):
    """Gaussian along axis 0 with edge padding (no shrinking towards zero at the ends)."""
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0:
        return x.copy()
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(r, r)] + [(0, 0)] * (x.ndim - 1)
    xp = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x)
    for i, kv in enumerate(k):
        out += kv * xp[i:i + len(x)]
    return out


# --------------------------------------------------------------------------
# Unchanged frames
# --------------------------------------------------------------------------
def link_or_copy(src, dst, hardlink=True):
    """Unchanged frame: hard link (no extra space, same file) or copy. Falls back to a copy across drives."""
    if hardlink:
        try:
            os.link(src, dst)
            return "link"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


# --------------------------------------------------------------------------
# Color for analysis (display-referred, HDR values kept)
# --------------------------------------------------------------------------
def encode(x):
    """linear -> sRGB curve, unclamped (values above 1 and below 0 keep the curve)."""
    s = np.sign(x)
    a = np.abs(x)
    return (s * np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1 / 2.4) - 0.055)).astype(np.float32)


def decode(x):
    s = np.sign(x)
    a = np.abs(x)
    return (s * np.where(a <= 0.04045, a / 12.92, np.power((a + 0.055) / 1.055, 2.4))).astype(np.float32)


def proxy(rgb, linear, size):
    """Small 8-bit luminance image for motion analysis; returns (proxy, scale full/proxy)."""
    need_cv2()
    h, w = rgb.shape[:2]
    f = max(w, h) / float(size) if max(w, h) > size else 1.0
    pw, ph = max(16, int(round(w / f))), max(16, int(round(h / f)))
    small = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
    if linear:
        small = encode(small)
    y = np.clip(small @ LUMA, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8), (w / float(pw), h / float(ph))


# --------------------------------------------------------------------------
# Motion
# --------------------------------------------------------------------------
def flow(a, b):
    """Dense optical flow a -> b (HxWx2, px): where each pixel of a is found in b. DIS (OpenCV), no download."""
    need_cv2()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(a, b, None)


def pair_homography(a, b, step=8, margin=0.04):
    """Global motion a -> b as a 3x3 homography, robust against moving objects (RANSAC on flow samples)."""
    fl = flow(a, b)
    h, w = a.shape
    mx, my = int(w * margin), int(h * margin)
    ys, xs = np.mgrid[my:h - my:step, mx:w - mx:step]
    p = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    q = p + fl[ys.ravel(), xs.ravel()]
    H, inl = cv2.findHomography(p, q, cv2.RANSAC, 1.0, maxIters=3000, confidence=0.995)
    if H is None or not np.all(np.isfinite(H)):
        return np.eye(3), 0.0
    return H / H[2, 2], float(inl.mean()) if inl is not None else 0.0


def apply_h(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def homography_from(p, q):
    return cv2.getPerspectiveTransform(np.float32(p), np.float32(q)).astype(np.float64)


def decompose(p, q):
    """Corner pin q (4x2) relative to the frame corners p: position (centroid), log scale, rotation and the
    perspective residual (4 complex corner offsets in px, after removing the similarity)."""
    pc = (p[:, 0] - p[:, 0].mean()) + 1j * (p[:, 1] - p[:, 1].mean())
    cq = q.mean(axis=0)
    qc = (q[:, 0] - cq[0]) + 1j * (q[:, 1] - cq[1])
    z = np.sum(np.conj(pc) * qc) / np.sum(np.abs(pc) ** 2)
    s, th = abs(z), np.angle(z)
    r = np.conj(np.exp(1j * th)) * qc / s - pc
    return cq, math.log(s), th, r


def compose(p, cq, logs, th, r):
    pc = (p[:, 0] - p[:, 0].mean()) + 1j * (p[:, 1] - p[:, 1].mean())
    qc = math.exp(logs) * np.exp(1j * th) * (pc + r)
    return np.stack([qc.real + cq[0], qc.imag + cq[1]], axis=1)


def scale_about(c, s):
    """Zoom by s about c (output -> source mapping): z' = c + (z - c) / s."""
    T = np.array([[1 / s, 0, c[0] * (1 - 1 / s)], [0, 1 / s, c[1] * (1 - 1 / s)], [0, 0, 1]], dtype=np.float64)
    return T


def fill_scale(A, w, h, steps=12):
    """Smallest zoom >= 1 so that the output frame samples only inside the source (no edges come in)."""
    pts = np.array([[0, 0], [w / 2, 0], [w, 0], [w, h / 2], [w, h], [w / 2, h], [0, h], [0, h / 2]], float)
    c = (w / 2.0, h / 2.0)

    def inside(s):
        x = apply_h(A @ scale_about(c, s), pts)
        return np.all((x[:, 0] >= -0.5) & (x[:, 0] <= w - 0.5) & (x[:, 1] >= -0.5) & (x[:, 1] <= h - 0.5))

    if inside(1.0):
        return 1.0
    lo, hi = 1.0, 1.05
    while not inside(hi) and hi < 4.0:
        lo, hi = hi, hi * 1.25
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if inside(mid) else (mid, hi)
    return hi


def window_motion(proxies, scale, w, h, progress=None):
    """Raw camera path of a window: corner pins q_k (full-res px) of the first frame's corners, chained
    frame to frame. Returns (p, q list, inlier ratios)."""
    sx, sy = scale
    S = np.diag([sx, sy, 1.0])
    Si = np.diag([1 / sx, 1 / sy, 1.0])
    p = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    C = np.eye(3)
    qs, inl = [p.copy()], []
    for k in range(len(proxies) - 1):
        Hk, ratio = pair_homography(proxies[k], proxies[k + 1])
        C = Hk @ C
        C /= C[2, 2]
        qs.append(apply_h(S @ C @ Si, p))
        inl.append(ratio)
        if progress:
            progress()
    return p, qs, inl


COMPONENTS = ("position", "rotation", "scale", "perspective")


def smooth_path(p, qs, enabled, strength, sigma, weights):
    """Target corner pins: for each enabled component raw + strength * w * (smoothed - raw).
    Returns (targets, curves) – curves holds raw / target per component for the report."""
    n = len(qs)
    parts = [decompose(p, q) for q in qs]
    pos = np.array([pt[0] for pt in parts])
    logs = np.array([pt[1] for pt in parts])
    th = np.unwrap(np.array([pt[2] for pt in parts]))
    res = np.array([np.r_[pt[3].real, pt[3].imag] for pt in parts])
    wv = (strength * weights)[:, None]
    t_pos, t_logs, t_th, t_res = pos.copy(), logs.copy(), th.copy(), res.copy()
    if enabled.get("position"):
        t_pos = pos + wv * (gauss1d(pos, sigma) - pos)
    if enabled.get("scale"):
        t_logs = logs + wv[:, 0] * (gauss1d(logs, sigma) - logs)
    if enabled.get("rotation"):
        t_th = th + wv[:, 0] * (gauss1d(th, sigma) - th)
    if enabled.get("perspective"):
        t_res = res + wv * (gauss1d(res, sigma) - res)
    targets = [compose(p, t_pos[k], t_logs[k], t_th[k], t_res[k][:4] + 1j * t_res[k][4:]) for k in range(n)]
    curves = {
        "x": (pos[:, 0], t_pos[:, 0]), "y": (pos[:, 1], t_pos[:, 1]),
        "scale": (np.exp(logs - logs[0]) * 100.0, np.exp(t_logs - logs[0]) * 100.0),
        "rotation": (np.degrees(th - th[0]), np.degrees(t_th - th[0])),
        "perspective": (np.hypot(res[:, :4], res[:, 4:]).mean(axis=1),
                        np.hypot(t_res[:, :4], t_res[:, 4:]).mean(axis=1)),
    }
    return targets, curves


def jump(values, idx):
    """Size of a jerk at a seam: second difference of the path at the seam (the frame before vs. the seam frame
    against their neighbours)."""
    v = np.asarray(values, dtype=np.float64)
    best = 0.0
    for i in (idx - 1, idx):
        if 1 <= i < len(v) - 1:
            best = max(best, abs(v[i + 1] - 2 * v[i] + v[i - 1]))
    return best


# --------------------------------------------------------------------------
# Warping
# --------------------------------------------------------------------------
def warp(img, A, disp=None, border="extend edges"):
    """img HxW or HxWxC float32; A maps output px -> source px (3x3). disp (optional, HxWx2): local displacement,
    output(z) = warped(z - disp(z)). One resampling step (bicubic)."""
    need_cv2()
    h, w = img.shape[:2]
    mode = cv2.BORDER_CONSTANT if border.startswith("black") else cv2.BORDER_REPLICATE
    if disp is None:
        return cv2.warpPerspective(img, A, (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=mode,
                                   borderValue=0)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    zx, zy = xs - disp[..., 0], ys - disp[..., 1]
    den = A[2, 0] * zx + A[2, 1] * zy + A[2, 2]
    mx = ((A[0, 0] * zx + A[0, 1] * zy + A[0, 2]) / den).astype(np.float32)
    my = ((A[1, 0] * zx + A[1, 1] * zy + A[1, 2]) / den).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=mode, borderValue=0)


def resize_flow(fl, w, h):
    sx, sy = w / float(fl.shape[1]), h / float(fl.shape[0])
    out = cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR)
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


def morph_fields(proxies, A_proxy, weights, strength, passes=3, blur_frac=0.012):
    """Local smoothing on top of the global correction: every interior frame is moved towards the midpoint of where
    its pixels are found in the previous and the next frame (a Laplacian step on each pixel's path), so a local
    jump at the seam is spread out while constant motion stays as it is. Returns displacement fields (proxy px)."""
    n = len(proxies)
    h, w = proxies[0].shape
    base = [cv2.warpPerspective(proxies[k], A_proxy[k], (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE) for k in range(n)]
    D = [np.zeros((h, w, 2), np.float32) for _ in range(n)]
    sig = max(1.0, blur_frac * max(w, h))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(passes):
        cur = [cv2.remap(base[k], xs - D[k][..., 0], ys - D[k][..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE) for k in range(n)]
        upd = []
        for k in range(1, n - 1):
            lam = 0.5 * strength * weights[k]
            if lam <= 0:
                upd.append(None)
                continue
            c = 0.5 * (flow(cur[k], cur[k - 1]) + flow(cur[k], cur[k + 1]))
            upd.append(lam * cv2.GaussianBlur(c, (0, 0), sig))
        for k, u in zip(range(1, n - 1), upd):
            if u is not None:
                D[k] += u
    return D


# --------------------------------------------------------------------------
# Detail match
# --------------------------------------------------------------------------
def _bands(y):
    need_cv2()
    levels = [y] + [cv2.GaussianBlur(y, (0, 0), s) for s in DETAIL_SIGMAS]
    return [levels[i] - levels[i + 1] for i in range(len(DETAIL_SIGMAS))]


def detail_energy(rgb, linear, crop=None):
    """Mean absolute value of each detail band of the luminance (display-referred), borders excluded.
    crop: measure only a centred crop of this size (full resolution) – much faster for long sequences."""
    if crop:
        h, w = rgb.shape[:2]
        ch, cw = min(h, crop), min(w, crop)
        rgb = rgb[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    y = (encode(rgb) if linear else rgb) @ LUMA
    h, w = y.shape
    m = max(8, int(0.03 * min(w, h)))
    return np.array([float(np.mean(np.abs(b[m:h - m, m:w - m]))) for b in _bands(np.ascontiguousarray(y))])


def deflicker_gains(energy, base_gains, sigma, strength=1.0, max_gain=2.5):
    """Per-frame band gains that bring every frame to the smoothed detail curve of the sequence.
    energy: (n, bands) measured per frame; base_gains: (n, bands) step gains of detail seams (1 = none) – applied
    first, so a hard step is matched and not just smeared. The curve is smoothed in log space over time."""
    e = np.maximum(np.asarray(energy, np.float64), 1e-7)
    b = np.log(e * np.asarray(base_gains, np.float64))
    target = np.exp(gauss1d(b, sigma))
    g = np.clip(target / e, 1.0 / max_gain, max_gain)
    return 1.0 + float(strength) * (g - 1.0)


def detail_gains(ref, tgt, strength=1.0, max_gain=2.5):
    g = np.where(tgt > 1e-7, ref / np.maximum(tgt, 1e-7), 1.0)
    g = np.clip(g, 1.0 / max_gain, max_gain)
    return 1.0 + float(strength) * (g - 1.0)


def apply_detail(rgb, linear, gains):
    """Adds (gain - 1) x each band of the luminance to all three channels (no color fringes), in display space."""
    d = encode(rgb) if linear else rgb
    y = np.ascontiguousarray(d @ LUMA)
    add = np.zeros_like(y)
    for g, b in zip(gains, _bands(y)):
        if abs(g - 1.0) > 1e-4:
            add += (g - 1.0) * b
    d = d + add[..., None]
    return decode(d) if linear else d


# --------------------------------------------------------------------------
# Overlap blend
# --------------------------------------------------------------------------
def blend_curve(t, curve):
    return t * t * (3 - 2 * t) if curve.startswith("smooth") else t


def flow_blend(a, b, t, size, linear, alpha_a=None, alpha_b=None):
    """Morphing transition: both frames are moved towards each other along the optical flow before they are mixed,
    so nothing doubles (ghosting) when the two takes differ slightly in position."""
    h, w = a.shape[:2]
    pa, _ = proxy(a, linear, size)
    pb, _ = proxy(b, linear, size)
    f_ab = resize_flow(flow(pa, pb), w, h)
    f_ba = resize_flow(flow(pb, pa), w, h)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    wa = cv2.remap(a, xs - t * f_ab[..., 0], ys - t * f_ab[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, xs - (1 - t) * f_ba[..., 0], ys - (1 - t) * f_ba[..., 1], cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_REPLICATE)
    rgb = (1 - t) * wa + t * wb
    alpha = None
    if alpha_a is not None or alpha_b is not None:
        aa = alpha_a if alpha_a is not None else np.ones((h, w), np.float32)
        ab = alpha_b if alpha_b is not None else np.ones((h, w), np.float32)
        aa = cv2.remap(aa, xs - t * f_ab[..., 0], ys - t * f_ab[..., 1], cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
        ab = cv2.remap(ab, xs - (1 - t) * f_ba[..., 0], ys - (1 - t) * f_ba[..., 1], cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
        alpha = (1 - t) * aa + t * ab
    return rgb.astype(np.float32), alpha
