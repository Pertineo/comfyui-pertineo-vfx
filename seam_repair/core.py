# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Seam detection, repair planning and format-preserving frame I/O (no torch in here)."""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..image_io.readwrite import HAS_OPENEXR, OpenEXR, cv2, is_exr, linear_to_srgb, read_rgba_raw


# --------------------------------------------------------------------------
# Color
# --------------------------------------------------------------------------
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _analysis_frame(path, size, exr_is_linear):
    """(small grey frame for motion, sharpness) – display-referred, so differences are perceptual.
    Sharpness = mean mid-frequency detail of a full-resolution centre crop (max. 1024 px)."""
    rgb, _ = read_rgba_raw(path)
    if is_exr(path) and exr_is_linear:
        rgb = linear_to_srgb(rgb)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    grey = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)
    h, w = grey.shape
    ch, cw = min(h, 1024), min(w, 1024)
    crop = grey[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]
    if cv2 is not None:  # mid-frequency detail (band 1-2.5 px), so grain does not dominate
        sharp = float(np.mean(np.abs(cv2.GaussianBlur(crop, (0, 0), 1.0) - cv2.GaussianBlur(crop, (0, 0), 2.5))))
    else:
        sharp = float(np.mean(np.abs(np.diff(crop, axis=0))) + np.mean(np.abs(np.diff(crop, axis=1))))
    s = size / float(max(h, w))
    if s < 1.0 and cv2 is not None:
        grey = cv2.resize(grey, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    elif s < 1.0:
        step = max(1, int(round(1.0 / s)))
        grey = grey[::step, ::step]
    return grey, sharp


def _flow(a, b):
    """Optical flow between two grey frames at analysis size -> (mean length px, median dx, median dy)."""
    a8 = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    b8 = np.clip(b * 255.0 + 0.5, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 4, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag)), float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))


def analyse(paths, size=256, exr_is_linear=True, workers=8, progress=None):
    """Per frame i, compared with frame i-1 (index 0 = no previous frame):
    diff   mean absolute grey difference (0-1)            -> duplicate frames
    motion mean optical-flow length (px at analysis size)  -> motion spikes
    vx, vy median flow = camera motion per frame           -> offsets, speed / direction changes
    sharp  sharpness of the frame itself                   -> quality changes"""
    n = len(paths)
    m = {k: np.zeros(n, dtype=np.float64) for k in ("diff", "motion", "vx", "vy", "sharp")}
    prev = None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for i, (g, sharp) in enumerate(ex.map(lambda p: _analysis_frame(p, size, exr_is_linear), paths)):
            m["sharp"][i] = sharp
            if prev is not None:
                if g.shape != prev.shape:
                    raise ValueError(f"Frame size changes inside the sequence at {os.path.basename(paths[i])}.")
                m["diff"][i] = float(np.mean(np.abs(g - prev)))
                if cv2 is not None:
                    m["motion"][i], m["vx"][i], m["vy"][i] = _flow(prev, g)
                else:
                    m["motion"][i] = m["diff"][i]
            prev = g
            if progress:
                progress(i + 1, n)
    return m


def _local_level(values, i, window, floor, exclude=0, lo_index=1):
    n = len(values)
    neigh = [values[j] for j in range(max(lo_index, i - window), min(n, i + window + 1)) if abs(j - i) > exclude]
    return max(float(np.median(neigh)) if neigh else floor, floor)


def _step(series, i, k, lo_index):
    """Median of k values from i on minus median of k values before i (shorter at the ends).
    Medians, so a single-frame spike (offset, duplicate) is not mistaken for a lasting change."""
    before = series[max(lo_index, i - k):i]
    after = series[i:i + k]
    if len(before) == 0 or len(after) == 0:
        return 0.0, None
    return np.median(after, axis=0) - np.median(before, axis=0), (len(before), len(after))


def _peaks(score, thr, absolute, min_abs, radius):
    """Indices that are above thr (and min_abs) and the maximum of their neighbourhood."""
    out = []
    n = len(score)
    for i in range(1, n):
        if score[i] > thr and absolute[i] > min_abs:
            left = score[max(1, i - radius):i]
            right = score[i + 1:min(n, i + radius + 1)]
            if (len(left) == 0 or score[i] > left.max()) and (len(right) == 0 or score[i] >= right.max()):
                out.append(i)
    return out


SEAM_TYPES = ("jump", "speed / direction", "quality", "duplicate")


def classify(m, sensitivity=3.0, window=8, k=5, min_motion=0.5, min_quality=0.08, dup_ratio=0.2,
             min_diff=0.002, detect_duplicates=True, detect_quality=True):
    """Scores every cut i (between frame i-1 and i) for four kinds of seams, each as
    'times above the normal level of this shot':
      jump              camera motion jumps for one frame (offset) or the motion spikes
      speed / direction camera motion runs on with another speed or direction
      quality           sharpness steps up or down and stays there
      duplicate         frame (almost) identical to the previous one while the shot moves
    Returns (combined_score[n], kind[n] (index into SEAM_TYPES or -1), found{idx: kind})."""
    n = len(m["motion"])
    found = {}
    if n < 4:
        return np.zeros(n), np.full(n, -1), found
    v = np.stack([m["vx"], m["vy"]], axis=1)

    # jump: deviation of the camera motion from the mean of its neighbours (+ motion magnitude spike)
    jerk = np.zeros(n)
    for i in range(1, n):
        nb = [v[j] for j in (i - 1, i + 1) if 1 <= j < n]
        jerk[i] = float(np.linalg.norm(v[i] - np.mean(nb, axis=0)))
    jerk_floor = max(float(np.median(jerk[1:])), 0.02)
    mot_floor = max(0.25 * float(np.median(m["motion"][1:])), 1e-3)
    s_jump = np.zeros(n)
    a_jump = np.zeros(n)
    for i in range(1, n):
        s1 = jerk[i] / _local_level(jerk, i, window, jerk_floor, exclude=1)
        s2 = m["motion"][i] / _local_level(m["motion"], i, window, mot_floor)
        s_jump[i] = max(s1, s2)
        a_jump[i] = max(jerk[i], m["motion"][i] - _local_level(m["motion"], i, window, mot_floor))

    # speed / direction: step in the camera motion
    stepv = np.zeros(n)
    for i in range(2, n):
        d, cnt = _step(v, i, k, 1)
        stepv[i] = 0.0 if cnt is None else float(np.linalg.norm(d))
    dv = np.linalg.norm(np.diff(v[1:], axis=0), axis=1) if n > 2 else np.zeros(1)
    step_floor = max(float(np.median(stepv[2:])), 0.5 * float(np.median(dv)), 0.03)
    s_speed = stepv / step_floor

    # quality: step in log sharpness
    s_qual = np.zeros(n)
    a_qual = np.zeros(n)
    if detect_quality:
        q = np.log(np.maximum(m["sharp"], 1e-6))
        stepq = np.zeros(n)
        kq = 3 * k  # a quality change lasts (a whole section): longer medians, so pulsing sharpness averages out
        for i in range(1, n):
            d, cnt = _step(q, i, kq, 0)
            stepq[i] = 0.0 if cnt is None else abs(float(d))
        # compared with the LOCAL level of the sharpness steps (+/- 50 frames, without the step itself): in a shot
        # whose sharpness pulses all the time only the steps that clearly stand out count as seams
        q_floor = max(float(np.median(stepq[1:])), 0.002)
        for i in range(1, n):
            s_qual[i] = stepq[i] / _local_level(stepq, i, 50, q_floor, exclude=kq)
        a_qual = stepq

    scores = np.stack([s_jump, s_speed, s_qual])
    absolutes = [a_jump, stepv, a_qual]
    mins = [min_motion, min_motion * 0.5, min_quality]
    combined = scores.max(axis=0)
    kind = scores.argmax(axis=0)
    for t in range(3):
        for i in _peaks(scores[t], sensitivity, absolutes[t], mins[t], 3 * k if t == 2 else k):
            if i not in found or scores[t][i] > scores[found[i]][i]:
                found[i] = t
    for i, t in found.items():
        kind[i] = t

    if detect_duplicates:
        d_floor = max(0.25 * float(np.median(m["diff"][1:])), 1e-6)
        for i in range(1, n):
            dl = _local_level(m["diff"], i, window, d_floor)
            if m["diff"][i] < dup_ratio * dl and dl > min_diff:
                found[i] = 3
                kind[i] = 3
                combined[i] = max(combined[i], sensitivity * 1.01, dl / max(m["diff"][i], 1e-9))
    return combined, kind, found


def merge_close(indices, score, min_distance):
    """Keeps only the strongest of seams that lie closer than min_distance to each other."""
    out = []
    for i in sorted(indices):
        if out and i - out[-1] < min_distance:
            if score[i] > score[out[-1]]:
                out[-1] = i
        else:
            out.append(i)
    return out


TYPE_COLORS = ((220, 60, 60), (90, 170, 230), (180, 110, 230), (230, 150, 40))


def draw_graph(frames, score, kind, sensitivity, seams, width=1200, height=360):
    """RGB uint8 graph of the seam score over the shot (log scale), threshold line and found seams
    coloured by type: jump red, speed / direction blue, quality violet, duplicate orange."""
    img = np.full((height, width, 3), 28, dtype=np.uint8)
    if cv2 is None or len(score) < 2:
        return img
    ml, mr, mt, mb = 50, 15, 26, 40
    pw, ph = width - ml - mr, height - mt - mb
    n = len(score)
    top = max(float(np.max(score)), sensitivity * 2.0)
    lt = np.log1p(top)

    def pt(i, v):
        x = ml + int(round((i / max(1, n - 1)) * pw))
        y = mt + ph - int(round(np.log1p(max(0.0, min(v, top))) / lt * ph))
        return x, y

    cv2.rectangle(img, (ml, mt), (ml + pw, mt + ph), (70, 70, 70), 1)
    for i in seams:
        x, _ = pt(i, 0)
        cv2.line(img, (x, mt), (x, mt + ph), TYPE_COLORS[max(0, int(kind[i]))], 1)
    _, ty = pt(0, sensitivity)
    cv2.line(img, (ml, ty), (ml + pw, ty), (120, 120, 120), 1)
    cv2.polylines(img, [np.array([pt(i, score[i]) for i in range(1, n)], np.int32)], False, (220, 220, 220), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k5 in range(5):
        i = int(round(k5 * (n - 1) / 4))
        x, _ = pt(i, 0)
        cv2.putText(img, str(frames[i]), (max(0, x - 18), height - 15), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"{sensitivity:g}x", (4, ty + 4), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    x = ml
    for label, col in (("score (grey, log)", (220, 220, 220)), ("threshold", (120, 120, 120)),
                       ("jump", TYPE_COLORS[0]), ("speed/direction", TYPE_COLORS[1]),
                       ("quality", TYPE_COLORS[2]), ("duplicate", TYPE_COLORS[3])):
        cv2.putText(img, label, (x, 16), font, 0.42, col, 1, cv2.LINE_AA)
        x += 12 + cv2.getTextSize(label, font, 0.42, 1)[0][0]
    return img


# --------------------------------------------------------------------------
# Repair planning
# --------------------------------------------------------------------------
def parse_frames(text):
    """'1097, 1193\\n1288  # comment' -> [1097, 1193, 1288]"""
    out = []
    for line in (text or "").splitlines():
        line = line.split("#", 1)[0]
        out += [int(x) for x in re.findall(r"\d+", line)]
    return sorted(set(out))


def plan_segments(seams, first, last, radius):
    """A seam at frame t is the cut between t-1 and t. Frames t-radius .. t+radius-1 are replaced,
    interpolated between the anchors t-radius-1 and t+radius. Overlapping windows are merged.
    Returns (segments [(anchor_a, anchor_b, [seams])], skipped [(seam, reason)])."""
    segs, skipped = [], []
    for t in sorted(set(seams)):
        if t <= first or t > last:
            skipped.append((t, f"outside the sequence ({first}-{last}) or its first frame"))
            continue
        lo, hi = max(first + 1, t - radius), min(last - 1, t + radius - 1)
        if lo > hi:
            skipped.append((t, "too close to the start/end of the sequence"))
            continue
        if segs and lo <= segs[-1][1] + 1:
            segs[-1][1] = max(segs[-1][1], hi)
            segs[-1][2].append(t)
        else:
            segs.append([lo, hi, [t]])
    return [(lo - 1, hi + 1, ts) for lo, hi, ts in segs], skipped


# --------------------------------------------------------------------------
# Format-preserving frame I/O
# --------------------------------------------------------------------------
class Frame:
    """One frame as stored: kind 'exr' or 'raster'. rgb/alpha float32 as stored (no color change)."""

    def __init__(self, path):
        self.path = path
        self.kind = "exr" if is_exr(path) else "raster"
        self.exr_channels = None   # name -> array (original dtype), OpenEXR path only
        self.exr_header = None
        self.dtype = None          # raster: np.uint8 / np.uint16 / float
        self.rgb, self.alpha = None, None
        if self.kind == "exr" and HAS_OPENEXR:
            with OpenEXR.File(path, separate_channels=True) as f:
                try:
                    self.exr_header = dict(f.header())
                except Exception:
                    self.exr_header = {}
                self.exr_channels = {k: np.array(v.pixels) for k, v in f.channels().items()}
            c = self.exr_channels
            if not all(k in c for k in ("R", "G", "B")):
                raise ValueError(f"EXR without plain R, G, B channels is not supported here: {path} "
                                 f"(channels: {', '.join(sorted(c))})")
            self.rgb = np.dstack([c["R"], c["G"], c["B"]]).astype(np.float32)
            self.alpha = c["A"].astype(np.float32) if "A" in c else None
        elif self.kind == "raster" and cv2 is not None:
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise IOError(f"Cannot read image: {path}")
            self.dtype = im.dtype
            f = im.astype(np.float32)
            if im.dtype == np.uint16:
                f /= 65535.0
            elif im.dtype == np.uint8:
                f /= 255.0
            if f.ndim == 2:
                self.rgb = np.dstack([f, f, f])
            else:
                self.rgb = np.ascontiguousarray(f[..., :3][..., ::-1])
                self.alpha = np.ascontiguousarray(f[..., 3]) if f.shape[2] == 4 else None
        else:
            self.rgb, self.alpha = read_rgba_raw(path)

    def write_like(self, path, rgb, alpha=None):
        """Writes rgb (+ alpha) in the same format, bit depth and channel layout as this frame.
        Extra EXR channels (not R, G, B, A) are taken from this frame."""
        if self.kind == "exr":
            if self.exr_channels is not None:
                chans = {}
                for name, arr in self.exr_channels.items():
                    src = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}.get(name)
                    if name == "A" and alpha is not None:
                        src = alpha
                    chans[name] = np.ascontiguousarray((arr if src is None else src).astype(arr.dtype))
                header = {"compression": self.exr_header.get("compression", OpenEXR.ZIP_COMPRESSION),
                          "type": OpenEXR.scanlineimage}
                with OpenEXR.File(header, chans) as f:
                    f.write(path)
                return
            if cv2 is None:
                raise IOError(f"Cannot write EXR without OpenEXR or OpenCV: {path}")
            bgr = rgb[..., ::-1]
            img = np.dstack([bgr, alpha]) if alpha is not None else bgr
            if not cv2.imwrite(path, np.ascontiguousarray(img.astype(np.float32)),
                               [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF]):
                raise IOError(f"Cannot write EXR: {path}")
            return
        if cv2 is None:
            raise IOError(f"Writing {os.path.splitext(path)[1]} needs OpenCV: {path}")
        bgr = rgb[..., ::-1]
        img = np.dstack([bgr, alpha]) if alpha is not None else bgr
        img = np.clip(img, 0.0, 1.0)
        if self.dtype == np.uint16:
            data = np.round(img * 65535.0).astype(np.uint16)
        elif self.dtype == np.uint8:
            data = np.round(img * 255.0).astype(np.uint8)
        else:
            data = img.astype(np.float32)
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.lower().endswith((".jpg", ".jpeg")) else []
        if not cv2.imwrite(path, np.ascontiguousarray(data), params):
            raise IOError(f"Cannot write: {path}")


# --------------------------------------------------------------------------
# Bridge window (generative seam repair)
# --------------------------------------------------------------------------
def bridge_window(seam, first, last, context, bridge, step=8):
    """Frame window for a video model around a seam (cut between seam-1 and seam).
    bridge frames centred on the cut are regenerated, context frames on both sides guide the model.
    Total length is raised to step*n+1 (LTX). Returns dict or raises ValueError."""
    total = 2 * context + bridge
    if step > 1 and (total - 1) % step:
        bridge += step - (total - 1) % step
        total = 2 * context + bridge
    b_first = seam - bridge // 2
    b_last = b_first + bridge - 1
    w_first, w_last = b_first - context, b_last + context
    if w_first < first:  # shift into the sequence, keep the bridge on the cut
        w_first, w_last = first, first + total - 1
    if w_last > last:
        w_last, w_first = last, last - total + 1
    if w_first < first:
        raise ValueError(f"The sequence ({first}-{last}) is shorter than the window ({total} frames).")
    b_first, b_last = max(b_first, w_first + 1), min(b_last, w_last - 1)
    return {"first": w_first, "last": w_last, "bridge_first": b_first, "bridge_last": b_last,
            "seam": seam, "length": w_last - w_first + 1}


def blend_weights(win, feather):
    """frame -> weight of the generated frame: 1 inside the bridge, ramps down over `feather`
    context frames on each side (crossfade into the original)."""
    w = {}
    for f in range(win["first"], win["last"] + 1):
        if win["bridge_first"] <= f <= win["bridge_last"]:
            w[f] = 1.0
        else:
            d = win["bridge_first"] - f if f < win["bridge_first"] else f - win["bridge_last"]
            if d <= feather:
                w[f] = 1.0 - d / (feather + 1.0)
    return w
