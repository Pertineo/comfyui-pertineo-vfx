# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – Compare: renders an A/B comparison of two image batches as a new batch (side by side, stacked,
wipe, difference, matte check) with labels and a frame-number burn-in. Watch it in Save Media's player or write it
out as a review video – no frontend extension needed."""

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..common import LOG
from .readwrite import cv2

CATEGORY = "Pertineo/Review"
MODES = ["side by side", "stacked (A top)", "wipe", "difference", "matte check"]
WIPE_DIRECTIONS = ["vertical line (A left)", "horizontal line (A top)"]
DIFF_STYLES = ["heatmap", "absolute (color)"]
MATTE_STYLES = ["red overlay", "green overlay", "checkerboard (cut out)", "matte only"]
BURN_IN = ["labels + frame", "labels", "frame", "off"]


def _np(t):
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return a.astype(np.float32, copy=False)


def _fit(frame, h, w):
    """Scales a frame into h x w without distortion (letterbox / pillarbox in black)."""
    fh, fw = frame.shape[:2]
    if (fh, fw) == (h, w):
        return frame
    s = min(h / fh, w / fw)
    nh, nw = max(1, int(round(fh * s))), max(1, int(round(fw * s)))
    if cv2 is not None:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        r = cv2.resize(frame, (nw, nh), interpolation=interp)
        if r.ndim == 2:
            r = r[..., None]
    else:
        idx_y = (np.arange(nh) / s).astype(int).clip(0, fh - 1)
        idx_x = (np.arange(nw) / s).astype(int).clip(0, fw - 1)
        r = frame[idx_y][:, idx_x]
    out = np.zeros((h, w) + frame.shape[2:], np.float32)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = r.reshape((nh, nw) + frame.shape[2:])
    return out


def _rgb(a):
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] == 1:
        return np.repeat(a, 3, -1)
    return a[..., :3]


def _heat(x):
    """0..1 -> inferno-like colormap (black -> purple -> orange -> yellow)."""
    x = np.clip(x, 0, 1)[..., None]
    stops = np.array([[0, 0, 0], [0.34, 0.06, 0.43], [0.86, 0.32, 0.23], [0.99, 0.67, 0.04], [0.99, 1.0, 0.64]],
                     np.float32)
    pos = x * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = pos - i
    return stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f


class _Burn:
    """Text burn-in with a dark box – rendered once per text with PIL, composited onto float frames."""
    _cache = {}

    def __init__(self, height):
        from .save_media import _font
        self.size = max(12, int(height / 36))
        self.font = _font(self.size)

    def stamp(self, frame, text, corner):
        if not text:
            return
        key = (text, self.size)
        if key not in self._cache:
            probe = ImageDraw.Draw(Image.new("L", (8, 8)))
            x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=self.font)
            pad = max(4, self.size // 3)
            img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
            ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=self.font, fill=255)
            self._cache[key] = np.asarray(img, np.float32) / 255.0
            if len(self._cache) > 4000:
                self._cache.clear()
        t = self._cache[key]
        th, tw = t.shape
        H, W = frame.shape[:2]
        m = max(6, self.size // 2)
        y = m if corner[0] == "t" else H - th - m
        x = m if corner[1] == "l" else W - tw - m
        if y < 0 or x < 0 or y + th > H or x + tw > W:
            return
        region = frame[y:y + th, x:x + tw]
        a = t[..., None]
        region *= 0.35                         # darkened box behind the text
        region[...] = region * (1 - a) + 0.95 * a  # light text


class PertineoCompare:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("The comparison as an image batch – connect it to Save Media (player / review video) or a "
                       "preview node.",)
    FUNCTION = "run"
    DESCRIPTION = ("Renders an A/B comparison of two image batches: side by side, stacked, wipe (fixed split), "
                   "difference (heatmap or colour, with gain) or matte check (a matte over A as colour overlay, "
                   "cut out on a checkerboard, or the matte alone). Labels and frame numbers are burnt in. Different "
                   "lengths: the shorter one holds its last frame. Different sizes: B is fitted into A's size "
                   "without distortion. View it in Save Media's player or write a review video.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "A – e.g. the source plate or the previous version."}),
                "image_b": ("IMAGE", {"tooltip": "B – e.g. the result. In matte check mode: a matte rendered as an "
                                                 "image (grey) – or connect the 'matte' input instead."}),
                "mode": (MODES, {
                    "tooltip": "side by side = A | B (double width). stacked = A above B (double height). wipe = A "
                               "and B split at wipe_position. difference = |A - B| (shows what changed). matte check "
                               "= the matte (B or 'matte') over A."}),
                "wipe_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "wipe: where the split line sits (0.5 = middle)."}),
                "wipe_direction": (WIPE_DIRECTIONS, {"tooltip": "wipe: vertical line (A left, B right) or "
                                                                "horizontal line (A top, B bottom)."}),
                "difference_style": (DIFF_STYLES, {
                    "tooltip": "difference: heatmap of the luminance difference (black = identical) or the colour "
                               "difference per channel."}),
                "difference_gain": ("FLOAT", {"advanced": True, "default": 4.0, "min": 1.0, "max": 100.0, "step": 0.5,
                    "tooltip": "difference: amplifies small changes (4 = a difference of 0.25 is shown at full "
                               "strength)."}),
                "matte_style": (MATTE_STYLES, {
                    "tooltip": "matte check: red / green overlay over A, A cut out on a checkerboard (shows edges "
                               "and holes), or the matte alone."}),
                "matte_opacity": ("FLOAT", {"advanced": True, "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "matte check: strength of the colour overlay."}),
                "burn_in": (BURN_IN, {"tooltip": "Text burnt into the frames: labels (A / B) and / or the frame "
                                                 "number (bottom right)."}),
                "label_a": ("STRING", {"default": "A", "tooltip": "Label of A (e.g. plate, v001)."}),
                "label_b": ("STRING", {"default": "B", "tooltip": "Label of B (e.g. comp, v002)."}),
                "first_frame": ("INT", {"advanced": True, "default": 1001, "min": 0, "max": 10_000_000,
                    "tooltip": "Frame number of the first frame for the burn-in (e.g. first_frame of the loader)."}),
            },
            "optional": {
                "matte": ("MASK", {"tooltip": "matte check: the matte to check (e.g. from ViTMatte). Without it, "
                                              "image_b is used as the matte (its luminance)."}),
            },
        }

    def run(self, image_a, image_b, mode, wipe_position, wipe_direction, difference_style, difference_gain,
            matte_style, matte_opacity, burn_in, label_a, label_b, first_frame, matte=None):
        A, Bimg = image_a, image_b
        n = max(int(A.shape[0]), int(Bimg.shape[0]), int(matte.shape[0]) if matte is not None else 0)
        H, W = int(A.shape[1]), int(A.shape[2])
        if mode == "side by side":
            oh, ow = H, 2 * W
        elif mode.startswith("stacked"):
            oh, ow = 2 * H, W
        else:
            oh, ow = H, W
        out = torch.empty((n, oh, ow, 3), dtype=torch.float32)
        burn = _Burn(oh) if burn_in != "off" else None
        show_labels = burn_in in ("labels + frame", "labels")
        show_frame = burn_in in ("labels + frame", "frame")
        line = max(1, H // 540)
        checker = None
        if (int(Bimg.shape[0]) != n or int(A.shape[0]) != n):
            print(f"{LOG} Compare: A has {int(A.shape[0])}, B {int(Bimg.shape[0])} frames – the shorter one holds "
                  f"its last frame.")
        for i in range(n):
            a = _rgb(_np(A[min(i, A.shape[0] - 1)]))
            b = _fit(_rgb(_np(Bimg[min(i, Bimg.shape[0] - 1)])), H, W)
            if mode == "side by side":
                f = np.concatenate([a, b], axis=1)
                if show_labels and burn:
                    burn.stamp(f[:, :W], label_a, "tl")
                    burn.stamp(f[:, W:], label_b, "tl")
            elif mode.startswith("stacked"):
                f = np.concatenate([a, b], axis=0)
                if show_labels and burn:
                    burn.stamp(f[:H], label_a, "tl")
                    burn.stamp(f[H:], label_b, "tl")
            elif mode == "wipe":
                f = a.copy()
                if wipe_direction.startswith("vertical"):
                    x = int(round(wipe_position * W))
                    f[:, x:] = b[:, x:]
                    f[:, max(0, x - line):min(W, x + line)] = 1.0
                    if show_labels and burn:
                        burn.stamp(f[:, :max(x, 1)], label_a, "tl")
                        burn.stamp(f[:, x:] if x < W else f, label_b, "tr")
                else:
                    y = int(round(wipe_position * H))
                    f[y:] = b[y:]
                    f[max(0, y - line):min(H, y + line)] = 1.0
                    if show_labels and burn:
                        burn.stamp(f[:max(y, 1)], label_a, "tl")
                        burn.stamp(f[y:] if y < H else f, label_b, "tl")
            elif mode == "difference":
                d = np.abs(a - b)
                if difference_style == "heatmap":
                    lum = 0.2126 * d[..., 0] + 0.7152 * d[..., 1] + 0.0722 * d[..., 2]
                    f = _heat(lum * difference_gain)
                else:
                    f = np.clip(d * difference_gain, 0, 1)
                f = np.ascontiguousarray(f, np.float32)
                if show_labels and burn:
                    burn.stamp(f, f"|{label_a} - {label_b}| x{difference_gain:g}", "tl")
            else:  # matte check
                if matte is not None:
                    m = _np(matte[min(i, matte.shape[0] - 1)])
                    m = _fit(m[..., None], H, W)[..., 0]
                else:
                    m = 0.2126 * b[..., 0] + 0.7152 * b[..., 1] + 0.0722 * b[..., 2]
                m = np.clip(m, 0, 1)[..., None]
                if matte_style.startswith("checkerboard"):
                    if checker is None:
                        s = max(8, H // 40)
                        yy, xx = np.mgrid[0:H, 0:W]
                        checker = np.where(((yy // s) + (xx // s)) % 2 == 0, 0.42, 0.58).astype(np.float32)[..., None]
                    f = a * m + checker * (1 - m)
                elif matte_style == "matte only":
                    f = np.repeat(m, 3, -1)
                else:
                    color = np.array([1, 0.1, 0.1] if matte_style.startswith("red") else [0.1, 1, 0.2], np.float32)
                    k = m * matte_opacity
                    f = a * (1 - k) + color * k
                f = np.ascontiguousarray(f, np.float32)
                if show_labels and burn:
                    burn.stamp(f, f"{label_a} + matte ({matte_style})", "tl")
            if show_frame and burn:
                burn.stamp(f, str(first_frame + i), "br")
            out[i] = torch.from_numpy(np.ascontiguousarray(np.clip(f, 0, 1), dtype=np.float32))
        print(f"{LOG} Compare: {mode}, {n} frames, {ow}x{oh}")
        return (out,)


# --------------------------------------------------------------------------
# Preview Frame – one frame of a batch, large on the node
# --------------------------------------------------------------------------
PICK = ["middle", "first", "last", "index"]
SHOW = ["image", "mask", "image | mask side by side", "mask over image (red)", "image cut out (checkerboard)"]


class PertineoPreviewFrame:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "mask", "index")
    OUTPUT_TOOLTIPS = ("ONLY the chosen frame of the images (a batch of 1) – not the sequence. Do not put this node "
                       "between a loader and a saver: connect the batch to the saver directly.",
                       "ONLY the chosen frame of the mask (a batch of 1).",
                       "Index of the chosen frame in the batch (0 = first).")
    OUTPUT_NODE = True
    FUNCTION = "run"
    DESCRIPTION = ("Shows ONE frame of an image and / or mask batch large on the node – the middle, first, last or "
                   "any frame by index – instead of the whole batch. To judge a matte: mask alone, side by side, red "
                   "overlay or cut out on a checkerboard. Full resolution (up to 4096 px). Note: the outputs carry only "
                   "this ONE frame, not the batch – use the node as a side branch for viewing, not in the chain.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame": (PICK, {"tooltip": "Which frame to show: middle, first, last, or 'index'."}),
                "index": ("INT", {"default": 0, "min": -100000, "max": 100000,
                    "tooltip": "Only for frame = index: position in the batch, 0 = first frame, -1 = last. Values "
                               "outside the batch are clamped."}),
                "show": (SHOW, {"tooltip": "image, mask, both side by side, the mask as red overlay over the image, "
                                           "or the image cut out by the mask on a checkerboard."}),
                "first_frame": ("INT", {"default": 1001, "min": 0, "max": 10_000_000,
                    "tooltip": "Frame number of the first frame – only for the label (e.g. first_frame of the "
                               "loader)."}),
                "label": ("BOOLEAN", {"default": True, "tooltip": "Burn the frame number into the preview."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Image batch."}),
                "mask": ("MASK", {"tooltip": "Mask / matte batch."}),
            },
        }

    def run(self, frame, index, show, first_frame, label, images=None, mask=None):
        import os
        import uuid
        import folder_paths
        if images is None and mask is None:
            raise ValueError(f"{LOG} Preview Frame: connect images and / or mask.")
        n = int(images.shape[0] if images is not None else mask.shape[0])
        if mask is not None and mask.ndim == 2:
            mask = mask[None]
        i = {"middle": n // 2, "first": 0, "last": n - 1}.get(frame)
        if i is None:
            i = index if index >= 0 else n + index
        i = max(0, min(n - 1, int(i)))
        img = _rgb(_np(images[min(i, images.shape[0] - 1)])) if images is not None else None
        m = _np(mask[min(i, mask.shape[0] - 1)]) if mask is not None else None
        if img is None:
            img = np.zeros(m.shape + (3,), np.float32)
        H, W = img.shape[:2]
        if m is not None and m.shape != (H, W):
            m = _fit(m[..., None], H, W)[..., 0]
        mm = None if m is None else np.clip(m, 0, 1)[..., None]
        if show == "mask" and mm is not None:
            f = np.repeat(mm, 3, -1)
        elif show.startswith("image | mask") and mm is not None:
            f = np.concatenate([img, np.repeat(mm, 3, -1)], axis=1)
        elif show.startswith("mask over") and mm is not None:
            k = mm * 0.5
            f = img * (1 - k) + np.array([1, 0.1, 0.1], np.float32) * k
        elif show.startswith("image cut out") and mm is not None:
            s = max(8, H // 40)
            yy, xx = np.mgrid[0:H, 0:W]
            checker = np.where(((yy // s) + (xx // s)) % 2 == 0, 0.42, 0.58).astype(np.float32)[..., None]
            f = img * mm + checker * (1 - mm)
        else:
            if show != "image":
                print(f"{LOG} Preview Frame: '{show}' needs a mask – showing the image.")
            f = img
        f = np.ascontiguousarray(np.clip(f, 0, 1), np.float32)
        if label:
            _Burn(f.shape[0]).stamp(f, f"{first_frame + i}  ({i + 1}/{n})", "br")
        pil = Image.fromarray((f * 255.0 + 0.5).astype(np.uint8))
        if max(pil.size) > 4096:
            pil.thumbnail((4096, 4096))
        name = f"pertineo_frame_{uuid.uuid4().hex[:10]}.png"
        pil.save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
        out_img = torch.from_numpy(np.ascontiguousarray(img, np.float32))[None]
        out_mask = torch.from_numpy(np.ascontiguousarray(m if m is not None else np.zeros((H, W), np.float32),
                                                         np.float32))[None]
        print(f"{LOG} Preview Frame: frame {first_frame + i} (index {i} of {n}), {show}")
        return {"ui": {"images": [{"filename": name, "subfolder": "", "type": "temp"}]},
                "result": (out_img, out_mask, i)}
