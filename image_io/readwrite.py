# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Image sequence I/O: PNG / TIFF / JPG (8 or 16 bit) and OpenEXR."""

import glob
import os
import re
from urllib.parse import unquote, urlparse

import numpy as np

# Allows OpenCV to read/write EXR as a fallback (must be set before cv2 is used).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import OpenEXR  # pip package "OpenEXR" >= 3.2

    HAS_OPENEXR = hasattr(OpenEXR, "File")
except ImportError:
    OpenEXR = None
    HAS_OPENEXR = False

from PIL import Image

EXR_HINT = ("EXR support needs the OpenEXR package. Install it into the Python environment of "
            "ComfyUI: <python> -m pip install OpenEXR")

IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".exr")
_FRAME_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


# --------------------------------------------------------------------------
# Color
# --------------------------------------------------------------------------
def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


# --------------------------------------------------------------------------
# EXR
# --------------------------------------------------------------------------
def _exr_channels(path):
    """Returns dict channel name -> HxW float32."""
    if HAS_OPENEXR:
        with OpenEXR.File(path, separate_channels=True) as f:
            chans = f.channels()
            return {k: np.asarray(v.pixels, dtype=np.float32) for k, v in chans.items()}
    im = None
    if cv2 is not None:
        try:
            im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except cv2.error:
            im = None
    if im is None:
        raise IOError(f"Cannot read EXR: {path}. {EXR_HINT}")
    im = im.astype(np.float32)
    if im.ndim == 2:
        return {"Y": im}
    names = ["B", "G", "R", "A"][: im.shape[2]]
    return {n: im[..., i] for i, n in enumerate(names)}


def _pick(chans, prefs):
    for p in prefs:
        if p in chans:
            return chans[p]
    # fall back to the first channel of the main layer
    for k in sorted(chans):
        if "." not in k:
            return chans[k]
    return chans[sorted(chans)[0]]


def _read_exr_rgb(path):
    c = _exr_channels(path)
    if all(k in c for k in ("R", "G", "B")):
        return np.dstack([c["R"], c["G"], c["B"]])
    y = _pick(c, ["Y", "R", "A"])
    return np.dstack([y, y, y])


def _read_exr_single(path):
    return _pick(_exr_channels(path), ["R", "Y", "A"])


def write_exr(path, alpha):
    a = np.ascontiguousarray(alpha.astype(np.float32))
    if HAS_OPENEXR:
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        channels = {"R": a, "G": a, "B": a, "A": a}
        with OpenEXR.File(header, channels) as f:
            f.write(path)
        return
    if cv2 is not None:
        img = np.dstack([a, a, a, a])
        params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]
        try:
            if cv2.imwrite(path, img, params):
                return
        except cv2.error:
            pass
    raise IOError(f"Cannot write EXR: {path}. {EXR_HINT}")


# --------------------------------------------------------------------------
# PNG / TIFF / JPG
# --------------------------------------------------------------------------
def _read_raster(path):
    if cv2 is not None:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"Cannot read image: {path}")
        if im.ndim == 3:
            im = im[..., :3][..., ::-1]  # BGR(A) -> RGB
    else:
        im = np.array(Image.open(path))
        if im.ndim == 3:
            im = im[..., :3]
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    elif im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    else:
        im = im.astype(np.float32)
    return np.ascontiguousarray(im)


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------
def is_exr(path):
    return path.lower().endswith(".exr")


def read_plate(path, exr_linear_to_srgb=True):
    """Returns HxWx3 float32 sRGB in [0,1]."""
    if is_exr(path):
        im = _read_exr_rgb(path)
        im = linear_to_srgb(im) if exr_linear_to_srgb else np.clip(im, 0, 1)
        return np.ascontiguousarray(im.astype(np.float32))
    im = _read_raster(path)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, axis=-1)
    return im


def _to_float(im):
    if im.dtype == np.uint16:
        return im.astype(np.float32) / 65535.0
    if im.dtype == np.uint8:
        return im.astype(np.float32) / 255.0
    return im.astype(np.float32)


def read_rgba_raw(path):
    """Returns (rgb HxWx3, alpha HxW or None) as stored in the file (no color conversion).
    Grayscale files are expanded to RGB."""
    if is_exr(path):
        c = _exr_channels(path)
        if all(k in c for k in ("R", "G", "B")):
            rgb = np.dstack([c["R"], c["G"], c["B"]])
        else:
            y = _pick(c, ["Y", "R", "A"])
            rgb = np.dstack([y, y, y])
        a = c.get("A")
        return np.ascontiguousarray(rgb.astype(np.float32)), (None if a is None else a.astype(np.float32))
    if cv2 is not None:
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise IOError(f"Cannot read image: {path}")
        im = _to_float(im)
        if im.ndim == 2:
            return np.ascontiguousarray(np.dstack([im, im, im])), None
        rgb = np.ascontiguousarray(im[..., :3][..., ::-1])
        a = np.ascontiguousarray(im[..., 3]) if im.shape[2] == 4 else None
        return rgb, a
    img = Image.open(path)
    im = _to_float(np.array(img))
    if im.ndim == 2:
        return np.dstack([im, im, im]), None
    a = im[..., 3].copy() if im.shape[2] == 4 else None
    return np.ascontiguousarray(im[..., :3]), a


def read_single(path):
    """Returns HxW float32 (trimap / matte), values as stored (raw)."""
    if is_exr(path):
        return np.ascontiguousarray(_read_exr_single(path).astype(np.float32))
    im = _read_raster(path)
    if im.ndim == 3:
        im = im[..., 0]
    return im


def write_png16(path, alpha):
    data = np.round(np.clip(alpha, 0, 1) * 65535.0).astype(np.uint16)
    if cv2 is not None:
        if not cv2.imwrite(path, data):
            raise IOError(f"Cannot write: {path}")
    else:
        Image.fromarray(data, mode="I;16").save(path)


OUTPUT_FORMATS = ["png (16 bit)", "exr (32 bit float)"]


def write_matte(path_no_ext, alpha, fmt):
    if fmt.startswith("exr"):
        path = path_no_ext + ".exr"
        write_exr(path, alpha)
    else:
        path = path_no_ext + ".png"
        write_png16(path, alpha)
    return path


_SEQ_PATTERN_RE = re.compile(r"(#+|%0?\d*d|@+|\$F\d*|\*)")


def clean_path(text):
    """Strips whitespace, quotes and file:// URIs (e.g. pasted from a file manager)."""
    p = (text or "").strip().strip("\"'").strip()
    if p.lower().startswith("file://"):
        p = unquote(urlparse(p).path)
    return os.path.expanduser(p)


def normalize_folder(text):
    """Accepts a folder, any file of a sequence, or a sequence pattern
    (name.####.png, name.%04d.png, name.$F4.exr, name.*.png) and returns the folder."""
    p = clean_path(text)
    if not p:
        return p
    if os.path.isfile(p):
        return os.path.dirname(p)
    if not os.path.isdir(p) and _SEQ_PATTERN_RE.search(os.path.basename(p)):
        return os.path.dirname(p)
    return p.rstrip(os.sep) or p


def normalize_output_folder(text):
    """Like normalize_folder, but the folder does not have to exist yet."""
    p = clean_path(text)
    if p and (os.path.isfile(p) or _SEQ_PATTERN_RE.search(os.path.basename(p))
              or os.path.splitext(p)[1].lower() in IMAGE_EXTS):
        return os.path.dirname(p)
    return p.rstrip(os.sep) or p


def check_sequence_folder(text, label):
    """Returns an error message for the UI, or None if the folder holds a sequence."""
    folder = normalize_folder(text)
    if not folder:
        return f"{label} path is empty."
    if not os.path.isdir(folder):
        return f"{label} folder not found: {folder}"
    try:
        index_frames(folder)
    except FileNotFoundError:
        return f"No numbered images (e.g. name.1001.png) in {label} folder: {folder}"
    return None


def describe_frames(frames):
    keys = sorted(frames)
    gaps = (keys[-1] - keys[0] + 1) - len(keys)
    s = f"{len(keys)} frames ({keys[0]}-{keys[-1]})"
    return s + (f", {gaps} missing in between" if gaps else "")


def sequence_prefix(folder):
    """Name of a sequence without frame number and extension: 'shot_v01.1001.png' -> 'shot_v01'."""
    frames, _ = index_frames(folder)
    name = os.path.basename(frames[sorted(frames)[0]])
    m = _FRAME_RE.search(name)
    base = name[:m.start()] if m else os.path.splitext(name)[0]
    return base.rstrip("._- ") or "sequence"


def index_frames(folder):
    """Maps frame number -> file path for all numbered images in a folder."""
    folder = normalize_folder(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    frames = {}
    pad = 4
    for f in sorted(glob.glob(os.path.join(folder, "*"))):
        if not f.lower().endswith(IMAGE_EXTS):
            continue
        m = _FRAME_RE.search(os.path.basename(f))
        if m:
            frames[int(m.group(1))] = f
            pad = len(m.group(1))
    if not frames:
        raise FileNotFoundError(f"No numbered images found in: {folder}")
    return frames, pad
