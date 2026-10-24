# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Image writers for the audit save node: PNG 8 / 16 bit with text chunks, EXR (RGB)."""

import struct
import zlib

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

FORMATS = ["png (8 bit)", "png (16 bit)", "exr (32 bit float, linear)"]


def ext_for(fmt):
    return ".exr" if fmt.startswith("exr") else ".png"


def _rgb(img):
    """HxWxC float [0,1] -> HxWx3 or HxWx4."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    elif a.shape[-1] == 2:
        a = np.concatenate([np.repeat(a[..., :1], 3, axis=-1), a[..., 1:2]], axis=-1)
    elif a.shape[-1] > 4:
        a = a[..., :4]
    # IMAGE tensors are often channel-last views of channel-first memory -> make them C-contiguous
    return np.ascontiguousarray(a)


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _text_chunk(key, text):
    k = key.encode("latin-1", "replace")[:79]
    try:
        return _chunk(b"tEXt", k + b"\0" + text.encode("latin-1"))
    except UnicodeEncodeError:
        return _chunk(b"iTXt", k + b"\0\0\0\0\0" + text.encode("utf-8"))


def write_png16(f, img, texts):
    """Minimal 16-bit PNG encoder (stdlib zlib), so text chunks can be embedded – cv2 cannot."""
    a = _rgb(img)
    h, w, c = a.shape
    color_type = {3: 2, 4: 6}[c]
    px = np.ascontiguousarray(np.round(np.clip(a, 0.0, 1.0) * 65535.0), dtype=">u2")
    raw = np.empty((h, 1 + w * c * 2), dtype=np.uint8)
    raw[:, 0] = 0  # filter: none
    raw[:, 1:] = px.view(np.uint8).reshape(h, w * c * 2)
    out = [b"\x89PNG\r\n\x1a\n", _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0))]
    out += [_text_chunk(k, v) for k, v in texts]
    out.append(_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
    out.append(_chunk(b"IEND", b""))
    for part in out:
        f.write(part)


def write_png8(f, img, texts):
    a = _rgb(img)
    px = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    info = PngInfo()
    for k, v in texts:
        info.add_text(k, v)
    Image.fromarray(px, mode="RGBA" if px.shape[-1] == 4 else "RGB").save(f, format="PNG", pnginfo=info,
                                                                         compress_level=4)


def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def write_exr_rgb(path, img):
    """Writes RGB(A) as 32-bit float EXR, sRGB-encoded IMAGE values converted to linear (alpha stays)."""
    from ..image_io.readwrite import EXR_HINT, HAS_OPENEXR, OpenEXR, cv2

    a = _rgb(img)
    rgb = srgb_to_linear(a[..., :3])
    chans = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
    if a.shape[-1] == 4:
        chans["A"] = a[..., 3]
    chans = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in chans.items()}
    if HAS_OPENEXR:
        header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
        with OpenEXR.File(header, chans) as f:
            f.write(path)
        return
    if cv2 is not None:
        order = ["B", "G", "R"] + (["A"] if "A" in chans else [])
        stack = np.dstack([chans[k] for k in order])
        try:
            if cv2.imwrite(path, stack, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
                return
        except cv2.error:
            pass
    raise IOError(f"Cannot write EXR. {EXR_HINT}")
