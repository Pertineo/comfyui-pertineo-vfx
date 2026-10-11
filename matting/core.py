# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Matting core: ViTMatte model loading and seamless tiled inference."""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import folder_paths

from ..common import LOG

MODEL_REPOS = {
    "base (best quality)": "hustvl/vitmatte-base-composition-1k",
    "small (faster)": "hustvl/vitmatte-small-composition-1k",
}
CUSTOM_MODEL = "custom (see custom_model)"
MODEL_CHOICES = list(MODEL_REPOS.keys()) + [CUSTOM_MODEL]

LICENSE_NOTICE = (
    f"{LOG} LICENSE NOTE: the hustvl ViTMatte weights were trained on the Adobe Deep Image Matting "
    "dataset (Composition-1k). Its license restricts models trained on it to NON-COMMERCIAL use. "
    "Check this before using them in paid work, or load your own weights via 'custom_model'. "
    "See README > Model license."
)

_OOM = getattr(torch, "OutOfMemoryError", RuntimeError)
_MODEL_CACHE = {}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
MODEL_FOLDER = "vitmatte"
_DEFAULT_DIR = os.path.join(folder_paths.models_dir, MODEL_FOLDER)

# Register "vitmatte" as a ComfyUI model folder. If extra_model_paths.yaml already
# defines a "vitmatte" entry, ComfyUI has created it before the custom nodes are
# loaded; in that case the default folder is only appended as a fallback.
if MODEL_FOLDER in folder_paths.folder_names_and_paths:
    _paths = folder_paths.folder_names_and_paths[MODEL_FOLDER][0]
    if _DEFAULT_DIR not in _paths:
        _paths.append(_DEFAULT_DIR)
else:
    folder_paths.folder_names_and_paths[MODEL_FOLDER] = ([_DEFAULT_DIR], set())


def _model_dirs():
    """All folders registered for 'vitmatte', in priority order."""
    try:
        dirs = list(folder_paths.get_folder_paths(MODEL_FOLDER))
    except Exception:
        dirs = []
    return dirs or [_DEFAULT_DIR]


def find_model_dir(repo):
    """Returns the local folder of a model, or None if it is not downloaded yet."""
    name = repo.split("/")[-1]
    for d in _model_dirs():
        cand = os.path.join(d, name)
        if os.path.isfile(os.path.join(cand, "config.json")):
            return cand
    return None


def resolve_custom_model(custom_model):
    """custom_model: absolute folder, or the name of a subfolder in a 'vitmatte' model folder.
    The folder must contain a ViTMatte model in Hugging Face format (config.json + weights)."""
    from ..image_io.readwrite import clean_path

    p = clean_path(custom_model)
    if not p:
        raise ValueError(f"{LOG} model is '{CUSTOM_MODEL}', but custom_model is empty.")
    cands = [p] if os.path.isabs(p) else [os.path.join(d, p) for d in _model_dirs()]
    for c in cands:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    raise FileNotFoundError(
        f"{LOG} custom_model not found (needs a folder with config.json and the weights in Hugging "
        f"Face format): tried {cands}"
    )


def load_vitmatte(variant, device, custom_model=""):
    key = (variant, str(device), custom_model if variant == CUSTOM_MODEL else "")
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from transformers import VitMatteForImageMatting

    if variant == CUSTOM_MODEL:
        local_dir = resolve_custom_model(custom_model)
        print(f"{LOG} Using custom model {local_dir}")
    else:
        repo = MODEL_REPOS[variant]
        local_dir = find_model_dir(repo)
        if local_dir is None:
            from huggingface_hub import snapshot_download

            target_root = _model_dirs()[0]  # first registered folder = download target
            os.makedirs(target_root, exist_ok=True)
            local_dir = os.path.join(target_root, repo.split("/")[-1])
            print(f"{LOG} Model not found in {_model_dirs()} - downloading {repo} to {local_dir} ...")
            snapshot_download(repo_id=repo, local_dir=local_dir)
        else:
            print(f"{LOG} Using model {local_dir}")
        print(LICENSE_NOTICE)

    model = VitMatteForImageMatting.from_pretrained(local_dir)
    model = model.to(device=device, dtype=torch.float32).eval()

    _MODEL_CACHE.clear()  # keep only one model in VRAM
    _MODEL_CACHE[key] = model
    return model


# --------------------------------------------------------------------------
# Tiled inference
# --------------------------------------------------------------------------
@torch.inference_mode()
def _infer_tile(model, img, tri, device, flip_tta):
    """img: HxWx3 float32 [0,1] sRGB, tri: HxW float32 [0,1] -> HxW alpha.

    Same preprocessing as the HF VitMatteImageProcessor: image normalized to
    [-1,1] (mean=std=0.5), trimap in [0,1] appended as 4th channel, padded with
    zeros to a multiple of 32.
    """
    h, w = tri.shape
    x = np.concatenate([(img - 0.5) / 0.5, tri[..., None]], axis=-1)
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
    t = t.to(device=device, dtype=torch.float32)
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), value=0.0)

    try:
        alpha = model(pixel_values=t).alphas[..., :h, :w]
        if flip_tta:
            a_flip = model(pixel_values=torch.flip(t, dims=[3])).alphas
            a_flip = torch.flip(a_flip, dims=[3])[..., :h, :w]
            alpha = 0.5 * (alpha + a_flip)
    except _OOM:
        mm.soft_empty_cache()
        raise RuntimeError(
            f"{LOG} Out of VRAM at tile size {max(h, w)} px. Lower tile_size "
            f"(e.g. in steps of 256) and restart ComfyUI to free the VRAM."
        )
    return alpha[0, 0].float().cpu().numpy()


def _tile_positions(total, tile, overlap):
    """Start positions spread so that the last tile ends flush with the edge."""
    if total <= tile:
        return [0], total
    n = max(math.ceil((total - overlap) / (tile - overlap)), 2)
    return [round(i * (total - tile) / (n - 1)) for i in range(n)], tile


def _ramp(length, fade_start, fade_end, overlap):
    """1D weight: 1 in the core, cosine ramp at edges that have a neighbour tile,
    no ramp at the image border. Never exactly 0, so every pixel has weight."""
    w = np.ones(length, np.float32)
    m = min(overlap, length)
    if m <= 0:
        return w
    r = (np.arange(m, dtype=np.float32) + 0.5) / m
    r = 0.5 - 0.5 * np.cos(np.pi * r)
    if fade_start:
        w[:m] = np.minimum(w[:m], r)
    if fade_end:
        w[-m:] = np.minimum(w[-m:], r[::-1])
    return w


def quantize_trimap(tri):
    return np.where(tri > 0.75, 1.0, np.where(tri > 0.25, 0.5, 0.0)).astype(np.float32)


def matte_frame(model, img, tri, tile, overlap, device, flip_tta=True,
                clamp_known=True, quantize=True, infer_fn=None, label=""):
    """Full-resolution ViTMatte with overlapping tiles and weighted blending."""
    infer_fn = infer_fn or _infer_tile
    if img.shape[:2] != tri.shape:
        raise ValueError(
            f"{LOG} Plate {img.shape[1]}x{img.shape[0]} and trimap "
            f"{tri.shape[1]}x{tri.shape[0]} have different resolutions."
        )
    overlap = min(overlap, tile // 2)
    H, W = tri.shape
    if quantize:
        tri = quantize_trimap(tri)
        unknown = tri == 0.5
    else:
        tri = np.clip(tri, 0.0, 1.0).astype(np.float32)
        unknown = (tri > 0.004) & (tri < 0.996)

    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    ys, th = _tile_positions(H, tile, overlap)
    xs, tw = _tile_positions(W, tile, overlap)
    n_inferred = 0

    for iy, y in enumerate(ys):
        wy = _ramp(th, iy > 0, iy < len(ys) - 1, overlap)
        for ix, x in enumerate(xs):
            mm.throw_exception_if_processing_interrupted()
            wx = _ramp(tw, ix > 0, ix < len(xs) - 1, overlap)
            sl = (slice(y, y + th), slice(x, x + tw))
            if unknown[sl].any():
                a = infer_fn(model, img[sl], tri[sl], device, flip_tta)
                n_inferred += 1
            else:
                a = tri[sl]  # tile without unknown area -> keep trimap
            wgt = wy[:, None] * wx[None, :]
            acc[sl] += a * wgt
            wsum[sl] += wgt

    print(f"{LOG} {label}trimap: {100 * (tri <= 0.004).mean():.1f}% black, "
          f"{100 * unknown.mean():.2f}% unknown, {100 * (tri >= 0.996).mean():.1f}% white"
          f" | tiles computed: {n_inferred}/{len(ys) * len(xs)}")
    if not unknown.any():
        print(f"{LOG} WARNING: no unknown (grey) area in the trimap - ViTMatte computed "
              f"nothing, output equals trimap.")

    alpha = acc / np.maximum(wsum, 1e-8)
    if clamp_known:
        alpha = np.where(unknown, alpha, tri)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)
