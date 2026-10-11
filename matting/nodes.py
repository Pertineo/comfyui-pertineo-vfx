# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – Matting nodes (ViTMatte tiled refiner, trimap builder)."""

import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils

from ..common import LOG
from .core import MODEL_CHOICES, load_vitmatte, matte_frame
from ..image_io.readwrite import (OUTPUT_FORMATS, check_sequence_folder, describe_frames, index_frames,
                                  normalize_folder, normalize_output_folder, read_plate, read_single,
                                  write_matte)
from .trimap import make_trimap, trimap_overlay

CATEGORY = "Pertineo/Matting"

# --------------------------------------------------------------------------
# Shared inputs
# --------------------------------------------------------------------------
MATTING_INPUTS = {
    "model": (MODEL_CHOICES, {
        "tooltip": "ViTMatte model. Base gives the best quality, Small is faster and needs less VRAM. "
                   "Note: the hustvl weights are restricted to non-commercial use by their training-data "
                   "license (see README). 'custom' loads your own weights from custom_model."}),
    "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 32,
        "tooltip": "Tile size in px. Use the largest size that fits into VRAM: more context = better matte. "
                   "VRAM grows quadratically (approx. 1536 fits into 32 GB, 2048 does not)."}),
    "overlap": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 16,
        "tooltip": "Overlap between tiles in px. Tiles are blended with smooth weights, so there are no seams. "
                   "More overlap = smoother transitions, slightly longer render."}),
    "flip_tta": ("BOOLEAN", {"default": True,
        "tooltip": "Test-time augmentation: every tile is computed a second time horizontally flipped and both "
                   "results are averaged. Cleaner, calmer edges at twice the render time."}),
    "clamp_known": ("BOOLEAN", {"default": True,
        "tooltip": "After matting, force alpha to 0 where the trimap is black and to 1 where it is white. "
                   "Removes noise in background and core, but trimap errors can no longer be corrected."}),
    "quantize_trimap": ("BOOLEAN", {"default": True,
        "tooltip": "Before matting, round the trimap to 0 / 0.5 / 1 (cleans dithered or color-shifted trimaps). "
                   "Turn off to pass a soft trimap unchanged, like the Nuke Cattery node does."}),
}


OPTIONAL_MATTING_INPUTS = {
    "custom_model": ("STRING", {"default": "", "multiline": True,
        "placeholder": "only for model = custom: folder of your own ViTMatte weights",
        "tooltip": "Only used when model is 'custom'. Absolute path, or the name of a subfolder in the "
                   "'vitmatte' model folder. Must contain a ViTMatte model in Hugging Face format "
                   "(config.json + model.safetensors), e.g. weights you trained or licensed yourself."}),
}


def _to_np_mask(t):
    return t.detach().cpu().numpy().astype(np.float32)


def _resize_hw(arr, h, w):
    """Bilinear resize of an HxW numpy array."""
    if arr.shape == (h, w):
        return arr
    t = torch.from_numpy(arr)[None, None]
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


MATTE_CHANNELS = ["red", "green", "blue", "alpha", "luminance"]


def _image_channel(image, channel):
    """IMAGE batch (B,H,W,C) -> MASK batch (B,H,W) from the chosen channel."""
    if channel == "luminance":
        if image.shape[-1] < 3:
            return image[..., 0]
        return 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    idx = {"red": 0, "green": 1, "blue": 2, "alpha": 3}[channel]
    if idx >= image.shape[-1]:
        raise ValueError(f"{LOG} matte_image has no {channel} channel ({image.shape[-1]} channels).")
    return image[..., idx]


# --------------------------------------------------------------------------
# 1) Trimap from Matte
# --------------------------------------------------------------------------
class PertineoTrimapFromMatte:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("trimap", "preview")
    FUNCTION = "run"
    DESCRIPTION = ("Builds a ViTMatte trimap from a coarse matte (e.g. an LTX alpha): the matte is cut at "
                   "the threshold, eroded inwards for the solid core (1) and dilated outwards for the "
                   "unknown band (0.5).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "The matte is cut at this value before eroding / dilating (0.5 = half transparency)."}),
                "core_erode": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Erode inwards in px. Everything inside stays solid white (1). "
                               "Increase it where hair or motion blur lies inside the coarse matte."}),
                "outer_dilate": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1,
                    "tooltip": "Dilate outwards in px. The area between core and this edge becomes unknown (0.5), "
                               "the only area ViTMatte refines."}),
                "matte_channel": (MATTE_CHANNELS, {
                    "tooltip": "Only used with matte_image: which channel of the image holds the matte. "
                               "For a matte rendered as grey RGB, red or luminance both work."}),
                "invert_matte": ("BOOLEAN", {"default": False,
                    "tooltip": "Invert the matte first (e.g. for masks from LoadImage alpha, which are inverted)."}),
                "invert_garbage": ("BOOLEAN", {"default": False,
                    "tooltip": "Off: white in garbage_mask = garbage (forced to background). "
                               "On: white = keep, everything OUTSIDE is garbage – like a roto drawn around the subject."}),
            },
            "optional": {
                "matte": ("MASK", {"tooltip": "Coarse matte batch as MASK (e.g. from LTX), white = foreground. "
                                              "Connect either this or matte_image."}),
                "matte_image": ("IMAGE", {"tooltip": "Coarse matte batch as IMAGE (e.g. a matte rendered as RGB "
                                                     "sequence). The channel is chosen with matte_channel. "
                                                     "Connect either this or matte."}),
                "reference_image": ("IMAGE", {"tooltip": "Plate. If connected, the matte is resized to the plate "
                                                         "resolution first and the preview shows an overlay."}),
                "core_mask": ("MASK", {"tooltip": "Optional: white areas are forced to solid foreground (1), "
                                                  "e.g. holes inside the body."}),
                "garbage_mask": ("MASK", {"tooltip": "Optional: garbage areas are forced to background (0), e.g. "
                                                     "objects the coarse matte caught by mistake. Wins over core_mask. "
                                                     "See invert_garbage."}),
            },
        }

    def run(self, threshold, core_erode, outer_dilate, matte_channel, invert_matte, invert_garbage,
            matte=None, matte_image=None, reference_image=None, core_mask=None, garbage_mask=None):
        if (matte is None) == (matte_image is None):
            raise ValueError(f"{LOG} Trimap from Matte: connect exactly one of 'matte' (MASK) or "
                             f"'matte_image' (IMAGE).")
        src = matte if matte is not None else _image_channel(matte_image, matte_channel)

        n = src.shape[0]
        if reference_image is not None:
            H, W = reference_image.shape[1], reference_image.shape[2]
        else:
            H, W = src.shape[1], src.shape[2]

        def frame(t, i):
            if t is None:
                return None
            a = _to_np_mask(t[min(i, t.shape[0] - 1)])
            return _resize_hw(a, H, W)

        tris, previews = [], []
        for i in range(n):
            m = frame(src, i)
            if invert_matte:
                m = 1.0 - m
            g = frame(garbage_mask, i)
            if g is not None and invert_garbage:
                g = 1.0 - g
            tri = make_trimap(m, threshold, core_erode, outer_dilate, frame(core_mask, i), g)
            tris.append(torch.from_numpy(tri))
            if reference_image is not None:
                img = reference_image[min(i, reference_image.shape[0] - 1), ..., :3].cpu().numpy()
                previews.append(torch.from_numpy(trimap_overlay(img.astype(np.float32), tri)))
            else:
                previews.append(torch.from_numpy(np.repeat(tri[..., None], 3, axis=-1)))

        tri_b = torch.stack(tris, 0)
        unk = ((tri_b > 0.25) & (tri_b < 0.75)).float().mean().item() * 100
        print(f"{LOG} Trimap from Matte: {n} frame(s), {W}x{H}, unknown area {unk:.2f}%")
        return (tri_b, torch.stack(previews, 0))


# --------------------------------------------------------------------------
# 2) Refiner (batch, for LTX / video workflows)
# --------------------------------------------------------------------------
class PertineoViTMatteRefiner:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("alpha",)
    FUNCTION = "run"
    DESCRIPTION = ("Refines a matte with ViTMatte at full resolution using seamless overlapping tiles. "
                   "Input: plate batch + trimap batch (e.g. from 'Trimap from Matte').")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Plate batch (RGB, sRGB). Use the full-resolution source plate."}),
                "trimap": ("MASK", {"tooltip": "Trimap batch: 0 = background, 0.5 = unknown, 1 = foreground. "
                                               "Resized to the plate if the resolution differs."}),
                **MATTING_INPUTS,
            },
            "optional": dict(OPTIONAL_MATTING_INPUTS),
        }

    def run(self, image, trimap, model, tile_size, overlap, flip_tta, clamp_known, quantize_trimap,
            custom_model=""):
        n = image.shape[0]
        if trimap.shape[0] not in (1, n):
            raise ValueError(f"{LOG} Plate batch has {n} frames, trimap batch {trimap.shape[0]}.")
        H, W = image.shape[1], image.shape[2]
        if tuple(trimap.shape[1:]) != (H, W):
            print(f"{LOG} Trimap {trimap.shape[2]}x{trimap.shape[1]} resized to plate {W}x{H}.")

        device = mm.get_torch_device()
        net = load_vitmatte(model, device, custom_model)
        pbar = comfy.utils.ProgressBar(n)
        out = []
        for i in range(n):
            img = image[i, ..., :3].cpu().numpy().astype(np.float32)
            tri = _resize_hw(_to_np_mask(trimap[min(i, trimap.shape[0] - 1)]), H, W)
            alpha = matte_frame(net, img, tri, tile_size, overlap, device, flip_tta,
                                clamp_known, quantize_trimap, label=f"frame {i + 1}/{n} ")
            out.append(torch.from_numpy(alpha))
            pbar.update_absolute(i + 1, n)
        mm.soft_empty_cache()
        return (torch.stack(out, 0),)


# --------------------------------------------------------------------------
# 3) ViTMatte Tiled (Sequence) – reads and writes folders frame by frame
# --------------------------------------------------------------------------
class PertineoViTMatteSequence:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("preview_last_frame", "output_folder")
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("ViTMatte at full resolution for long, high-resolution shots: reads plate and trimap "
                   "folders frame by frame (PNG/TIFF/EXR), writes the matte sequence. Never holds the whole "
                   "sequence in memory.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plate_folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "PLATE: folder of the source image sequence",
                    "tooltip": "PLATE – folder with the image sequence ViTMatte is applied to (PNG/TIFF in sRGB or EXR). You can also paste the path of any frame of the sequence or a pattern like name.####.png."}),
                "trimap_folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "TRIMAP: folder of the trimap sequence",
                    "tooltip": "TRIMAP – folder with the trimap sequence (0 = background, 0.5 = unknown, "
                               "1 = foreground), written raw/linear. Frame numbers must match the plate. "
                               "You can also paste the path of any frame of the sequence."}),
                "output_folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "OUTPUT: target folder for the matte sequence",
                    "tooltip": "OUTPUT – the finished matte sequence is written here. Created if it does not exist."}),
                "filename_prefix": ("STRING", {"default": "alpha",
                    "tooltip": "File name of the output: <prefix>.<frame>.<ext>"}),
                "output_format": (OUTPUT_FORMATS, {
                    "tooltip": "PNG 16 bit (read as raw/linear) or EXR 32-bit float (alpha in R, G, B and A)."}),
                "exr_plate_to_srgb": ("BOOLEAN", {"default": True,
                    "tooltip": "EXR plates are scene-linear; ViTMatte expects sRGB. On = convert linear to sRGB "
                               "(recommended). Has no effect on PNG/TIFF plates."}),
                **MATTING_INPUTS,
                "skip_existing": ("BOOLEAN", {"default": True,
                    "tooltip": "Skip frames that already exist in the output folder (resume after an abort). "
                               "Turn off to re-render."}),
                "start_frame": ("INT", {"default": -1, "min": -1, "max": 10_000_000,
                    "tooltip": "First frame number to render, as in the file names. -1 = from the first frame."}),
                "end_frame": ("INT", {"default": -1, "min": -1, "max": 10_000_000,
                    "tooltip": "Last frame number to render, as in the file names. -1 = to the last frame."}),
            },
            "optional": dict(OPTIONAL_MATTING_INPUTS),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # files on disk may have changed -> always run

    @classmethod
    def VALIDATE_INPUTS(cls, plate_folder=None, trimap_folder=None, output_folder=None):
        # Runs when Queue/Run is pressed: wrong paths are shown on the node
        # before anything is loaded or computed. Inputs connected to other nodes
        # are not known yet (None) and are checked at run time instead.
        for text, label in ((plate_folder, "PLATE"), (trimap_folder, "TRIMAP")):
            if text is None:
                continue
            err = check_sequence_folder(text, label)
            if err:
                return err
        if output_folder is not None and not normalize_output_folder(output_folder):
            return "OUTPUT path is empty."
        if plate_folder is None or trimap_folder is None:
            return True
        plates, _ = index_frames(plate_folder)
        trimaps, _ = index_frames(trimap_folder)
        if not set(plates) & set(trimaps):
            return (f"PLATE ({describe_frames(plates)}) and TRIMAP ({describe_frames(trimaps)}) "
                    f"have no frame numbers in common.")
        return True

    def run(self, plate_folder, trimap_folder, output_folder, filename_prefix, output_format,
            exr_plate_to_srgb, model, tile_size, overlap, flip_tta, clamp_known, quantize_trimap,
            skip_existing, start_frame, end_frame, custom_model=""):
        plate_folder = normalize_folder(plate_folder)
        trimap_folder = normalize_folder(trimap_folder)
        output_folder = normalize_output_folder(output_folder)
        plates, pad = index_frames(plate_folder)
        trimaps, _ = index_frames(trimap_folder)
        print(f"{LOG} PLATE  : {plate_folder}  [{describe_frames(plates)}]")
        print(f"{LOG} TRIMAP : {trimap_folder}  [{describe_frames(trimaps)}]")
        print(f"{LOG} OUTPUT : {output_folder}")

        frames = sorted(set(plates) & set(trimaps))
        missing = sorted(set(plates) ^ set(trimaps))
        if missing:
            print(f"{LOG} Warning: {len(missing)} frame(s) only exist in one of the folders and are "
                  f"skipped: {missing[:10]}")
        if start_frame >= 0:
            frames = [f for f in frames if f >= start_frame]
        if end_frame >= 0:
            frames = [f for f in frames if f <= end_frame]
        if not frames:
            raise ValueError(f"{LOG} No matching plate/trimap frame pairs in the selected frame range.")
        print(f"{LOG} Rendering {len(frames)} frame pair(s): {frames[0]}-{frames[-1]}")

        out_dir = output_folder
        if not out_dir:
            raise ValueError(f"{LOG} output_folder is empty.")
        os.makedirs(out_dir, exist_ok=True)
        ext = ".exr" if output_format.startswith("exr") else ".png"

        device = mm.get_torch_device()
        net = load_vitmatte(model, device, custom_model)
        pbar = comfy.utils.ProgressBar(len(frames))
        last = None

        for i, fr in enumerate(frames):
            base = os.path.join(out_dir, f"{filename_prefix}.{fr:0{pad}d}")
            if skip_existing and os.path.isfile(base + ext):
                print(f"{LOG} frame {fr} exists, skipped (turn off skip_existing to re-render)")
                pbar.update_absolute(i + 1, len(frames))
                continue
            img = read_plate(plates[fr], exr_plate_to_srgb)
            tri = read_single(trimaps[fr])
            alpha = matte_frame(net, img, tri, tile_size, overlap, device, flip_tta,
                                clamp_known, quantize_trimap, label=f"frame {fr} ")
            path = write_matte(base, alpha, output_format)
            last = alpha
            print(f"{LOG} frame {fr} -> {path}")
            pbar.update_absolute(i + 1, len(frames))

        mm.soft_empty_cache()
        if last is None:
            last = np.zeros((64, 64), np.float32)
        return (torch.from_numpy(last)[None], out_dir)


NODE_CLASS_MAPPINGS = {
    "PertineoTrimapFromMatte": PertineoTrimapFromMatte,
    "PertineoViTMatteRefiner": PertineoViTMatteRefiner,
    "PertineoViTMatteSequence": PertineoViTMatteSequence,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PertineoTrimapFromMatte": "Pertineo Trimap from Matte",
    "PertineoViTMatteRefiner": "Pertineo ViTMatte Tiled Refiner",
    "PertineoViTMatteSequence": "Pertineo ViTMatte Tiled (Sequence)",
}
