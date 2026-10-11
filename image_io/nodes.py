# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – I/O nodes (high bit-depth image sequences)."""

import os
import uuid

import numpy as np
import torch
from PIL import Image

import comfy.utils
import folder_paths

from ..common import LOG
from .readwrite import (OUTPUT_FORMATS, check_sequence_folder, describe_frames, index_frames,
                        is_exr, linear_to_srgb, normalize_folder, normalize_output_folder,
                        read_rgba_raw, sequence_prefix, write_matte)

CATEGORY = "Pertineo/IO"
MASK_CHANNELS = ["red", "green", "blue", "alpha", "luminance"]
MISSING_MODES = ["error", "hold previous", "skip"]


def _to_np_mask(t):
    return t.detach().cpu().numpy().astype(np.float32)


def _select_frames(frames, start_frame, end_frame):
    keys = sorted(frames)
    first = keys[0] if start_frame < 0 else start_frame
    last = keys[-1] if end_frame < 0 else end_frame
    return first, last


def _save_preview(image, frame, max_size=768):
    """Writes a small 8-bit PNG of one frame to ComfyUI's temp folder for the node preview."""
    arr = np.clip(image.cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil.thumbnail((max_size, max_size))
    name = f"pertineo_seq_preview_{frame}_{uuid.uuid4().hex[:8]}.png"
    pil.save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    return {"filename": name, "subfolder": "", "type": "temp"}


# --------------------------------------------------------------------------
# Load Image Sequence
# --------------------------------------------------------------------------
class PertineoLoadImageSequence:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "first_frame", "frame_count", "folder")
    FUNCTION = "run"
    DESCRIPTION = ("Loads a numbered image sequence (PNG / TIFF 8 or 16 bit, EXR) from a folder in full "
                   "precision, with first and last frame given as real frame numbers. Outputs the images "
                   "and a mask from a chosen channel.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "SEQUENCE: folder, any frame of it, or name.####.png",
                    "tooltip": "Folder with the numbered image sequence. You can also paste the path of any "
                               "frame of the sequence or a pattern like name.####.png."}),
                "first_frame": ("INT", {"default": -1, "min": -1, "max": 10_000_000,
                    "tooltip": "First frame number to load, as in the file names (e.g. 1001). -1 = first frame "
                               "of the sequence."}),
                "last_frame": ("INT", {"default": -1, "min": -1, "max": 10_000_000,
                    "tooltip": "Last frame number to load, as in the file names. -1 = last frame of the sequence."}),
                "mask_channel": (MASK_CHANNELS, {
                    "tooltip": "Channel for the mask output, read raw from the file (no color conversion). "
                               "For a matte rendered as grey RGB use red or luminance; 'alpha' needs an alpha "
                               "channel in the files."}),
                "exr_to_srgb": ("BOOLEAN", {"default": True,
                    "tooltip": "EXR files are scene-linear. On = convert the IMAGE output to sRGB (what most "
                               "ComfyUI nodes and ViTMatte expect). Has no effect on PNG / TIFF and never on "
                               "the mask output."}),
                "missing_frames": (MISSING_MODES, {
                    "tooltip": "If a frame inside the range is missing: stop with an error (safest, keeps plate "
                               "and matte in sync), repeat the previous frame, or skip it."}),
                "show_preview": ("BOOLEAN", {"default": True,
                    "tooltip": "After loading, show the middle frame of the range as a small preview on the node."}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, folder=None, first_frame=None, last_frame=None):
        # Inputs connected to other nodes are not known yet at validation time (None) –
        # those checks are done when the node runs.
        if folder is None:
            return True
        err = check_sequence_folder(folder, "SEQUENCE")
        if err:
            return err
        if first_frame is None or last_frame is None:
            return True
        frames, _ = index_frames(folder)
        first, last = _select_frames(frames, first_frame, last_frame)
        if last < first:
            return f"last_frame ({last}) is before first_frame ({first})."
        if not any(first <= f <= last for f in frames):
            return f"No frames between {first} and {last} in the sequence ({describe_frames(frames)})."
        return True

    @classmethod
    def IS_CHANGED(cls, folder=None, first_frame=None, last_frame=None, **kwargs):
        # Reload only when the files in the range change (not on every run).
        # Linked first/last frame are not known here; then the whole folder is checked
        # (a changed range itself makes ComfyUI re-run the node anyway).
        if folder is None:
            return float("nan")
        try:
            frames, _ = index_frames(folder)
        except Exception:
            return float("nan")
        first, last = _select_frames(frames, -1 if first_frame is None else first_frame,
                                     -1 if last_frame is None else last_frame)
        sig = []
        for f in sorted(frames):
            if first <= f <= last:
                st = os.stat(frames[f])
                sig.append((f, st.st_size, st.st_mtime_ns))
        return str(sig)

    def run(self, folder, first_frame, last_frame, mask_channel, exr_to_srgb, missing_frames,
            show_preview=True):
        folder = normalize_folder(folder)
        frames, _ = index_frames(folder)
        first, last = _select_frames(frames, first_frame, last_frame)
        if last < first:
            raise ValueError(f"{LOG} last_frame ({last}) is before first_frame ({first}).")
        wanted = list(range(first, last + 1))
        missing = [f for f in wanted if f not in frames]
        if missing:
            msg = f"{LOG} {len(missing)} frame(s) missing between {first} and {last}: {missing[:10]}"
            if missing_frames == "error":
                raise ValueError(msg + " (set missing_frames to 'hold previous' or 'skip' to continue)")
            print(msg + f" -> {missing_frames}")
        if missing_frames == "skip":
            wanted = [f for f in wanted if f in frames]
        if not wanted or wanted[0] not in frames:
            raise ValueError(f"{LOG} The first frame of the range ({first}) does not exist.")

        print(f"{LOG} SEQUENCE: {folder}  [{describe_frames(frames)}] -> loading {first}-{last} "
              f"({len(wanted)} frames)")
        images, masks = [], []
        prev = None
        size = None
        pbar = comfy.utils.ProgressBar(len(wanted))
        for i, f in enumerate(wanted):
            if f in frames:
                path = frames[f]
                rgb, alpha = read_rgba_raw(path)
                if size is None:
                    size = rgb.shape[:2]
                    est = len(wanted) * size[0] * size[1] * 4 * 4 / 1024 ** 3
                    print(f"{LOG} {size[1]}x{size[0]}, approx. {est:.1f} GB RAM for image + mask")
                elif rgb.shape[:2] != size:
                    raise ValueError(f"{LOG} Frame {f} has {rgb.shape[1]}x{rgb.shape[0]}, "
                                     f"expected {size[1]}x{size[0]}.")
                if mask_channel == "alpha":
                    if alpha is None:
                        raise ValueError(f"{LOG} mask_channel is 'alpha', but {os.path.basename(path)} "
                                         f"has no alpha channel.")
                    m = alpha
                elif mask_channel == "luminance":
                    m = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                else:
                    m = rgb[..., {"red": 0, "green": 1, "blue": 2}[mask_channel]]
                img = linear_to_srgb(rgb) if (exr_to_srgb and is_exr(path)) else np.clip(rgb, 0, 1)
                prev = (torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
                        torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32)))
            images.append(prev[0])
            masks.append(prev[1])
            pbar.update_absolute(i + 1, len(wanted))

        result = (torch.stack(images, 0), torch.stack(masks, 0), wanted[0], len(wanted), folder)
        if not show_preview:
            return result
        mid = len(images) // 2
        return {"ui": {"images": [_save_preview(images[mid], wanted[mid])]}, "result": result}


# --------------------------------------------------------------------------
# Save Matte Sequence (16-bit PNG / 32-bit EXR)
# --------------------------------------------------------------------------
class PertineoSaveMatteSequence:
    CATEGORY = CATEGORY
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Writes a matte batch as an image sequence in 16-bit PNG or 32-bit float EXR "
                   "(the standard Save Image node only writes 8 bit).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha": ("MASK", {"tooltip": "Matte batch to write."}),
                "output_folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "OUTPUT: target folder (ignored when source_folder is connected)",
                    "tooltip": "OUTPUT – folder the sequence is written to. Created if it does not exist. "
                               "Ignored when source_folder is connected."}),
                "filename_prefix": ("STRING", {"default": "",
                    "placeholder": "empty = <source name><suffix>, e.g. shot_v01_matte",
                    "tooltip": "File name: <prefix>.<frame>.<ext>. Leave empty to name the files after the source "
                               "sequence plus folder_suffix (needs source_folder), e.g. shot_v01_matte.1001.png. "
                               "Without source_folder an empty prefix becomes 'alpha'."}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 10_000_000,
                    "tooltip": "Frame number of the first image in the batch (match your plate numbering)."}),
                "frame_padding": ("INT", {"default": 4, "min": 1, "max": 8,
                    "tooltip": "Digits of the frame number (4 = 1001, 0001)."}),
                "format": (OUTPUT_FORMATS, {
                    "tooltip": "PNG 16 bit (read as raw/linear) or EXR 32-bit float (alpha in R, G, B and A)."}),
                "overwrite": ("BOOLEAN", {"default": True,
                    "tooltip": "Overwrite existing files. Off = existing frames are kept."}),
                "folder_suffix": ("STRING", {"default": "_matte",
                    "tooltip": "Only used when source_folder is connected: the output folder is named "
                               "<source folder name><suffix> next to the source (shot_v01 -> shot_v01_matte); "
                               "with an empty filename_prefix the files get <source name><suffix> as well."}),
            },
            "optional": {
                "source_folder": ("STRING", {"forceInput": True,
                    "tooltip": "Connect 'folder' from Pertineo Load Image Sequence: the matte is then written to a "
                               "sibling folder <source folder><folder_suffix> and output_folder is ignored."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, output_folder=None):
        # output_folder may stay empty when source_folder is connected – checked at run time.
        return True

    def run(self, alpha, output_folder, filename_prefix, start_frame, frame_padding, format, overwrite,
            folder_suffix="_matte", source_folder=None):
        if source_folder:
            src = normalize_folder(source_folder).rstrip(os.sep)
            suffix = (folder_suffix or "").strip()
            if not suffix:
                raise ValueError(f"{LOG} folder_suffix is empty – the matte would overwrite the source folder.")
            out_dir = os.path.join(os.path.dirname(src), os.path.basename(src) + suffix)
            if not filename_prefix.strip():
                filename_prefix = sequence_prefix(src) + suffix
            print(f"{LOG} Output next to source: {out_dir}  (output_folder field ignored)")
        else:
            out_dir = normalize_output_folder(output_folder)
            if not filename_prefix.strip():
                filename_prefix = "alpha"
        filename_prefix = filename_prefix.strip()
        print(f"{LOG} Files: {filename_prefix}.{start_frame:0{frame_padding}d}.* ...")
        if not out_dir:
            raise ValueError(f"{LOG} output_folder is empty (or connect source_folder).")
        os.makedirs(out_dir, exist_ok=True)
        ext = ".exr" if format.startswith("exr") else ".png"
        n = alpha.shape[0]
        for i in range(n):
            base = os.path.join(out_dir, f"{filename_prefix}.{start_frame + i:0{frame_padding}d}")
            if not overwrite and os.path.isfile(base + ext):
                print(f"{LOG} {base + ext} exists, skipped")
                continue
            path = write_matte(base, _to_np_mask(alpha[i]), format)
            print(f"{LOG} wrote {path}")
        return ()



# --------------------------------------------------------------------------
# Empty Audio (silent track of matching length, e.g. for LTX audio inputs)
# --------------------------------------------------------------------------
class PertineoEmptyAudio:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration_sec")
    FUNCTION = "run"
    DESCRIPTION = ("Creates a silent AUDIO track whose length matches an image sequence "
                   "(frame_count / fps) – e.g. for workflows like LTX that expect an audio input.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {"default": 121, "min": 1, "max": 1_000_000,
                    "tooltip": "Number of frames. Connect frame_count from 'Pertineo Load Image Sequence' "
                               "so the length always matches."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate of the sequence (e.g. 23.976, 24, 25, 29.97, 30)."}),
                "sample_rate": (["48000", "44100", "96000", "22050", "16000"], {
                    "tooltip": "Sample rate of the silent track. 48000 is the usual video standard."}),
                "channels": (["stereo", "mono"], {
                    "tooltip": "Number of audio channels."}),
            }
        }

    def run(self, frame_count, fps, sample_rate, channels):
        sample_rate = int(sample_rate)
        duration = frame_count / float(fps)
        samples = max(1, int(round(duration * sample_rate)))
        ch = 2 if channels == "stereo" else 1
        waveform = torch.zeros((1, ch, samples), dtype=torch.float32)
        print(f"{LOG} Empty audio: {frame_count} frames @ {fps} fps = {duration:.3f} s, "
              f"{samples} samples @ {sample_rate} Hz, {channels}")
        return ({"waveform": waveform, "sample_rate": sample_rate}, duration)


# --------------------------------------------------------------------------
# Frames for Model (central frame range with valid lengths for video models)
# --------------------------------------------------------------------------
# length = step * n + offset   (n >= 0)
MODEL_FRAME_RULES = {
    "LTX (8n+1)": (8, 1),
    "Wan (4n+1)": (4, 1),
    "MiniMax H3 (17n+5)": (17, 5),
    "custom (step / offset)": None,
}
ROUNDING = ["down (fewer frames)", "up (more frames)", "nearest"]


def model_length(frame_count, step, offset, rounding):
    if frame_count <= offset:
        return offset
    n_float = (frame_count - offset) / step
    if rounding.startswith("down"):
        n = int(n_float // 1)
    elif rounding.startswith("up"):
        n = -int(-n_float // 1)
    else:
        n = int(n_float + 0.5)
    return step * n + offset


class PertineoFramesForModel:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("first_frame", "last_frame", "length", "difference")
    FUNCTION = "run"
    DESCRIPTION = ("Central frame range for a workflow: set first and last frame here; the last frame is "
                   "adjusted so the length fits the video model (LTX 8n+1, Wan 4n+1, MiniMax H3 17n+5 or a "
                   "custom rule). Connect first_frame / last_frame to the loaders (and first_frame to Save "
                   "Matte Sequence), length to the model and Empty Audio.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_frame": ("INT", {"default": 1001, "min": 0, "max": 10_000_000,
                    "tooltip": "First frame of the range (e.g. 1001). Passed through on the first_frame output "
                               "for the loaders and Save Matte Sequence."}),
                "last_frame": ("INT", {"default": 1121, "min": 0, "max": 10_000_000,
                    "tooltip": "Desired last frame. It is adjusted to the next valid length; use the "
                               "last_frame OUTPUT for the loaders."}),
                "model": (list(MODEL_FRAME_RULES.keys()), {
                    "tooltip": "Video model whose length rule is applied. LTX: 8n+1 (97, 121, ...), "
                               "Wan: 4n+1 (81, ...), MiniMax H3: 17n+5 (124, ...)."}),
                "rounding": (ROUNDING, {
                    "tooltip": "down = fewer frames, all of them exist (safest). up = more frames – they must "
                               "exist in the folder, otherwise the loader stops. nearest = closest valid length."}),
                "custom_step": ("INT", {"default": 8, "min": 1, "max": 1000,
                    "tooltip": "Only for model = custom: length = step * n + offset."}),
                "custom_offset": ("INT", {"default": 1, "min": 0, "max": 1000,
                    "tooltip": "Only for model = custom: length = step * n + offset."}),
            }
        }

    def run(self, first_frame, last_frame, model, rounding, custom_step, custom_offset):
        if last_frame < first_frame:
            raise ValueError(f"{LOG} last_frame ({last_frame}) is before first_frame ({first_frame}).")
        rule = MODEL_FRAME_RULES[model]
        step, offset = rule if rule is not None else (custom_step, custom_offset)
        frame_count = last_frame - first_frame + 1
        length = model_length(frame_count, step, offset, rounding)
        diff = length - frame_count
        new_last = first_frame + length - 1
        note = "fits" if diff == 0 else (f"{-diff} frame(s) fewer" if diff < 0 else f"{diff} frame(s) more")
        print(f"{LOG} {model}: range {first_frame}-{last_frame} ({frame_count} frames) -> "
              f"{first_frame}-{new_last} = length {length} ({note})")
        return (first_frame, new_last, length, diff)


NODE_CLASS_MAPPINGS = {
    "PertineoLoadImageSequence": PertineoLoadImageSequence,
    "PertineoSaveMatteSequence": PertineoSaveMatteSequence,
    "PertineoEmptyAudio": PertineoEmptyAudio,
    "PertineoFramesForModel": PertineoFramesForModel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PertineoLoadImageSequence": "Pertineo Load Image Sequence",
    "PertineoSaveMatteSequence": "Pertineo Save Matte Sequence",
    "PertineoEmptyAudio": "Pertineo Empty Audio",
    "PertineoFramesForModel": "Pertineo Frames for Model",
}
