# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – I/O nodes (high bit-depth image sequences)."""

import os
import time
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
from .media import folder_root_for, write_preview_video, write_sequence_parallel
from .compare import PertineoCompare, PertineoPreviewFrame
from .load_video import PertineoLoadVideo
from .save_media import PertineoLoadImage, PertineoOutputName, PertineoSaveMedia

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


def _save_preview(image, frame, max_size=768, label=None):
    """Writes a small 8-bit PNG of one frame to ComfyUI's temp folder for the node preview.
    label: text burnt in at the bottom (which frame is shown, which range was loaded)."""
    arr = np.clip(image.cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil.thumbnail((max_size, max_size))
    if label:
        try:
            from PIL import ImageDraw
            from .save_media import _font
            pil = pil.convert("RGB")
            d = ImageDraw.Draw(pil)
            size = max(11, pil.size[1] // 28)
            while True:  # shrink until the label fits the width (narrow / portrait frames)
                font = _font(size)
                x0, y0, x1, y1 = d.textbbox((0, 0), label, font=font)
                if x1 - x0 + 10 <= pil.size[0] or size <= 7:
                    break
                size -= 1
            pad = 5
            box = (0, pil.size[1] - (y1 - y0) - 2 * pad, min(pil.size[0], x1 - x0 + 2 * pad), pil.size[1])
            d.rectangle(box, fill=(20, 20, 20))
            d.text((pad - x0, box[1] + pad - y0), label, font=font, fill=(235, 235, 235))
        except Exception:
            pass
    name = f"pertineo_seq_preview_{frame}_{uuid.uuid4().hex[:8]}.png"
    pil.save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    return {"filename": name, "subfolder": "", "type": "temp"}


# --------------------------------------------------------------------------
# Load Image Sequence
# --------------------------------------------------------------------------
# "video" was the name of "player" up to 1.5.0 – still accepted, so saved workflows keep working
LOADER_PREVIEW = ["still frame", "player"]


class PertineoLoadImageSequence:
    CATEGORY = CATEGORY
    # 1.5.0: folder_root and name come after folder – the first five slots are the same as in 1.4.x
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "first_frame", "frame_count", "folder", "folder_root", "name")
    OUTPUT_TOOLTIPS = ("The images (sRGB for EXR when exr_to_srgb is on).", "Mask from mask_channel.",
                       "Frame number of the first loaded frame.", "Number of loaded frames.",
                       "The sequence folder – connect it to Save Matte Sequence (source_folder).",
                       "The folder ABOVE the sequence folder (and above a date folder like 2026-11-06) – connect it "
                       "to Output Name (output_folder), so the next version is written next to this one instead of "
                       "inside it. Only steps out of the frame folder when it is named like the sequence or has a "
                       "version in its name (sh010_comp_v01/); otherwise it is the sequence folder itself. A "
                       "sequence inside ComfyUI's input folder gives the ComfyUI output folder instead.",
                       "Name of the sequence without frame number and extension (shot010_plate.1001.png -> "
                       "shot010_plate) – connect it to Save Media (name) to name the outputs after the source.")
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
                               "ComfyUI nodes and ViTMatte expect). Only affects .exr files – PNG / TIFF / JPG "
                               "are loaded unchanged – and never the mask output. If an EXR looks flat or washed out with this on, it was saved with sRGB values (some ComfyUI savers do that) – turn it off."}),
                "missing_frames": (MISSING_MODES, {"advanced": True,
                    "tooltip": "If a frame inside the range is missing: stop with an error (safest, keeps plate "
                               "and matte in sync), repeat the previous frame, or skip it."}),
                "show_preview": ("BOOLEAN", {"default": True,
                    "tooltip": "After loading, show the middle frame of the range as a small preview on the node, "
                               "labelled with its frame number and the loaded range. Pasting one frame of a "
                               "sequence loads the whole sequence (use first_frame / last_frame for a part)."}),
            },
            "optional": {  # 1.5.0 – optional and last, so 1.4.x workflows keep their values
                "preview_type": (LOADER_PREVIEW, {
                    "tooltip": "With show_preview on: still frame = the labelled middle frame (default, instant). "
                               "player = a small H.264 copy of the loaded range to play on the node (max. 540 px high, in "
                               "ComfyUI's temp folder) – takes a moment longer for long sequences."}),
                "preview_fps": ("FLOAT", {"advanced": True, "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Playback speed of the video preview only – the loaded frames are not affected."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, folder=None, first_frame=None, last_frame=None, preview_type=None):
        # Inputs connected to other nodes are not known yet at validation time (None) –
        # those checks are done when the node runs.
        if preview_type is not None and preview_type not in LOADER_PREVIEW + ["video"]:  # "video" = up to 1.5.0
            return f"preview_type: '{preview_type}' is not one of {', '.join(LOADER_PREVIEW)}"
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
            show_preview=True, preview_type=None, preview_fps=24.0):
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
        # 1.5.0: frames go straight into one preallocated tensor (list + torch.stack needed twice the RAM)
        images = masks = None
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
            if images is None:
                images = torch.empty((len(wanted),) + tuple(prev[0].shape), dtype=torch.float32)
                masks = torch.empty((len(wanted),) + tuple(prev[1].shape), dtype=torch.float32)
            images[i] = prev[0]
            masks[i] = prev[1]
            pbar.update_absolute(i + 1, len(wanted))

        seq_name = sequence_prefix(folder)
        result = (images, masks, wanted[0], len(wanted), folder, folder_root_for(folder, seq_name), seq_name)
        if not show_preview:
            return result
        if preview_type in ("player", "video") and len(images) > 1:
            try:
                name = f"pertineo_seq_preview_{uuid.uuid4().hex[:10]}.mp4"
                write_preview_video(os.path.join(folder_paths.get_temp_directory(), name), images,
                                    float(preview_fps or 24.0))
                # same form as ComfyUI's own Save Video: the frontend shows a player for an animated mp4
                return {"ui": {"images": [{"filename": name, "subfolder": "", "type": "temp",
                                           "format": "video/h264-mp4"}], "animated": (True,)}, "result": result}
            except Exception as e:  # the preview must never stop the load
                print(f"{LOG} SEQUENCE: video preview failed ({type(e).__name__}: {e}) – showing a still frame.")
        mid = len(images) // 2
        label = (f"{wanted[mid]} (middle) · loaded {wanted[0]}-{wanted[-1]} ({len(wanted)} frames)"
                 if len(wanted) > 1 else f"{wanted[0]} (1 frame)")
        return {"ui": {"images": [_save_preview(images[mid], wanted[mid], label=label)]}, "result": result}


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
                "frame_padding": ("INT", {"advanced": True, "default": 4, "min": 1, "max": 8,
                    "tooltip": "Digits of the frame number (4 = 1001, 0001)."}),
                "format": (OUTPUT_FORMATS, {
                    "tooltip": "PNG 16 bit (read as raw/linear) or EXR 32-bit float (alpha in R, G, B and A)."}),
                "overwrite": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Overwrite existing files. Off = existing frames are kept."}),
                "folder_suffix": ("STRING", {"advanced": True, "default": "_matte",
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
        frames = _to_np_mask(alpha)  # one transfer from the GPU / tensor for the whole batch
        jobs, skipped = [], 0
        for i in range(n):
            base = os.path.join(out_dir, f"{filename_prefix}.{start_frame + i:0{frame_padding}d}")
            if not overwrite and os.path.isfile(base + ext):
                skipped += 1
                continue
            jobs.append((base + ext, lambda b=base, a=frames[i]: write_matte(b, a, format)))
        # 1.5.0: frames are written in parallel (several cores compress at once)
        pbar = comfy.utils.ProgressBar(max(1, len(jobs)))
        t0 = time.time()
        write_sequence_parallel(jobs, hash_files=False, progress=lambda d, t: pbar.update_absolute(d, t))
        print(f"{LOG} wrote {len(jobs)} frame(s) to {out_dir} in {time.time() - t0:.1f} s"
              + (f", {skipped} existing frame(s) kept" if skipped else ""))
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
                "custom_step": ("INT", {"advanced": True, "default": 8, "min": 1, "max": 1000,
                    "tooltip": "Only for model = custom: length = step * n + offset."}),
                "custom_offset": ("INT", {"advanced": True, "default": 1, "min": 0, "max": 1000,
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


# --------------------------------------------------------------------------
# Fit Frames to Model (adjusts an image batch to a valid model length)
# --------------------------------------------------------------------------
EXTEND_MODES = ["hold last frame", "hold first frame", "bounce (ping-pong)", "loop"]
TRIM_FROM = ["end", "start"]


def _fit_index(n_src, length, extend, trim_from):
    """Source frame index for every output frame. Trimming drops frames at the end (or start); extending adds
    frames after the last one: hold (repeat it), bounce (play backwards from the second-to-last) or loop.
    'hold first frame' adds the extra frames BEFORE the first one."""
    if length <= n_src:
        start = n_src - length if trim_from == "start" else 0
        return list(range(start, start + length))
    extra = length - n_src
    idx = list(range(n_src))
    if extend == "hold first frame":
        return [0] * extra + idx
    if extend == "hold last frame" or n_src == 1:
        return idx + [n_src - 1] * extra
    if extend == "loop":
        return idx + [i % n_src for i in range(extra)]
    cycle = list(range(n_src - 2, 0, -1)) + list(range(0, n_src))  # bounce: ... 3 2 1 0 1 2 3 ...
    return idx + [cycle[i % len(cycle)] for i in range(extra)]


class PertineoFitFramesToModel:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "mask", "audio", "length", "original_length", "difference")
    OUTPUT_TOOLTIPS = ("The batch with a valid length for the model.", "The mask, adjusted the same way (empty "
                       "when not connected).", "The audio, cut or extended with silence to the new length (silence "
                       "when not connected).", "New number of frames (fits the model rule).",
                       "Number of frames that came in – use it to cut the result back to the original length "
                       "after generation.", "length - original_length (positive = frames added).")
    FUNCTION = "run"
    DESCRIPTION = ("Brings an image batch (and its mask and audio) to a length the video model accepts – LTX 8n+1, "
                   "Wan 4n+1, MiniMax H3 17n+5 or a custom rule. Too long: frames are cut at the end (or start). "
                   "Too short: extended by holding the last frame (or first frame, bounce, loop); audio is "
                   "extended with silence (looped with loop). original_length tells you how many frames to keep afterwards.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The frames (e.g. from Load Video / Load Image Sequence)."}),
                "model": (list(MODEL_FRAME_RULES.keys()), {
                    "tooltip": "Video model whose length rule is applied. LTX: 8n+1 (97, 121, ...), "
                               "Wan: 4n+1 (81, ...), MiniMax H3: 17n+5 (124, ...)."}),
                "rounding": (ROUNDING, {"default": ROUNDING[1],
                    "tooltip": "up = add frames (nothing of the source is lost – default here). down = cut frames. "
                               "nearest = whichever is closer."}),
                "extend": (EXTEND_MODES, {
                    "tooltip": "How missing frames are made: hold last frame (repeat it at the end), hold first frame "
                               "(repeat it at the start – the source then starts later), bounce (play backwards from "
                               "the end, then forwards again), loop (start again from the first frame). Audio: loop "
                               "repeats the sound too, the other modes add silence."}),
                "trim_from": (TRIM_FROM, {"advanced": True,
                    "tooltip": "When frames are cut: drop them at the end (default) or at the start."}),
                "custom_step": ("INT", {"advanced": True, "default": 8, "min": 1, "max": 1000,
                    "tooltip": "Only for model = custom: length = step * n + offset."}),
                "custom_offset": ("INT", {"advanced": True, "default": 1, "min": 0, "max": 1000,
                    "tooltip": "Only for model = custom: length = step * n + offset."}),
                "fps": ("FLOAT", {"advanced": True, "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate – only used to cut or extend the audio (connect fps of Load Video)."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Mask / matte batch – adjusted exactly like the images."}),
                "audio": ("AUDIO", {"tooltip": "Audio – cut or extended with silence to the new length."}),
            },
        }

    def run(self, images, model, rounding, extend, trim_from, custom_step, custom_offset, fps, mask=None, audio=None):
        rule = MODEL_FRAME_RULES[model]
        step, offset = rule if rule is not None else (custom_step, custom_offset)
        n_src = int(images.shape[0])
        length = model_length(n_src, step, offset, rounding)
        idx = _fit_index(n_src, length, extend, trim_from)
        out_img = images[idx] if idx != list(range(n_src)) else images
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None]
            m_idx = [min(i, mask.shape[0] - 1) for i in idx]
            out_mask = mask[m_idx]
        else:
            out_mask = torch.zeros((length,) + tuple(images.shape[1:3]), dtype=torch.float32)

        rate, ch = 48000, 2
        wave = None
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            wave, rate = audio["waveform"], int(audio.get("sample_rate") or 48000)
            ch = int(wave.shape[1])
        samples_per_frame = rate / float(fps)
        want = max(1, int(round(length * samples_per_frame)))
        if wave is None:
            out_audio = {"waveform": torch.zeros((1, ch, want), dtype=torch.float32), "sample_rate": rate}
        else:
            # audio follows the picture: where frames were held, repeated or looped, there is silence
            shift = 0
            if length > n_src and extend == "hold first frame":
                shift = int(round((length - n_src) * samples_per_frame))
            elif length < n_src and trim_from == "start":
                shift = -int(round((n_src - length) * samples_per_frame))
            buf = torch.zeros((wave.shape[0], ch, want), dtype=torch.float32)
            src0 = max(0, -shift)
            dst0 = max(0, shift)
            n = max(0, min(wave.shape[-1] - src0, want - dst0, int(round(min(n_src, length) * samples_per_frame))))
            if n:
                buf[..., dst0:dst0 + n] = wave[..., src0:src0 + n]
                if extend == "loop" and length > n_src:  # repeat the sound with the pictures
                    pos = dst0 + n
                    while pos < want:
                        k = min(n, want - pos)
                        buf[..., pos:pos + k] = buf[..., dst0:dst0 + k]
                        pos += k
            out_audio = {"waveform": buf, "sample_rate": rate}

        diff = length - n_src
        how = "fits" if diff == 0 else (f"{-diff} frame(s) cut at the {trim_from}" if diff < 0
                                        else f"{diff} frame(s) added ({extend})")
        print(f"{LOG} Fit Frames to Model: {model}: {n_src} -> {length} frames ({how})")
        return (out_img, out_mask, out_audio, length, n_src, diff)


NODE_CLASS_MAPPINGS = {
    "PertineoLoadImageSequence": PertineoLoadImageSequence,
    "PertineoSaveMatteSequence": PertineoSaveMatteSequence,
    "PertineoEmptyAudio": PertineoEmptyAudio,
    "PertineoFramesForModel": PertineoFramesForModel,
    "PertineoSaveMedia": PertineoSaveMedia,
    "PertineoOutputName": PertineoOutputName,
    "PertineoLoadImage": PertineoLoadImage,
    "PertineoLoadVideo": PertineoLoadVideo,
    "PertineoFitFramesToModel": PertineoFitFramesToModel,
    "PertineoCompare": PertineoCompare,
    "PertineoPreviewFrame": PertineoPreviewFrame,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PertineoLoadImageSequence": "Pertineo Load Image Sequence",
    "PertineoSaveMatteSequence": "Pertineo Save Matte Sequence",
    "PertineoEmptyAudio": "Pertineo Empty Audio",
    "PertineoFramesForModel": "Pertineo Frames for Model",
    "PertineoSaveMedia": "Pertineo Save Media",
    "PertineoOutputName": "Pertineo Output Name",
    "PertineoLoadImage": "Pertineo Load Image",
    "PertineoLoadVideo": "Pertineo Load Video",
    "PertineoFitFramesToModel": "Pertineo Fit Frames to Model",
    "PertineoCompare": "Pertineo Compare",
    "PertineoPreviewFrame": "Pertineo Preview Frame",
}
