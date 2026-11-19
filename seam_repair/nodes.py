# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – Seam Repair nodes: find and repair the seams of chained renders (detect, stabilize, interpolate,
regenerate, blend overlaps)."""

import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

import comfy.utils
import folder_paths

from ..common import LOG
from ..image_io.readwrite import (cv2, check_sequence_folder, describe_frames, index_frames, linear_to_srgb,
                                  normalize_folder, normalize_output_folder)
from .core import (SEAM_TYPES, Frame, analyse, blend_weights, bridge_window, classify, draw_graph,
                   merge_close, parse_frames, plan_segments, srgb_to_linear)
from .stabilize import PertineoOverlapBlend, PertineoSeamStabilize

CATEGORY = "Pertineo/Seam Repair"


def _range(frames, first_frame, last_frame):
    keys = sorted(frames)
    first = keys[0] if first_frame < 0 else first_frame
    last = keys[-1] if last_frame < 0 else last_frame
    if last < first:
        raise ValueError(f"{LOG} last_frame ({last}) is before first_frame ({first}).")
    wanted = list(range(first, last + 1))
    missing = [f for f in wanted if f not in frames]
    if missing:
        raise ValueError(f"{LOG} {len(missing)} frame(s) missing between {first} and {last}: {missing[:10]} – "
                         f"seams can only be found and smoothed in a gap-free sequence.")
    return wanted


def _signature(folder):
    try:
        frames, _ = index_frames(folder)
    except Exception:
        return float("nan")
    return str([(f, os.stat(p).st_size, os.stat(p).st_mtime_ns) for f, p in sorted(frames.items())])


# --------------------------------------------------------------------------
# Seam Detect
# --------------------------------------------------------------------------
class PertineoSeamDetect:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "INT", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("seams", "seam_count", "report", "graph", "folder")
    OUTPUT_TOOLTIPS = (
        "Frame numbers of the found seams, comma separated. A seam at frame t is the cut between t-1 and t. "
        "Connect it to Seam Stabilize, Seam Interpolate or Seam Window (seams).",
        "Number of found seams.",
        "Readable list: every seam with its type (jump, speed / direction, quality, duplicate) and score, plus the strongest "
        "frames below the threshold – useful to tune the sensitivity.",
        "Graph of the seam score over the shot (log scale) with the threshold; found seams coloured by type: "
        "red = jump, blue = speed / direction change, violet = quality step, orange = duplicate frame.",
        "The sequence folder – connect it to Seam Stabilize / Seam Interpolate (folder).")
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Finds motion jumps and duplicate frames in an image sequence – typically where chained "
                   "renders (extend / upscale chains) were joined. Compares every frame with the previous one at "
                   "low resolution (optical flow) and measures its sharpness. Finds four kinds of seams: one-frame jumps of "
                   "the camera motion, changes of speed or direction, steps in sharpness (quality) and duplicate frames "
                   "(stutter). Reads the files itself, so long "
                   "sequences need very little RAM.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "SEQUENCE: folder, any frame of it, or name.####.exr",
                    "tooltip": "Folder with the numbered image sequence (PNG / TIFF / JPG / EXR)."}),
                "first_frame": ("INT", {"default": -1, "min": -1, "max": 10_000_000,
                    "tooltip": "First frame to analyse, as in the file names. -1 = first frame of the sequence."}),
                "last_frame": ("INT", {"default": -1, "min": -1, "max": 10_000_000,
                    "tooltip": "Last frame to analyse, as in the file names. -1 = last frame of the sequence."}),
                "sensitivity": ("FLOAT", {"default": 3.0, "min": 1.2, "max": 20.0, "step": 0.1,
                    "tooltip": "A frame counts as a jump when its difference to the previous frame is this many "
                               "times higher than the local level of the shot. Lower = finds weaker jumps (and more "
                               "false alarms). Check the graph and the report, then adjust."}),
                "detect_duplicates": ("BOOLEAN", {"default": True,
                    "tooltip": "Also report frames that are (almost) identical to the previous one while the shot "
                               "moves – a typical stutter where two chain parts overlap by one frame."}),
                "min_distance": ("INT", {"default": 4, "min": 1, "max": 100,
                    "tooltip": "Seams closer to each other than this many frames are counted as one (the stronger "
                               "one is kept). The repair nodes merge overlapping windows anyway."}),
            },
            "optional": {
                "analysis_size": ("INT", {"advanced": True, "default": 256, "min": 64, "max": 1024, "step": 32,
                    "tooltip": "Long side in pixels the frames are scaled to for the analysis. 256 is enough for "
                               "motion jumps and fast."}),
                "min_motion": ("FLOAT", {"advanced": True, "default": 0.5, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Jumps with less motion than this (pixels at analysis_size) are ignored, so almost "
                               "static shots do not report grain or boiling as jumps."}),
                "detect_quality": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Also report frames where the sharpness steps up or down and stays there (e.g. a chain "
                               "part with more or less detail). Seam Interpolate hides such a step over its window; "
                               "Seam Stabilize (detail_seams) matches the sharpness after it."}),
                "exr_is_linear": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "EXR files are scene-linear (as written by Save Media with exr_to_linear). They are "
                               "converted to sRGB for the analysis. Turn off for EXRs saved with sRGB values."}),
                "min_quality": ("FLOAT", {"advanced": True, "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Smallest sharpness step that counts as a quality seam (0.08 = about 8 %). Raise it "
                               "when a shot whose sharpness pulses all the time reports too many quality seams; for "
                               "sharpness that pulses through the whole shot use Seam Stabilize with detail_mode "
                               "deflicker instead."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, folder=None):
        if folder is None:
            return True
        return check_sequence_folder(folder, "SEQUENCE") or True

    @classmethod
    def IS_CHANGED(cls, folder=None, **kwargs):
        return float("nan") if folder is None else _signature(normalize_folder(folder))

    def run(self, folder, first_frame, last_frame, sensitivity, detect_duplicates, min_distance,
            analysis_size=256, min_motion=0.5, detect_quality=True, exr_is_linear=True, min_quality=0.08):
        folder = normalize_folder(folder)
        frames, _ = index_frames(folder)
        wanted = _range(frames, first_frame, last_frame)
        if len(wanted) < 3:
            raise ValueError(f"{LOG} Seam Detect needs at least 3 frames.")
        print(f"{LOG} SEAM DETECT: {folder} [{describe_frames(frames)}] -> analysing {wanted[0]}-{wanted[-1]}")
        pbar = comfy.utils.ProgressBar(len(wanted))
        m = analyse([frames[f] for f in wanted], size=analysis_size, exr_is_linear=exr_is_linear,
                    progress=lambda i, n: pbar.update_absolute(i, n))
        score, kind, found_d = classify(m, sensitivity=sensitivity, min_motion=min_motion, min_quality=min_quality,
                                        detect_duplicates=detect_duplicates, detect_quality=detect_quality)
        found = merge_close(list(found_d), score, min_distance)
        seams = [wanted[i] for i in found]

        lines = [f"Sequence: {folder}", f"Analysed: {wanted[0]}-{wanted[-1]} ({len(wanted)} frames), "
                 f"sensitivity {sensitivity:g}", "Score = how many times above the normal level of this shot.", ""]
        if seams:
            lines.append(f"{len(seams)} seam(s) (cut between frame t-1 and t):")
            for i in found:
                if found_d[i] == 3:
                    lines.append(f"  {wanted[i]}  duplicate          (almost identical to {wanted[i] - 1})")
                else:
                    lines.append(f"  {wanted[i]}  {SEAM_TYPES[found_d[i]]:<17}  score {score[i]:.1f}")
            top = sorted(found, key=lambda i: -score[i])[:8]
            if len(found) > 8:
                lines += ["", "Strongest: " + ", ".join(f"{wanted[i]} ({score[i]:.0f})" for i in top)]
        else:
            lines.append("No seams found.")
        below = [i for i in np.argsort(-score) if i not in found_d and 0 < score[i] < sensitivity][:8]
        if below:
            lines += ["", "Strongest frames below the threshold:"]
            lines += [f"  {wanted[i]}  {SEAM_TYPES[kind[i]] if kind[i] >= 0 else '-':<17}  score {score[i]:.1f}"
                      for i in below]
        report = "\n".join(lines)
        print(f"{LOG} SEAM DETECT: " + (", ".join(map(str, seams)) if seams else "no seams"))

        graph = draw_graph(wanted, score, kind, sensitivity, found)
        graph_t = torch.from_numpy(graph.astype(np.float32) / 255.0)[None]
        name = f"pertineo_seam_graph_{uuid.uuid4().hex[:8]}.png"
        Image.fromarray(graph).save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
        result = (", ".join(map(str, seams)), len(seams), report, graph_t, folder)
        return {"ui": {"images": [{"filename": name, "subfolder": "", "type": "temp"}]}, "result": result}


# --------------------------------------------------------------------------
# Seam Interpolate
# --------------------------------------------------------------------------
def _interpolate(interp_model, a, b, multiplier):
    """Runs ComfyUI's own FrameInterpolate (FILM / RIFE) on two frames (HxWx3 float 0-1).
    Returns the multiplier-1 in-between frames as a list of HxWx3 float32 arrays."""
    from comfy_extras.nodes_frame_interpolation import FrameInterpolate
    images = torch.from_numpy(np.ascontiguousarray(np.stack([a, b]), dtype=np.float32))
    out = FrameInterpolate.execute(interp_model, images, multiplier)
    args = getattr(out, "args", None) or getattr(out, "result", None)
    res = args[0]
    if res.shape[0] != multiplier + 1:
        raise RuntimeError(f"{LOG} Frame interpolation returned {res.shape[0]} frames, expected {multiplier + 1}.")
    return [res[k].float().cpu().numpy() for k in range(1, multiplier)]


class PertineoSeamInterpolate:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder", "report")
    OUTPUT_TOOLTIPS = ("The new sequence folder – connect it to Load Image Sequence to check the result.",
                       "Which frames were replaced, between which anchors.")
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Repairs short jumps and stutter (duplicate frames) at the given seams by replacing frames, and "
                   "writes a complete new sequence. Around every seam "
                   "the frames are replaced by in-betweens from a frame interpolation model (FILM or RIFE, loaded with "
                   "ComfyUI's 'Load Frame Interpolation Model'), calculated from the clean frames on both sides – the "
                   "jump becomes an even movement. All other frames are copied unchanged (bit-identical). Same file "
                   "names, format, bit depth and channels as the source. For zoom steps and drift over many frames "
                   "use Seam Stabilize – it keeps the real frames and only moves them.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interp_model": ("INTERP_MODEL", {
                    "tooltip": "Frame interpolation model from 'Load Frame Interpolation Model'. FILM handles large "
                               "motion best."}),
                "folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "SEQUENCE: folder, any frame of it, or name.####.exr",
                    "tooltip": "Folder with the source sequence (gap-free)."}),
                "seams": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "e.g. 1097, 1193  – or connect Seam Detect",
                    "tooltip": "Frame numbers of the seams (cut between t-1 and t), separated by commas, spaces or "
                               "new lines. Text after # is ignored. Connect Seam Detect (seams) or type them in."}),
                "radius": ("INT", {"default": 3, "min": 1, "max": 24,
                    "tooltip": "Frames replaced on each side of a seam. 3 = frames t-3 .. t+2 are replaced (6 frames), "
                               "interpolated between t-4 and t+3. Larger = smoother for big jumps, but more invented "
                               "frames (the model assumes straight, even motion between the anchors)."}),
                "output_folder": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "empty = next to the source: <folder>_smooth",
                    "tooltip": "Folder for the complete new sequence. Empty = a folder next to the source with "
                               "'_smooth' added. Must not be the source folder."}),
                "if_exists": (["overwrite", "error"], {
                    "tooltip": "If the output folder already holds files of this sequence: overwrite them or stop."}),
            },
            "optional": {
                "add_seams": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Extra seam frames to smooth in addition to the connected list (e.g. one that Seam "
                               "Detect missed)."}),
                "skip_seams": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Seam frames from the list to leave untouched (false alarms)."}),
                "exr_is_linear": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "EXR files are scene-linear: converted to sRGB for the model and back to linear when "
                               "written. Note: repaired EXR frames are limited to 0-1 (like Save Media with "
                               "clamp on). Turn off for EXRs saved with sRGB values."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, folder=None):
        if folder is None:
            return True
        return check_sequence_folder(folder, "SEQUENCE") or True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always write when queued

    def run(self, interp_model, folder, seams, radius, output_folder, if_exists,
            add_seams="", skip_seams="", exr_is_linear=True):
        folder = normalize_folder(folder)
        frames, _ = index_frames(folder)
        wanted = _range(frames, -1, -1)
        first, last = wanted[0], wanted[-1]

        out_dir = normalize_output_folder(output_folder) if (output_folder or "").strip() else \
            folder.rstrip(os.sep) + "_smooth"
        if os.path.realpath(out_dir) == os.path.realpath(folder):
            raise ValueError(f"{LOG} output_folder is the source folder – choose another folder.")
        names = {f: os.path.basename(frames[f]) for f in wanted}
        if os.path.isdir(out_dir) and if_exists == "error":
            clash = [n for n in names.values() if os.path.exists(os.path.join(out_dir, n))]
            if clash:
                raise ValueError(f"{LOG} {len(clash)} file(s) already exist in {out_dir} (e.g. {clash[0]}). "
                                 f"Set if_exists to overwrite or choose another folder.")

        skip = set(parse_frames(skip_seams))
        todo = [t for t in parse_frames(seams) + parse_frames(add_seams) if t not in skip]
        segments, skipped = plan_segments(todo, first, last, radius)

        if not segments:
            msg = "No seams to smooth – nothing written."
            if skipped:
                msg += "\n" + "\n".join(f"Seam {t} skipped: {why}" for t, why in skipped)
            print(f"{LOG} SEAM INTERPOLATE: {msg}")
            return {"ui": {"text": [msg]}, "result": (folder, msg)}

        os.makedirs(out_dir, exist_ok=True)
        replaced = {f for a, b, _ in segments for f in range(a + 1, b)}
        total = len(wanted)
        pbar = comfy.utils.ProgressBar(total)
        done = [0]

        def tick(k=1):
            done[0] += k
            pbar.update_absolute(min(done[0], total), total)

        lines = [f"Source: {folder} ({first}-{last})", f"Output: {out_dir}", f"Radius {radius}", ""]
        print(f"{LOG} SEAM INTERPOLATE: {len(todo)} seam(s) -> {len(segments)} window(s), "
              f"{len(replaced)} frame(s) replaced -> {out_dir}")

        # 1. Repair windows
        for a, b, ts in segments:
            fa, fb = Frame(frames[a]), Frame(frames[b])
            if fa.rgb.shape != fb.rgb.shape:
                raise ValueError(f"{LOG} Frames {a} and {b} have different sizes.")
            lin = fa.kind == "exr" and exr_is_linear

            def prep(x):
                return linear_to_srgb(x) if lin else np.clip(x, 0.0, 1.0)

            m = b - a
            mids = _interpolate(interp_model, prep(fa.rgb), prep(fb.rgb), m)
            amids = None
            if fa.alpha is not None or fb.alpha is not None:
                aa = fa.alpha if fa.alpha is not None else np.ones(fa.rgb.shape[:2], np.float32)
                ab = fb.alpha if fb.alpha is not None else np.ones(fb.rgb.shape[:2], np.float32)
                amids = [x[..., 0] for x in _interpolate(interp_model, np.dstack([aa] * 3),
                                                          np.dstack([ab] * 3), m)]
            for k, f in enumerate(range(a + 1, b)):
                rgb = srgb_to_linear(mids[k]) if lin else mids[k]
                ref = fa if (f - a) <= (b - f) else fb   # extra channels / layout from the nearer anchor
                ref.write_like(os.path.join(out_dir, names[f]), rgb, None if amids is None else amids[k])
                tick()
            lines.append(f"Seam {', '.join(map(str, ts))}: replaced {a + 1}-{b - 1} ({m - 1} frames), "
                         f"anchors {a} -> {b}, x{m}")

        # 2. Copy everything else unchanged
        def copy(f):
            shutil.copy2(frames[f], os.path.join(out_dir, names[f]))

        rest = [f for f in wanted if f not in replaced]
        with ThreadPoolExecutor(max_workers=8) as ex:
            for _ in ex.map(copy, rest):
                tick()

        for t, why in skipped:
            lines.append(f"Seam {t} skipped: {why}")
        lines += ["", f"{len(replaced)} frame(s) replaced, {len(rest)} copied unchanged."]
        report = "\n".join(lines)
        print(f"{LOG} SEAM INTERPOLATE: done – {out_dir}")
        return {"ui": {"text": [report]}, "result": (out_dir, report)}


# --------------------------------------------------------------------------
# Seam Bridge: window for a video model + writer
# --------------------------------------------------------------------------
class PertineoSeamWindow:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("INT", "INT", "MASK", "PERTINEO_SEAM_WINDOW", "INT", "STRING")
    RETURN_NAMES = ("first_frame", "last_frame", "mask", "window", "run_count", "info")
    OUTPUT_TOOLTIPS = (
        "First frame of the window – connect it to Load Image Sequence (first_frame).",
        "Last frame of the window – connect it to Load Image Sequence (last_frame).",
        "Temporal inpaint mask for the window: white on the bridge frames (regenerated), black on the context "
        "frames (kept, they guide the model). Small resolution – resize it to the working size.",
        "Window description for Seam Write.",
        "Number of seams in the list = batch count for one run over all seams.",
        "Which seam this run works on and which frames are regenerated.")
    FUNCTION = "run"
    DESCRIPTION = ("Picks seam number seam_index from the list and returns the frame window around it for a video "
                   "model (e.g. an LTX inpaint workflow): bridge frames on the cut are regenerated, context frames "
                   "on both sides guide motion and look. Set seam_index to increment and queue run_count runs to "
                   "bridge all seams one after the other.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "SEQUENCE: folder, any frame of it, or name.####.exr",
                    "tooltip": "The source sequence (the one with the seams)."}),
                "seams": ("STRING", {"default": "", "multiline": True, "placeholder": "e.g. 1270, 1321, 1481",
                    "tooltip": "Seam frames (cut between t-1 and t). Type them or connect Seam Detect (seams)."}),
                "seam_index": ("INT", {"default": 0, "min": 0, "max": 999, "control_after_generate": "increment",
                    "tooltip": "Which seam of the list this run bridges (0 = first). Counts up per run – set it "
                               "back to 0 before a new shot."}),
                "context": ("INT", {"default": 8, "min": 1, "max": 64,
                    "tooltip": "Real frames on each side of the bridge that guide the model (motion, look)."}),
                "bridge": ("INT", {"default": 9, "min": 1, "max": 121,
                    "tooltip": "Frames regenerated across the cut. Raised automatically so the whole window is "
                               "8n+1 frames long (LTX)."}),
                "frame_step": ("INT", {"advanced": True, "default": 8, "min": 1, "max": 16,
                    "tooltip": "Window length rule of the video model: step*n+1 (LTX = 8). 1 = any length."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, folder, seams, seam_index, context, bridge, frame_step=8):
        folder = normalize_folder(folder)
        frames, _ = index_frames(folder)
        wanted = _range(frames, -1, -1)
        lst = parse_frames(seams)
        if not lst:
            raise ValueError(f"{LOG} Seam Window: no seam frames given.")
        if seam_index >= len(lst):
            raise ValueError(f"{LOG} Seam Window: seam_index {seam_index}, but the list has only {len(lst)} seam(s) "
                             f"({', '.join(map(str, lst))}) – all done. Set seam_index back to 0 for a new shot.")
        win = bridge_window(lst[seam_index], wanted[0], wanted[-1], context, bridge, frame_step)
        win["folder"] = folder
        h, w = Frame(frames[win["first"]]).rgb.shape[:2]
        m = np.zeros((win["length"], max(8, h // 8), max(8, w // 8)), np.float32)
        m[win["bridge_first"] - win["first"]:win["bridge_last"] - win["first"] + 1] = 1.0
        mask = torch.from_numpy(m)
        win["size"] = (h, w)
        info = (f"Seam {seam_index + 1}/{len(lst)}: {win['seam']}  window {win['first']}-{win['last']} "
                f"({win['length']} frames), regenerated {win['bridge_first']}-{win['bridge_last']}")
        print(f"{LOG} SEAM WINDOW: {info}")
        return {"ui": {"text": [info]}, "result": (win["first"], win["last"], mask, win, len(lst), info)}


class PertineoSeamWrite:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder", "report")
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Writes the regenerated bridge frames of one seam window into a complete new sequence. On the "
                   "first run every source frame is copied (bit-identical); bridge frames are replaced, and over "
                   "`feather` frames on each side the result is crossfaded into the original. Same file names, "
                   "format and bit depth as the source; the images are scaled back to the source size.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The generated window (all frames of the window, sRGB)."}),
                "window": ("PERTINEO_SEAM_WINDOW", {"tooltip": "From Seam Window."}),
                "output_folder": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "empty = next to the source: <folder>_bridge",
                    "tooltip": "Folder for the complete new sequence. Empty = '<source folder>_bridge'. All seams of "
                               "a shot go into the same folder."}),
                "feather": ("INT", {"default": 2, "min": 0, "max": 16,
                    "tooltip": "Context frames on each side that crossfade from the generated frames back into the "
                               "original. 0 = hard cut at the bridge edge."}),
            },
            "optional": {
                "exr_is_linear": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "EXR files are scene-linear: the sRGB images are converted back to linear. Replaced "
                               "frames are limited to 0-1."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, images, window, output_folder, feather, exr_is_linear=True):
        win = window
        folder = win["folder"]
        frames, _ = index_frames(folder)
        out_dir = normalize_output_folder(output_folder) if (output_folder or "").strip() else \
            folder.rstrip(os.sep) + "_bridge"
        if os.path.realpath(out_dir) == os.path.realpath(folder):
            raise ValueError(f"{LOG} output_folder is the source folder – choose another folder.")
        imgs = images.float().cpu().numpy() if hasattr(images, "cpu") else np.asarray(images)
        if imgs.shape[0] != win["length"]:
            raise ValueError(f"{LOG} Seam Write: got {imgs.shape[0]} images, the window has {win['length']} frames "
                             f"({win['first']}-{win['last']}).")
        os.makedirs(out_dir, exist_ok=True)

        # 1. complete copy (only files that are not there yet – earlier seams stay)
        missing = [f for f in sorted(frames) if not os.path.exists(os.path.join(out_dir, os.path.basename(frames[f])))]
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda f: shutil.copy2(frames[f], os.path.join(out_dir, os.path.basename(frames[f]))), missing))

        # 2. bridge + crossfade
        weights = blend_weights(win, feather)
        h, w = win["size"]
        pbar = comfy.utils.ProgressBar(len(weights))
        for k, (f, wt) in enumerate(sorted(weights.items())):
            src = Frame(frames[f])
            lin = src.kind == "exr" and exr_is_linear
            gen = imgs[f - win["first"]][..., :3]
            if gen.shape[:2] != (h, w):
                if cv2 is None:
                    raise RuntimeError(f"{LOG} Seam Write needs OpenCV to scale the images back to {w}x{h}.")
                interp = cv2.INTER_AREA if gen.shape[0] > h else cv2.INTER_LANCZOS4
                gen = np.clip(cv2.resize(gen, (w, h), interpolation=interp), 0.0, 1.0)
            orig = linear_to_srgb(src.rgb) if lin else np.clip(src.rgb, 0.0, 1.0)
            out = wt * gen + (1.0 - wt) * orig
            src.write_like(os.path.join(out_dir, os.path.basename(frames[f])),
                           srgb_to_linear(out) if lin else out, src.alpha)
            pbar.update_absolute(k + 1, len(weights))
        report = (f"Seam {win['seam']}: regenerated {win['bridge_first']}-{win['bridge_last']}, crossfade "
                  f"{feather} frame(s) each side -> {out_dir}" + (f"  (copied {len(missing)} source frames)" if missing else ""))
        print(f"{LOG} SEAM WRITE: {report}")
        return {"ui": {"text": [report]}, "result": (out_dir, report)}


NODE_CLASS_MAPPINGS = {
    "PertineoSeamDetect": PertineoSeamDetect,
    "PertineoSeamStabilize": PertineoSeamStabilize,
    "PertineoSeamInterpolate": PertineoSeamInterpolate,
    "PertineoSeamWindow": PertineoSeamWindow,
    "PertineoSeamWrite": PertineoSeamWrite,
    "PertineoOverlapBlend": PertineoOverlapBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PertineoSeamDetect": "Pertineo Seam Detect",
    "PertineoSeamStabilize": "Pertineo Seam Stabilize",
    "PertineoSeamInterpolate": "Pertineo Seam Interpolate",
    "PertineoSeamWindow": "Pertineo Seam Window",
    "PertineoSeamWrite": "Pertineo Seam Write",
    "PertineoOverlapBlend": "Pertineo Overlap Blend",
}
