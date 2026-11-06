# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – Load Video: MOV / MP4 / MXF … through ffmpeg in full bit depth (10 / 12 bit stay 16 bit
precision), with alpha (ProRes 4444), exact frame range and the color matrix of the file."""

import json
import os
import subprocess

import numpy as np
import torch

import comfy.utils
import folder_paths

from ..common import LOG
from .media import ffmpeg_path, folder_root_for
from .readwrite import clean_path

CATEGORY = "Pertineo/IO"
VIDEO_EXTS = (".mov", ".mp4", ".mxf", ".mkv", ".webm", ".avi", ".m4v")
VIDEO_MASKS = ["alpha", "luminance", "red", "green", "blue"]
_NO_FILES = "– no videos in the input folder –"
_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt601": "bt601", "fcc": "fcc",
           "smpte240m": "smpte240m", "bt2020nc": "bt2020", "bt2020c": "bt2020"}


def _ffprobe():
    exe = ffmpeg_path()
    if not exe:
        return None
    cand = os.path.join(os.path.dirname(exe), "ffprobe")
    return cand if os.path.isfile(cand) else "ffprobe"


def probe(path):
    """Size, pixel format, fps, frame count, color matrix / range of the first video stream."""
    exe = _ffprobe()
    if not exe:
        raise IOError(f"{LOG} Load Video needs ffmpeg / ffprobe (Linux Mint / Ubuntu: sudo apt install ffmpeg).")
    out = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_frames,color_space,"
                          "color_range,duration:format=duration", "-of", "json", path],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise IOError(f"{LOG} Cannot read video {path}: {out.stderr.strip()[-300:]}")
    data = json.loads(out.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise IOError(f"{LOG} No video stream in {path}")
    s = streams[0]

    def rate(text):
        try:
            a, b = str(text).split("/")
            return float(a) / float(b) if float(b) else 0.0
        except Exception:
            return 0.0

    fps = rate(s.get("r_frame_rate")) or rate(s.get("avg_frame_rate"))
    frames = None
    try:
        frames = int(s.get("nb_frames"))
    except (TypeError, ValueError):
        dur = s.get("duration") or (data.get("format") or {}).get("duration")
        try:
            frames = int(round(float(dur) * fps)) if dur and fps else None
        except ValueError:
            frames = None
    pix = s.get("pix_fmt") or ""
    return {"width": int(s["width"]), "height": int(s["height"]), "pix_fmt": pix, "fps": fps, "frames": frames,
            "alpha": pix.startswith(("yuva", "rgba", "argb", "bgra", "abgr", "gbrap", "ya", "pal8")),
            "color_space": s.get("color_space") or "", "color_range": s.get("color_range") or ""}


def probe_audio(path):
    """(sample_rate, channels) of the first audio stream, or None when the file has no audio."""
    exe = _ffprobe()
    out = subprocess.run([exe, "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=sample_rate,channels", "-of", "json", path],
                         capture_output=True, text=True, timeout=60)
    try:
        s = (json.loads(out.stdout or "{}").get("streams") or [])[0]
        return int(s["sample_rate"]), max(1, int(s.get("channels") or 1))
    except (IndexError, KeyError, ValueError, TypeError):
        return None


def load_audio(path, start, duration, fallback_rate=48000):
    """Audio of [start, start + duration) seconds as ComfyUI AUDIO ({'waveform': [1, C, S], 'sample_rate'}),
    float32, original sample rate and channels. Without an audio stream: silence of the same length (stereo,
    48 kHz), so nodes after it always get a valid input. Returns (audio, note)."""
    info = probe_audio(path)
    if info is None:
        rate, ch = fallback_rate, 2
        samples = max(1, int(round(duration * rate)))
        return {"waveform": torch.zeros((1, ch, samples), dtype=torch.float32), "sample_rate": rate}, \
            "no audio in the file – silence"
    rate, ch = info
    cmd = [ffmpeg_path(), "-v", "error", "-ss", f"{start:.6f}", "-i", path, "-t", f"{duration:.6f}",
           "-map", "0:a:0", "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(ch), "-ar", str(rate), "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=600)
    data = np.frombuffer(raw.stdout, dtype="<f4")
    want = max(1, int(round(duration * rate)))
    frames = data[: (len(data) // ch) * ch].reshape(-1, ch).T if len(data) >= ch else np.zeros((ch, 0), np.float32)
    if frames.shape[1] < want:  # audio shorter than the picture: pad with silence, keep the length in sync
        frames = np.pad(frames, ((0, 0), (0, want - frames.shape[1])))
    frames = frames[:, :want]
    wave = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))[None]
    return {"waveform": wave, "sample_rate": rate}, f"{rate} Hz, {ch} ch"


def _input_videos():
    d = folder_paths.get_input_directory()
    try:
        return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(VIDEO_EXTS))
    except OSError:
        return []


def _resolve(video, path):
    p = clean_path(path or "")
    if p:
        return p
    return folder_paths.get_annotated_filepath(video)


class PertineoLoadVideo:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "FLOAT", "STRING", "STRING", "STRING", "INT", "INT", "AUDIO")
    RETURN_NAMES = ("image", "mask", "frame_count", "fps", "folder", "folder_root", "name", "width", "height",
                    "audio")
    OUTPUT_TOOLTIPS = ("The frames as sRGB-encoded images (16-bit precision, so 10 / 12-bit sources keep their "
                       "steps).", "Mask from mask_channel (alpha needs a video with alpha, e.g. ProRes 4444).",
                       "Number of loaded frames.", "Frame rate of the loaded frames (divided by every_nth).",
                       "Folder the video is in.",
                       "Folder for Output Name (output_folder): the video's folder, or the one above it when the "
                       "video sits in a date folder like 2026-11-06 – so the next version is written next to this "
                       "one. A video from ComfyUI's input folder (upload) gives the ComfyUI output folder instead.",
                       "File name without folder and extension – connect it to Save Media (name).",
                       "Width in pixels.", "Height in pixels.",
                       "The audio of the loaded frame range (original sample rate and channels, 32-bit float). "
                       "Without an audio track: silence of the same length, so nodes after it always work.")
    FUNCTION = "run"
    DESCRIPTION = ("Loads a video (MOV, MP4, MXF, MKV …) through ffmpeg in full bit depth: 10- and 12-bit sources "
                   "keep their precision, ProRes 4444 keeps its alpha. Exact frame range by frame index (not by "
                   "time), color matrix from the file (untagged HD = Rec.709). No scaling. For long or "
                   "high-resolution clips an image sequence is the better master – this loader is for reference, "
                   "review and delivery clips. The audio of the loaded range comes out as 'audio'.")

    @classmethod
    def INPUT_TYPES(cls):
        files = _input_videos()
        return {
            "required": {
                "video": (files or [_NO_FILES], {"video_upload": True,
                    "tooltip": "Video in the ComfyUI input folder (or upload one). Ignored when 'path' is set."}),
                "first_frame": ("INT", {"default": 0, "min": 0, "max": 10_000_000,
                    "tooltip": "First frame to load, counted from 0 = first frame of the video (a frame index, not "
                               "a timecode)."}),
                "frame_count": ("INT", {"default": 0, "min": 0, "max": 10_000_000,
                    "tooltip": "Number of frames to load (output frames – with every_nth 2 and frame_count 10 you get "
                               "10 frames from a range of 19). 0 = everything up to the end."}),
                "every_nth": ("INT", {"advanced": True, "default": 1, "min": 1, "max": 100,
                    "tooltip": "Load only every n-th frame (1 = every frame). fps is divided accordingly."}),
                "mask_channel": (VIDEO_MASKS, {"advanced": True,
                    "tooltip": "Channel for the mask output. 'alpha' needs a video with alpha (e.g. ProRes 4444); "
                               "without one the mask is empty. luminance / red / green / blue read a matte "
                               "rendered as grey video."}),
            },
            "optional": {
                "path": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "optional: full path to a video outside the input folder",
                    "tooltip": "Full path to a video anywhere on disk (e.g. on the NAS). Overrides 'video'."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, video=None, path=None):
        p = clean_path(path or "")
        if p:
            return True if os.path.isfile(p) else f"Video not found: {p}"
        if video is None:
            return True
        if video == _NO_FILES or not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True

    @classmethod
    def IS_CHANGED(cls, video=None, path=None, **kwargs):
        try:
            st = os.stat(_resolve(video, path))
            return f"{st.st_size}:{st.st_mtime_ns}"
        except Exception:
            return float("nan")

    def run(self, video, first_frame, frame_count, every_nth, mask_channel, path=""):
        src = _resolve(video, path)
        info = probe(src)
        w, h = info["width"], info["height"]
        alpha = info["alpha"]
        total = info["frames"]
        if total is not None and first_frame >= total:
            raise ValueError(f"{LOG} first_frame {first_frame} is beyond the end of the video ({total} frames).")
        avail = (total - first_frame) if total is not None else None
        # frame_count = frames that come OUT; with every_nth they span frame_count * every_nth source frames
        want = (frame_count - 1) * every_nth + 1 if frame_count > 0 else avail
        if want is not None and avail is not None:
            want = min(want, avail)
        expected = None if want is None else (want + every_nth - 1) // every_nth

        matrix = _MATRIX.get(info["color_space"]) or ("bt709" if h >= 720 else "bt601")
        rng = "pc" if info["color_range"] in ("pc", "jpeg") else "tv"
        sel = f"gte(n\\,{first_frame})"
        if want is not None:
            sel += f"*lt(n\\,{first_frame + want})"
        if every_nth > 1:
            sel += f"*not(mod(n-{first_frame}\\,{every_nth}))"
        out_fmt = "rgba64le" if alpha else "rgb48le"
        ch = 4 if alpha else 3
        vf = f"select='{sel}',scale=in_color_matrix={matrix}:in_range={rng},format={out_fmt}"
        cmd = [ffmpeg_path(), "-v", "error", "-i", src, "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough"]
        if expected is not None:
            cmd += ["-frames:v", str(expected)]
        cmd += ["-f", "rawvideo", "-pix_fmt", out_fmt, "-"]
        print(f"{LOG} Load Video: {os.path.basename(src)} {w}x{h} {info['pix_fmt']} {info['fps']:.3f} fps, "
              f"{total if total is not None else '?'} frames -> loading from {first_frame}"
              + (f", {expected} frames" if expected is not None else "") + f" (matrix {matrix}, {rng} range)")

        frame_bytes = w * h * ch * 2
        images = torch.empty((expected, h, w, 3), dtype=torch.float32) if expected else None
        masks = torch.empty((expected, h, w), dtype=torch.float32) if expected else None
        extra_i, extra_m = [], []
        n = 0
        pbar = comfy.utils.ProgressBar(max(1, expected or 1))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if not buf or len(buf) < frame_bytes:
                    break
                px = np.frombuffer(buf, dtype="<u2").reshape(h, w, ch).astype(np.float32) / 65535.0
                rgb = px[..., :3]
                if mask_channel == "alpha":
                    m = px[..., 3] if alpha else np.zeros((h, w), np.float32)
                elif mask_channel == "luminance":
                    m = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                else:
                    m = rgb[..., {"red": 0, "green": 1, "blue": 2}[mask_channel]]
                ti = torch.from_numpy(np.ascontiguousarray(rgb))
                tm = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))
                if images is not None and n < len(images):
                    images[n] = ti
                    masks[n] = tm
                else:
                    extra_i.append(ti)
                    extra_m.append(tm)
                n += 1
                pbar.update_absolute(min(n, expected or n), expected or n)
        finally:
            err = proc.stderr.read().decode("utf-8", "replace").strip()
            proc.stdout.close()
            proc.wait()
        if n == 0:
            raise IOError(f"{LOG} No frames decoded from {src}" + (f": {err[-300:]}" if err else ""))
        if mask_channel == "alpha" and not alpha:
            print(f"{LOG} Load Video: {os.path.basename(src)} has no alpha channel – mask is empty.")
        if images is None:
            images, masks = torch.stack(extra_i, 0), torch.stack(extra_m, 0)
        else:
            if n < len(images):
                images, masks = images[:n], masks[:n]
            if extra_i:
                images = torch.cat([images, torch.stack(extra_i, 0)], 0)
                masks = torch.cat([masks, torch.stack(extra_m, 0)], 0)
        fps = info["fps"] / every_nth if info["fps"] else 0.0
        name = os.path.splitext(os.path.basename(src))[0]
        # audio covers the whole loaded range in time (with every_nth the picture skips frames, the sound does not)
        src_fps = info["fps"] or 24.0
        span = (n - 1) * every_nth + 1
        audio, note = load_audio(src, first_frame / src_fps, span / src_fps)
        print(f"{LOG} Load Video: audio {note}, {audio['waveform'].shape[-1] / audio['sample_rate']:.3f} s "
              f"from {first_frame / src_fps:.3f} s")
        folder = os.path.dirname(os.path.abspath(src))
        return (images, masks, n, float(fps), folder, folder_root_for(folder), name, w, h, audio)
