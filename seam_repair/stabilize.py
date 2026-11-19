# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – Seam Stabilize and Overlap Blend: repair the joins of chained video generations (extend / upscale
in sections) directly on the image sequence. Only a window around each seam is loaded, so long 4K shots never
have to fit into memory; every other frame is taken over unchanged."""

import datetime
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

import comfy.utils
import folder_paths

from ..common import LOG
from ..image_io import media as M
from ..image_io.readwrite import check_sequence_folder, index_frames, normalize_folder, sequence_prefix
from . import stabilize_core as S
from .core import Frame

CATEGORY = "Pertineo/Seam Repair"

BORDERS = ["extend edges", "black", "auto scale to fill"]
UNCHANGED = ["hard link (no extra space)", "copy"]
BLEND_MODES = ["flow (morph)", "cross-dissolve", "hard cut (later section)"]
CURVES = ["smoothstep", "linear"]
DETAIL_MODES = ["match at detail_seams", "deflicker (whole sequence)"]

_UNCHANGED_TIP = ("Frames outside the seam windows are taken over bit for bit. hard link: the new sequence takes "
                  "no extra disk space (both names point to the same file – fine as long as nobody overwrites "
                  "frames in place; renders from Nuke and ComfyUI write new files). copy: fully independent files. "
                  "Across drives a copy is always made.")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _plan(output_name, src_folder, suffix):
    """Target of the new sequence: from Output Name, otherwise next to the source sequence as
    <source name>_<suffix>_v###/ (next free version)."""
    if isinstance(output_name, dict):
        root, full = output_name["root"], output_name["full"]
        P = M.plan_paths(root, output_name["date_folder"], full, now=output_name.get("now"))
        note = f"from Output Name ({output_name['note']})"
    else:
        seq = sequence_prefix(src_folder)
        root = os.path.dirname(os.path.normpath(src_folder)) or src_folder  # right next to the source folder
        base = M.clean_name(f"{seq}_{suffix}") or suffix
        # the source keeps its version, the new sequence gets its own: shot_ext_v002 -> shot_ext_v002_stabilized_v001
        full, _ver, _n, note = M.resolve_version("next free", 1, root, base, M.VERSION_IN_NAME[1])
        P = M.plan_paths(root, "off", full)
    if os.path.isdir(P["seq_dir"]) and os.listdir(P["seq_dir"]):
        raise RuntimeError(f"{LOG} Target folder already holds files: {P['seq_dir']} – choose another version "
                           "(Output Name: next free) so nothing is overwritten.")
    os.makedirs(P["seq_dir"], exist_ok=True)
    return P, note


def _frame_path(P, frame, pad, ext):
    return os.path.join(P["seq_dir"], f"{P['base']}.{frame:0{pad}d}{ext}")


def _finish(P, written, write_checksums, write_workflow, report, prompt, extra_pnginfo, report_name):
    """.sha256 list, .workflow.json and the text report next to the new sequence folder."""
    extras = []
    if write_checksums:
        with ThreadPoolExecutor(max_workers=M.worker_count()) as ex:
            digests = dict(zip(written, ex.map(M.sha256_file, written)))
        cpath = os.path.join(P["folder"], P["base"] + ".sha256")
        M.write_checksums(cpath, digests, P["folder"])
        extras.append(os.path.basename(cpath))
    if write_workflow:
        try:
            from ..audit import sidecar as SC
            wf = SC.workflow_record(prompt, extra_pnginfo)
            wp, _ = SC.write_workflow_file(P["folder"], P["base"], wf)
            if wp:
                extras.append(os.path.basename(wp))
        except Exception as e:
            print(f"{LOG} workflow not written ({type(e).__name__}: {e})")
    rpath = os.path.join(P["folder"], P["base"] + report_name)
    with open(rpath, "w", encoding="utf-8", newline="\n") as f:
        f.write(report + "\n")
    extras.append(os.path.basename(rpath))
    return extras


def _temp_png(img, prefix):
    name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    img.save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    return {"filename": name, "subfolder": "", "type": "temp"}


def _font(size=13):
    try:
        from ..image_io.save_media import _font as f
        return f(size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _curve_image(rows):
    """Small chart per window: deviation from a straight line between the anchors, raw (grey) and result (orange),
    for position, scale, rotation and perspective. Shown on the node (no frontend extension needed)."""
    panels = [("position px", ("x", "y")), ("scale %", ("scale",)), ("rotation deg", ("rotation",)),
              ("perspective px", ("perspective",))]
    pw, ph, gap, top = 230, 120, 10, 22
    W = len(panels) * (pw + gap) + gap
    H = max(1, len(rows)) * (ph + top + gap) + gap
    img = Image.new("RGB", (W, H), (30, 30, 30))
    d = ImageDraw.Draw(img)
    font = _font(12)
    for r, row in enumerate(rows):
        y0 = gap + r * (ph + top + gap)
        d.text((gap, y0), row["title"], font=font, fill=(225, 225, 225))
        n = len(row["frames"])
        for c, (label, keys) in enumerate(panels):
            x0 = gap + c * (pw + gap)
            box = (x0, y0 + top, x0 + pw, y0 + top + ph)
            d.rectangle(box, fill=(42, 42, 42))
            d.text((x0 + 4, y0 + top + 2), label, font=font, fill=(150, 150, 150))
            series = []
            for k in keys:
                raw, res = row["curves"][k]
                line = np.linspace(raw[0], raw[-1], n)
                series.append((np.asarray(raw) - line, np.asarray(res) - line))
            vals = np.concatenate([np.r_[a, b] for a, b in series]) if series else np.zeros(1)
            span = max(float(np.max(np.abs(vals))), 1e-6)
            tag = f"±{span:.3g}"
            d.text((box[2] - 6 - d.textlength(tag, font=font), y0 + top + 2), tag, font=font, fill=(150, 150, 150))

            def pts(v):
                xs = np.linspace(box[0] + 4, box[2] - 4, n)
                ys = (box[1] + box[3]) / 2 - np.asarray(v) / span * (ph / 2 - 16)
                return list(zip(xs.tolist(), ys.tolist()))

            for s in row["seams"]:
                if row["frames"][0] <= s <= row["frames"][-1]:
                    xs = box[0] + 4 + (s - row["frames"][0]) / max(1, n - 1) * (pw - 8)
                    d.line([(xs, box[1] + 14), (xs, box[3])], fill=(90, 90, 140))
            for raw, res in series:
                if n > 1:
                    d.line(pts(raw), fill=(140, 140, 140), width=1)
                    d.line(pts(res), fill=(255, 150, 40), width=2)
    return img


# --------------------------------------------------------------------------
# Seam Stabilize
# --------------------------------------------------------------------------
class PertineoSeamStabilize:
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder", "report")
    OUTPUT_TOOLTIPS = (
        "The new sequence folder – connect it to Load Image Sequence, Seam Interpolate, Seam Window or another "
        "Seam Stabilize.",
        "What was measured and changed at every seam (also written as <name>.seam_report.txt).")
    FUNCTION = "run"
    DESCRIPTION = (
        "Smooths jumps at the joins of chained generations (H3 extend, LTX upscale in sections …) directly on the "
        "image sequence – like Tracker > smooth > Stabilize on the difference in Nuke. Around each seam only a "
        "window is loaded; the camera path (position, rotation, scale, perspective) is measured with optical flow, "
        "smoothed, and only the difference is applied as a warp. The anchor frames at both ends of each window and "
        "every frame outside stay exactly as they are. Optional: morph (local jumps), detail match (sharpness of "
        "the section after a seam matched to the frames before it). Writes a new, complete sequence. Seams can come "
        "from Seam Detect.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "SEQUENCE: folder, any frame of it, or name.####.exr",
                    "tooltip": "The image sequence with the seams (EXR, PNG, TIFF, JPG). It is never changed – the "
                               "result is a new sequence."}),
                "seams": ("STRING", {"default": "", "placeholder": "1668, 1872, 2076   or ranges 1650-1700",
                    "tooltip": "Frame numbers of the seams = first frame of the new section (the cut between t-1 and t), "
                               "separated by commas – typed or from Seam Detect (seams): "
                               "+/- window frames around each are smoothed. Or ranges 'a-b': a and b are the anchor "
                               "frames that stay unchanged, only the frames between them are worked on. Windows that "
                               "overlap are joined."}),
                "window": ("INT", {"default": 25, "min": 2, "max": 500,
                    "tooltip": "Frames before and after a single seam (25 = 51 frames incl. both anchors). Larger = "
                               "the correction is spread more gently; needs more RAM (4K float: ~130 MB per frame)."}),
                "position": ("BOOLEAN", {"default": True, "tooltip": "Smooth jumps in X / Y."}),
                "rotation": ("BOOLEAN", {"default": True, "tooltip": "Smooth jumps in rotation."}),
                "scale": ("BOOLEAN", {"default": True, "tooltip": "Smooth jumps in scale (zoom)."}),
                "perspective": ("BOOLEAN", {"default": False,
                    "tooltip": "Also smooth what position / rotation / scale do not explain: keystone, tilt (the "
                               "full homography, like a corner pin)."}),
                "morph": ("BOOLEAN", {"default": False,
                    "tooltip": "Local smoothing on top: parts of the image that jump on their own (a person, a "
                               "car) while the rest is calm. Every frame in the window is pulled towards the "
                               "middle of where its pixels are in the frames before and after – constant motion "
                               "stays. Experimental: can drag at edges where something gets covered."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the global correction is applied (1 = fully smoothed path)."}),
                "detail_seams": ("STRING", {"default": "",
                    "placeholder": "detail match after these seams, e.g. 1668 (empty = off)",
                    "tooltip": "Detail match: from each frame listed here up to the next listed one (or the end), "
                               "the sharpness of the luminance is matched band by band (fine to coarse detail) to "
                               "the frames right before it – e.g. original plate > extension from a 768p draft. "
                               "Not a blanket sharpen: each band gets its own gain. Can be any frame, also one "
                               "without a seam window."}),
            },
            "optional": {
                "output_name": ("PERTINEO_OUTPUT_NAME", {
                    "tooltip": "From Pertineo Output Name: folder, name and version of the new sequence. Without it: "
                               "right next to the source folder as <source name>_stabilized_v001/ (next free; the source keeps its own version in the name)."}),
                "smoothing": ("FLOAT", {"advanced": True, "default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                    "tooltip": "Width of the smoothing in frames (Gaussian sigma). 0 = auto: a sixth of the window, "
                               "so the correction has faded out before it reaches the anchors."}),
                "morph_strength": ("FLOAT", {"advanced": True, "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only with morph: how far each frame is pulled per pass (3 passes)."}),
                "border": (BORDERS, {"advanced": True,
                    "tooltip": "What fills the edge that a warp uncovers: extend edges (repeat the outermost "
                               "pixels), black, or auto scale to fill (zoom in just enough per window, faded in and "
                               "out like the correction)."}),
                "analysis_size": ("INT", {"advanced": True, "default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "Motion is measured on a copy of this size (long side); the warp is always applied "
                               "at full resolution. Larger = more precise, slower."}),
                "detail_strength": ("FLOAT", {"advanced": True, "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "1 = match the reference detail; less = partly; more = sharper than the reference."}),
                "detail_frames": ("INT", {"advanced": True, "default": 12, "min": 1, "max": 200,
                    "tooltip": "How many frames before (reference) and after (target) a detail seam are measured."}),
                "detail_max_gain": ("FLOAT", {"advanced": True, "default": 2.5, "min": 1.0, "max": 8.0, "step": 0.1,
                    "tooltip": "Upper limit per band, so noise is not blown up where the detail is simply gone."}),
                "detail_mode": (DETAIL_MODES, {
                    "tooltip": "match at detail_seams: only the sections after the frames in detail_seams are matched "
                               "to the frames before them (hard steps, e.g. original plate > draft extension). "
                               "deflicker (whole sequence): the sharpness of EVERY frame is brought to its smoothed "
                               "course over time (deflicker_window) – for sharpness that pulses through the whole shot "
                               "(chunked upscales, frame interpolation). Soft frames are sharpened band by band, "
                               "frames that stick out are calmed down. detail_seams still count as hard steps that are "
                               "matched first. Every frame is rewritten."}),
                "deflicker_window": ("INT", {"default": 25, "min": 2, "max": 500,
                    "tooltip": "Only with deflicker: frames before and after over which the sharpness is averaged. "
                               "Larger = calmer, but a slow, intended change of sharpness (focus pull) is evened out "
                               "too."}),
                "unchanged_frames": (UNCHANGED, {"advanced": True, "tooltip": _UNCHANGED_TIP}),
                "write_checksums": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Write <name>.sha256 with the SHA256 of every frame (sha256sum -c compatible)."}),
                "write_workflow": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Write <name>.workflow.json next to the new sequence."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, output_name=None, **kwargs):
        return float("nan")  # always writes a new version

    @classmethod
    def VALIDATE_INPUTS(cls, folder=None):
        if folder is not None:
            err = check_sequence_folder(folder, "Seam Stabilize")
            if err:
                return err
        return True

    def run(self, folder, seams, window, position, rotation, scale, perspective, morph, strength, detail_seams,
            output_name=None, smoothing=0.0, morph_strength=0.5, border=BORDERS[0], analysis_size=1024,
            detail_strength=1.0, detail_frames=12, detail_max_gain=2.5, detail_mode=DETAIL_MODES[0],
            deflicker_window=25,
            unchanged_frames=UNCHANGED[0], write_checksums=True, write_workflow=True, prompt=None,
            extra_pnginfo=None):
        S.need_cv2()
        t0 = time.time()
        src = normalize_folder(folder)
        frames, pad = index_frames(src)
        keys = sorted(frames)
        first, last = keys[0], keys[-1]
        wins = S.parse_seams(seams, window, first, last)
        dseams = S.parse_frames(detail_seams)
        deflicker = str(detail_mode).startswith("deflicker")
        if not wins and not dseams and not deflicker:
            raise RuntimeError(f"{LOG} Seam Stabilize: nothing to do – enter seams, detail_seams or set detail_mode "
                               "to deflicker.")
        for w in wins:
            missing = [f for f in range(w["a"], w["b"] + 1) if f not in frames]
            if missing:
                raise RuntimeError(f"{LOG} Seam Stabilize: frames missing in the window {w['a']}-{w['b']}: "
                                   f"{missing[:8]}{' …' if len(missing) > 8 else ''}")
        for s in dseams:
            if not first < s <= last:
                raise RuntimeError(f"{LOG} Seam Stabilize: detail seam {s} is outside the sequence ({first}-{last}) or "
                                   "has no frames before it.")
        ext = os.path.splitext(frames[first])[1].lower()
        f0 = Frame(frames[first])
        linear = f0.kind == "exr"
        extra = sorted(k for k in (f0.exr_channels or {}) if k not in ("R", "G", "B", "A"))
        kind = "exr " + ("half" if f0.exr_channels and f0.exr_channels["R"].dtype == np.float16 else "float") \
            if linear else f"{os.path.splitext(frames[first])[1][1:]} {f0.dtype}"
        del f0
        P, note = _plan(output_name, src, "stabilized")
        enabled = {"position": position, "rotation": rotation, "scale": scale, "perspective": perspective}
        hard = unchanged_frames.startswith("hard")

        lines = [f"Seam Stabilize – {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                 f"source:   {src}  ({len(keys)} frames, {first}-{last}, {kind})",
                 f"output:   {P['seq_dir']}  ({note})",
                 "smoothed: " + (", ".join(k for k in S.COMPONENTS if enabled[k]) or "nothing (global off)")
                 + (" + morph" if morph else "") + f" · strength {strength:g} · border {border}"]
        if extra:
            lines.append(f"EXR channels besides RGBA are kept as they are (not warped): {', '.join(extra)}")

        # ---- detail gains (measured on the source) -----------------------
        sections = []  # (start, end, gains)
        frame_gains = {}  # deflicker: frame -> gains
        if deflicker:
            # measure every frame (centred 1024 px crop at full resolution, read in parallel)
            def measure(f):
                return S.detail_energy(Frame(frames[f]).rgb, linear, crop=1024)

            mbar = comfy.utils.ProgressBar(len(keys))
            energy = []
            with ThreadPoolExecutor(max_workers=min(8, M.worker_count())) as ex:
                for e in ex.map(measure, keys):
                    energy.append(e)
                    mbar.update(1)
            energy = np.array(energy)
            base = np.ones_like(energy)
            for i, s in enumerate(dseams):
                end = (dseams[i + 1] - 1) if i + 1 < len(dseams) else last
                idx = np.array(keys)
                ref = (idx >= s - detail_frames) & (idx < s)
                tgt = (idx >= s) & (idx < s + detail_frames) & (idx <= end)
                g = S.detail_gains(energy[ref].mean(axis=0), (energy[tgt] * base[tgt]).mean(axis=0), 1.0,
                                   detail_max_gain)
                sec = (idx >= s) & (idx <= end)
                base[sec] *= g
                lines.append(f"detail step at {s}: gains fine->coarse " + " / ".join(f"{x:.2f}" for x in g))
            dgains = S.deflicker_gains(energy, base, deflicker_window / 3.0, detail_strength, detail_max_gain)
            frame_gains = {f: dgains[i] for i, f in enumerate(keys)}
            # flicker = frame-to-frame deviation of the finest band from its smoothed course (slow changes excluded)
            le = np.log(np.maximum(energy[:, 0], 1e-7))
            la = le + np.log(dgains[:, 0])
            fl0 = np.std(le - S.gauss1d(le, deflicker_window / 3.0)) * 100
            fl1 = np.std(la - S.gauss1d(la, deflicker_window / 3.0)) * 100
            lines.append(f"deflicker over +/- {deflicker_window} frames: sharpness flicker {fl0:.1f} % -> about "
                         f"{fl1:.1f} % (std, finest band); gains per frame {dgains.min():.2f} - {dgains.max():.2f}")
        else:
            for i, s in enumerate(dseams):
                end = (dseams[i + 1] - 1) if i + 1 < len(dseams) else last
                ref = [f for f in keys if s - detail_frames <= f < s]
                tgt = [f for f in keys if s <= f < s + detail_frames and f <= end]
                e_ref = np.mean([S.detail_energy(Frame(frames[f]).rgb, linear) for f in ref], axis=0)
                tgt_imgs = [Frame(frames[f]).rgb for f in tgt]
                e_tgt = np.mean([S.detail_energy(im, linear) for im in tgt_imgs], axis=0)
                g = S.detail_gains(e_ref, e_tgt, 1.0, detail_max_gain)
                # one correction step: measure again after applying (bands interact)
                e_new = np.mean([S.detail_energy(S.apply_detail(im, linear, g), linear) for im in tgt_imgs], axis=0)
                g = np.clip(g * np.where(e_new > 1e-7, e_ref / np.maximum(e_new, 1e-7), 1.0), 1 / detail_max_gain,
                            detail_max_gain)
                g = 1.0 + detail_strength * (g - 1.0)
                del tgt_imgs
                sections.append((s, end, g))
                lines.append(f"detail from {s} to {end}: reference {ref[0]}-{ref[-1]}, gains fine->coarse "
                             + " / ".join(f"{x:.2f}" for x in g))

        def gains_for(f):
            if deflicker:
                g = frame_gains.get(f)
                return None if g is None or np.all(np.abs(g - 1.0) < 1e-4) else g
            for s, e, g in sections:
                if s <= f <= e:
                    return g
            return None

        total = len(keys)
        pbar = comfy.utils.ProgressBar(total)
        written, done = [], set()
        rows = []

        # ---- windows -----------------------------------------------------
        for wi, w in enumerate(wins, 1):
            a, b = w["a"], w["b"]
            fr = list(range(a, b + 1))
            n = len(fr)
            data = [Frame(frames[f]) for f in fr]
            h, wd = data[0].rgb.shape[:2]
            prox, sc = [], (1.0, 1.0)
            for fd in data:
                p_, sc = S.proxy(fd.rgb, linear, analysis_size)
                prox.append(p_)
            wts = S.taper(n)
            sigma = smoothing if smoothing > 0 else n / 6.0
            p, qs, inl = S.window_motion(prox, sc, wd, h)
            targets, curves = S.smooth_path(p, qs, enabled, strength, sigma, wts)
            A = [S.homography_from(p, qs[k]) @ np.linalg.inv(S.homography_from(p, targets[k])) for k in range(n)]
            if border.startswith("auto"):
                need = np.array([S.fill_scale(A[k], wd, h) for k in range(n)])
                smax = float(need.max())
                zs = np.maximum(1.0 + (smax - 1.0) * wts, need)
                zs[0] = zs[-1] = 1.0
                A = [A[k] @ S.scale_about((wd / 2.0, h / 2.0), zs[k]) for k in range(n)]
            disp = [None] * n
            if morph and morph_strength > 0:
                sx, sy = sc
                Sm, Si = np.diag([sx, sy, 1.0]), np.diag([1 / sx, 1 / sy, 1.0])
                A_p = [Si @ A[k] @ Sm for k in range(n)]
                D = S.morph_fields(prox, A_p, wts, morph_strength)
                disp = [S.resize_flow(D[k], wd, h) if wts[k] > 0 else None for k in range(n)]

            for k, f in enumerate(fr):
                fd = data[k]
                rgb, al = fd.rgb, fd.alpha
                dst = _frame_path(P, f, pad, ext)
                g = gains_for(f)
                # no visible change (< 1/200 px at the corners) = the frame stays bit for bit
                ident = wts[k] == 0 or (disp[k] is None and
                                        np.abs(S.apply_h(A[k], p) - p).max() < 0.005)
                if ident and g is None:
                    S.link_or_copy(frames[f], dst, hard)
                else:
                    if not ident:
                        rgb = S.warp(rgb, A[k], disp[k], border)
                        al = None if al is None else S.warp(al, A[k], disp[k], border)
                    if g is not None:
                        rgb = S.apply_detail(rgb, linear, g)
                    fd.write_like(dst, rgb, al)
                written.append(dst)
                done.add(f)
                pbar.update(1)
                data[k] = None  # free the frame
            del data

            # report: jump at each seam, before -> after
            seam_txt = []
            check = list(w["seams"])
            if not check:  # a range: report where the biggest jump in the raw path is
                score = [S.jump(curves["scale"][0], i) * 50 + S.jump(curves["x"][0], i) + S.jump(curves["y"][0], i)
                         for i in range(n)]
                check = [a + int(np.argmax(score))]
            for s in check:
                i = s - a
                parts = []
                for key, unit, fmt in (("x", "px", ".2f"), ("y", "px", ".2f"), ("scale", "%", ".3f"),
                                       ("rotation", "deg", ".3f"), ("perspective", "px", ".2f")):
                    r0, r1 = S.jump(curves[key][0], i), S.jump(curves[key][1], i)
                    parts.append(f"{key} {r0:{fmt}}->{r1:{fmt}} {unit}")
                seam_txt.append(f"  {'seam' if w['seams'] else 'largest jump at'} {s}: " + " · ".join(parts))
            lines.append(f"window {wi}: {a}-{b} ({n} frames, anchors {a} and {b} unchanged), sigma {sigma:.1f}, "
                         f"tracking inliers {np.mean(inl) * 100:.0f}%"
                         + (f", zoom up to {max(zs):.3f}" if border.startswith('auto') else ""))
            if np.mean(inl) < 0.5:
                lines.append("  WARNING: less than half of the image follows one motion – the measurement may be "
                             "pulled by a large moving object; check the result.")
            lines += seam_txt
            rows.append({"title": f"window {wi}: {a}-{b}" + (f"  seams {', '.join(map(str, w['seams']))}"
                                                              if w["seams"] else ""),
                         "frames": fr, "seams": w["seams"], "curves": curves})

        # ---- all other frames: detail only, or taken over unchanged ----------
        rest = [f for f in keys if f not in done]

        def take(f):
            dst = _frame_path(P, f, pad, ext)
            g = gains_for(f)
            if g is None:
                S.link_or_copy(frames[f], dst, hard)
            else:
                fd = Frame(frames[f])
                fd.write_like(dst, S.apply_detail(fd.rgb, linear, g), fd.alpha)
            return dst

        with ThreadPoolExecutor(max_workers=min(8, M.worker_count())) as ex:
            for dst in ex.map(take, rest):
                written.append(dst)
                pbar.update(1)

        lines.append(f"done in {time.time() - t0:.1f} s – {len(written)} frames written")
        report = "\n".join(lines)
        extras = _finish(P, written, write_checksums, write_workflow, report, prompt, extra_pnginfo,
                         ".seam_report.txt")
        print(f"{LOG} Seam Stabilize: {P['seq_dir']} ({len(written)} frames, {len(wins)} windows) – "
              f"{', '.join(extras)}")

        ui = {}
        if rows:
            try:
                ui = {"images": [_temp_png(_curve_image(rows), "pertineo_seam")]}
            except Exception:
                traceback.print_exc()
        return {"ui": ui, "result": (P["seq_dir"], report)}


# --------------------------------------------------------------------------
# Overlap Blend
# --------------------------------------------------------------------------
class PertineoOverlapBlend:
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("folder", "report")
    OUTPUT_TOOLTIPS = ("The joined sequence folder – e.g. into Seam Stabilize or Load Image Sequence.",
                       "Which sections were joined where (also written as <name>.blend_report.txt).")
    FUNCTION = "run"
    DESCRIPTION = (
        "Joins sections that were saved as separate sequences with overlapping frame numbers (e.g. each generation "
        "run as its own version v001, v002 …) into one sequence, blending across the overlap instead of cutting. "
        "flow (morph) moves both takes towards each other along the optical flow while they are mixed, so nothing "
        "doubles. Frames covered by only one section are taken over unchanged.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folders": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "one sequence folder per line, in order (earliest first)",
                    "tooltip": "The sections, one folder (or any frame / name.####.exr) per line, in time order. "
                               "Frame numbers must be on the same timeline – the overlap is where two sections "
                               "have the same frame numbers. Same format and resolution in all sections."}),
                "blend": (BLEND_MODES, {
                    "tooltip": "flow (morph): both takes are warped towards each other along the optical flow and "
                               "mixed – no ghosting. cross-dissolve: plain mix. hard cut: the later section wins "
                               "(what the chains do today)."}),
                "blend_frames": ("INT", {"default": 0, "min": 0, "max": 1000,
                    "tooltip": "Length of the transition, centred in the overlap. 0 = the whole overlap. Before it "
                               "the earlier section is used, after it the later one."}),
            },
            "optional": {
                "output_name": ("PERTINEO_OUTPUT_NAME", {
                    "tooltip": "From Pertineo Output Name. Without it: right next to the first section's folder as "
                               "<name>_blend_v001/ (next free)."}),
                "curve": (CURVES, {"advanced": True,
                    "tooltip": "smoothstep = soft start and end of the transition; linear = even."}),
                "analysis_size": ("INT", {"advanced": True, "default": 1024, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "Flow is measured on a copy of this size (long side), applied at full resolution."}),
                "unchanged_frames": (UNCHANGED, {"advanced": True, "tooltip": _UNCHANGED_TIP}),
                "write_checksums": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Write <name>.sha256 with the SHA256 of every frame (sha256sum -c compatible)."}),
                "write_workflow": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Write <name>.workflow.json next to the new sequence."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, folders=None):
        if folders is None:
            return True
        lines = [ln for ln in folders.splitlines() if ln.strip()]
        if len(lines) < 2:
            return "Overlap Blend: enter at least two sequence folders, one per line."
        for i, ln in enumerate(lines, 1):
            err = check_sequence_folder(ln, f"Overlap Blend section {i}")
            if err:
                return err
        return True

    def run(self, folders, blend, blend_frames, output_name=None, curve=CURVES[0], analysis_size=1024,
            unchanged_frames=UNCHANGED[0], write_checksums=True, write_workflow=True, prompt=None,
            extra_pnginfo=None):
        S.need_cv2()
        t0 = time.time()
        srcs = [normalize_folder(ln) for ln in folders.splitlines() if ln.strip()]
        secs = []
        for s in srcs:
            fr, pad = index_frames(s)
            secs.append({"folder": s, "frames": fr, "pad": pad, "ext": os.path.splitext(fr[min(fr)])[1].lower()})
        exts = {s["ext"] for s in secs}
        if len(exts) > 1:
            raise RuntimeError(f"{LOG} Overlap Blend: all sections need the same file format, found {sorted(exts)}.")
        shapes = {Frame(s["frames"][min(s["frames"])]).rgb.shape for s in secs}
        if len(shapes) > 1:
            raise RuntimeError(f"{LOG} Overlap Blend: the sections have different resolutions {sorted(shapes)}.")
        ext, pad = secs[0]["ext"], max(s["pad"] for s in secs)
        linear = ext == ".exr"
        P, note = _plan(output_name, srcs[0], "blend")
        hard = unchanged_frames.startswith("hard")

        lines = [f"Overlap Blend – {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
                 f"output:  {P['seq_dir']}  ({note})", f"blend:   {blend}, curve {curve}"]
        for i, s in enumerate(secs, 1):
            k = sorted(s["frames"])
            lines.append(f"section {i}: {s['folder']}  ({k[0]}-{k[-1]})")

        # plan every frame: ("copy", sec) or ("mix", earlier, later, t)
        allf = sorted(set().union(*[s["frames"].keys() for s in secs]))
        plan = {}
        for f in allf:
            have = [i for i, s in enumerate(secs) if f in s["frames"]]
            if len(have) == 1:
                plan[f] = ("copy", have[0])
            elif len(have) > 2:
                raise RuntimeError(f"{LOG} Overlap Blend: frame {f} is in more than two sections "
                                   f"({', '.join(str(i + 1) for i in have)}) – shorten the sections.")
        for i in range(len(secs) - 1):
            j = i + 1
            ov = sorted(set(secs[i]["frames"]) & set(secs[j]["frames"]))
            if not ov:
                lines.append(f"sections {i + 1}/{j + 1}: no overlap – hard cut")
                continue
            if blend.startswith("hard"):
                region = []
            elif blend_frames <= 0 or blend_frames >= len(ov):
                region = ov
            else:
                start = (len(ov) - blend_frames) // 2
                region = ov[start:start + blend_frames]
            for f in ov:
                if f in region:
                    t = (region.index(f) + 1) / (len(region) + 1.0)
                    plan[f] = ("mix", i, j, S.blend_curve(t, curve))
                elif region and f < region[0]:
                    plan[f] = ("copy", i)
                else:
                    plan[f] = ("copy", j)
            lines.append(f"sections {i + 1}/{j + 1}: overlap {ov[0]}-{ov[-1]} ({len(ov)} frames), "
                         + (f"blend {region[0]}-{region[-1]} ({len(region)} frames)" if region else
                            "hard cut to the later section"))
        for f in allf:  # frames in two non-neighbouring sections
            if f not in plan:
                raise RuntimeError(f"{LOG} Overlap Blend: frame {f} is in two sections that are not neighbours in "
                                   "the list – put the folders in time order.")

        pbar = comfy.utils.ProgressBar(len(allf))

        def job(f):
            dst = _frame_path(P, f, pad, ext)
            step = plan[f]
            if step[0] == "copy":
                S.link_or_copy(secs[step[1]]["frames"][f], dst, hard)
                return dst
            _, i, j, t = step
            fa, fb = Frame(secs[i]["frames"][f]), Frame(secs[j]["frames"][f])
            a_rgb, a_al, b_rgb, b_al = fa.rgb, fa.alpha, fb.rgb, fb.alpha
            if blend.startswith("flow"):
                rgb, al = S.flow_blend(a_rgb, b_rgb, t, analysis_size, linear, a_al, b_al)
            else:
                rgb = (1 - t) * a_rgb + t * b_rgb
                al = None
                if a_al is not None or b_al is not None:
                    one = np.ones(a_rgb.shape[:2], np.float32)
                    al = (1 - t) * (a_al if a_al is not None else one) + t * (b_al if b_al is not None else one)
            fa.write_like(dst, rgb, al)
            return dst

        written = []
        with ThreadPoolExecutor(max_workers=min(4, M.worker_count())) as ex:
            for dst in ex.map(job, allf):
                written.append(dst)
                pbar.update(1)
        n_mix = sum(1 for v in plan.values() if v[0] == "mix")
        lines.append(f"done in {time.time() - t0:.1f} s – {len(written)} frames, {n_mix} blended")
        report = "\n".join(lines)
        extras = _finish(P, written, write_checksums, write_workflow, report, prompt, extra_pnginfo,
                         ".blend_report.txt")
        print(f"{LOG} Overlap Blend: {P['seq_dir']} ({len(written)} frames, {n_mix} blended) – {', '.join(extras)}")
        return {"ui": {}, "result": (P["seq_dir"], report)}
