# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – Save Media (sequence + mask + video in one go, versioned), Output Name, Load Image."""

import datetime
import json
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.utils
import folder_paths

from ..common import LOG
from . import media as M
from .readwrite import clean_path, is_exr, linear_to_srgb, read_rgba_raw

CATEGORY = "Pertineo/IO"


def _np(t):
    if t is None:
        return None
    a = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    return a.astype(np.float32, copy=False)


def _preview(arr, max_size=768):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = M.as_channels(a)[..., :3]
    pil = Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8))
    pil.thumbnail((max_size, max_size))
    name = f"pertineo_media_preview_{uuid.uuid4().hex[:10]}.png"
    pil.save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    return {"filename": name, "subfolder": "", "type": "temp"}


PREVIEW_MODES = ["video", "still frame", "off"]

_VERSION_TIP = ("next free = one more than the highest version already in the output folder (and its date folders) "
                "for this name – counting every output: image or mask sequence, video, audit file. So all outputs "
                "of one run share the same number, even if only a matte exists of the previous version. Every "
                "queue writes a new version. fixed = exactly the number in 'version'. off = no _v### in the name.")


# --------------------------------------------------------------------------
# Save Media
# --------------------------------------------------------------------------
class PertineoSaveMedia:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("base_name", "version", "folder", "saved_paths")
    OUTPUT_TOOLTIPS = ("Full name of this output, e.g. shot010_comp_v003.",
                       "The version of this run (v003).",
                       "Folder that holds the video, the sequence folders and the audit files (incl. date folder).",
                       "All written files and folders, one per line.")
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Saves an image sequence, a mask sequence and videos in one go – all with the same name and "
                   "version: <name>_v003.mov, folder <name>_v003/ with <name>_v003.1001.png next to it, the mask "
                   "as <name>_v003_mask/ (and _mask.mov). Folders are created automatically. PNG / TIFF / EXR / JPG "
                   "in 8, 16 or 32 bit, written in parallel on several cores. Video through ffmpeg (H.264 preview, "
                   "H.265 10 bit, ProRes 422 HQ / 4444). SHA256 list of all files, one audit report for the whole "
                   "output (connect Pertineo Collect Metadata). Folder, name, version and date folder come from "
                   "Pertineo Output Name (with a preview of the names); without it: <output>/render_v###.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seq_write": ("BOOLEAN", {"default": True,
                    "tooltip": "Write the images as a numbered sequence into the folder <name>_v###/."}),
                "seq_format": (M.SEQ_FORMATS, {
                    "tooltip": "PNG 16 bit = full quality for sRGB images (default). EXR half / float = linear, "
                               "keeps values above 1 (HDR) unless seq_clamp is on. TIFF 32 bit float = float TIFF. "
                               "JPG = small previews only."}),
                "seq_compression": (M.COMPRESSION, {"advanced": True,
                    "tooltip": "Lossless compression of PNG / TIFF / EXR (image and mask sequence). balanced = fast and small enough "
                               "(PNG level 3, TIFF LZW, EXR ZIP). fast = bigger files (PNG 1, TIFF none, EXR ZIPS). "
                               "small = slowest (PNG 9, TIFF Deflate, EXR PIZ). For JPG it sets the quality."}),
                "seq_start_frame": ("INT", {"default": 1001, "min": 0, "max": 10_000_000,
                    "tooltip": "Frame number of the first image, for the image and the mask sequence (match "
                               "your plate numbering, e.g. first_frame from Load Image Sequence)."}),
                "seq_padding": ("INT", {"advanced": True, "default": 4, "min": 1, "max": 8,
                    "tooltip": "Digits of the frame number (4 = 1001)."}),
                "seq_exr_to_linear": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "EXR is scene-linear by convention: on = convert the sRGB IMAGE values to linear "
                               "(like Save Image + Metadata). Off = write the values as they are. Never applied to "
                               "masks or other formats."}),
                "seq_clamp": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Clip float formats (EXR, TIFF 32 bit) to 0–1. Off = values above 1 and below 0 are "
                               "kept (HDR). Integer formats (PNG, TIFF 8/16, JPG) and video are always 0–1."}),
                "seq_alpha_channel": (M.ALPHA_CHANNEL, {
                    "tooltip": "RGB or RGBA for the image sequence and a ProRes 4444 video. from mask (default) = the "
                               "connected mask becomes the alpha (white = opaque) – no Join Image with Alpha needed "
                               "(which inverts the mask); without a mask connected it works like 'from image'. "
                               "from image = RGBA images keep their alpha, RGB stays RGB. none = "
                               "always RGB, an alpha is dropped. JPG, H.264 / H.265 and ProRes 422 never have alpha. "
                               "Independent of mask_write / video_of, which write the mask as separate files."}),
                "seq_input_premultiplied": ("BOOLEAN", {"advanced": True, "default": False,
                    "tooltip": "On when the incoming RGB is already multiplied by its alpha (e.g. some background "
                               "removal settings). It is un-premultiplied first, so nothing is premultiplied twice. "
                               "Off for most ComfyUI nodes (Join Image with Alpha gives straight RGB)."}),
                "seq_alpha": (M.ALPHA_MODES, {
                    "tooltip": "Image sequence with alpha (see seq_alpha_channel): straight (default) = the full "
                               "RGB image plus a separate alpha – premultiply in the comp (in Nuke: Premult after the "
                               "Read). premultiplied = RGB x alpha (EXR in linear light). auto = EXR premultiplied "
                               "(the OpenEXR convention, as renderers write it), PNG / TIFF straight."}),
                "mask_write": ("BOOLEAN", {"default": True,
                    "tooltip": "Write the mask (if connected) as a sequence into <name>_v###_mask/."}),
                "mask_format": (M.SEQ_FORMATS, {
                    "tooltip": "Format of the mask sequence. Masks are written as stored (no color conversion); "
                               "EXR puts the matte into R, G, B and A, like Save Matte Sequence."}),
                "video_write": ("BOOLEAN", {"default": False,
                    "tooltip": "Also write a video next to the sequence folder (needs ffmpeg)."}),
                "video_of": (M.VIDEO_OF, {
                    "tooltip": "Which inputs become a video: the images (<name>_v###.mov/.mp4), the mask as a grey "
                               "video (<name>_v###_mask.mov/.mp4), or both as two separate videos. This is NOT the "
                               "alpha channel: with 'image' and ProRes 4444 the mask is still used – as the alpha of "
                               "the image video, when seq_alpha_channel is 'from mask'. H.264 / H.265 / ProRes 422 have "
                               "no alpha, there 'image' uses only the images."}),
                "video_quality": (M.VIDEO_QUALITY, {
                    "tooltip": "preview = H.264 8 bit, small and fast (NVENC on NVIDIA GPUs, otherwise x264). "
                               "review = H.265 10 bit. ProRes 422 HQ = 10 bit for editing / delivery. ProRes 4444 "
                               "= 10 bit 4:4:4 with the alpha channel (see seq_alpha_channel; without alpha it is written "
                               "as RGB – players like Nuke then show a white alpha). ffmpeg's ProRes encoder takes "
                               "10-bit input. Video is always 0–1 (sRGB, Rec.709 tags)."}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate of the video (23.976, 24, 25 …). Also stored in the audit report."}),
                "video_alpha": (M.VIDEO_ALPHA_MODES, {
                    "tooltip": "ProRes 4444 with alpha: straight = RGB as it is, premultiplied = RGB x alpha (shows "
                               "the cut-out over black in players that ignore the alpha). Videos without alpha are "
                               "not affected."}),
                "video_audio": (M.VIDEO_AUDIO, {
                    "tooltip": "Sound track of the image video. only if connected = the audio input, no track without "
                               "it. always = with nothing connected, a silent stereo track (48 kHz, exact video "
                               "length) is written – a base track for editing, as many video tools expect one. .mov: "
                               "24-bit PCM, .mp4: AAC (note: DaVinci Resolve on Linux does not play AAC – use .mov)."}),
                "if_exists": (M.IF_EXISTS, {"advanced": True,
                    "tooltip": "What to do when the output already exists (only possible with version fixed / off "
                               "or Output Name set to fixed / off): stop with an error (safe) or overwrite."}),
                "meta_embed_workflow": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Embed workflow and prompt in the video's metadata (mov / mp4, credentials redacted), "
                               "the way Video Helper Suite does – the video then carries its own recipe. Turn off for "
                               "anonymised sharing: the workflow contains local paths typed into nodes."}),
                "meta_write_workflow": ("BOOLEAN", {"default": True,
                    "tooltip": "Write <name>_v###.workflow.json next to the outputs (opens directly in ComfyUI, "
                               "credentials redacted). The workflow contains everything typed into the nodes, incl. "
                               "local paths. Frames do not get the workflow embedded."}),
                "meta_json": ("BOOLEAN", {"default": False,
                    "tooltip": "With metadata connected: write <name>_v###.audit.json (machine-readable, with the "
                               "workflow and the SHA256 list – needed for the audit chain)."}),
                "meta_txt": ("BOOLEAN", {"default": True,
                    "tooltip": "With metadata connected: write <name>_v###.audit.txt – one report for the whole "
                               "output, not one per frame."}),
                "meta_pdf": ("BOOLEAN", {"default": False,
                    "tooltip": "With metadata connected: write <name>_v###.audit.pdf with a thumbnail of the middle "
                               "frame."}),
                "meta_checksums": ("BOOLEAN", {"advanced": True, "default": True,
                    "tooltip": "Write <name>_v###.sha256 with the SHA256 of every frame and video (sha256sum "
                               "format – check later with: sha256sum -c <file>). Also added to the audit .json."}),
                "preview": (PREVIEW_MODES, {
                    "tooltip": "What the node shows after the run. video = a player with a small H.264 copy (max. 540 "
                               "px high, with audio; only in ComfyUI's temp folder, deleted on restart – your outputs "
                               "are untouched). still frame = the middle frame. off = nothing (fastest)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Image batch (the frames of the sequence / video)."}),
                "mask": ("MASK", {"tooltip": "Mask / matte batch."}),
                "audio": ("AUDIO", {
                    "tooltip": "Sound for the image video (e.g. the audio output of Pertineo Load Video). Cut or padded "
                               "with silence to the video length; starts with the first frame. .mov: 24-bit PCM "
                               "(lossless), .mp4: AAC 320 kbit/s. Not added to the mask video or the sequences."}),
                "output_name": ("PERTINEO_OUTPUT_NAME", {
                    "tooltip": "From Pertineo Output Name: folder, name, version and date folder are set there, with a "
                               "preview of the file names before rendering. Connect one Output Name to several Save "
                               "Media so all write the same name and version. Not connected = ComfyUI output folder, "
                               "name 'render', next free version (render_v001, render_v002 …)."}),
                "metadata_text": ("STRING", {"forceInput": True,
                    "tooltip": "metadata_text from Pertineo Collect Metadata."}),
                "metadata_json": ("STRING", {"forceInput": True,
                    "tooltip": "metadata_json from Pertineo Collect Metadata."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, output_name=None, **kwargs):
        # without Output Name: next free writes a new version on every queue (otherwise ComfyUI would skip an
        # unchanged run); with Output Name connected, that node decides
        if output_name is None:
            return float("nan")
        return ""

    def run(self, seq_write, seq_format, seq_compression, seq_start_frame, seq_padding, seq_exr_to_linear,
            seq_clamp, seq_alpha_channel, seq_input_premultiplied, seq_alpha,
            mask_write, mask_format,
            video_write, video_of, video_quality, video_fps, video_alpha, video_audio,
            if_exists, meta_embed_workflow, meta_write_workflow, meta_json, meta_txt, meta_pdf, meta_checksums,
            preview="video", images=None, mask=None, audio=None, output_name=None, metadata_text=None,
            metadata_json=None, prompt=None, extra_pnginfo=None):
        # widget names carry their group (seq_ / mask_ / video_ / meta_); the code below uses the plain names
        write_sequence, sequence_format, compression = seq_write, seq_format, seq_compression
        start_frame, frame_padding, exr_to_linear, clamp = seq_start_frame, seq_padding, seq_exr_to_linear, seq_clamp
        alpha_channel, input_premultiplied, sequence_alpha = seq_alpha_channel, seq_input_premultiplied, seq_alpha
        write_mask, write_video, fps = mask_write, video_write, video_fps
        embed_workflow, write_workflow, write_json = meta_embed_workflow, meta_write_workflow, meta_json
        write_txt, write_pdf, write_checksums = meta_txt, meta_pdf, meta_checksums
        t0 = time.time()
        imgs = _np(images)
        masks = _np(mask)
        if masks is not None and masks.ndim == 2:
            masks = masks[None]
        if imgs is not None and imgs.ndim == 3:
            imgs = imgs[None]

        do_seq = bool(write_sequence and imgs is not None)
        do_mask = bool(write_mask and masks is not None)
        vid_img = bool(write_video and imgs is not None and video_of in ("image", M.VIDEO_OF[2]))
        vid_mask = bool(write_video and masks is not None and video_of in ("mask", M.VIDEO_OF[2]))
        if not (do_seq or do_mask or vid_img or vid_mask):
            raise ValueError(f"{LOG} Save Media: nothing to write – connect images and / or mask and turn on "
                             f"seq_write, mask_write or video_write.")
        if imgs is not None and masks is not None and len(imgs) != len(masks):
            print(f"{LOG} Save Media: {len(imgs)} images but {len(masks)} masks – each is written with its own length.")

        if isinstance(output_name, dict):
            root, full, date_folder = output_name["root"], output_name["full"], output_name["date_folder"]
            ver, ver_note = output_name["version"], f"from Output Name ({output_name['note']})"
            P = M.plan_paths(root, date_folder, full, now=output_name.get("now"))
        else:  # no Output Name: ComfyUI output folder, 'render', next free version
            root = os.path.normpath(folder_paths.get_output_directory())
            full, ver, _n, ver_note = M.resolve_version("next free", 1, root, M.DEFAULT_NAME, M.VERSION_IN_NAME[1])
            ver_note += " – no Output Name connected"
            P = M.plan_paths(root, "off", full)
        folder, base, mbase = P["folder"], P["base"], P["mask_base"]
        seq_ext, mask_ext, vext = M.ext_for(sequence_format), M.ext_for(mask_format), M.video_ext(video_quality)
        vid_path = os.path.join(folder, base + vext)
        vid_mask_path = os.path.join(folder, mbase + vext)
        has_meta = bool(metadata_text or metadata_json)
        # companion files (.sha256, .workflow.json, .audit.*) are named after the main output: the image name, or
        # the mask name when this node writes only the mask – so a second Save Media for the mask can share one
        # Output Name without clashing with the first
        xbase = base if (do_seq or vid_img) else mbase

        targets = []
        if do_seq:
            targets.append(P["seq_dir"])
        if do_mask:
            targets.append(P["mask_dir"])
        if vid_img:
            targets.append(vid_path)
        if vid_mask:
            targets.append(vid_mask_path)
        if write_checksums:
            targets.append(os.path.join(folder, xbase + ".sha256"))
        if write_workflow:
            targets.append(os.path.join(folder, xbase + ".workflow.json"))
        if has_meta:
            targets += [os.path.join(folder, xbase + e) for e, on in
                        ((".audit.txt", write_txt), (".audit.json", write_json), (".audit.pdf", write_pdf)) if on]
        existing = [t for t in targets if os.path.isfile(t) or (os.path.isdir(t) and os.listdir(t))]
        overwrite = if_exists == "overwrite"
        if existing and not overwrite:
            raise ValueError(f"{LOG} Save Media: {base} already exists ({', '.join(os.path.basename(e) for e in existing)}"
                             f" in {folder}). Use version_mode 'next free', another version, or if_exists 'overwrite'.")
        print(f"{LOG} Save Media: {base} ({ver_note}) -> {folder}")
        os.makedirs(folder, exist_ok=True)

        # ---- frames (parallel) --------------------------------------------
        jobs = []

        def add_jobs(arr, out_dir, b, fmt, ext, is_mask):
            os.makedirs(out_dir, exist_ok=True)
            for i in range(len(arr)):
                path = os.path.join(out_dir, f"{b}.{start_frame + i:0{frame_padding}d}{ext}")
                jobs.append((path, lambda p=path, src=arr, k=i: M.write_frame(
                    p, src[k], fmt, compression, clamp, exr_to_linear, is_mask, sequence_alpha, input_premultiplied)))

        alpha_from_mask = str(alpha_channel).startswith("from mask")
        src_imgs = M.FrameSource(imgs, masks if alpha_from_mask else None,
                                 rgb_only=str(alpha_channel).startswith("none")) if imgs is not None else None
        if do_seq:
            add_jobs(src_imgs, P["seq_dir"], base, sequence_format, seq_ext, False)
        if do_mask:
            add_jobs(masks, P["mask_dir"], mbase, mask_format, mask_ext, True)

        # ---- videos run next to the frame writing ------------------------
        video_jobs = []
        if vid_img:
            video_jobs.append(("video", vid_path, src_imgs, False))
        if vid_mask:
            video_jobs.append(("mask video", vid_mask_path, masks, True))
        encoders = {}
        # the workflow (credentials redacted) – embedded in the videos and written as .workflow.json
        wf = None
        if write_workflow or has_meta or (embed_workflow and video_jobs):
            try:
                from ..audit import sidecar as S
                wf = S.workflow_record(prompt, extra_pnginfo)
            except Exception as e:
                print(f"{LOG} Save Media: workflow not available ({type(e).__name__}: {e})")
        video_meta = None
        if embed_workflow and video_jobs and wf:
            video_meta = {"comment": json.dumps({k: wf[k] for k in ("prompt", "workflow") if k in wf},
                                                ensure_ascii=False, separators=(",", ":"), default=str)}
            for k in ("prompt", "workflow"):
                if k in wf:
                    video_meta[k] = json.dumps(wf[k], ensure_ascii=False, separators=(",", ":"), default=str)

        if audio is None and vid_img and str(video_audio).startswith("always"):
            audio = {"waveform": np.zeros((1, 2, 1), dtype=np.float32), "sample_rate": 48000}  # padded to length
        sound = M.audio_for_video(audio, len(imgs), fps) if (vid_img and audio is not None) else None
        if audio is not None and not vid_img:
            print(f"{LOG} Save Media: audio is connected but no image video is written – audio not used.")

        def encode_all():
            for label, path, arr, is_mask in video_jobs:
                encoders[path] = M.write_video(path, arr, fps, video_quality, is_mask=is_mask, overwrite=overwrite,
                                               metadata=video_meta, audio=None if is_mask else sound,
                                               alpha_mode=video_alpha, input_premult=input_premultiplied)

        pbar = comfy.utils.ProgressBar(max(1, len(jobs)))
        with ThreadPoolExecutor(max_workers=1) as vex:
            vfut = vex.submit(encode_all) if video_jobs else None
            digests = M.write_sequence_parallel(jobs, hash_files=write_checksums,
                                                progress=lambda d, t: pbar.update_absolute(d, t))
            if vfut is not None:
                vfut.result()
        if write_checksums:
            for _l, path, _a, _m in video_jobs:
                digests[path] = M.sha256_file(path)
        written = [d for d in (P["seq_dir"] if do_seq else None, P["mask_dir"] if do_mask else None) if d]
        written += [p for _l, p, _a, _m in video_jobs]

        # ---- warn about old frames left over by an overwrite --------------
        if overwrite:
            for out_dir, b, n in ((P["seq_dir"], base, len(imgs) if do_seq else 0),
                                  (P["mask_dir"], mbase, len(masks) if do_mask else 0)):
                if n and os.path.isdir(out_dir) and len(os.listdir(out_dir)) > n:
                    print(f"{LOG} Save Media: {out_dir} holds more files than were written now – older frames "
                          f"were not deleted.")

        # ---- SHA256 list ----------------------------------------------------
        checksum_name = None
        if write_checksums:
            cpath = os.path.join(folder, xbase + ".sha256")
            count = M.write_checksums(cpath, digests, folder)
            checksum_name = f"{os.path.basename(cpath)} – SHA256 of {count} files (check: sha256sum -c " \
                            f"{os.path.basename(cpath)})"
            written.append(cpath)

        # ---- workflow + audit ---------------------------------------------
        wf_rec = None
        if write_workflow or has_meta:
            try:
                from ..audit import sidecar as S
                if write_workflow:
                    wp, wf_rec = S.write_workflow_file(folder, xbase, wf, overwrite)
                    if wp:
                        written.append(wp)
                else:
                    wf_rec = wf
                if has_meta:
                    meta = S.parse_meta(metadata_json)
                    first = imgs if imgs is not None else masks
                    h, w = first.shape[1], first.shape[2]
                    outputs = []
                    last = lambda n: start_frame + n - 1
                    if do_seq:
                        outputs.append({"kind": "image sequence", "path": f"{base}/{base}.{'#' * frame_padding}{seq_ext}",
                                        "format": sequence_format, "frames": len(imgs),
                                        "text": f"image sequence {base}/{base}.{'#' * frame_padding}{seq_ext} · "
                                                f"{sequence_format} · {len(imgs)} frames ({start_frame}-{last(len(imgs))})"})
                    if do_mask:
                        outputs.append({"kind": "mask sequence", "path": f"{mbase}/{mbase}.{'#' * frame_padding}{mask_ext}",
                                        "format": mask_format, "frames": len(masks),
                                        "text": f"mask sequence {mbase}/{mbase}.{'#' * frame_padding}{mask_ext} · "
                                                f"{mask_format} · {len(masks)} frames ({start_frame}-{last(len(masks))})"})
                    for label, path, arr, _m in video_jobs:
                        outputs.append({"kind": label, "path": os.path.basename(path), "format": video_quality,
                                        "frames": len(arr), "fps": fps, "encoder": encoders.get(path),
                                        "sha256": digests.get(path),
                                        "audio": (None if _m or sound is None else
                                                  {"sample_rate": sound[1], "channels": int(sound[0].shape[0])}),
                                        "text": f"{label} {os.path.basename(path)} · {video_quality} · {len(arr)} frames "
                                                f"@ {fps:g} fps · {encoders.get(path)}"
                                                + ("" if _m or sound is None else
                                                   f" · audio {sound[1]} Hz {sound[0].shape[0]} ch")})
                    file_info = {
                        "name": xbase, "basename": xbase, "batch_index": 0, "batch_size": 1, "width": w, "height": h,
                        "format": ", ".join(o["kind"] for o in outputs), "folder": S.display_folder(folder, meta),
                        "saved": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                        "version": ver, "first_frame": start_frame, "fps": fps, "outputs": outputs,
                    }
                    if checksum_name:
                        file_info["checksums"] = checksum_name
                        file_info["checksum_list"] = {os.path.relpath(p, folder).replace(os.sep, "/"): d
                                                      for p, d in sorted(digests.items()) if d}
                    files = [p for p in digests if digests.get(p)]
                    if len(files) == 1:  # a single file: its hash is the link to the next workflow
                        file_info["sha256"] = digests[files[0]]
                    thumb = first[len(first) // 2]
                    written += S.write_audit(folder, xbase, file_info, meta, metadata_text, thumb, write_txt,
                                             write_json, write_pdf, workflow=wf_rec, overwrite=overwrite)
            except Exception as e:
                traceback.print_exc()
                print(f"{LOG} Save Media: audit / workflow files could not be written ({type(e).__name__}: {e})")

        n_frames = len(jobs)
        print(f"{LOG} Save Media: wrote {n_frames} frame(s)" + (f" and {len(video_jobs)} video(s)" if video_jobs else "")
              + f" in {time.time() - t0:.1f} s")
        for p in written:
            print(f"{LOG}   {p}")
        prev_src = imgs if imgs is not None else masks
        ui = {}
        if preview == "video" and len(prev_src) > 1:
            try:
                name = f"pertineo_media_preview_{uuid.uuid4().hex[:10]}.mp4"
                M.write_preview_video(os.path.join(folder_paths.get_temp_directory(), name), prev_src, fps,
                                      audio=(sound if sound is not None else M.audio_for_video(audio, len(imgs), fps))
                                      if (imgs is not None and audio is not None) else None, is_mask=imgs is None)
                # same form as ComfyUI's own Save Video: the frontend shows a player for an animated mp4
                ui = {"images": [{"filename": name, "subfolder": "", "type": "temp", "format": "video/h264-mp4"}],
                      "animated": (True,)}
            except Exception as e:  # the preview must never fail the save
                print(f"{LOG} Save Media: preview video failed ({type(e).__name__}: {e}) – showing a still frame.")
        if not ui and preview != "off":
            ui = {"images": [_preview(prev_src[len(prev_src) // 2])]}
        return {"ui": ui, "result": (base, ver, folder, "\n".join(written))}


# --------------------------------------------------------------------------
# Output Name – folder, name and version in one place, with a preview of the file names
# --------------------------------------------------------------------------
def preview_text(P, root, note):
    b, m = P["base"], P["mask_base"]
    return "\n".join([
        f"folder:          {P['folder']}",
        f"name:            {b}",
        f"version:         {note}",
        f"video:           {b}.mov / .mp4",
        f"image sequence:  {b}/{b}.####.<ext>",
        f"mask sequence:   {m}/{m}.####.<ext>",
        f"mask video:      {m}.mov / .mp4",
        f"audit:           {b}.audit.txt / .json / .pdf",
        f"extras:          {b}.sha256 · .workflow.json",
        "(#### = frame number · <ext> and .mov / .mp4 = format chosen in Save Media)",
    ])


_FONT = None


_FONTS = {}


def _font(size=15):
    """Monospace font in the given size (cached per size)."""
    if size in _FONTS:
        return _FONTS[size]
    global _FONT
    _FONT = None
    if _FONT is None:
        from PIL import ImageFont
        for name in ("DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                     "LiberationMono-Regular.ttf", "Menlo.ttc", "consola.ttf", "cour.ttf"):
            try:
                _FONT = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _FONT = ImageFont.load_default()
    _FONTS[size] = _FONT
    return _FONT


def _text_image(text, max_chars=84):
    """Renders the preview text as a small PNG for the node itself – ComfyUI shows images on any node without a
    frontend extension (text output is only shown by special nodes such as Preview Any)."""
    from PIL import ImageDraw
    lines = []
    for line in text.split("\n"):
        while len(line) > max_chars:  # long paths: wrap, keep the label column
            cut = line.rfind("/", 0, max_chars)
            cut = cut + 1 if cut > 20 else max_chars
            lines.append(line[:cut])
            line = " " * 17 + line[cut:]
        lines.append(line)
    font = _font()
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    box = [probe.textbbox((0, 0), ln or " ", font=font) for ln in lines]
    lh = max(b[3] - b[1] for b in box) + 6
    w = max(b[2] for b in box) + 28
    img = Image.new("RGB", (max(w, 320), lh * len(lines) + 24), (30, 30, 30))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        color = (150, 150, 150) if ln.startswith("(") else (225, 225, 225)
        d.text((14, 12 + i * lh), ln, font=font, fill=color)
    name = f"pertineo_output_name_{uuid.uuid4().hex[:10]}.png"
    img.save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    return {"filename": name, "subfolder": "", "type": "temp"}


class PertineoOutputName:
    CATEGORY = CATEGORY
    OUTPUT_NODE = True  # can be queued on its own (right-click > Queue Selected Output Nodes) to check the names
    RETURN_TYPES = ("PERTINEO_OUTPUT_NAME", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("output_name", "preview", "base_name", "version", "folder")
    OUTPUT_TOOLTIPS = ("Connect to output_name of Save Media (one or several – all get the same name and version).",
                       "The file names Save Media will write, as text (they are also shown on the node itself).",
                       "Name with version, e.g. shot010_comp_v003.", "The version (v003), empty with mode off.",
                       "Target folder incl. date folder.")
    FUNCTION = "run"
    DESCRIPTION = ("Sets folder, name and version of the outputs in one place and shows the resulting file names on "
                   "the node. To check them before rendering, select this node and use right-click > Queue Selected "
                   "Output Nodes – only this node (and a loader feeding 'name') runs. Connect output_name to Save "
                   "Media (also to several – they then share name and version).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "OUTPUT: empty = ComfyUI output folder; relative = inside it; absolute allowed",
                    "tooltip": "Target folder. Empty = ComfyUI output folder, a relative path is created inside it, "
                               "absolute paths are allowed."}),
                "filename": ("STRING", {"default": "{name}", "placeholder": "{name}  (empty = the linked name)",
                    "tooltip": "Name of the outputs. {name} is replaced by the linked 'name' input (e.g. from Load "
                               "Image Sequence / Load Video / Load Image) – '{name}_comp' -> shot010_plate_comp. "
                               "Without a linked name, {name} becomes 'render'. Empty = {name}. The result is made "
                               "pipeline-safe: "
                               "ä -> ae, ß -> ss, accents dropped, spaces and other characters -> '_' "
                               "('clip 002 (1)' -> 'clip_002_1')."}),
                "suffix": ("STRING", {"default": "", "placeholder": "suffix, e.g. upscaled, edit, 48fps",
                    "tooltip": "Added to the end of the name with '_' (upscaled -> shot010_plate_upscaled). Empty = "
                               "nothing. With a version already in the name it stays where it is "
                               "(shot010_v02_upscaled); otherwise _v### follows the suffix (shot010_upscaled_v001). "
                               "With version_mode off: no version at all."}),
                "version_mode": (M.VERSION_MODES, {"tooltip": _VERSION_TIP}),
                "version": ("INT", {"advanced": True, "default": 1, "min": 0, "max": 9999,
                    "tooltip": "fixed: the version to write (1 = v001). next free: the lowest version to start from."}),
                "date_folder": (M.DATE_FOLDERS, {
                    "tooltip": "Put everything into a folder of today's date inside output_folder (e.g. 2026-11-04/)."}),
                "version_in_name": (M.VERSION_IN_NAME, {"advanced": True,
                    "tooltip": "replace existing: if the name already has a version (sh010_comp_v01, .v003, -V2), "
                               "exactly that one is counted up – digits and spelling are kept (v01 -> v02); with "
                               "several, the last one. next free then goes above the version in the name and above "
                               "everything in the folder. Without a version in the name, _v001 is appended. "
                               "append: always add _v### at the end."}),
            },
            "optional": {
                "name": ("STRING", {"forceInput": True, "tooltip": "Replaces {name} in filename."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, version_mode=None, **kwargs):
        # next free looks at the folder again on every queue
        return float("nan") if version_mode == "next free" else ""

    def run(self, output_folder, filename, suffix, version_mode, version, date_folder,
            version_in_name=M.VERSION_IN_NAME[0], name=None):
        root = M.output_root(output_folder, folder_paths.get_output_directory())
        base0 = M.base_name(filename, name)
        sfx = M.clean_name(suffix or "")
        if sfx:
            base0 = f"{base0}_{sfx}"
        full, ver, n, note = M.resolve_version(version_mode, version, root, base0, version_in_name)
        now = datetime.datetime.now()  # the same date for preview and save, even across midnight
        P = M.plan_paths(root, date_folder, full, now=now)
        text = preview_text(P, root, f"{ver or '–'} – {note}")
        print(f"{LOG} Output Name: {P['base']} -> {P['folder']} ({note})")
        data = {"root": root, "full": full, "date_folder": date_folder, "version": ver, "note": note, "now": now}
        try:
            ui = {"images": [_text_image(text)]}
        except Exception as e:  # the preview must never stop a render
            print(f"{LOG} Output Name: preview image failed ({type(e).__name__}: {e})")
            ui = {}
        return {"ui": ui, "result": (data, text, P["base"], ver, P["folder"])}


# --------------------------------------------------------------------------
# Load Image (full precision, with name output)
# --------------------------------------------------------------------------
_RAW_EXTS = (".png", ".tif", ".tiff", ".exr")
_LOAD_EXTS = _RAW_EXTS + (".jpg", ".jpeg", ".webp", ".bmp")
LOAD_MASKS = ["alpha inverted (like Load Image)", "alpha", "red", "green", "blue", "luminance"]


class PertineoLoadImage:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "folder", "folder_root", "name", "width", "height")
    OUTPUT_TOOLTIPS = ("The image – 16-bit PNG / TIFF and EXR in full precision (the core Load Image reduces them "
                       "to 8 bit).", "Mask from mask_channel (empty when the channel does not exist).",
                       "Folder the image is in.",
                       "Folder for Output Name (output_folder): the image's folder, or the one above it when the "
                       "image sits in a date folder like 2026-11-06 – so the next version is written next to this "
                       "one. Most useful with 'path' (an image outside the input folder, e.g. a render); an image "
                       "from ComfyUI's input folder gives the ComfyUI output folder instead.",
                       "File name without folder and extension – connect it to Save Media (name).",
                       "Width in pixels.", "Height in pixels.")
    FUNCTION = "run"
    DESCRIPTION = ("Loads one image from the ComfyUI input folder (upload works like Load Image) or from any 'path' in "
                   "full precision: PNG / TIFF 8 or 16 bit and EXR, plus JPG / WebP. Outputs the file name and "
                   "folder for Output Name.")

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        try:
            files = sorted(f for f in os.listdir(input_dir)
                           if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(_LOAD_EXTS))
        except OSError:
            files = []
        return {
            "required": {
                "image": (files, {"image_upload": True,
                    "tooltip": "Image in the ComfyUI input folder (or upload one)."}),
                "mask_channel": (LOAD_MASKS, {
                    "tooltip": "Mask output: 'alpha inverted' matches the core Load Image (transparent = 1, for "
                               "inpainting). 'alpha' = the matte as stored (opaque = 1). red / green / blue / "
                               "luminance read a matte rendered as grey RGB."}),
                "exr_to_srgb": ("BOOLEAN", {"default": True,
                    "tooltip": "EXR files are scene-linear: on = convert the image to sRGB (what most ComfyUI nodes "
                               "expect). Only affects .exr files, never the mask. If an EXR looks flat or washed out with this on, it was saved with sRGB values (some ComfyUI savers do that) – turn it off."}),
            },
            "optional": {
                "path": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "optional: full path to an image outside the input folder",
                    "tooltip": "Full path to an image anywhere on disk (e.g. a render on the NAS) – paste it with "
                               "Ctrl+C in the file manager, Ctrl+V here. Overrides 'image'."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, image=None, path=None):
        p = clean_path(path or "")
        if p:
            return True if os.path.isfile(p) else f"Image not found: {p}"
        if image is not None and not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True

    @classmethod
    def IS_CHANGED(cls, image=None, path=None, **kwargs):
        try:
            st = os.stat(clean_path(path or "") or folder_paths.get_annotated_filepath(image))
            return f"{st.st_size}:{st.st_mtime_ns}"
        except Exception:
            return float("nan")

    def run(self, image, mask_channel, exr_to_srgb, path=""):
        path = clean_path(path or "") or folder_paths.get_annotated_filepath(image)
        if path.lower().endswith(_RAW_EXTS):
            rgb, alpha = read_rgba_raw(path)
        else:
            pil = ImageOps.exif_transpose(Image.open(path))
            arr = np.asarray(pil.convert("RGBA" if "A" in pil.getbands() else "RGB"), dtype=np.float32) / 255.0
            rgb, alpha = arr[..., :3], (arr[..., 3] if arr.shape[-1] == 4 else None)
        h, w = rgb.shape[:2]
        if mask_channel.startswith("alpha"):
            m = np.zeros((h, w), np.float32) if alpha is None else np.clip(alpha, 0, 1)
            if alpha is not None and "inverted" in mask_channel:
                m = 1.0 - m
            if alpha is None:
                print(f"{LOG} Load Image: {os.path.basename(path)} has no alpha channel – mask is empty.")
        elif mask_channel == "luminance":
            m = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        else:
            m = rgb[..., {"red": 0, "green": 1, "blue": 2}[mask_channel]]
        img = linear_to_srgb(rgb) if (exr_to_srgb and is_exr(path)) else np.clip(rgb, 0, 1)
        name = os.path.splitext(os.path.basename(path))[0]
        folder = os.path.dirname(os.path.abspath(path))
        return (torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None], folder, M.folder_root_for(folder),
                name, w, h)
