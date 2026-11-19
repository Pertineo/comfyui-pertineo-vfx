# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo Audit Chain – one breakdown for a whole job: from the final file back through every audited step."""

import json
import os
import re
import traceback

import comfy.utils
import folder_paths

from ..common import LOG
from ..image_io.readwrite import clean_path
from . import chain as CH

from . import hashing
from .collect import own_version
from .chain_pdf import write_chain_pdf

CATEGORY = "Pertineo/Audit"
IF_EXISTS = ["new version", "overwrite"]


def _wrap(text, width, indent=25):
    """Wraps long lines at spaces, continuation lines under the value column."""
    out = []
    for line in text.split("\n"):
        while len(line) > width:
            cut = line.rfind(" ", indent + 10, width)
            if cut < 0:
                cut = width
            out.append(line[:cut])
            line = " " * indent + line[cut:].lstrip()
        out.append(line)
    return "\n".join(out)


class PertineoAuditChain:
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("report", "folder")
    OUTPUT_TOOLTIPS = ("The chain breakdown as text (also written as <name>.chain.txt).",
                       "The folder with the chain files (<name>_chain/ next to the final file, versioned "
                       "<name>.chain_v001.*).")
    FUNCTION = "run"
    DESCRIPTION = (
        "Connects the audit reports of several work steps into one breakdown. Start from the final file (or its "
        ".audit.json): its inputs are looked up by SHA256 among all audit reports in search_folders – an input of "
        "one step is exactly the file another step wrote – and so on back to the first step. Lists every step "
        "(step / step_description, models and licenses, inputs and where they came from, triage), the strictest "
        "triage of the whole chain, outside sources (inputs not made in an audited step), gaps (a file changed "
        "outside ComfyUI) and the iterations that were not used. Run this node on its own – it does not need the "
        "workflow that made the files.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "FINAL: the final image / video / audio file, or its .audit.json",
                    "tooltip": "The end of the chain: the delivered file (PNG, EXR, MOV, MP4, WAV …, also one frame "
                               "of a sequence) or its audit report (.audit.json, or .json from 1.4.0). Paste the path "
                               "(in Nemo: Ctrl+C on the file, Ctrl+V here)."}),
                "search_folders": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "folders with the reports of the earlier steps, one per line (empty = folder "
                                   "of the final file + ComfyUI output + ComfyUI input)",
                    "tooltip": "Where the reports of the earlier steps are: one folder per line, searched with all "
                               "sub folders. Empty = the folder of the final file, the ComfyUI output folder and the ComfyUI input "
                               "folder (declared external steps next to loaded files). "
                               "Steps saved elsewhere (another drive, an older output folder) must be listed here."}),
                "write_txt": ("BOOLEAN", {"default": True, "tooltip": "Write <name>.chain.txt (complete)."}),
                "write_json": ("BOOLEAN", {"default": True,
                    "tooltip": "Write <name>.chain.json (machine-readable, with every hash)."}),
                "write_pdf": ("BOOLEAN", {"default": True,
                    "tooltip": "Write <name>.chain.pdf: overview with the final result, the strictest triage and "
                               "the chain as small pictures, then every step with its picture."}),
            },
            "optional": {
                "if_exists": (IF_EXISTS, {
                    "tooltip": "The files are versioned: <name>.chain_v001.txt / .json / .pdf. new version = every "
                               "run writes the next version (v002 …), earlier breakdowns stay. overwrite = the newest "
                               "version is replaced."}),
                "write_checklist": ("BOOLEAN", {"default": True,
                    "tooltip": "Write <name>.chain_v###.checklist.txt (and .checklist.pdf with write_pdf): the 30 disclosure items over the whole chain, "
                               "incomplete ones first with how to close them. For your own check – not part of the "
                               "PDF."}),
                "pdf_content": (["audit", "breakdown", "audit + breakdown (two files)"], {
                    "tooltip": "audit = the complete record in one PDF: overview, chain graph and per step the full "
                               "disclosure (models, licenses, prompts, settings, resolutions …) and, with "
                               "pdf_assessment, the assessment. breakdown = the overview: graph and per step what was "
                               "done, where the inputs come from, the main models and the grade – no prompts or "
                               "settings. audit + breakdown = both as separate files. The .txt / .json always contain "
                               "everything."}),
                "pdf_prompts": (["shortened", "full"], {
                    "tooltip": "Audit only: prompts shortened (about 400 characters, the complete text is in the .txt) "
                               "or full – for an audit that has to stand on its own."}),
                "pdf_assessment": (["with triage", "without triage"], {
                    "tooltip": "with triage = per step the production context, approvals needed (covered / OPEN), "
                               "every finding with severity and department, the guiding principles, and the grade "
                               "legend. without triage = documentation only, no grades or findings."}),
                "write_overview": ("BOOLEAN", {"default": False,
                    "tooltip": "Also write <name>.chain_v###.overview.pdf: a neutral listing – graph and per step what "
                               "was done, inputs and models by name – with no grades, no findings and no licenses."}),
                "title": ("STRING", {"default": "", "placeholder": "title for the PDFs (empty = the file name)",
                    "tooltip": "Your own heading for the PDFs, e.g. 'SOYLENT feline – commercial spot'. The small line "
                               "above it still says which version it is (audit / breakdown / overview); the file name "
                               "moves to the line below. Empty = the file name as heading."}),
                "audit_layout": (["by topic", "by step"], {
                    "tooltip": "How the audit is ordered. by topic = the whole chain gathered per subject: summary, "
                               "responsibilities, graph, production context (one field per line), approvals, "
                               "findings and principles, all models and licenses in one table, inputs and provenance, "
                               "environment and node packages, generation settings, prompts, outputs. by step = step "
                               "after step, like the breakdown but with everything."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # reports may have changed on disk

    @classmethod
    def VALIDATE_INPUTS(cls, final=None):
        if final is None:
            return True
        p = clean_path(final)
        if not p:
            return "Audit Chain: enter the final file or its .audit.json."
        if not os.path.isfile(p):
            return f"Audit Chain: file not found: {p}"
        return True

    def run(self, final, search_folders, write_txt, write_json, write_pdf, if_exists=IF_EXISTS[0],
            write_checklist=True, pdf_content="audit", pdf_prompts="shortened", pdf_assessment="with triage",
            write_overview=False, title="", audit_layout="by topic"):
        final = os.path.realpath(clean_path(final))
        folders = [os.path.realpath(clean_path(x)) for x in (search_folders or "").splitlines() if x.strip()]
        if not folders:
            folders = [os.path.dirname(final), os.path.realpath(folder_paths.get_output_directory()),
                       os.path.realpath(folder_paths.get_input_directory())]
        folders = list(dict.fromkeys(folders))
        missing = [f for f in folders if not os.path.isdir(f)]
        if missing:
            raise ValueError(f"{LOG} Audit Chain: folder not found: {missing[0]}")
        # a folder inside another one is searched once
        folders = [f for f in folders if not any(f != g and f.startswith(g + os.sep) for g in folders)]

        pbar = comfy.utils.ProgressBar(100)
        reports, by_hash = CH.build_index(folders, progress=lambda i, n: pbar.update_absolute(int(80 * i / n), 100))
        print(f"{LOG} Audit Chain: {len(reports)} audit reports in {', '.join(folders)}")
        start, how = CH.find_start(final, reports, by_hash,
                                   lambda p: hashing.sha256_file(p, label=os.path.basename(p))[0])
        steps = CH.build_chain(start, reports, by_hash)
        summary = CH.summarize(steps, reports, final)
        version = own_version()
        text = CH.render_text(steps, summary, folders, len(reports), version)
        pbar.update_absolute(90, 100)

        base = CH._stem(start["file"].get("name") or start["path"]) or "chain"
        out_dir = os.path.join(os.path.dirname(final), base + "_chain")
        os.makedirs(out_dir, exist_ok=True)
        used = [int(m.group(1)) for m in (re.match(re.escape(base) + r"\.chain_v(\d+)\.", n) for n in os.listdir(out_dir))
                if m]
        ver = (max(used) if used else 1) if if_exists == "overwrite" else (max(used) + 1 if used else 1)
        stem = f"{base}.chain_v{ver:03d}"
        written = []
        if write_txt:
            p = os.path.join(out_dir, stem + ".txt")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(text + "\n")
            written.append(p)
        if write_json:
            p = os.path.join(out_dir, stem + ".json")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(CH.chain_json(steps, summary, folders, len(reports), version), f, indent=2,
                          ensure_ascii=False, default=str)
            written.append(p)
        if write_checklist:
            p = os.path.join(out_dir, stem + ".checklist.txt")
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(CH.render_checklist(steps, os.path.basename(final), version) + "\n")
            written.append(p)
            if write_pdf:
                p = os.path.join(out_dir, stem + ".checklist.pdf")
                try:
                    from .chain_pdf import write_checklist_pdf
                    write_checklist_pdf(p, steps, os.path.basename(final), version)
                    written.append(p)
                except Exception:
                    print(f"{LOG} Audit Chain: checklist PDF not written")
                    traceback.print_exc()
        if write_pdf:
            thumbs = None
            modes = list(("audit", "breakdown") if pdf_content.startswith("audit +") else (pdf_content,))
            if write_overview:
                modes.append("overview")
            for mode in modes:
                p = os.path.join(out_dir, f"{stem}.{mode}.pdf")
                try:
                    if thumbs is None:
                        thumbs = {s["id"]: step_picture(s["report"]) for s in steps}
                        thumbs.update(input_pictures(steps, folders, final))
                    write_chain_pdf(p, steps, summary, thumbs, version, full_prompts=(pdf_prompts == "full"),
                                    mode=mode, assess=(pdf_assessment == "with triage"), title=title,
                                    layout=audit_layout)
                    written.append(p)
                except Exception:
                    print(f"{LOG} Audit Chain: {mode} PDF not written")
                    traceback.print_exc()
        pbar.update_absolute(100, 100)
        print(f"{LOG} Audit Chain: {len(steps)} step(s), worst triage {summary['worst_grade'] or '-'} ({how}) -> "
              f"{out_dir}/{stem}.*")

        ui = {}
        try:  # the summary as an image on the node (no frontend extension needed)
            from ..image_io.save_media import _text_image
            head = text.split("\nSTEPS ────")[0]
            ui = {"images": [_text_image(_wrap(head, 110), max_chars=120)]}
        except Exception:
            traceback.print_exc()
        return {"ui": ui, "result": (text, out_dir)}


# --------------------------------------------------------------------------
# Pictures for the PDF
# --------------------------------------------------------------------------
def _checker(w, h, cell=16):
    import numpy as np
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((yy // cell + xx // cell) % 2).astype(np.float32)
    return (0.62 + 0.18 * c)[..., None].repeat(3, axis=-1)


def _still(path, max_size=900):
    """An image file (PNG / TIFF / EXR / JPG, 8 / 16 bit / float, alpha on a checkerboard) -> PIL RGB."""
    import numpy as np
    from PIL import Image
    from ..image_io.readwrite import cv2, is_exr, linear_to_srgb, read_rgba_raw

    rgb, alpha = read_rgba_raw(path)
    h, w = rgb.shape[:2]
    s = min(1.0, max_size / float(max(w, h)))
    if s < 1.0 and cv2 is not None:
        size = (max(1, int(w * s)), max(1, int(h * s)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = None if alpha is None else cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    rgb = linear_to_srgb(rgb) if is_exr(path) else np.clip(rgb, 0, 1)
    if alpha is not None:
        a = np.clip(alpha, 0, 1)[..., None]
        rgb = rgb * a + _checker(rgb.shape[1], rgb.shape[0]) * (1 - a)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def _video_frame(path, seconds):
    import io
    import subprocess
    from PIL import Image
    from ..image_io.media import ffmpeg_path

    exe = ffmpeg_path()
    if not exe:
        return None
    r = subprocess.run([exe, "-v", "error", "-ss", f"{max(0.0, seconds):.3f}", "-i", path, "-frames:v", "1",
                        "-vf", "scale='min(900,iw)':-2", "-f", "image2pipe", "-vcodec", "png", "-"],
                       capture_output=True, timeout=60)
    return Image.open(io.BytesIO(r.stdout)).convert("RGB") if r.returncode == 0 and r.stdout else None


def _waveform(path):
    import subprocess
    import numpy as np
    from PIL import Image
    from ..image_io.media import ffmpeg_path, waveform_image

    exe = ffmpeg_path()
    if not exe:
        return None
    r = subprocess.run([exe, "-v", "error", "-i", path, "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
                       capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        return None
    wav = np.frombuffer(r.stdout, dtype=np.float32)[None]
    img = waveform_image((wav, 8000), width=900, height=420, ruler=True)
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def input_pictures(steps, folders, final):
    """Pictures of inputs that no audited step made (outside sources, gaps, declared sources), keyed like the graph
    nodes ('S3:1'). The file is looked for where it was loaded from, in the ComfyUI input folder, next to the final
    file and in the search folders."""
    want = {}
    for s in steps:
        for i, it in enumerate(s["inputs"]):
            if it["from"] is None:
                want[f"{s['id']}:{i}"] = str(it["asset"].get("file") or "")
    if not want:
        return {}
    try:
        inp = folder_paths.get_input_directory()
    except Exception:
        inp = ""
    names = {os.path.basename(f.replace("<input>/", "")) for f in want.values()}
    found = {}
    for root in [inp, os.path.dirname(final)] + list(folders):  # direct hits first, then a walk
        for n in names:
            p = os.path.join(root, n) if root else ""
            if p and n not in found and os.path.isfile(p):
                found[n] = p
    missing = names - set(found)
    for root in list(folders) + ([inp] if inp else []):
        if not missing:
            break
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.endswith("_chain")]
            for n in missing & set(filenames):
                found[n] = os.path.join(dirpath, n)
            missing -= set(found)
            if not missing:
                break
    out = {}
    for key, f in want.items():
        p = f if os.path.isabs(f) and os.path.isfile(f) else found.get(os.path.basename(f.replace("<input>/", "")))
        if not p:
            continue
        try:
            low = p.lower()
            if low.endswith(CH.RAW_EXTS):
                out[key] = raw_preview(p)
            elif low.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".webp", ".bmp")):
                try:
                    out[key] = _still(p)
                except Exception:  # formats the EXR / TIFF reader does not take: plain PIL
                    from PIL import Image
                    im = Image.open(p).convert("RGB")
                    im.thumbnail((900, 900))
                    out[key] = im
            elif low.endswith((".mp4", ".mov", ".mkv", ".webm")):
                out[key] = _video_frame(p, 1.0)
        except Exception as e:
            print(f"{LOG} Audit Chain: no picture for {os.path.basename(p)} ({type(e).__name__}: {e})")
    return {k: v for k, v in out.items() if v is not None}


def raw_preview(path, max_size=900):
    """The largest JPEG preview embedded in a camera raw (RW2, CR2, NEF, ARW, DNG …) -> PIL RGB, or None."""
    import io
    from PIL import Image
    with open(path, "rb") as f:
        data = f.read()
    best, pos = None, 0
    for _ in range(64):
        i = data.find(b"\xff\xd8\xff", pos)
        if i < 0:
            break
        pos = i + 3
        try:
            im = Image.open(io.BytesIO(data[i:]))
            if best is None or im.size[0] * im.size[1] > best.size[0] * best.size[1]:
                im.load()
                best = im
        except Exception:
            continue
    if best is None or best.size[0] < 160:
        return None
    best = best.convert("RGB")
    best.thumbnail((max_size, max_size))
    return best


def step_picture(rep):
    """The output of a step as a small picture: the image itself, the middle frame of a video or sequence, or the
    waveform of a sound; for a camera original the preview embedded in the raw. None if the file is not there."""
    if rep.get("kind") == CH.PHOTO:
        f = rep.get("file") or {}
        p = os.path.join(f.get("folder") or "", f.get("name") or "")
        try:
            if os.path.isfile(p):
                return raw_preview(p) if p.lower().endswith(CH.RAW_EXTS) else _still(p)
        except Exception as e:
            print(f"{LOG} Audit Chain: no preview from {os.path.basename(p)} ({type(e).__name__}: {e})")
        return None
    folder = os.path.dirname(rep["path"])
    try:
        for m in rep.get("media") or []:
            p = os.path.join(folder, m["path"])
            if not os.path.isfile(p):
                continue
            kind = str(m.get("kind") or "")
            if "video" in kind and "mask" not in kind:
                frames, fps = m.get("frames") or 0, m.get("fps") or 24.0
                return _video_frame(p, (frames / 2.0) / fps if frames else 1.0)
            if "audio" in kind:
                return _waveform(p)
        if rep.get("frames"):
            mid = rep["frames"][len(rep["frames"]) // 2]
            p = os.path.join(folder, mid)
            if os.path.isfile(p):
                return _still(p)
        name = (rep.get("file") or {}).get("name")
        if name:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                low = p.lower()
                if low.endswith((".mp4", ".mov", ".mkv", ".webm")):
                    return _video_frame(p, 1.0)
                if low.endswith((".wav", ".flac", ".mp3")):
                    return _waveform(p)
                return _still(p)
        for m in rep.get("media") or []:  # e.g. only a mask video was written
            p = os.path.join(folder, m["path"])
            if os.path.isfile(p) and "video" in str(m.get("kind")):
                return _video_frame(p, 1.0)
    except Exception as e:
        print(f"{LOG} Audit Chain: no picture for {os.path.basename(rep['path'])} ({type(e).__name__}: {e})")
    return None
