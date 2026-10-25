# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – Audit nodes (collect metadata, save image + metadata with the same name)."""

import datetime
import hashlib
import json
import os
import re
import traceback
import uuid

import numpy as np
from PIL import Image

import folder_paths

from ..common import LOG
from ..image_io.readwrite import clean_path
from . import collect as C
from . import context as K
from .pdf import write_pdf as write_pdf_file
from .imagefile import FORMATS, ext_for, write_exr_rgb, write_png8, write_png16
from .report import HEADER, render

CATEGORY = "Pertineo/Audit"
SIDECAR_EXTS = (".txt", ".json", ".pdf")
ALL_EXTS = (".png", ".exr") + SIDECAR_EXTS
_IMAGE_EXT_RE = re.compile(r"\.(png|jpe?g|webp|tiff?|exr|bmp|gif)$", re.I)

_EMBED_NOTE_SHOWN = False
_MANUAL_TIP = (" Write 'none' if there is nothing to declare. Empty = the report shows 'UNKNOWN / NOT "
               "VERIFIED – not provided' (notes and planned work: optional, empty stays empty).")


# --------------------------------------------------------------------------
# Collect Metadata
# --------------------------------------------------------------------------
class PertineoCollectMetadata:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("metadata_text", "metadata_json")
    FUNCTION = "run"
    DESCRIPTION = ("Collects an audit record of the current run: environment, all model files with SHA256 / "
                   "AutoV2 hash, source and license from your model_info file, custom node packages and "
                   "versions, API nodes, sampler settings, prompts and resolutions, plus manual audit fields. "
                   "Outputs a readable text and a JSON with the same content. Never stops the generation – "
                   "problems are listed as warnings in the report.")

    @classmethod
    def INPUT_TYPES(cls):
        def text(tip, placeholder):
            return ("STRING", {"default": "", "multiline": True, "placeholder": placeholder,
                               "tooltip": tip + _MANUAL_TIP})

        return {
            "required": {
                "workflow_purpose": text("What this workflow / image is for (job, shot, client deliverable).",
                                         "PURPOSE: e.g. character insert for shot 010, client X"),
                "reference_image_sources": text(
                    "Origin and RIGHTS of the input / reference images – not the file name (input files are listed "
                    "automatically with their SHA256). E.g. 'Own layout sketch, drawn by me in Krita, no third-party "
                    "material' or 'Plates delivered by the production, 2026-10-20'.",
                    "REFERENCES: origin and rights of input images, e.g. own sketch / client plates / licensed stock"),
                "external_software": text("WHICH other programs were used before or after ComfyUI, e.g. 'Krita (input "
                                          "layout)', 'Nuke 16 (comp)'.",
                                          "EXTERNAL SOFTWARE: e.g. Krita (input layout), Nuke 16 (comp)"),
                "changes_outside_comfyui": text(
                    "PLANNED work on the result after ComfyUI, e.g. 'comp in Nuke, grading', 'typography finalised in "
                    "Krita' – tells a reviewer that this image is not the final picture. 'none' if it is delivered as "
                    "is. Work that has already been done belongs to its own record: an edited image that goes back "
                    "into a workflow is listed there as an input with its hash.",
                    "PLANNED WORK OUTSIDE COMFYUI: e.g. comp in Nuke – or 'none'"),
                "notes": text("Anything else worth recording.", "NOTES"),
                "compute_model_hashes": ("BOOLEAN", {"default": True,
                    "tooltip": "Compute SHA256 and AutoV2 of every model file. Large files take a while the first "
                               "time; results are cached (path + size + modification time) in "
                               "<ComfyUI user dir>/pertineo_vfx/model_hash_cache.json."}),
                "model_info_file": ("STRING", {"default": "",
                    "placeholder": "empty = <ComfyUI user dir>/pertineo_vfx/model_info.ods / .xlsx / .csv / .json",
                    "tooltip": "Spreadsheet (.ods / .xlsx / .csv, e.g. LibreOffice Calc) or .json file you maintain with source, "
                               "license, base model, trigger words … per model, keyed by SHA256 or file name (see "
                               "model_info.example.ods). Empty = <ComfyUI user dir>/pertineo_vfx/model_info.ods, "
                               ".xlsx, .csv or .json – an empty model_info.csv is created there if none "
                               "exists. Relative paths are relative to that folder. Missing entries are UNKNOWN / NOT VERIFIED."}),
                "include_full_paths": ("BOOLEAN", {"default": False,
                    "tooltip": "Off (recommended for sharing): only file names and sub folders relative to the "
                               "model folder appear in the report. On: absolute local paths are included."}),
            },
            "optional": {
                "source_image": ("IMAGE", {"tooltip": "Optional: the source / input image, for the source resolution."}),
                "output_image": ("IMAGE", {"tooltip": "Optional: the final image (batch), for the final resolution "
                                                      "and, with fps, frame count and duration."}),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Frame rate for video output (e.g. 24, 25, 23.976). 0 = still image(s), no video "
                               "section in the report."}),
                "author": ("STRING", {"default": "", "placeholder": "AUTHOR: e.g. your name",
                    "tooltip": "Person responsible for the image. Shown in the report title and header when "
                               "include_author is on."}),
                "company": ("STRING", {"default": "", "placeholder": "COMPANY: optional, e.g. studio",
                    "tooltip": "Optional company or studio, shown in brackets after the author."}),
                "include_author": ("BOOLEAN", {"default": True,
                    "tooltip": "On: author and company appear in the .txt and .json. Off: both are left out "
                               "of the report (they stay in the node). Note: an embedded workflow still contains "
                               "every widget value – turn off embed_workflow on the save node as well if the name "
                               "must not appear anywhere."}),
                "production_context": ("PERTINEO_CONTEXT", {
                    "tooltip": "Optional: connect 'Pertineo Production Context' for the declared purpose, rights and "
                               "consent. Without it the triage is D (indeterminate)."}),
            },
            # EXECUTION_LIST gives access to values other nodes produced in this run (e.g. a prompt
            # joined by a text node) – used only where a value cannot be read from the graph itself.
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID",
                       "execution_list": "EXECUTION_LIST"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The graph, model files and model_info file are not visible to ComfyUI's cache.
        return float("nan")

    def run(self, workflow_purpose, reference_image_sources, external_software, changes_outside_comfyui, notes,
            compute_model_hashes, model_info_file, include_full_paths, source_image=None, output_image=None,
            fps=0.0, author="", company="", include_author=True, prompt=None, extra_pnginfo=None, unique_id=None,
            execution_list=None, production_context=None):
        manual = {"workflow_purpose": workflow_purpose, "reference_image_sources": reference_image_sources,
                  "external_software": external_software, "changes_outside_comfyui": changes_outside_comfyui,
                  "notes": notes}
        if include_author:
            for key, val in (("author", author), ("company", company)):
                if isinstance(val, str) and val.strip():
                    manual[key] = val.strip()
        try:
            data = C.collect(prompt, extra_pnginfo, str(unique_id) if unique_id is not None else None, manual,
                             compute_model_hashes, model_info_file, include_full_paths,
                             source_image, output_image, fps, execution_list, production_context)
        except Exception as e:  # last line of defence: never stop the generation
            traceback.print_exc()
            data = {"schema": C.SCHEMA, "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "manual": manual, "warnings": [f"Audit collection failed: {type(e).__name__}: "
                                                   f"{C.PathDisplay(include_full_paths).text(str(e))}"],
                    "options": {"include_full_paths": bool(include_full_paths)}}
        try:
            text = render(data)
        except Exception as e:
            traceback.print_exc()
            text = f"{HEADER}\nReport rendering failed ({type(e).__name__}). See metadata_json.\n"
        n_models = len(data.get("models") or [])
        print(f"{LOG} Audit: {n_models} model file(s), {len(data.get('warnings') or [])} warning(s)")
        return (text, json.dumps(data, indent=2, ensure_ascii=False, default=str))


# --------------------------------------------------------------------------
# Production Context
# --------------------------------------------------------------------------
class PertineoProductionContext:
    """Declared production context (purpose, rights, consent, review). Everything here is a declaration
    by the user – the report marks it as 'declared', never as verified."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("PERTINEO_CONTEXT",)
    RETURN_NAMES = ("production_context",)
    FUNCTION = "run"
    DESCRIPTION = ("Declares the production context of this run – production, shot, purpose, whether material "
                   "goes into the final cut, rights and consent, reviewers. Connect it to 'Pertineo Collect "
                   "Metadata'. It feeds the findings, the indicative triage (A–D) and the guiding principles in "
                   "the report. Declarations are recorded as such – this is not an approval.")

    @classmethod
    def INPUT_TYPES(cls):
        req = {}
        for key, label in K.TEXT_FIELDS:
            req[key] = ("STRING", {"default": "", "placeholder": label.upper(),
                                   "tooltip": K.TOOLTIPS.get(key, label + ".") + " Leave empty if not applicable."})
        for key, label, answers, default in K.CHOICE_FIELDS:
            req[key] = (list(answers), {"default": default,
                                        "tooltip": K.TOOLTIPS.get(key, label + ".")})
        for key, label in K.REVIEWER_FIELDS:
            req[key] = ("STRING", {"default": "", "placeholder": label.upper(),
                                   "tooltip": K.TOOLTIPS.get(key, f"{label}: name or role. Empty = not reviewed yet.")})
        req["context_notes"] = ("STRING", {"default": "", "multiline": True, "placeholder": "CONTEXT NOTES",
                                           "tooltip": K.TOOLTIPS["context_notes"]})
        return {"required": req}

    def run(self, **kw):
        ctx = {}
        for key, _l in K.TEXT_FIELDS + K.REVIEWER_FIELDS + [("context_notes", "")]:
            v = kw.get(key, "")
            ctx[key] = v.strip() if isinstance(v, str) else ""
        for key, _l, answers, default in K.CHOICE_FIELDS:
            v = kw.get(key, default)
            ctx[key] = v if v in answers else default
        return (ctx,)


# --------------------------------------------------------------------------
# Save Image + Metadata
# --------------------------------------------------------------------------
def _resolve_output(output_folder, filename_prefix):
    out_root = folder_paths.get_output_directory()
    p = clean_path(output_folder)
    if not p:
        folder = out_root
    elif os.path.isabs(p):
        folder = p
    else:
        folder = os.path.join(out_root, p)
    prefix = (filename_prefix or "").strip().replace("\\", "/")
    if "/" in prefix:
        sub, prefix = prefix.rsplit("/", 1)
        folder = os.path.join(folder, sub)
    prefix = _IMAGE_EXT_RE.sub("", prefix.strip()) or "audit"
    prefix = re.sub(r'[<>:"|?*\x00-\x1f]', "_", prefix)
    return os.path.normpath(folder), prefix


def _existing_max(folder, prefix):
    pat = re.compile(re.escape(prefix) + r"_(\d+)\.[A-Za-z0-9]+$")
    best = 0
    try:
        for name in os.listdir(folder):
            m = pat.match(name)
            if m:
                best = max(best, int(m.group(1)))
    except FileNotFoundError:
        pass
    return best


def _reserve(folder, prefix, digits, counter, ext):
    """Finds the next number where no image, .txt or .json exists and creates the image file
    exclusively (O_EXCL), so parallel runs can never get the same name."""
    while True:
        base = f"{prefix}_{counter:0{digits}d}"
        if not any(os.path.exists(os.path.join(folder, base + e)) for e in ALL_EXTS):
            path = os.path.join(folder, base + ext)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                return base, path, fd, counter
            except FileExistsError:
                pass
        counter += 1


def _display_folder(folder, full):
    if full:
        return folder
    out_root = os.path.normpath(folder_paths.get_output_directory())
    if folder == out_root:
        return "<output>"
    if folder.startswith(out_root + os.sep):
        return "<output>/" + os.path.relpath(folder, out_root).replace(os.sep, "/")
    return os.path.basename(folder)


def _file_section(name, idx, n, w, h, fmt, folder_disp, saved_at):
    return (f"File: {name}    Batch: {idx + 1}/{n} (index {idx})    Saved: {saved_at}\n"
            f"Image: {w}x{h} · {fmt} · folder {folder_disp}")


def _text_for(text, file_line):
    if text and text.startswith(HEADER):
        first, _, rest = text.partition("\n")
        return f"{first}\n{file_line}\n{rest}"
    body = text if text else "(no metadata_text connected)\n"
    return f"{HEADER}\n{file_line}\n\n{body}"


def _output_preview(path, fmt):
    """PNG 8 bit inside the ComfyUI output folder: show the saved file itself (full size, like the core
    Save Image node). Everything else (16 bit, EXR, folders outside output) gets a small 8-bit preview."""
    if not fmt.startswith("png (8"):
        return None
    out_root = os.path.realpath(folder_paths.get_output_directory())
    real = os.path.realpath(path)
    if not real.startswith(out_root + os.sep):
        return None
    sub = os.path.relpath(os.path.dirname(real), out_root)
    return {"filename": os.path.basename(real), "subfolder": "" if sub == "." else sub, "type": "output"}


def _save_preview(img, max_size=1024):
    arr = np.clip(np.asarray(img)[..., :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    pil.thumbnail((max_size, max_size))
    name = f"pertineo_audit_preview_{uuid.uuid4().hex[:10]}.png"
    pil.save(os.path.join(folder_paths.get_temp_directory(), name), compress_level=1)
    return {"filename": name, "subfolder": "", "type": "temp"}


class PertineoSaveImageMetadata:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_basename", "saved_paths")
    FUNCTION = "run"
    OUTPUT_NODE = True
    DESCRIPTION = ("Saves images and their audit report with guaranteed identical base names "
                   "(name_001.png + name_001.txt / .json). The counter skips every number where an image, "
                   ".txt or .json already exists; nothing is ever overwritten. Every image of a batch gets "
                   "its own file pair.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Image batch to save."}),
                "output_folder": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "OUTPUT: empty = ComfyUI output folder; relative = inside it; absolute allowed",
                    "tooltip": "Target folder. Empty = ComfyUI output folder, a relative path is created inside it, "
                               "absolute paths are allowed. Created if it does not exist."}),
                "filename_prefix": ("STRING", {"default": "audit",
                    "tooltip": "File name before the counter: <prefix>_001.png. Can be linked (e.g. from a "
                               "file name). A trailing image extension is removed; 'sub/name' writes into a "
                               "sub folder."}),
                "counter_digits": ("INT", {"default": 3, "min": 1, "max": 8,
                    "tooltip": "Digits of the counter (3 = _001)."}),
                "format": (FORMATS, {
                    "tooltip": "PNG 8 bit (like Save Image), PNG 16 bit, or EXR 32-bit float. EXR values are "
                               "converted from sRGB to linear; the workflow is not embedded in EXR files."}),
                "embed_workflow": ("BOOLEAN", {"default": True,
                    "tooltip": "PNG only: embed prompt and workflow like the core Save Image node (the image can "
                               "be dropped back into ComfyUI), plus the audit JSON if connected. Note: the "
                               "embedded workflow contains everything typed into the nodes, including local "
                               "paths – turn off for anonymised sharing."}),
                "write_txt": ("BOOLEAN", {"default": True, "tooltip": "Write <name>.txt with the readable report."}),
                "write_json": ("BOOLEAN", {"default": False, "tooltip": "Write <name>.json with the machine-readable report."}),
            },
            "optional": {
                "metadata_text": ("STRING", {"forceInput": True,
                    "tooltip": "metadata_text from Pertineo Collect Metadata."}),
                "metadata_json": ("STRING", {"forceInput": True,
                    "tooltip": "metadata_json from Pertineo Collect Metadata."}),
                "write_pdf": ("BOOLEAN", {"default": False,
                    "tooltip": "Also write <name>.pdf: the readable report on A4 pages with a thumbnail of the "
                               "image. Uses a standard PDF font – characters outside Western European scripts "
                               "(e.g. Chinese) are shown as '?' with a note; the .txt / .json keep them."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def run(self, images, output_folder, filename_prefix, counter_digits, format, embed_workflow, write_txt,
            write_json, metadata_text=None, metadata_json=None, write_pdf=False, prompt=None,
            extra_pnginfo=None):
        folder, prefix = _resolve_output(output_folder, filename_prefix)
        os.makedirs(folder, exist_ok=True)
        ext = ext_for(format)

        meta = None
        if metadata_json:
            try:
                meta = json.loads(metadata_json)
            except Exception:
                meta = {"metadata_raw": metadata_json}
        full_paths = bool(((meta or {}).get("options") or {}).get("include_full_paths", False))
        folder_disp = _display_folder(folder, full_paths)

        texts_png = []
        if embed_workflow and ext == ".png":
            if prompt is not None:
                texts_png.append(("prompt", json.dumps(prompt)))
            if isinstance(extra_pnginfo, dict):
                for k, v in extra_pnginfo.items():
                    texts_png.append((k, json.dumps(v)))

        n = images.shape[0]
        counter = _existing_max(folder, prefix) + 1
        basenames, paths, previews = [], [], []
        for i in range(n):
            img = images[i].detach().cpu().numpy()
            h, w = img.shape[0], img.shape[1]
            base, path, fd, counter = _reserve(folder, prefix, counter_digits, counter, ext)
            saved_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
            file_line = _file_section(base + ext, i, n, w, h, format, folder_disp, saved_at)
            file_info = {"name": base + ext, "basename": base, "batch_index": i, "batch_size": n,
                         "width": w, "height": h, "format": format, "folder": folder_disp, "saved": saved_at}
            per_json = None
            if meta is not None or write_json:
                per_json = dict(meta or {})
                per_json["file"] = file_info
            try:
                with os.fdopen(fd, "wb") as f:
                    if ext == ".png":
                        chunks = list(texts_png)
                        if embed_workflow and per_json is not None and meta is not None:
                            chunks.append(("pertineo_audit", json.dumps(per_json)))
                        (write_png16 if "16" in format else write_png8)(f, img, chunks)
                if ext == ".exr":
                    write_exr_rgb(path, img)
            except Exception:
                try:
                    os.remove(path)
                except OSError:
                    pass
                raise
            written = [path]
            # SHA256 of the saved image: links this report to the next workflow that loads the image
            # (the same hash then appears there under INPUT FILES). Not part of the PNG-embedded copy.
            try:
                hasher = hashlib.sha256()
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        hasher.update(chunk)
                file_info["sha256"] = hasher.hexdigest()
                if per_json is not None:
                    per_json["file"] = dict(file_info)
            except OSError as e:
                print(f"{LOG} Audit: could not hash {path} ({type(e).__name__})")
            txt = None
            if write_txt or write_pdf:
                if isinstance(meta, dict) and meta.get("schema") == C.SCHEMA:
                    try:  # re-render from the JSON so the file block sits in the header
                        txt = render(per_json)
                    except Exception:
                        traceback.print_exc()
                        txt = _text_for(metadata_text, file_line)
                else:
                    txt = _text_for(metadata_text, file_line)
            if write_txt:
                p = os.path.join(folder, base + ".txt")
                with open(p, "x", encoding="utf-8") as f:
                    f.write(txt)
                written.append(p)
            if write_pdf:
                p = os.path.join(folder, base + ".pdf")
                try:
                    thumb = (np.clip(img[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                    if thumb.shape[-1] == 1:
                        thumb = np.repeat(thumb, 3, axis=-1)
                    author = ((per_json or {}).get("manual") or {}).get("author")
                    bad = write_pdf_file(p, txt, title=f"Audit report {base}", thumbnail=thumb, author=author)
                    if bad:
                        print(f"{LOG} Audit: {bad} character(s) cannot be shown in the PDF font – replaced by '?' "
                              f"(see note in the PDF; the .txt keeps them).")
                    written.append(p)
                except Exception as e:
                    traceback.print_exc()
                    print(f"{LOG} Audit: PDF could not be written ({type(e).__name__}: {e})")
            if write_json:
                p = os.path.join(folder, base + ".json")
                with open(p, "x", encoding="utf-8") as f:
                    json.dump(per_json, f, indent=2, ensure_ascii=False, default=str)
                written.append(p)
            for p in written:
                print(f"{LOG} Audit: wrote {p}")
            basenames.append(base)
            paths += written
            counter += 1
            try:
                previews.append(_output_preview(path, format) or _save_preview(img))
            except Exception as e:
                print(f"{LOG} Audit: preview failed ({type(e).__name__})")

        global _EMBED_NOTE_SHOWN
        if embed_workflow and ext == ".png" and not full_paths and not _EMBED_NOTE_SHOWN:
            _EMBED_NOTE_SHOWN = True
            print(f"{LOG} Audit: note – the embedded workflow may contain local paths typed into nodes; "
                  f"turn off embed_workflow for anonymised sharing.")
        return {"ui": {"images": previews}, "result": ("\n".join(basenames), "\n".join(paths))}


# --------------------------------------------------------------------------
# LoRA Loader + Info (trigger words etc. from the model_info file)
# --------------------------------------------------------------------------
def _info_file_mtime(model_info_file):
    try:
        text = (model_info_file or "").strip().strip("\"'")
        base = C.hashing.pertineo_user_dir()
        if not text:
            path, _note = C.default_model_info_path(create=False)
        else:
            path = os.path.expanduser(clean_path(text))
            if not os.path.isabs(path):
                path = os.path.join(base, path)
        return f"{path}:{os.stat(path).st_mtime_ns}" if path else ""
    except Exception:
        return ""


def model_info_lookup(folder, rel_name, model_info_file, lookup_by_hash):
    """Returns (extras dict, info entry or None, note). Never raises."""
    show = C.PathDisplay(False)
    notes = []
    try:
        entries, _name = C.load_model_info(model_info_file, show, notes.append)
        sha = None
        if lookup_by_hash and entries:
            full = folder_paths.get_full_path(folder, rel_name)
            if full:
                sha, _cached = C.hashing.sha256_file(full, label=os.path.basename(rel_name))
        info, _matched = C.lookup_model_info(entries, sha, rel_name, f"{folder}/{rel_name}")
        if info is None:
            notes.append(f"No model_info entry for {folder}/{rel_name}.")
            return {}, None, " ".join(notes)
        return C.info_extras(info), info, " ".join(notes)
    except Exception as e:
        return {}, None, f"model_info lookup failed ({type(e).__name__}: {show.text(str(e))})"


def info_outputs(folder, rel_name, model_info_file, lookup_by_hash, label):
    """(name, base_model, trigger_words, recommended, license, info text) for one model file."""
    extras, info, note = model_info_lookup(folder, rel_name, model_info_file, lookup_by_hash)
    triggers = ", ".join(extras.get("trigger_words") or [])
    base = extras.get("base_model", "")
    name = extras.get("name", "")
    lic = ""
    lines = [f"{label}: {name or rel_name}"]
    if info is not None:
        cu = info.get("commercial_use")
        cu = ("yes" if cu else "no") if isinstance(cu, bool) else (cu or C.UNKNOWN_INFO)
        lic = str(info.get("license") or "")
        lines.append(f"File: {folder}/{rel_name}")
        lines.append(f"Base model: {base or C.NOT_PROVIDED}")
        if triggers or folder == "loras":
            lines.append(f"Trigger words: {triggers or '– none –'}")
        if extras.get("version") or extras.get("released"):
            lines.append("Version: " + " · ".join(x for x in (extras.get("version"), extras.get("released") and
                                                              f"released {extras['released']}") if x))
        if extras.get("recommended"):
            lines.append(f"Recommended: {extras['recommended']}")
        lines.append(f"License: {lic or C.UNKNOWN_INFO}")
        lines.append(f"Commercial use: {cu}")
        if info.get("separate_license") not in (None, ""):
            sl = "yes" if C._truthy(info["separate_license"]) else "no"
            ref = f" – {info['license_reference']}" if info.get("license_reference") else ""
            lines.append(f"Separate license: {sl}{ref}")
        if info.get("_family"):
            lines.append(f"Family row: {', '.join(info['_family'])}")
        if info.get("source"):
            lines.append(f"Source: {info['source']}")
        if extras.get("checked"):
            lines.append(f"License checked: {extras['checked']}")
        if info.get("notes"):
            lines.append(f"Notes: {info['notes']}")
    if note:
        lines.append(note)
        print(f"{LOG} Model info: {note}")
    return name, base, triggers, extras.get("recommended", ""), lic, "\n".join(lines)


class PertineoLoraLoaderInfo:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "trigger_words", "base_model", "info")
    FUNCTION = "run"
    DESCRIPTION = ("Loads a LoRA like the core LoRA loader and additionally outputs what your model_info file "
                   "says about it: trigger words (ready to join into a prompt), base model and a short info "
                   "text with license and source. The LoRA is chosen only once, so loader and info always "
                   "match. A missing entry never stops the run.")

    def __init__(self):
        self._loader = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to apply the LoRA to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "LoRA strength on the diffusion model."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "LoRA strength on the text encoder. Ignored when no clip is connected."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional text encoder. Leave unconnected for model-only LoRAs "
                                             "(the clip output is then empty)."}),
                "model_info_file": ("STRING", {"default": "",
                    "placeholder": "empty = <ComfyUI user dir>/pertineo_vfx/model_info.ods / .xlsx / .csv / .json",
                    "tooltip": "Same file as in Pertineo Collect Metadata. Empty = "
                               "<ComfyUI user dir>/pertineo_vfx/model_info.ods / .xlsx / .csv / .json."}),
                "lookup_by_hash": ("BOOLEAN", {"default": True,
                    "tooltip": "Find the entry by SHA256 (survives renaming; hash is cached). Off = by file "
                               "name only."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, model_info_file="", **kwargs):
        # re-run when the model_info file was edited, otherwise ComfyUI's cache applies
        return _info_file_mtime(model_info_file)

    def run(self, model, lora_name, strength_model, strength_clip, clip=None, model_info_file="",
            lookup_by_hash=True):
        import nodes as comfy_nodes

        if self._loader is None:
            self._loader = comfy_nodes.LoraLoader()
        out_model, out_clip = self._loader.load_lora(model, clip, lora_name, strength_model,
                                                     strength_clip if clip is not None else 0.0)

        _name, base, triggers, _rec, _lic, text = info_outputs("loras", lora_name, model_info_file, lookup_by_hash,
                                                              "LoRA")
        return (out_model, out_clip, triggers, base, text)


# --------------------------------------------------------------------------
# Model Info – one small node per model folder, each with a loader-like dropdown
# --------------------------------------------------------------------------
MODEL_INFO_TYPES = [  # (class suffix, display label, ComfyUI model folder)
    ("Lora", "LoRA", "loras"),
    ("Checkpoint", "Checkpoint", "checkpoints"),
    ("DiffusionModel", "Diffusion Model", "diffusion_models"),
    ("VAE", "VAE", "vae"),
    ("TextEncoder", "Text Encoder", "text_encoders"),
    ("ControlNet", "ControlNet", "controlnet"),
    ("Upscaler", "Upscaler", "upscale_models"),
]


class _ModelInfoBase:
    CATEGORY = CATEGORY + "/Model Info"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("name", "base_model", "trigger_words", "recommended", "license", "info")
    FUNCTION = "run"
    OUTPUT_NODE = True  # runs on its own, so you can look a model up without wiring it into a graph
    FOLDER = None
    LABEL = None

    @classmethod
    def INPUT_TYPES(cls):
        try:
            files = folder_paths.get_filename_list(cls.FOLDER)
        except Exception:
            files = []
        return {
            "required": {
                "model": (files or ["– no files in this model folder –"], {
                    "tooltip": f"{cls.LABEL} file from the '{cls.FOLDER}' model folder – the same list as in "
                               f"the loaders."}),
            },
            "optional": {
                "model_info_file": ("STRING", {"default": "",
                    "placeholder": "empty = <ComfyUI user dir>/pertineo_vfx/model_info.ods / .xlsx / .csv / .json",
                    "tooltip": "Same file as in Pertineo Collect Metadata. Empty = "
                               "<ComfyUI user dir>/pertineo_vfx/model_info.ods / .xlsx / .csv / .json."}),
                "lookup_by_hash": ("BOOLEAN", {"default": True,
                    "tooltip": "Find the entry by SHA256 (survives renaming; hash is cached). Off = by file "
                               "name only."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, model_info_file="", **kwargs):
        return _info_file_mtime(model_info_file)

    def run(self, model, model_info_file="", lookup_by_hash=True):
        out = info_outputs(self.FOLDER, model, model_info_file, lookup_by_hash, self.LABEL)
        return {"ui": {"text": [out[-1]]}, "result": out}


MODEL_INFO_CLASSES = {}
for _suffix, _label, _folder in MODEL_INFO_TYPES:
    _cls = type(f"PertineoModelInfo{_suffix}", (_ModelInfoBase,), {
        "FOLDER": _folder, "LABEL": _label,
        "DESCRIPTION": (f"Shows what your model_info file says about a {_label} – name, base model, trigger "
                        f"words, recommended settings, license and source. Only reads the file; nothing is "
                        f"loaded into memory. Connect 'info' to a text preview or use the single outputs."),
    })
    MODEL_INFO_CLASSES[_cls.__name__] = (_cls, f"Pertineo Model Info ({_label})")


NODE_CLASS_MAPPINGS = {
    "PertineoCollectMetadata": PertineoCollectMetadata,
    "PertineoProductionContext": PertineoProductionContext,
    "PertineoSaveImageMetadata": PertineoSaveImageMetadata,
    "PertineoLoraLoaderInfo": PertineoLoraLoaderInfo,
    **{k: v[0] for k, v in MODEL_INFO_CLASSES.items()},
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PertineoCollectMetadata": "Pertineo Collect Metadata",
    "PertineoProductionContext": "Pertineo Production Context",
    "PertineoSaveImageMetadata": "Pertineo Save Image + Metadata",
    "PertineoLoraLoaderInfo": "Pertineo LoRA Loader + Info",
    **{k: v[1] for k, v in MODEL_INFO_CLASSES.items()},
}
