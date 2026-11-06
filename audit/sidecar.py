# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Audit files for one Save Media output (a sequence, a video or both): one report per output, not per frame.
Same files as Save Image + Metadata: <base>.audit.txt / .audit.json / .audit.pdf + <base>.workflow.json."""

import hashlib
import json
import os
import traceback

from ..common import LOG
from . import assess as A
from . import collect as C
from .nodes import _display_folder, _pdf_thumbnail, _text_for
from .pdf import write_pdf as write_pdf_file
from .report import render


def parse_meta(metadata_json):
    if not metadata_json:
        return None
    try:
        return json.loads(metadata_json)
    except Exception:
        return {"metadata_raw": metadata_json}


def display_folder(folder, meta):
    full = bool(((meta or {}).get("options") or {}).get("include_full_paths", False))
    return _display_folder(os.path.normpath(folder), full)


def workflow_record(prompt, extra_pnginfo):
    """The workflow as run (credentials redacted), as stored in .audit.json – or None."""
    try:
        return A.workflow_for_json(prompt, extra_pnginfo)
    except Exception as e:
        print(f"{LOG} Audit: workflow could not be read ({type(e).__name__}: {e})")
        return None


def write_workflow_file(folder, base, wf, overwrite=False):
    """Writes <base>.workflow.json (opens directly in ComfyUI). Returns (path, record for the .audit.json)."""
    if not wf or not wf.get("workflow"):
        return None, wf
    wp = os.path.join(folder, base + ".workflow.json")
    raw = json.dumps(wf["workflow"], indent=2, ensure_ascii=False, default=str).encode("utf-8")
    with open(wp, "wb" if overwrite else "xb") as f:
        f.write(raw)
    rec = dict(wf, file=os.path.basename(wp), file_sha256=hashlib.sha256(raw).hexdigest(),
               note="The workflow as it was run (credentials redacted). Open "
                    f"{os.path.basename(wp)} in ComfyUI to load it.")
    return wp, rec


def write_audit(folder, base, file_info, meta, metadata_text, thumb, write_txt, write_json, write_pdf,
                workflow=None, overwrite=False):
    """Writes the audit files next to the outputs and returns their paths. Never raises for the PDF.
    workflow: record from write_workflow_file (added to the .audit.json) or None."""
    mode = "w" if overwrite else "x"
    per_json = dict(meta or {})
    per_json["file"] = file_info
    written = []
    file_line = (f"File: {file_info['name']}    Saved: {file_info['saved']}\n"
                 f"Image: {file_info['width']}x{file_info['height']} · {file_info['format']} · "
                 f"folder {file_info['folder']}")
    txt = pdf_txt = None
    if write_txt or write_pdf:
        if isinstance(meta, dict) and meta.get("schema") == C.SCHEMA:
            try:
                txt = render(per_json)
                pdf_txt = render(per_json, short=True)
            except Exception:
                traceback.print_exc()
                txt = _text_for(metadata_text, file_line)
        else:
            txt = _text_for(metadata_text, file_line)
    pdf_txt = pdf_txt or txt
    if write_txt:
        p = os.path.join(folder, base + ".audit.txt")
        with open(p, mode, encoding="utf-8") as f:
            f.write(txt)
        written.append(p)
    if write_pdf:
        p = os.path.join(folder, base + ".audit.pdf")
        try:
            if overwrite and os.path.exists(p):
                os.remove(p)
            author = ((per_json or {}).get("manual") or {}).get("author")
            bad = write_pdf_file(p, pdf_txt, title=f"Audit report {base}",
                                 thumbnail=None if thumb is None else _pdf_thumbnail(thumb), author=author)
            if bad:
                print(f"{LOG} Audit: {bad} character(s) cannot be shown in the PDF font – replaced by '?'.")
            written.append(p)
        except Exception as e:
            traceback.print_exc()
            print(f"{LOG} Audit: PDF could not be written ({type(e).__name__}: {e})")
    if write_json:
        p = os.path.join(folder, base + ".audit.json")
        out_json = dict(per_json)
        if workflow:
            out_json["comfyui_workflow"] = workflow
        with open(p, mode, encoding="utf-8") as f:
            json.dump(out_json, f, indent=2, ensure_ascii=False, default=str)
        written.append(p)
    return written
