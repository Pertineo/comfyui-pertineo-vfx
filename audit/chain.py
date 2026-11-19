# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Audit Chain: connects the audit reports of several work steps through SHA256 – the output of one step is an
input of the next – and writes one breakdown for the whole chain. Pure Python (no ComfyUI imports), so it can be
tested on its own."""

import datetime
import difflib
import json
import os
import re
import shutil
import subprocess

SCHEMA_PREFIX = "pertineo-audit"
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "X": 4}
_COUNTER_RE = re.compile(r"([_.\-]v?\d+)+$", re.I)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------
_CHAIN_FILE_RE = re.compile(r"\.chain(_v\d+)?\.json$", re.I)
EXTERNAL = "external_step"
PROJECT_EXTS = (".kra", ".psd", ".psb", ".xcf", ".ora", ".nk", ".nknc", ".blend", ".aep", ".afphoto", ".pdn",
                ".drp", ".hip", ".hiplc")


def is_report_name(name):
    n = name.lower()
    return (n.endswith(".json") and not n.endswith((".workflow.json", ".amend.json"))
            and not _CHAIN_FILE_RE.search(n))


def amend_path(report_path):
    """<name>.audit.json -> <name>.audit.amend.json (written by Pertineo Audit External Step, mode amend report)."""
    return re.sub(r"\.json$", ".amend.json", report_path, flags=re.I)


def load_amend(report_path):
    """(fields, inputs) added to a report later: fields {key: {"value", "date"}}, inputs [asset + "added"] –
    ({}, []) if there is no amendment."""
    p = amend_path(report_path)
    if not os.path.isfile(p):
        return {}, []
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return {}, []
    if str(d.get("schema", "")).split("/")[0] != "pertineo-audit-amend":
        return {}, []
    fields = {k: v for k, v in (d.get("fields") or {}).items()
              if isinstance(v, dict) and str(v.get("value", "")).strip()}
    inputs = [a for a in d.get("inputs") or [] if isinstance(a, dict) and a.get("sha256")]
    return fields, inputs


def load_report(path):
    """The audit report in a JSON file (.audit.json from 1.4.1 on, .json from 1.4.0) – None if it is none."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    schema = str(d.get("schema", "")) if isinstance(d, dict) else ""
    if schema.split("/")[0] != SCHEMA_PREFIX:  # not the chain / amend files
        return None
    return d


def output_hashes(rep):
    """Every file this report wrote: image, video, audio, every frame of a sequence."""
    f = rep.get("file") or {}
    out = set()
    if f.get("sha256"):
        out.add(f["sha256"].lower())
    for o in f.get("outputs") or []:
        if isinstance(o, dict) and o.get("sha256"):
            out.add(o["sha256"].lower())
    for v in (f.get("checksum_list") or {}).values():
        if isinstance(v, str):
            out.add(v.lower())
    return out


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def _png_prompt(path):
    """The API prompt ComfyUI embeds in a saved PNG (text chunk 'prompt') – {} if there is none."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            raw = (im.info or {}).get("prompt")
        d = json.loads(raw) if raw else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def recover_from_workflow(rep, path=None):
    """Older reports carry the workflow as it was run (comfyui_workflow.prompt, from 1.4.1 on; before that only
    the PNG they saved carries it) but did not yet extract every value from it. Reads what is plainly there –
    constant widget values, no guessing – and marks it 'from the recorded workflow': the tile size of tiled
    upscalers, the CFG of a sampler, that a BasicGuider has no CFG, and the generation size where the latent
    comes from a node with a constant width / height or from an encoded input of known size. Works on a copy."""
    P = ((rep.get("comfyui_workflow") or {}).get("prompt")) or {}
    src = "the recorded workflow"
    if (not isinstance(P, dict) or not P) and path:
        f = rep.get("file") or {}
        png = os.path.join(os.path.dirname(path), str(f.get("name") or ""))
        if png.lower().endswith(".png") and os.path.isfile(png):
            P = _png_prompt(png)
            src = "the workflow embedded in the PNG"
    if not isinstance(P, dict) or not P:
        return rep
    rep = json.loads(json.dumps(rep, default=str))
    asset_size = {str(a.get("node_id")): (a.get("width"), a.get("height"))
                  for a in rep.get("input_assets") or [] if isinstance(a, dict) and a.get("width")}

    def decoded_straight_to_save(sampler_id):
        """True if a save node gets its images from a VAE decode of exactly this sampler's output."""
        for n in P.values():
            if not isinstance(n, dict) or "save" not in str(n.get("class_type") or "").lower():
                continue
            im = (n.get("inputs") or {}).get("images", (n.get("inputs") or {}).get("image"))
            if not _is_link(im):
                continue
            dec = P.get(str(im[0])) if isinstance(P.get(str(im[0])), dict) else {}
            if "vaedecode" not in str(dec.get("class_type") or "").lower().replace("_", ""):
                continue
            smp = (dec.get("inputs") or {}).get("samples")
            if _is_link(smp) and str(smp[0]) == str(sampler_id):
                return True
        return False

    def latent_size(link):
        """Follows a latent link upstream (max. 8 nodes) to a constant width / height or a loaded image."""
        for _ in range(8):
            if not _is_link(link):
                return None
            nid = str(link[0])
            n = P.get(nid) if isinstance(P.get(nid), dict) else {}
            ins = n.get("inputs") or {}
            if isinstance(ins.get("width"), int) and isinstance(ins.get("height"), int):
                return ins["width"], ins["height"], n.get("class_type")
            if nid in asset_size:
                return asset_size[nid] + (n.get("class_type"),)
            nxt = next((ins[k] for k in ("samples", "latent", "latent_image", "pixels", "image") if _is_link(ins.get(k))),
                       None)
            link = nxt
        return None

    def node(nid):
        n = P.get(str(nid))
        return n if isinstance(n, dict) else {}

    for x in rep.get("sampling") or []:
        if not isinstance(x, dict):
            continue
        params = x.setdefault("params", {})
        ins = node(x.get("node_id")).get("inputs") or {}
        if "cfg" not in params and "guidance" not in params:
            if isinstance(ins.get("cfg"), (int, float)):
                params["cfg"] = f"{ins['cfg']} (from {src})"
            elif _is_link(ins.get("guider")):
                g = node(ins["guider"][0])
                gt = str(g.get("class_type") or "")
                gi = g.get("inputs") or {}
                if "basicguider" in gt.lower().replace("_", ""):
                    params["cfg"] = f"none ({gt} – no CFG; from {src})"
                elif isinstance(gi.get("cfg"), (int, float)):
                    params["cfg"] = f"{gi['cfg']} (from {src})"
    for g in (rep.get("resolution") or {}).get("generation") or []:
        if not isinstance(g, dict) or g.get("width") or g.get("tile_width"):
            continue
        m = re.search(r"#(\d+)\s*$", str(g.get("sampler") or ""))
        ins = node(m.group(1)).get("inputs") if m else None
        if ins and isinstance(ins.get("tile_width"), int) and isinstance(ins.get("tile_height"), int):
            g.update(tile_width=ins["tile_width"], tile_height=ins["tile_height"], image_based=True,
                     from_workflow=src)
        elif ins:
            sz = latent_size(ins.get("latent_image", ins.get("latent")))
            if sz and sz[0] and sz[1]:
                g.update(width=sz[0], height=sz[1], from_workflow=src)
            elif m and decoded_straight_to_save(m.group(1)):
                # the sampler's latent is decoded and saved as it is (no resize, no upscale in between): the
                # generation size is the size of the saved image
                f = rep.get("file") or {}
                if f.get("width") and f.get("height"):
                    g.update(width=f["width"], height=f["height"],
                             from_workflow=f"{src} – decoded and saved without resizing, so = output size")
    return rep


def _slim(rep, path):
    """Keeps only what the chain needs (reports can be large)."""
    rep = recover_from_workflow(rep, path)
    f = rep.get("file") or {}
    manual, amended = dict(rep.get("manual") or {}), {}
    am_fields, am_inputs = load_amend(path)
    for k, v in am_fields.items():  # declarations added later – the report itself is not changed
        manual[k] = str(v["value"]).strip()
        amended[k] = str(v.get("date") or "")[:10]
    if am_inputs:
        amended["inputs"] = max(str(a.get("added") or "")[:10] for a in am_inputs)
    return {
        "path": path,
        "kind": rep.get("kind") or "comfyui",
        "amended": amended,
        "generated": rep.get("generated") or f.get("saved") or "",
        "generator": (rep.get("generator") or {}).get("version", ""),
        "manual": manual,
        "context": rep.get("context") or {},
        "file": {k: f.get(k) for k in ("name", "basename", "format", "width", "height", "folder", "saved",
                                       "version", "first_frame", "fps", "camera", "last_frame", "frame_count",
                                       "missing_frames")},
        "outputs": sorted(output_hashes(rep)),
        # files to show in the PDF: video / audio / sequence frames (relative to the report folder)
        "media": [{"kind": o.get("kind"), "path": o.get("path"), "frames": o.get("frames"), "fps": o.get("fps"),
                   "format": o.get("format"), "sample_rate": o.get("sample_rate"), "channels": o.get("channels"),
                   "duration_sec": o.get("duration_sec")}
                  for o in f.get("outputs") or [] if isinstance(o, dict) and o.get("path")],
        "frames": sorted(k for k in (f.get("checksum_list") or {})
                         if re.search(r"\d+\.(png|exr|tiff?|jpe?g|dpx|webp)$", k, re.I)),
        "inputs": [{k: a.get(k) for k in ("file", "role", "sha256", "node_type", "node_id", "width", "height", "added",
                                                   "camera", "frame_sha256", "path", "files_in_folder",
                                                   "first_frame", "last_frame")}
                   for a in rep.get("input_assets") or [] if isinstance(a, dict)]
                  + [{k: a.get(k) for k in ("file", "role", "sha256", "width", "height", "added", "camera")}
                     for a in am_inputs
                     if str(a["sha256"]).lower() not in {str(x.get("sha256") or "").lower()
                                                         for x in rep.get("input_assets") or [] if isinstance(x, dict)}],
        "models": [{k: m.get(k) for k in ("name", "file", "type", "role", "sha256", "license", "commercial_use",
                                          "license_notes", "creator", "source", "version", "base_model",
                                          "strength", "strength_model", "strength_clip")}
                   for m in rep.get("models") or [] if isinstance(m, dict)],
        "environment": rep.get("environment") or {},
        "packages": [{"package": c.get("package"), "version": c.get("version")}
                     for c in rep.get("custom_nodes") or [] if isinstance(c, dict)],
        "api_nodes": rep.get("api_nodes") or [],
        "external_services": rep.get("external_services") or [],
        "sampling": [{"node": f"{x.get('node_type')} #{x.get('node_id')}", "params": x.get("params") or {},
                      "prompts": {k: _prompt_texts(v, k) for k, v in (x.get("prompts") or {}).items()}}
                     for x in rep.get("sampling") or [] if isinstance(x, dict)],
        "prompt_writers": [{"node": f"{w.get('node_type')} #{w.get('node_id')}", "feeds": w.get("feeds") or []}
                           for w in rep.get("prompt_writers") or [] if isinstance(w, dict)],
        "resolution": rep.get("resolution") or {},
        "audio": rep.get("audio") or {},
        "seed_mode": rep.get("seed_mode") or {},
        "workflow_type": rep.get("workflow_type") or [],
        "triage": rep.get("triage") or {},
        "findings": [{k: x.get(k) for k in ("id", "text", "severity", "route", "evidence", "need")}
                     for x in rep.get("findings") or [] if isinstance(x, dict)],
        "principles": [{k: x.get(k) for k in ("id", "principle", "status", "basis")}
                       for x in rep.get("principles") or [] if isinstance(x, dict)],
        "reproducibility": (rep.get("reproducibility") or {}).get("status", ""),
        "probe": _probe_file(rep, path),
    }


_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".mxf", ".m4v")
_PROBE_CACHE = {}


def _probe_video(p):
    """(frames, fps, seconds) of a video file, read with PyAV (part of ComfyUI) or ffprobe – None if neither can
    read it. Nothing is decoded except when the container does not store a frame count."""
    key = (p, os.path.getmtime(p), os.path.getsize(p))
    if key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    res = None
    try:
        import av  # noqa: PLC0415
        with av.open(p) as c:
            st = c.streams.video[0]
            fps = float(st.average_rate or st.base_rate or 0) or None
            sec = (float(st.duration * st.time_base) if st.duration and st.time_base else
                   float(c.duration) / 1_000_000 if c.duration else None)
            n = int(st.frames or 0) or (round(sec * fps) if sec and fps else None)
            if fps and n:
                res = (n, fps, sec or n / fps)
    except Exception:
        res = None
    if res is None and shutil.which("ffprobe"):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
                                  "-of", "json", p], capture_output=True, text=True, timeout=30)
            d = json.loads(out.stdout or "{}")
            st = (d.get("streams") or [{}])[0]

            def rate(x):
                a, _, b = str(x or "0/0").partition("/")
                return float(a) / float(b) if b and float(b) else (float(a) if a and not b else None)
            fps = rate(st.get("avg_frame_rate")) or rate(st.get("r_frame_rate"))
            sec = float(st.get("duration") or (d.get("format") or {}).get("duration") or 0) or None
            n = int(st.get("nb_frames") or 0) or (round(sec * fps) if sec and fps else None)
            if fps and n:
                res = (n, fps, sec or n / fps)
        except Exception:
            res = None
    _PROBE_CACHE[key] = res
    return res


def _probe_file(rep, report_path):
    """Frames / fps / duration of the video a step wrote, read from the file on disk (the report may not have
    them – external steps, older reports). {} if there is no such file."""
    f = rep.get("file") or {}
    folders = [x for x in (f.get("folder"), os.path.dirname(report_path)) if x]
    names = [str(f.get("name") or "")] + [str(o.get("path") or "") for o in f.get("outputs") or []
                                          if isinstance(o, dict) and "video" in str(o.get("kind"))]
    for nm in names:
        if not nm.lower().endswith(_VIDEO_EXTS):
            continue
        for fo in folders:
            p = os.path.join(fo, nm)
            if os.path.isfile(p):
                t = _probe_video(p)
                if t:
                    return {"frames": t[0], "fps": t[1], "seconds": t[2]}
    return {}


def _prompt_texts(entries, side):
    """Prompt texts of one side. Some encoders hold both prompts (TextEncodeQwenImage21: prompt +
    negative_prompt) and appear on both sides – take only the matching field."""
    out = []
    for p in entries or []:
        texts = {k: t for k, t in (p.get("texts") or {}).items() if isinstance(t, str)}
        neg = {k: t for k, t in texts.items() if "negative" in k.lower()}
        if side == "negative":
            out += list(neg.values()) if neg else list(texts.values())
        else:
            out += [t for k, t in texts.items() if k not in neg and "system" not in k.lower()]
    return out


def build_index(folders, progress=None):
    """All audit reports below the folders (recursive). Returns (reports, by_output_hash)."""
    seen, reports, by_hash = set(), [], {}
    paths = []
    for root in folders:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.endswith("_chain")]
            for n in filenames:
                if is_report_name(n):
                    p = os.path.realpath(os.path.join(dirpath, n))
                    if p not in seen:
                        seen.add(p)
                        paths.append(p)
    for i, p in enumerate(sorted(paths)):
        rep = load_report(p)
        if rep is not None:
            s = _slim(rep, p)
            reports.append(s)
            for h in s["outputs"]:
                by_hash.setdefault(h, []).append(s)
        if progress:
            progress(i + 1, len(paths))
    # several reports for the same file (re-saved): the newest one counts
    for h, lst in by_hash.items():
        lst.sort(key=lambda r: r["generated"], reverse=True)
    return reports, by_hash


# --------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------
def _stem(name):
    n = os.path.basename(str(name or ""))
    n = re.sub(r"(\.audit)?\.[A-Za-z0-9]+$", "", n)
    return n


def series_key(rep):
    """Name without counters / versions: audit_007 -> audit, shot_comp_v003 -> shot_comp (iterations of one step)."""
    return _COUNTER_RE.sub("", _stem((rep["file"] or {}).get("name") or rep["path"])).lower()


_EDIT_SUFFIX_RE = re.compile(r"([_\-. ](edited|edit|retouched|retouch|final|fixed|paintover|comp))+$", re.I)


def _folder_hint(stem, reports, exclude):
    """Second try: the name of the input looks like the FOLDER of a step (02_character_upscaled_edited.png <->
    folder 02_character_upscale) – e.g. a result renamed while it was edited in Krita."""
    s = _EDIT_SUFFIX_RE.sub("", stem).lower()
    best, score = None, 0.0
    for r in reports:
        if r["path"] in exclude:
            continue
        folder = os.path.basename(os.path.dirname(r["path"])).lower()
        sc = difflib.SequenceMatcher(None, s, folder).ratio()
        if sc > score or (sc == score and best and r["generated"] > best["generated"]):
            best, score = r, sc
    if best is None or score < 0.85:
        return None
    folder = os.path.dirname(best["path"])
    n = sum(1 for r in reports if os.path.dirname(r["path"]) == folder)
    step = (best["manual"].get("step") or "").strip()
    return (f"name close to the folder {os.path.basename(folder)}/ ({n} report(s)"
            + (f", step '{step}'" if step else "") + ") – not verified")


def _name_hint(stem, reports, exclude):
    """For an input no audited step produced: a report whose output name is very close (e.g. an edit in Krita
    saved as <name>_edited). Only a hint – nothing is verified."""
    best, score = None, 0.0
    s = stem.lower()
    for r in reports:
        if r["path"] in exclude:
            continue
        o = _stem((r["file"] or {}).get("name") or "").lower()
        if not o:
            continue
        sc = difflib.SequenceMatcher(None, s, o).ratio()
        if s.startswith(o) or o.startswith(s):
            sc = max(sc, 0.9)
        if sc > score:
            best, score = r, sc
    return best if score >= 0.8 else None


def _short_path(p):
    return "/".join(str(p).split(os.sep)[-2:])


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def frame_range_text(r):
    """'frames 1001–1240 (240)' for a step that wrote an image sequence – '' otherwise."""
    f = r.get("file") or {}
    first, last, n = f.get("first_frame"), f.get("last_frame"), f.get("frame_count")
    if not (first is not None and last is not None):
        nums = []
        for k in r.get("frames") or []:
            m = re.search(r"(\d+)\.[A-Za-z0-9]+$", k)
            if m:
                nums.append(int(m.group(1)))
        if not nums:
            return ""
        first, last, n = min(nums), max(nums), len(nums)
    miss = f.get("missing_frames")
    return f"frames {first}–{last} ({n or last - first + 1})" + (f", {miss} missing" if miss else "")


def asset_name(a):
    """File name of an input; a folder / sequence as 'name/' or its pattern, with frame range / count when known."""
    f = str(a.get("file") or "?")
    if is_folder_input(a) or a.get("role") == "image sequence":
        n = a.get("files_in_folder")
        nm = os.path.basename(f.rstrip("/"))
        nm = nm if "#" in nm else nm + "/"
        if a.get("first_frame") is not None and a.get("last_frame") is not None:
            return f"{nm} frames {a['first_frame']}–{a['last_frame']} ({n})"
        return nm + (f" ({n} frames)" if n else "")
    return os.path.basename(f)


def is_folder_input(a):
    """An input that is a folder / image sequence (a sequence loader) rather than a single file."""
    f = str(a.get("file") or "")
    return (f.endswith("/") or "folder" in str(a.get("sha256") or "") or a.get("files_in_folder") is not None
            or a.get("role") == "image sequence")


def _sequence_names(rep):
    """Sub folders this report wrote sequences into (Save Media: <name>/<name>.####.exr)."""
    return {k.split("/")[0] for k in rep.get("frames") or [] if "/" in k}


def build_chain(start, reports, by_hash, max_depth=50, folder_hashes=None):
    """Walks back from the final report through the inputs. Returns the steps in order (sources first, the final
    step last), each {id, report, inputs:[{asset, from (step id) | None, hint}], depth}.
    folder_hashes(asset, report) -> [sha256 of frames]: for sequence / folder inputs, which are linked through the
    hashes of their frames (the report that wrote the sequence lists every frame)."""
    steps, by_path = [], {}

    def visit(rep, depth):
        if rep["path"] in by_path:
            return by_path[rep["path"]]
        node = {"report": rep, "inputs": [], "depth": depth}
        by_path[rep["path"]] = node  # set before the inputs: protects against loops
        for a in rep["inputs"]:
            h = (a.get("sha256") or "").lower()
            hs = [h] if _HEX64.match(h) else []
            seq = is_folder_input(a)
            if seq:  # an image sequence: linked through the hashes of its frames
                hs += [str(x).lower() for x in (a.get("frame_sha256") or {}).values() if _HEX64.match(str(x).lower())]
                if not hs and folder_hashes:
                    try:
                        hs += [x.lower() for x in folder_hashes(a, rep) or [] if x]
                    except Exception:
                        pass
                h = hs[0] if hs else h
            prod = []
            for x in hs:
                prod = [r for r in by_hash.get(x, []) if r["path"] != rep["path"]]
                if prod:
                    break
            item = {"asset": a, "from": None, "hint": None}
            if seq and not prod:
                name = os.path.basename(str(a.get("file") or "").rstrip("/"))
                cand = [r for r in reports if name and name in _sequence_names(r) and r["path"] != rep["path"]]
                if cand:
                    item["hint"] = (f"sequence folder '{name}' has the name of a sequence written by "
                                    f"{_short_path(cand[0]['path'])} – frames not found to verify")
                    node["inputs"].append(item)
                    continue
            if prod and depth < max_depth:
                item["from"] = visit(prod[0], depth + 1)
            elif not h:
                item["hint"] = "no SHA256 recorded"
            elif rep.get("kind") == EXTERNAL and is_photo_asset(a):
                # the camera original of a declared step: a start point of its own (step "photo")
                key = "photo:" + h
                if key not in by_path:
                    by_path[key] = {"report": photo_report(a, rep, key), "inputs": [], "depth": depth + 1}
                    steps.append(by_path[key])
                item["from"] = by_path[key]
            node["inputs"].append(item)
        steps.append(node)  # post-order: producers before consumers
        return node

    visit(start, 0)
    for i, s in enumerate(steps, 1):
        s["id"] = f"S{i}"
    # number the steps from the start of the chain: producers before the step that uses them, the longest branch
    # first and each branch in one run – S1 is the first link of the longest branch, the final step the last number
    depth = {}

    def dep(n, seen=()):
        k = id(n)
        if k not in depth:
            depth[k] = 1 + max((dep(it["from"], seen + (k,)) for it in n["inputs"]
                                if it["from"] is not None and id(it["from"]) not in seen), default=-1)
        return depth[k]

    order, done = [], set()

    def walk(n):
        if id(n) in done:
            return
        done.add(id(n))
        prods = [it["from"] for it in n["inputs"] if it["from"] is not None]
        for p in sorted(prods, key=lambda p: -dep(p)):
            walk(p)
        order.append(n)

    final = steps[-1]
    walk(final)
    order += [n for n in steps if id(n) not in done]
    steps[:] = [n for n in order if n is not final] + [final]
    for i, s in enumerate(steps, 1):
        s["id"] = f"S{i}"
    # older reports did not record the size of their inputs: an input made by a linked step has exactly the size
    # of that step's output (same SHA256 = same file)
    for s in steps:
        for item in s["inputs"]:
            a, src = item["asset"], item["from"]
            if src is not None and not a.get("width") and (src["report"]["file"] or {}).get("width"):
                a["width"], a["height"] = src["report"]["file"]["width"], src["report"]["file"].get("height")
                a["size_from"] = src["id"]
    in_chain = set(by_path)
    for s in steps:
        for item in s["inputs"]:
            if item["from"] is None and is_external(s["report"]):
                # a source of a declared external step (raw photo, .kra / .psd / .nk project file): documented by
                # that declaration – not an outside source and not a gap
                f = str(item["asset"].get("file") or "").lower()
                item["declared"] = "project file" if f.endswith(PROJECT_EXTS) else "declared source"
                item["hint"] = None
                continue
            if item["from"] is None and item["hint"] is None:
                stem = _stem(item["asset"].get("file"))
                h = _name_hint(stem, reports, in_chain)
                if h:
                    item["hint"] = f"name close to {h['file'].get('name')} ({_rel(h['path'])}) – not verified"
                else:
                    item["hint"] = _folder_hint(stem, reports, in_chain)
    return steps


def alternatives(step_rep, reports, chain_paths):
    """Other outputs of the same step that are not part of this chain (alternatives, or other shots done with
    the same step): same folder and the same 'step' field –
    or, without a step, the same name without counter (audit_001 … audit_007)."""
    st = (step_rep["manual"].get("step") or "").strip()
    folder = os.path.dirname(step_rep["path"])
    key = series_key(step_rep)
    out = []
    for r in reports:
        if r["path"] in chain_paths or os.path.dirname(r["path"]) != folder:
            continue
        rst = (r["manual"].get("step") or "").strip()
        if (st and rst == st) or (not st and not rst and series_key(r) == key):
            out.append(r)
    return sorted(out, key=lambda r: r["generated"])


def _rel(p):
    parts = os.path.normpath(p).split(os.sep)
    return os.sep.join(parts[-3:])


def is_external(r):
    """Declared steps (Audit External Step) and the camera originals they name – no ComfyUI run behind them."""
    return r.get("kind") in (EXTERNAL, PHOTO)


PHOTO = "photo"
RAW_EXTS = (".rw2", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".raf", ".pef", ".srw", ".rwl",
            ".3fr", ".iiq", ".x3f", ".raw")


def is_photo_asset(a):
    return bool(a.get("camera")) or str(a.get("file") or "").lower().endswith(RAW_EXTS)


def photo_report(a, consumer, key):
    """A slim 'report' for the camera original a declared step was made from – shown as a step of its own."""
    cm = consumer.get("manual") or {}
    cam = a.get("camera") or {}
    return {
        "path": key, "kind": PHOTO, "amended": {}, "generated": cam.get("captured") or "", "generator": "",
        "manual": {"step": "photo (camera original)", "workflow_purpose": "original photograph",
                   "external_software": "none (camera original)", "changes_outside_comfyui": "none (camera original)",
                   "reference_image_sources": cm.get("reference_image_sources") or "",
                   **({"author": cm["author"]} if cm.get("author") else {})},
        "context": {}, "file": {"name": os.path.basename(str(a.get("file") or "")), "folder":
                                os.path.dirname(str(a.get("file") or "")), "format": "camera raw"
                                if str(a.get("file") or "").lower().endswith(RAW_EXTS) else "photo",
                                "width": a.get("width"), "height": a.get("height"), "camera": cam,
                                "saved": cam.get("captured") or ""},
        "outputs": [str(a.get("sha256") or "").lower()], "media": [], "frames": [], "inputs": [], "models": [],
        "environment": {}, "packages": [], "api_nodes": [], "external_services": [], "sampling": [],
        "prompt_writers": [], "resolution": {}, "audio": {}, "seed_mode": {}, "workflow_type": [], "triage": {},
        "reproducibility": ""}


def worst_grade(grades):
    g = [x for x in grades if x in GRADE_ORDER]
    return max(g, key=lambda x: GRADE_ORDER[x]) if g else None


def _lic_status(m):
    cu = str(m.get("commercial_use") or "").strip().lower()
    if not m.get("license") or "unknown" in str(m.get("license")).lower():
        return "unknown"
    return cu or "unknown"


# --------------------------------------------------------------------------
# Breakdown
# --------------------------------------------------------------------------
def summarize(steps, reports, final_path):
    chain_paths = {s["report"]["path"] for s in steps}
    models, external, gaps, open_points = {}, [], [], []
    alt_total, alt_seen = 0, set()
    for s in steps:
        r = s["report"]
        # steps with the same step name in the same folder share their alternatives: list them once (first step)
        s["alternatives"] = [a for a in alternatives(r, reports, chain_paths) if a["path"] not in alt_seen]
        alt_seen.update(a["path"] for a in s["alternatives"])
        alt_total += len(s["alternatives"])
        for m in r["models"]:
            key = (m.get("sha256") or m.get("file") or m.get("name") or "?").lower()
            e = models.setdefault(key, dict(m, steps=[]))
            e["steps"].append(s["id"])
        for item in s["inputs"]:
            if item["from"] is None and not item.get("declared"):
                a = item["asset"]
                entry = {"step": s["id"], "file": a.get("file"), "role": a.get("role"), "sha256": a.get("sha256"),
                         "hint": item["hint"]}
                (gaps if item["hint"] else external).append(entry)
        for need in (r["triage"].get("needs") or []):
            if not need.get("covered"):
                open_points.append({"step": s["id"], "need": need.get("need")})
    grades = [s["report"]["triage"].get("grade") for s in steps]
    lic = {"yes": 0, "restricted": 0, "no": 0, "unknown": 0}
    for m in models.values():
        st = _lic_status(m)
        lic[st if st in lic else "unknown"] += 1
    return {
        "final": final_path,
        "steps": len(steps),
        "worst_grade": worst_grade(grades),
        "grades": {s["id"]: s["report"]["triage"].get("grade") for s in steps},
        "models": list(models.values()),
        "licenses": lic,
        "external_inputs": external,
        "gaps": gaps,
        "open_points": open_points,
        "alternatives": alt_total,
        "no_triage": [s["id"] for s in steps if not s["report"]["triage"].get("grade") and not is_external(s["report"])],
        "external_steps": [s["id"] for s in steps if is_external(s["report"])],
    }


def _line(label, value, w=22):
    return f"{label:<{w}}   {value}"


def render_text(steps, summary, folders, n_reports, version=""):
    L = []
    final = steps[-1]["report"] if steps else None
    title = (final["file"].get("name") if final else "") or "?"
    L += [f"AUDIT CHAIN – {title}", ""]
    L.append(_line("Created", datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M") +
                   (f" · Pertineo VFX {version}" if version else "")))
    L.append(_line("Final file", summary["final"]))
    L.append(_line("Searched", f"{n_reports} audit reports in " + ", ".join(folders)))
    L += ["", "SUMMARY ────"]
    wg = summary["worst_grade"]
    L.append(_line("Work steps", f"{summary['steps']} (S1 = earliest, S{summary['steps']} = final)"))
    L.append(_line("Worst triage", (f"{wg} – the strictest step decides for the whole chain" if wg else
                                    "none (no Production Context in any step)")
                   + (f" · without triage: {', '.join(summary['no_triage'])}" if summary["no_triage"] and wg else "")))
    lic = summary["licenses"]
    L.append(_line("Models", f"{len(summary['models'])} · commercial use yes {lic['yes']} · restricted "
                             f"{lic['restricted']} · no {lic['no']} · unknown {lic['unknown']}"))
    L.append(_line("Undocumented", f"{len(summary['external_inputs'])} input(s) of unknown origin – not made in an "
                                   "audited or declared step; rights / origin must be documented"
                   if summary["external_inputs"] else
                   "none – every input comes from an audited ComfyUI step or a declared step"))
    L.append(_line("Gaps", f"{len(summary['gaps'])} input(s) probably from an earlier step, but the file changed in "
                           "between (edit outside ComfyUI?) – link not verified" if summary["gaps"] else "none – every link verified by SHA256"))
    L.append(_line("Variants", f"{summary['alternatives']} other output(s) of these steps in total (alternatives, "
                               "iterations, other shots of a step) – not used for this result"
                   if summary["alternatives"] else "none"))
    if summary["external_steps"]:
        L.append(_line("External steps", f"{', '.join(summary['external_steps'])} – work outside ComfyUI, declared "
                                         "with Pertineo Audit External Step (not recorded by ComfyUI)"))
    if summary["open_points"]:
        L.append(_line("Open points", f"{len({o['need'] for o in summary['open_points']})} (see OPEN POINTS)"))
    miss = [c for c in checklist(steps) if c[2] == "missing"]
    L.append(_line("Disclosure", (f"{len(miss)} of {len(CHECK_ITEMS)} items incomplete – see the .checklist.txt"
                                  if miss else f"all {len(CHECK_ITEMS)} items given or not applicable")))

    L += ["", "STEPS ────"]
    for s in steps:
        r = s["report"]
        man, f = r["manual"], r["file"]
        step = (man.get("step") or "").strip()
        L.append("")
        L.append(f"{s['id']}  {step or '(no step name)'}  –  {f.get('name') or _rel(r['path'])}"
                 + ("  [camera original]" if r.get("kind") == PHOTO else "  [external step]" if is_external(r) else ""))
        if man.get("step_description"):
            L.append(_line("    What", man["step_description"]))
        if r["workflow_type"]:
            L.append(_line("    Workflow", ", ".join(map(str, r["workflow_type"]))))
        size = f"{f.get('width')}x{f.get('height')} · " if f.get("width") and f.get("height") else ""
        L.append(_line("    Output", f"{size}{f.get('format') or ''}" + (f" · {frame_range_text(r)}" if frame_range_text(r)
                                                                         else "")
                       + f" · saved {str(f.get('saved') or r['generated'])[:16]}"))
        L.append(_line("    Report", ("none – file " + _rel(os.path.join(f.get("folder") or "", f.get("name") or ""))
                                          + " · named as source in a declared step") if r.get("kind") == PHOTO else
                       _rel(r["path"]) + (f" (Pertineo VFX {r['generator']})" if r["generator"] else "")
                       + (" · declared with Audit External Step" if is_external(r) else "")
                       + (f" · amended {', '.join(sorted(set(r['amended'].values())))}" if r["amended"] else "")))
        for item in s["inputs"]:
            a = item["asset"]
            src = (f"<- {item['from']['id']}" if item["from"] else
                   f"{item['declared']} (documented by this declaration)" if item.get("declared") else
                   (f"GAP – {item['hint']}" if item["hint"] else "outside source (not audited)"))
            add = f"  (added later, {str(a['added'])[:10]})" if a.get("added") else ""
            L.append(_line("    Input", f"{a.get('role') or 'input'} · {asset_name(a)}  "
                                        f"{src}{add}"))
        for label, value in step_details(r, full=True):
            L.append(_line("    " + label, value))
        t = r["triage"]
        if t.get("grade"):
            L.append(_line("    Triage", f"{t['grade']} {t.get('label', '')} – {t.get('reason', '')}"))
        for label, value, _lv in audit_details(r, full=True):
            L.append(_line("    " + label, value))
        if r["reproducibility"]:
            L.append(_line("    Reproducible", r["reproducibility"]))
        if s.get("alternatives"):
            names = [a["file"].get("name") or _rel(a["path"]) for a in s["alternatives"]]
            shown = ", ".join(names[:6]) + (f" … (+{len(names) - 6})" if len(names) > 6 else "")
            L.append(_line("    Same step", f"{len(names)} not in this chain: {shown}"))

    if summary["models"]:
        L += ["", "MODELS IN THE CHAIN ────"]
        for m in summary["models"]:
            name = m.get("name") or os.path.basename(str(m.get("file") or "?"))
            L.append(f"{name}  ({', '.join(m['steps'])})")
            L.append(_line("    License", f"{_val(m.get('license'))} · commercial use {_val(m.get('commercial_use'))}"))
            L.append(_line("    Source", _val(m.get("source"))))
            if m.get("license_notes"):
                L.append(_line("    Notes", m["license_notes"]))
    if summary["external_inputs"] or summary["gaps"]:
        L += ["", "OUTSIDE SOURCES AND GAPS ────"]
        for e in summary["external_inputs"]:
            L.append(_line(e["step"], f"{e.get('role') or 'input'} · {e.get('file')} – outside source: document "
                                      "origin and rights"))
        for e in summary["gaps"]:
            L.append(_line(e["step"], f"{e.get('role') or 'input'} · {e.get('file')} – {e['hint']}"))
    if summary["open_points"]:
        L += ["", "OPEN POINTS ────"]
        grouped = {}
        for o in summary["open_points"]:
            grouped.setdefault(o["need"], []).append(o["step"])
        for need, sts in grouped.items():
            L.append(_line(", ".join(sts), need))
    L += ["", "TRIAGE GRADES ────"]
    for g, label, text in GRADE_LEGEND + [("-", "NO GRADE", NO_GRADE_TEXT)]:
        L.append(_line(("> " if g == summary["worst_grade"] else "  ") + f"{g}  {label}", text, 32))
    L += ["", GRADE_NOTE]
    L += ["", "NOTE ────",
          "Links between steps are made by SHA256: an input of one step is exactly the file another step wrote.",
          "An edit outside ComfyUI (Krita, Nuke, Photoshop) changes the file – the link breaks there and shows as a",
          "gap or an outside source, unless the edit was declared with Pertineo Audit External Step. Declared",
          "steps and fields added later are statements of the user, not recorded by ComfyUI. Triage and licenses",
          "are copied from the single reports: an indicative first assessment, not an approval and not legal advice."]
    return "\n".join(L)


GRADE_LEGEND = [
    ("A", "LOW-RISK CANDIDATE", "declared as ideation / temporary reference / R&D, not in the final result, no "
                                "trigger, no open point of medium severity or higher"),
    ("B", "REVIEW REQUIRED", "open points remain (unknown licenses, API nodes, undeclared answers, internal asset or "
                             "marketing use, restricted model licenses …)"),
    ("C", "WRITTEN APPROVAL INDICATED", "a typical trigger for written approval (final deliverable / final result, "
                                        "talent likeness or voice, third-party material, trademark clearance, a model "
                                        "license unknown for published material …); covered points count as done, "
                                        "OPEN points need the approval"),
    ("D", "INDETERMINATE", "Production Context connected, but purpose and final result both unknown"),
    ("X", "BLOCKED", "a critical finding: the use as declared is not permitted – e.g. a model whose license does not "
                     "allow commercial use, without a separate license, in published material. An approval cannot "
                     "fix this; only a license from the rights holder or a change (e.g. another model)"),
]
NO_GRADE_TEXT = ("the step has no grade of its own: no Production Context was connected in that run, or it is a "
                 "declared external step / camera original (graded by the steps that use it)")
GRADE_NOTE = ("Each step is graded in its own audit report – from its Production Context, the models and their "
              "licenses and the graph. The chain takes the strictest step for the whole result. No grade = no "
              "Production Context in that step; declared external steps have no grade of their own.")


def render_checklist(steps, final_name, version=""):
    """The disclosure checklist as its own file: what is missing first, with how to close it."""
    items = checklist(steps)
    miss = [c for c in items if c[2] == "missing"]
    L = [f"DISCLOSURE CHECKLIST – {final_name}", ""]
    L.append(_line("Created", datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M") +
                   (f" · Pertineo VFX {version}" if version else "")))
    L.append(_line("Steps", ", ".join(f"{s['id']} {(s['report']['manual'].get('step') or '').strip()}".strip()
                                      for s in steps)))
    L.append(_line("Result", f"{len(miss)} of {len(items)} items incomplete" if miss else
                   f"complete – all {len(items)} items given or not applicable"))
    if miss:
        L += ["", "INCOMPLETE ────"]
        for item, status, _lv in miss:
            L.append(_line(f"[!] {item}", status, 38))
        L += ["", "How to close a gap:",
              "  - purpose, reference / origin sources, external software, changes outside ComfyUI of an existing",
              "    report: Pertineo Audit External Step, mode 'amend report' (the report itself is not changed).",
              "  - a file edited or made outside ComfyUI (photo, Krita / Nuke / Photoshop edit): Pertineo Audit External",
              "    Step, mode 'external step', next to that file – the chain then links it by SHA256.",
              "  - values the workflow did not record (resolution, CFG …) cannot be added later: they stay",
              "    UNKNOWN / NOT VERIFIED for this report; new reports record more."]
    L += ["", "ALL ITEMS ────"]
    for item, status, level in items:
        mark = {"ok": "[x]", "na": "[-]", "missing": "[!]"}[level]
        L.append(_line(f"{mark} {item}", status, 38))
    return "\n".join(L)


def chain_json(steps, summary, folders, n_reports, version=""):
    return {
        "schema": "pertineo-audit-chain/1",
        "generator": {"package": "comfyui-pertineo-vfx", "version": version},
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "searched": {"folders": folders, "reports": n_reports},
        "summary": {k: v for k, v in summary.items() if k != "models"},
        "models": summary["models"],
        "steps": [{
            "id": s["id"], "report": s["report"]["path"], "file": s["report"]["file"],
            "kind": s["report"]["kind"], "amended": s["report"]["amended"],
            "step": s["report"]["manual"].get("step"), "step_description": s["report"]["manual"].get("step_description"),
            "workflow_purpose": s["report"]["manual"].get("workflow_purpose"),
            "workflow_type": s["report"]["workflow_type"], "triage": s["report"]["triage"],
            "outputs_sha256": s["report"]["outputs"],
            "inputs": [{**{k: v for k, v in it["asset"].items() if v is not None},
                        "from_step": it["from"]["id"] if it["from"] else None,
                        "note": it.get("declared") or it["hint"]}
                       for it in s["inputs"]],
            "models": s["report"]["models"],
            "details": [{"label": k, "value": v} for k, v in step_details(s["report"], full=True)],
            "environment": s["report"]["environment"], "node_packages": s["report"]["packages"],
            "sampling": s["report"]["sampling"], "prompt_writers": s["report"]["prompt_writers"],
            "resolution": s["report"]["resolution"],
            "not_used": [a["path"] for a in s.get("alternatives", [])],
        } for s in steps],
        "checklist": [{"item": i, "status": st, "level": lv} for i, st, lv in checklist(steps)],
    }


# --------------------------------------------------------------------------
# Start
# --------------------------------------------------------------------------
def find_start(final, reports, by_hash, sha256_of):
    """final: an audit report (.audit.json / .json) or the final file itself (image, video, audio, a frame of a
    sequence). Returns (report, note) or raises ValueError."""
    real = os.path.realpath(final)
    if is_report_name(real):
        for r in reports:
            if r["path"] == real:
                return r, "report given"
        rep = load_report(real)
        if rep is None:
            raise ValueError(f"{final} is not a Pertineo audit report.")
        return _slim(rep, real), "report given (outside the searched folders)"
    if os.path.isdir(real) or "#" in os.path.basename(real) or re.search(r"%0?\d*d", os.path.basename(real)):
        # an image sequence: its report lies next to the folder (<folder>.audit.json) – else find it by a frame
        folder = real if os.path.isdir(real) else os.path.dirname(real)
        for cand in (folder + ".audit.json", folder + ".json"):
            if os.path.isfile(cand) and load_report(cand) is not None:
                return next((r for r in reports if r["path"] == os.path.realpath(cand)),
                            _slim(load_report(cand), cand)), "report next to the sequence"
        frames = sorted(f for f in os.listdir(folder) if re.search(r"\d+\.(png|exr|tiff?|jpe?g|dpx|webp)$", f, re.I))
        for f in frames[len(frames) // 2:len(frames) // 2 + 1] + frames[:1]:
            h = sha256_of(os.path.join(folder, f)).lower()
            if h in by_hash:
                return by_hash[h][0], "found by the SHA256 of a frame"
        raise ValueError(f"No audit report wrote the sequence {os.path.basename(folder)} – searched {len(reports)} "
                         "reports. Add the folder of its report to search_folders.")
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", real)
    for cand in (stem + ".audit.json", stem + ".json"):
        if os.path.isfile(cand):
            rep = load_report(cand)
            if rep is not None:
                return next((r for r in reports if r["path"] == os.path.realpath(cand)), _slim(rep, cand)), \
                    "report next to the file"
    h = sha256_of(real).lower()
    if h in by_hash:
        return by_hash[h][0], "found by the file's SHA256"
    raise ValueError(f"No audit report wrote {os.path.basename(final)} (SHA256 {h[:12]}…) – searched "
                     f"{len(reports)} reports. Add the folder of its report to search_folders.")


# --------------------------------------------------------------------------
# Disclosure details per step and the checklist over the whole chain
# --------------------------------------------------------------------------
UNK = "UNKNOWN / NOT VERIFIED"
ROLE_LABELS = {"checkpoint": "checkpoint", "diffusion_model": "foundation model", "lora": "LoRA",
               "controlnet": "ControlNet", "ip_adapter": "IP-Adapter", "clip_vision": "CLIP vision",
               "upscaler": "upscaler", "vlm": "VLM", "text_encoder": "text encoder", "vae": "VAE",
               "segmentation_model": "segmentation", "style_model": "style model"}
_EMPTY = {"", "none", "-", "n/a", "na", "unknown"}


def _val(x):
    """A declared / recorded value, or UNKNOWN / NOT VERIFIED."""
    t = str(x).strip() if x is not None else ""
    return t if t and "not verified" not in t.lower() else UNK


def _declared(x):
    """For manual fields: 'none' is an answer, empty is not."""
    t = str(x or "").strip()
    return t if t else UNK


def _is_video(r):
    wt = " ".join(map(str, r.get("workflow_type") or [])).lower()
    return ("video" in wt or any("video" in str(m.get("kind")) for m in r.get("media") or [])
            or bool(r.get("probe")) or _seq_count(r) > 1)


def _seq_count(r):
    """Number of frames of an image sequence the step wrote (0 = no sequence)."""
    f = r.get("file") or {}
    if f.get("frame_count"):
        return int(f["frame_count"])
    return len(r.get("frames") or [])


def _fps_value(x):
    try:
        v = float(str(x).strip().replace(",", "."))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _timing(r):
    """(frames, fps, seconds) of a video / sound output, or None."""
    for m in r.get("media") or []:
        if "video" in str(m.get("kind")) and m.get("frames") and m.get("fps"):
            return int(m["frames"]), float(m["fps"]), m["frames"] / float(m["fps"])
    p = r.get("probe") or {}
    if p.get("frames") and p.get("fps"):
        return int(p["frames"]), float(p["fps"]), float(p["seconds"])
    f = r.get("file") or {}
    n = _seq_count(r)
    fps = _fps_value((r.get("manual") or {}).get("fps")) or _fps_value(f.get("fps"))
    if n > 1 and fps:
        return n, fps, n / fps
    a = r.get("audio") or {}
    if f.get("fps") and a.get("duration_sec"):
        return None, float(f["fps"]), float(a["duration_sec"])
    return None


def timing_text(r):
    """'241 frames @ 24 fps = 10.04 s' with where the values come from – UNKNOWN if a video / sequence has none,
    '' for a still image."""
    if not _is_video(r):
        return ""
    t = _timing(r)
    if not t:
        n = _seq_count(r)
        return (f"{n} frames · frame rate {UNK} (an image sequence carries none – add it with Audit External Step, "
                f"field fps)") if n > 1 else UNK
    txt = f"{t[0]} frames @ {t[1]:g} fps = {t[2]:.2f} s" if t[0] else f"{t[2]:.2f} s @ {t[1]:g} fps"
    if (r.get("probe") or {}).get("fps") and not any("video" in str(m.get("kind")) and m.get("fps")
                                                      for m in r.get("media") or []):
        txt += " (read from the file)"
    elif _fps_value((r.get("manual") or {}).get("fps")) and not _fps_value((r.get("file") or {}).get("fps")):
        am = (r.get("amended") or {}).get("fps")
        txt += " (fps declared" + (f", added later {am})" if am else ")")
    return txt


def ai_text(r):
    """Declaration whether an AI outside the ComfyUI run helped (e.g. Claude / ChatGPT writing the prompts)."""
    man, amended = r.get("manual") or {}, r.get("amended") or {}
    used = str(man.get("external_ai") or "").strip().lower()
    tools = str(man.get("external_ai_tools") or "").strip()
    later = sorted({amended[k] for k in ("external_ai", "external_ai_tools") if k in amended})
    tail = f" (declared, added later {later[-1]})" if later else " (declared)"
    if used == "yes":
        return "YES – " + (tools or f"which AI: {UNK}") + tail
    if used == "no":
        return "no external AI used" + (f" – {tools}" if tools else "") + tail
    if tools:
        return tools + tail
    return UNK


def ai_relevant(r):
    """External-AI line: always for steps with prompts, else only when something was declared."""
    man = r.get("manual") or {}
    return bool(r.get("sampling") or r.get("prompt_writers") or str(man.get("external_ai") or "").strip()
                or str(man.get("external_ai_tools") or "").strip())


def _short(text, n=400):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n].rstrip() + " …"


def model_line(m):
    role = ROLE_LABELS.get(m.get("role"), m.get("role") or "model")
    name = m.get("name") or os.path.basename(str(m.get("file") or "")) or UNK
    st = m.get("strength_model", m.get("strength"))
    extra = f" · strength {st}" if st not in (None, "") else ""
    ver = f" · version {m['version']}" if m.get("version") else ""
    base = f" · base model {m['base_model']}" if m.get("base_model") else ""
    return (f"{role}: {name}{extra} · file {_val(m.get('file'))} · sha256 {_val(m.get('sha256'))[:16]}{ver}{base} · "
            f"source {_val(m.get('source'))} · "
            f"license {_val(m.get('license'))} · commercial use {_val(m.get('commercial_use'))}")


def sampler_line(x, seed_mode=None):
    p = x["params"]
    cfg = p.get("cfg", p.get("guidance"))
    extra = {k: v for k, v in p.items() if k not in ("seed", "sampler", "scheduler", "steps", "cfg", "guidance",
                                                      "denoise", "noise_seed")}
    seed = p.get("seed", p.get("noise_seed"))
    mode = f" ({seed_mode})" if seed_mode else ""
    return (f"{x['node']} · seed {_val(seed)}{mode if seed is not None else ''} · sampler {_val(p.get('sampler'))} / "
            f"scheduler {_val(p.get('scheduler'))} · steps {_val(p.get('steps'))} · "
            f"{'cfg' if 'cfg' in p else 'guidance'} {_val(cfg)} · denoise {_val(p.get('denoise'))}"
            + "".join(f" · {k} {v}" for k, v in extra.items()))


NA_GEN = "not applicable (no generation in this step)"


def audio_out(r):
    """The written sound file of a step (Save Media audio output) – None if it wrote none."""
    return next((m for m in r.get("media") or [] if str(m.get("kind")) == "audio" and m.get("sample_rate")), None)


def audio_text(m):
    dur = m.get("duration_sec")
    return " · ".join(x for x in (str(m.get("format") or "") or None, f"{m['sample_rate']} Hz",
                                   f"{m['channels']} ch" if m.get("channels") else None,
                                   f"{dur:.2f} s" if isinstance(dur, (int, float)) else None) if x)


ROUTES = {"legal": "Legal / rights", "talent": "Talent & consent", "data": "Data protection / IT security",
          "production": "Production / studio approval", "vfx": "VFX supervision / QC", "docs": "Documentation"}
_CONTEXT_SHORT = ("purpose", "in_final_cut", "story_critical", "execution_environment", "studio_approval")


def breakdown_details(r, licenses=True):
    """(label, value) rows of the overview: the main models with their commercial-use status, the declaration of
    an external step – no prompts, settings, hashes or environment (those are in the audit)."""
    rows = []
    man = r["manual"]
    if is_external(r):
        rows.append(("Kind", "camera original" if r.get("kind") == PHOTO else "external step – work outside ComfyUI"))
        for k, label in (("external_software", "Software"), ("changes_outside_comfyui", "Changes"),
                         ("reference_image_sources", "Origin / rights")):
            if str(man.get(k) or "").strip():
                rows.append((label, man[k]))
        if (r["file"] or {}).get("camera"):
            rows.append(("Camera", camera_text(r["file"]["camera"])))
        if r.get("kind") != PHOTO and ai_relevant(r):
            rows.append(("External AI", ai_text(r)))
        if timing_text(r):
            rows.append(("Duration", timing_text(r)))
        return rows
    main = [m for m in r["models"] if m.get("role") not in ("vae", "text_encoder", "clip_vision")] or r["models"]
    for m in main:
        role = ROLE_LABELS.get(m.get("role"), m.get("role") or "model")
        nm = m.get("name") or os.path.basename(str(m.get("file") or "")) or UNK
        rows.append(("Model", f"{role}: {nm}" + (f" – commercial use {_val(m.get('commercial_use'))}"
                                                  if licenses else "")))
    other = len(r["models"]) - len(main)
    if other > 0:
        rows.append(("Model", f"+ {other} supporting model(s) (VAE, text encoder …) – see the audit"))
    if r["sampling"]:
        rows.append(("Generation", ", ".join(x["node"].split(" #")[0] for x in r["sampling"])))
    if ai_relevant(r):
        rows.append(("External AI", ai_text(r)))
    if timing_text(r):
        rows.append(("Duration", timing_text(r)))
    out = man.get("changes_outside_comfyui") or man.get("planned_work_outside")
    if str(out or "").strip() and str(out).strip().lower() not in ("none", "-"):
        rows.append(("Outside ComfyUI", out))
    return rows


def audit_details(r, full=False):
    """(label, value, level) rows of the assessment of one step – production context, approvals needed, findings,
    guiding principles. level: 'bad' / 'warn' / None (for colours). short = findings of medium severity and up and
    the main context fields; full = everything."""
    rows = []
    if is_external(r):
        return [("Assessment", "none – declared / camera original; assessed through the steps that use it", None)]
    ctx = {k: v for k, v in (r.get("context") or {}).items() if str(v or "").strip()}
    if not ctx and not r["triage"].get("grade"):
        return [("Assessment", "not assessed – no Production Context connected in this run", "warn")]
    keys = list(ctx) if full else [k for k in _CONTEXT_SHORT if k in ctx]
    if keys:
        rows.append(("Context", " · ".join(f"{k.replace('_', ' ')}: {ctx[k]}" for k in keys), None))
    for n in r["triage"].get("needs") or []:
        cov = n.get("covered")
        rows.append(("Approval", f"{n.get('need')} – " + (f"covered ({cov})" if cov else "OPEN"),
                     None if cov else "warn"))
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    fs = sorted(r.get("findings") or [], key=lambda x: sev_order.get(str(x.get("severity")), 5))
    shown = fs if full else [x for x in fs if str(x.get("severity")) in ("critical", "high", "medium")]
    for x in shown:
        sev = str(x.get("severity") or "")
        route = ROUTES.get(str(x.get("route")), x.get("route") or "")
        rows.append(("Finding", f"[{sev}] {x.get('text')}" + (f" ({route} · {x.get('evidence')})" if route else ""),
                     "bad" if sev == "critical" else "warn" if sev == "high" else None))
    if len(shown) < len(fs):
        rows.append(("Finding", f"+ {len(fs) - len(shown)} of low severity – in the full version and the .txt", None))
    for pr in r.get("principles") or []:
        st = str(pr.get("status") or "")
        rows.append(("Principle", f"{pr.get('id')} {pr.get('principle')}: {st.upper()} – {pr.get('basis')}",
                     "warn" if st == "unknown" else None))
    return rows


def repro_text(r):
    """Reproducibility of a step – never empty: not applicable for work outside ComfyUI, else UNKNOWN."""
    if r.get("kind") == PHOTO:
        return "not applicable (camera original)"
    if is_external(r):
        return "not applicable (made outside ComfyUI)"
    return str(r.get("reproducibility") or "").strip() or UNK


def camera_text(c):
    """Camera data of a photo (from its EXIF, recorded by Audit External Step) as one line."""
    expo = " ".join(v for v in (c.get("exposure"), c.get("f_number"), c.get("iso")) if v)
    return " · ".join(x for x in (c.get("camera"), c.get("lens"),
                                   f"captured {c['captured']}" if c.get("captured") else None, expo or None) if x)


def _gen_text(g):
    t = _gen_text_raw(g)
    if t and g.get("width") and g.get("from_workflow"):
        t += f" (from {g['from_workflow'] if isinstance(g['from_workflow'], str) else 'the recorded workflow'})"
    return t


def _gen_text_raw(g):
    if g.get("audio"):
        return f"audio latent {g['audio_seconds']:g} s" if isinstance(g.get("audio_seconds"), (int, float)) else None
    if g.get("width"):
        return f"{g.get('width')}x{g.get('height')}" + (f"x{g['length']}" if g.get("length") else "")
    if g.get("image_based"):
        tile = (f", tiles {g['tile_width']}x{g['tile_height']}" if g.get("tile_width") and g.get("tile_height")
                else "")
        res = f" {g['result_width']}x{g['result_height']}" if g.get("result_width") else ""
        fw = g.get("from_workflow")
        return f"tiled on the image{res}{tile}" + (
            f" (from {fw if isinstance(fw, str) else 'the recorded workflow'})" if fw else "")
    return None


def step_details(r, full=False):
    """(label, value) rows with everything a disclosure asks for – shared by .txt (full = complete prompts) and
    PDF (prompts shortened; the complete text is in the .txt / .json)."""
    lim = 10 ** 9 if full else 400
    man, env = r["manual"], r["environment"]
    amended = r.get("amended") or {}

    def decl(*keys):
        k = next((k for k in keys if str(man.get(k) or "").strip()), keys[0])
        v = _declared(man.get(k))
        return v + (f" (added later, {amended[k]})" if k in amended and v != UNK else "")

    f = r["file"]
    final_res = f"{f.get('width')}x{f.get('height')}" if f.get("width") else UNK
    src = [f"{a.get('width')}x{a.get('height')}" + (f" (size of the {a['size_from']} output)" if a.get("size_from")
                                                     else "") for a in r["inputs"] if a.get("width")]
    if is_external(r):
        rows = [("Purpose", decl("workflow_purpose")),
                ("Kind", ("camera original – the photo a declared step was made from (not recorded by ComfyUI)"
                          if r.get("kind") == PHOTO else
                          "external step – work outside ComfyUI, declared"
                          + (f" by {man['author']}" if man.get("author") else "")
                          + f" on {str(r['generated'])[:10]} (not recorded by ComfyUI)")),
                ("Software", decl("external_software")),
                ("Changes", decl("changes_outside_comfyui")),
                ("Origin / rights", decl("reference_image_sources")),
                *([("External AI", ai_text(r))] if r.get("kind") != PHOTO and ai_relevant(r) else []),
                *([("Duration", timing_text(r))] if timing_text(r) else []),
                *([("Camera", camera_text(f["camera"]))] if f.get("camera") and (
                    r.get("kind") == PHOTO or not any(is_photo_asset(x) for x in r["inputs"])) else []),
                ("Resolution", (f"{final_res} ({f.get('format')})" if r.get("kind") == PHOTO else
                                f"source {', '.join(src) if src else ('none (no source file)' if not r['inputs'] else UNK)}"
                                f" · final {final_res}"))]
        if man.get("notes"):
            rows.append(("Notes", man["notes"]))
        return rows
    rows = [("Purpose", decl("workflow_purpose"))]
    if env:
        rows.append(("Environment", f"ComfyUI {_val(env.get('comfyui'))}"
                     + (f" ({env['comfyui_commit']})" if env.get("comfyui_commit") else "")
                     + f" · frontend {_val(env.get('frontend'))} · Python {_val(env.get('python'))} · torch "
                       f"{_val(env.get('torch'))} · {_val(env.get('gpu'))} · {_val(env.get('os'))} · Pertineo VFX "
                       f"{_val(env.get('pertineo_vfx'))}"))
    else:
        rows.append(("Environment", UNK))
    pk = [f"{x['package']} {x.get('version') or ''}".strip() for x in r["packages"] if x.get("package") != "comfy-core"]
    rows.append(("Node packages", ", ".join(pk) if pk else "comfy-core only"))
    apis = [str(a.get("node_type") or a) for a in r["api_nodes"]] + [str(a.get("service") or a.get("name") or a)
                                                                     for a in r["external_services"]]
    rows.append(("APIs", ", ".join(apis) if apis else "none (local)"))
    for m in r["models"]:
        rows.append(("Model", model_line(m)))
    if not r["models"]:
        rows.append(("Model", "none recorded"))
    roles = {m.get("role") for m in r["models"]}
    wt = " ".join(map(str, r["workflow_type"])).lower()
    unused = [label for label, used in (
        ("LoRA", "lora" in roles),
        ("ControlNet", "controlnet" in roles),
        ("IP-Adapter", bool(roles & {"ip_adapter", "clip_vision"}) or "ip-adapter" in wt),
        ("upscaler", "upscaler" in roles),
        ("VLM / prompt writer", "vlm" in roles or bool(r["prompt_writers"]))) if not used]
    if unused:
        rows.append(("Not used", ", ".join(unused) + " – none in this step"))
    rows.append(("Reference sources", decl("reference_image_sources")))
    sm = (r.get("seed_mode") or {}).get("mode")
    for x in r["sampling"]:
        rows.append(("Sampler", sampler_line(x, sm)))
        pos = [t for t in x["prompts"].get("positive", []) if t.strip()]
        neg = x["prompts"].get("negative", [])
        rows.append(("Prompt", _short(" | ".join(pos), lim) if pos else UNK))
        if "negative" in x["prompts"]:
            negt = [t for t in neg if t.strip()]
            rows.append(("Negative", _short(" | ".join(negt), lim) if negt else "(empty)"))
    if not r["sampling"]:
        rows.append(("Sampler", NA_GEN + " – no seed, prompt, CFG or generation resolution"))
    for w in r["prompt_writers"]:
        rows.append(("Prompt writer", f"{w['node']} -> {', '.join(w['feeds']) or '?'} (prompt written by an LLM / "
                                      "VLM in the run – recorded above)"))
    if ai_relevant(r):
        rows.append(("External AI", ai_text(r)))
    res = r["resolution"] or {}
    gen = [t for t in (_gen_text(g) for g in res.get("generation") or []) if t]
    gen_s = ", ".join(gen) if gen else (UNK if r["sampling"] else "not applicable")
    aud = audio_out(r)
    if aud and not f.get("width"):
        # a sound: sample rate, bit depth / format and channels instead of a picture size
        rows.append(("Audio", audio_text(aud) + (f" · generation {gen_s}" if r["sampling"] else "")))
    else:
        rows.append(("Resolution", f"source {', '.join(src) if src else ('none (no image input)' if not r['inputs'] else UNK)}"
                                   f" · generation {gen_s} · final {final_res}"))
    if _is_video(r):
        rows.append(("Duration", timing_text(r)))
    elif (r.get("audio") or {}).get("duration_sec") and not aud:
        t = _timing(r)
        rows.append(("Duration", f"{t[2]:.2f} s @ {t[1]:g} fps" if t else UNK))
    rows.append(("External software", decl("external_software")))
    rows.append(("Outside ComfyUI", decl("changes_outside_comfyui", "planned_work_outside")))
    return rows


CHECK_ITEMS = [
    "workflow purpose", "ComfyUI version / environment", "checkpoint / foundation model", "exact model / version",
    "source", "licence", "known commercial-use status", "LoRAs", "ControlNet models", "IP-Adapters", "upscalers",
    "VLMs", "video models", "APIs", "custom nodes", "reference-image sources", "prompts",
    "negative prompts where relevant", "seed", "sampler / scheduler", "steps", "CFG or equivalent", "denoise values",
    "source resolution", "generation resolution", "final resolution", "video duration", "frame rate",
    "external software used", "changes performed outside ComfyUI", "external AI (e.g. prompts)"]


def checklist(steps):
    """One line per disclosure item: (item, status text, level) – level 'ok', 'na' or 'missing'."""
    out = []

    def per_step(test):
        """test(r) -> True (given) / False (missing) / None (not relevant)."""
        given, missing = [], []
        for s in steps:
            t = test(s["report"])
            if t is True:
                given.append(s["id"])
            elif t is False:
                missing.append(s["id"])
        return given, missing

    def add(item, test, used_label=None, external=None):
        """external: test for declared external steps (None = not applicable there)."""
        def t(r):
            if is_external(r):
                return external(r) if external else None
            return test(r)
        given, missing = per_step(t)
        na = [s["id"] for s in steps if s["id"] not in given and s["id"] not in missing]
        na_txt = f" · not applicable in {', '.join(na)}" if na and (given or missing) else ""
        if missing:
            out.append((item, f"{UNK} in {', '.join(missing)}" + (f" · given in {', '.join(given)}" if given else "")
                        + na_txt, "missing"))
        elif given:
            out.append((item, ("all steps" if len(given) == len(steps) else ", ".join(given))
                        + (f" ({used_label})" if used_label else "") + na_txt, "ok"))
        else:
            out.append((item, "not relevant / not used in this chain", "na"))

    def models_of(r, roles):
        return [m for m in r["models"] if m.get("role") in roles]

    def uses(roles):
        def t(r):
            return True if models_of(r, roles) else None
        return t

    has = lambda x: bool(str(x or "").strip())  # noqa: E731
    decl = lambda r, *k: any(has(r["manual"].get(x)) for x in k)  # noqa: E731
    add("workflow purpose", lambda r: has(r["manual"].get("workflow_purpose")),
        external=lambda r: decl(r, "workflow_purpose"))
    add("ComfyUI version / environment", lambda r: bool(r["environment"].get("comfyui")))
    add("checkpoint / foundation model",
        lambda r: (True if models_of(r, ("checkpoint", "diffusion_model")) else False) if r["sampling"] else None)
    add("exact model / version", lambda r: all(has(m.get("file")) and has(m.get("sha256")) and
                                               "not" not in str(m.get("sha256")) for m in r["models"])
        if r["models"] else None)
    add("source", lambda r: all(has(m.get("source")) for m in r["models"]) if r["models"] else None)
    add("licence", lambda r: all(_val(m.get("license")) != UNK for m in r["models"]) if r["models"] else None)
    add("known commercial-use status", lambda r: all(_val(m.get("commercial_use")) != UNK and
                                                     str(m.get("commercial_use")).lower() != "unclear"
                                                     for m in r["models"]) if r["models"] else None)
    add("LoRAs", uses(("lora",)), "listed with strength")
    add("ControlNet models", uses(("controlnet",)))
    add("IP-Adapters", lambda r: True if (models_of(r, ("ip_adapter", "clip_vision")) or any(
        "ip-adapter" in str(w).lower() for w in r["workflow_type"])) else None, "reference images in the conditioning")
    add("upscalers", uses(("upscaler",)))
    add("VLMs", lambda r: True if (models_of(r, ("vlm",)) or r["prompt_writers"]) else None)
    add("video models", lambda r: (True if models_of(r, ("diffusion_model", "checkpoint")) else False)
        if _is_video(r) and r["sampling"] else None)
    add("APIs", lambda r: True, "none = local only")
    add("custom nodes", lambda r: bool(r["packages"]))
    add("reference-image sources", lambda r: has(r["manual"].get("reference_image_sources")),
        external=lambda r: decl(r, "reference_image_sources"))
    add("prompts", lambda r: any(t.strip() for x in r["sampling"] for t in x["prompts"].get("positive", []))
        if r["sampling"] else None)
    add("negative prompts where relevant", lambda r: True if any("negative" in x["prompts"] for x in r["sampling"])
        else None, "empty where not used")
    for item, keys in (("seed", ("seed", "noise_seed")), ("sampler / scheduler", ("sampler",)), ("steps", ("steps",)),
                       ("CFG or equivalent", ("cfg", "guidance")), ("denoise values", ("denoise",))):
        add(item, lambda r, keys=keys: all(any(x["params"].get(k) is not None for k in keys) for x in r["sampling"])
            if r["sampling"] else None)
    src_res = lambda r: all(a.get("width") for a in r["inputs"]  # noqa: E731
                            if str(a.get("role")) in ("image", "video", "mask")) if r["inputs"] else None
    add("source resolution", src_res, external=src_res)
    add("generation resolution", lambda r: any(_gen_text(g) for g in (r["resolution"].get("generation") or []))
        if r["sampling"] else None)
    fin_res = lambda r: bool(r["file"].get("width")) if r["file"].get("format") not in (None, "audio") else None  # noqa: E731
    add("final resolution", fin_res, external=fin_res)
    vdur = lambda r: (_timing(r) is not None) if _is_video(r) else None  # noqa: E731
    vfps = lambda r: bool((_timing(r) or (None, None))[1]) if _is_video(r) else None  # noqa: E731
    add("video duration", vdur, external=vdur)
    add("frame rate", vfps, external=vfps)
    add("external software used", lambda r: decl(r, "external_software"),
        external=lambda r: decl(r, "external_software"))
    add("changes performed outside ComfyUI", lambda r: decl(r, "changes_outside_comfyui", "planned_work_outside"),
        external=lambda r: decl(r, "changes_outside_comfyui"))
    ai = lambda r: str(r["manual"].get("external_ai") or "").strip().lower() in ("yes", "no")  # noqa: E731
    add("external AI (e.g. prompts)", lambda r: ai(r) if ai_relevant(r) else None, "declared yes / no",
        external=lambda r: True if ai(r) else None)
    return out


# --------------------------------------------------------------------------
# Graph layout for the PDF: sources at the top, the final step at the bottom (like a node graph in Nuke) –
# parallel branches side by side, a step with several linked inputs is where they merge.
# --------------------------------------------------------------------------
def graph_layout(steps):
    """Returns (nodes, edges). nodes: {key: {"kind": "step" | "outside" | "gap", "step": step or None,
    "asset": asset or None, "hint": str, "layer": int, "col": int}}; edges: [(from_key, to_key)].
    Layer 0 = top; columns from 0 (the final step's trunk) to the right."""
    nodes, edges, prods = {}, [], {}
    for s in steps:
        nodes[s["id"]] = {"kind": "step", "step": s, "asset": None, "hint": None}
        prods[s["id"]] = []
        for i, it in enumerate(s["inputs"]):
            if it["from"] is not None:
                k = it["from"]["id"]
            else:
                k = f"{s['id']}:{i}"
                nodes[k] = {"kind": "declared" if it.get("declared") else "gap" if it["hint"] else "outside",
                            "step": None, "asset": it["asset"], "hint": it.get("declared") or it["hint"]}
                prods[k] = []
            if k not in prods[s["id"]]:
                prods[s["id"]].append(k)
                edges.append((k, s["id"]))
    cons = {k: [] for k in nodes}
    for a, b in edges:
        cons[a].append(b)
    # longest path from the sources, then every node as low as its consumers allow (sources sit next to the
    # step that uses them instead of all at the top)
    depth = {}

    def d(k, seen=()):
        if k in depth:
            return depth[k]
        if k in seen:
            return 0
        depth[k] = 1 + max((d(p, seen + (k,)) for p in prods[k]), default=-1)
        return depth[k]

    for k in nodes:
        d(k)
    order = sorted(nodes, key=lambda k: -depth[k])
    layer = dict(depth)
    for k in order:
        if cons[k]:
            layer[k] = min(layer[c] for c in cons[k]) - 1
    lo = min(layer.values()) if layer else 0
    for k in nodes:
        nodes[k]["layer"] = layer[k] - lo
    # columns: the first input continues the column of its consumer (the trunk), every further input opens a
    # new column to the right
    final = steps[-1]["id"]
    col, nxt = {}, [0]

    def place(k, c):
        if k in col:
            return
        col[k] = c
        # the longest branch continues straight up (the trunk), every other input opens a column to the right –
        # keeps long lines out of the way, like the B pipe in Nuke
        trunk = True
        for p in sorted(prods[k], key=lambda p: -depth.get(p, 0)):
            if p in col:
                continue
            if trunk:
                place(p, c)
                trunk = False
            else:
                nxt[0] += 1
                place(p, nxt[0])

    place(final, 0)
    for k in nodes:  # anything not reachable from the final step (should not happen)
        if k not in col:
            nxt[0] += 1
            col[k] = nxt[0]
    for k in nodes:
        nodes[k]["col"] = col[k]
    return nodes, edges


# --------------------------------------------------------------------------
# Audit by topic: the whole chain gathered per subject (models, inputs, findings …) instead of per step
# --------------------------------------------------------------------------
CONTEXT_LABELS = [
    ("production_title", "Production"), ("project_episode", "Project / episode"), ("shot_asset", "Shot / asset"),
    ("vendor_department", "Vendor / department"), ("purpose", "Purpose"), ("in_final_cut", "In the final result"),
    ("story_critical", "Story-critical"), ("execution_environment", "Execution environment"),
    ("unreleased_material", "Unreleased material"), ("personal_data", "Personal data"),
    ("talent_likeness_or_voice", "Talent likeness or voice"), ("third_party_copyright", "Third-party copyright"),
    ("trademarks_brands", "Trademarks / brands"), ("union_covered_work", "Union-covered work"),
    ("external_tools_train", "External tools train on data"), ("input_rights_documented", "Input rights documented"),
    ("talent_consent", "Talent consent"), ("union_approval", "Union approval"),
    ("studio_approval", "Studio / client approval"), ("legal_approval", "Legal approval")]
PEOPLE_LABELS = [("audit_requester", "Audit requested by"), ("creative_reviewer", "Creative reviewer"),
                 ("technical_reviewer", "Technical reviewer"), ("rights_reviewer", "Rights reviewer")]
SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def id_range(ids):
    """['S6','S7','S8','S10'] -> 'S6–S8, S10'."""
    nums = sorted(int(i[1:]) for i in ids if str(i)[1:].isdigit())
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"S{nums[i]}" if i == j else f"S{nums[i]}–S{nums[j]}" if j > i + 1 else f"S{nums[i]}, S{nums[j]}")
        i = j + 1
    return ", ".join(out)


def context_groups(steps):
    """[(step ids, context dict)] – steps with the same production context share one entry."""
    groups = []
    for s in steps:
        ctx = {k: str(v).strip() for k, v in (s["report"].get("context") or {}).items() if str(v or "").strip()}
        if not ctx:
            continue
        for g in groups:
            if g[1] == ctx:
                g[0].append(s["id"])
                break
        else:
            groups.append(([s["id"]], ctx))
    return groups


def people(steps):
    """Who requested / reviewed – from the production contexts, each name once per role."""
    out = []
    for key, label in PEOPLE_LABELS:
        names = []
        for s in steps:
            v = str((s["report"].get("context") or {}).get(key) or "").strip()
            if v and v not in names:
                names.append(v)
        if names:
            out.append((label, ", ".join(names)))
    a = []
    for s in steps:
        v = str(s["report"]["manual"].get("author") or "").strip()
        if v and v not in a:
            a.append(v)
    if a:
        out.insert(0, ("Author", ", ".join(a)))
    return out


def merged_findings(steps):
    """Findings of all steps, the same text once with all its steps; sorted by severity, then department."""
    by = {}
    for s in steps:
        for f in s["report"].get("findings") or []:
            k = (f.get("text"), f.get("severity"), f.get("route"))
            e = by.setdefault(k, dict(f, steps=[]))
            e["steps"].append(s["id"])
    return sorted(by.values(), key=lambda f: (SEV_ORDER.get(str(f.get("severity")), 9),
                                              ROUTES.get(str(f.get("route")), str(f.get("route") or "")),
                                              str(f.get("text"))))


def merged_approvals(steps):
    """need -> {'covered': {by: [steps]}, 'open': [steps]}."""
    out = {}
    for s in steps:
        for n in s["report"]["triage"].get("needs") or []:
            e = out.setdefault(n.get("need"), {"covered": {}, "open": []})
            if n.get("covered"):
                e["covered"].setdefault(n["covered"], []).append(s["id"])
            else:
                e["open"].append(s["id"])
    return out


def principle_matrix(steps):
    """(principle rows [(id, text)], step ids with principles, {(pid, step): (status, basis)})."""
    ps, cols, cell = [], [], {}
    for s in steps:
        pr = s["report"].get("principles") or []
        if pr:
            cols.append(s["id"])
        for p in pr:
            if (p.get("id"), p.get("principle")) not in ps:
                ps.append((p.get("id"), p.get("principle")))
            cell[(p.get("id"), s["id"])] = (str(p.get("status") or ""), p.get("basis"))
    return sorted(ps, key=lambda x: str(x[0])), cols, cell


def merged_environment(steps):
    """[(text, steps)] for ComfyUI / Python / torch / GPU / OS, and [(package version, steps)], [(api, steps)]."""
    env, pk, api = {}, {}, {}
    for s in steps:
        r = s["report"]
        e = r.get("environment") or {}
        if e:
            t = (f"ComfyUI {_val(e.get('comfyui'))}" + (f" ({e['comfyui_commit']})" if e.get("comfyui_commit") else "")
                 + f" · frontend {_val(e.get('frontend'))} · Python {_val(e.get('python'))} · torch "
                   f"{_val(e.get('torch'))} · {_val(e.get('gpu'))} · {_val(e.get('os'))}")
            env.setdefault(t, []).append(s["id"])
        for x in r.get("packages") or []:
            if x.get("package") and x.get("package") != "comfy-core":
                pk.setdefault(f"{x['package']} {x.get('version') or ''}".strip(), []).append(s["id"])
        for a in list(r.get("api_nodes") or []) + list(r.get("external_services") or []):
            t = str(a.get("node_type") or a.get("service") or a.get("name") or a) if isinstance(a, dict) else str(a)
            api.setdefault(t, []).append(s["id"])
    return list(env.items()), sorted(pk.items()), list(api.items())
