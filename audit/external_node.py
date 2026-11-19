# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo Audit External Step – declares work done outside ComfyUI (a photo, an edit in Krita / Nuke /
Photoshop) as an audit report next to the file, so the Audit Chain links it by SHA256; or adds missing declarations
to an existing report as a separate amendment file (the report itself is never changed)."""

import datetime
import json
import os
import re
import subprocess
import traceback

from ..common import LOG
from ..image_io.readwrite import clean_path
from . import chain as CH
from . import hashing
from .collect import own_version

CATEGORY = "Pertineo/Audit"
MODES = ["external step", "amend report"]
_LABELS = {"step": "Step", "workflow_purpose": "Purpose", "external_software": "Software",
           "changes_outside_comfyui": "Changes", "reference_image_sources": "Origin / rights", "notes": "Notes"}
_IMG = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".exr")
_VID = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".mxf")
_AUD = (".wav", ".flac", ".mp3", ".aac", ".ogg", ".m4a")
# node input -> manual field of the report
FIELDS = (("step", "step"), ("purpose", "workflow_purpose"), ("software", "external_software"),
          ("changes", "changes_outside_comfyui"), ("origin_and_rights", "reference_image_sources"))


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _role(path):
    low = path.lower()
    return "image" if low.endswith(_IMG) else "video" if low.endswith(_VID) else "audio" if low.endswith(_AUD) \
        else "file"


RAW = (".rw2", ".cr2", ".nef", ".nrw", ".arw", ".dng", ".orf", ".pef", ".srw", ".rwl", ".3fr", ".iiq", ".raw")


def _tiff_info(path):
    """Header of a TIFF-based camera raw (RW2, CR2, NEF, ARW, DNG, ORF …): (pixel size or None, camera dict).
    Size = the largest image in the IFDs (Panasonic RW2: sensor crop tags). Camera = make, model, lens, capture
    time, exposure, f-number, ISO from the EXIF tags. Reads only the header."""
    import struct
    cam = {}
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8 or head[:2] not in (b"II", b"MM"):
            return None, cam
        e = "<" if head[:2] == b"II" else ">"
        first = struct.unpack(e + "I", head[4:8])[0]
        best, rw2, seen, todo = (0, 0), {}, set(), [first]
        sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}
        want = {271: "make", 272: "model", 36867: "captured", 306: "datetime", 42036: "lens", 33434: "exposure",
                33437: "f_number", 34855: "iso", 42033: "body_serial"}

        def value(typ, cnt, data_or_off):
            n = sizes.get(typ, 1) * cnt
            if n <= 4:
                data = data_or_off[:n]
            else:
                pos = f.tell()
                f.seek(struct.unpack(e + "I", data_or_off)[0])
                data = f.read(min(n, 4096))
                f.seek(pos)
            try:
                if typ == 2:
                    return data.split(b"\x00")[0].decode("latin-1").strip()
                if typ == 3:
                    return struct.unpack(e + "H", data[:2])[0]
                if typ in (4, 9):
                    return struct.unpack(e + "I", data[:4])[0]
                if typ in (5, 10):
                    num, den = struct.unpack(e + "II", data[:8])
                    return (num, den)
            except Exception:
                return None
            return None

        while todo and len(seen) < 32:
            off = todo.pop()
            if not off or off in seen:
                continue
            seen.add(off)
            f.seek(off)
            raw = f.read(2)
            if len(raw) < 2:
                continue
            n = struct.unpack(e + "H", raw)[0]
            tags = {}
            for _ in range(min(n, 500)):
                ent = f.read(12)
                if len(ent) < 12:
                    break
                tag, typ, cnt = struct.unpack(e + "HHI", ent[:8])
                if tag == 330 and typ in (4, 13):  # SubIFDs
                    if cnt == 1:
                        todo.append(struct.unpack(e + "I", ent[8:12])[0])
                    else:
                        pos = f.tell()
                        f.seek(struct.unpack(e + "I", ent[8:12])[0])
                        todo += list(struct.unpack(e + "I" * min(cnt, 8), f.read(4 * min(cnt, 8))))
                        f.seek(pos)
                elif tag in (34665,) and typ in (4, 13):  # Exif IFD
                    todo.append(struct.unpack(e + "I", ent[8:12])[0])
                elif tag in (256, 257, 2, 3, 4, 5, 6, 7) and typ in (3, 4):
                    tags[tag] = value(typ, 1, ent[8:12])
                elif tag in want:
                    v = value(typ, cnt, ent[8:12])
                    if v not in (None, "", (0, 0)):
                        cam.setdefault(want[tag], v)
            nxt = f.read(4)
            if len(nxt) == 4:
                todo.append(struct.unpack(e + "I", nxt)[0])
            w, h = tags.get(256), tags.get(257)
            if w and h and w * h > best[0] * best[1]:
                best = (w, h)
            for t in (2, 3, 4, 5, 6, 7):
                if tags.get(t) is not None:
                    rw2.setdefault(t, tags[t])
        size = None
        if head[2:4] in (b"U\x00", b"\x00U") and all(rw2.get(t) is not None for t in (4, 5, 6, 7)):
            size = (rw2[7] - rw2[5], rw2[6] - rw2[4])  # right - left, bottom - top
        elif head[2:4] in (b"U\x00", b"\x00U") and rw2.get(2) and rw2.get(3):
            size = (rw2[2], rw2[3])
        elif best[0] > 64:
            size = best
    return size, _camera_text(cam)


def _camera_text(cam):
    """EXIF values -> readable camera dict (make/model joined, exposure as 1/x s, f-number, ISO)."""
    out = {}
    make, model = str(cam.get("make") or "").strip(), str(cam.get("model") or "").strip()
    if model:
        out["camera"] = model if make and model.lower().startswith(make.lower().split()[0]) else \
            f"{make} {model}".strip()
    elif make:
        out["camera"] = make
    if cam.get("lens"):
        out["lens"] = str(cam["lens"])
    when = cam.get("captured") or cam.get("datetime")
    if when:
        t = str(when)
        out["captured"] = (t[:10].replace(":", "-") + t[10:16]) if len(t) >= 16 else t
    ex = cam.get("exposure")
    if isinstance(ex, tuple) and ex[1]:
        out["exposure"] = f"1/{round(ex[1] / ex[0])} s" if 0 < ex[0] < ex[1] else f"{ex[0] / ex[1]:g} s"
    fn = cam.get("f_number")
    if isinstance(fn, tuple) and fn[1]:
        out["f_number"] = f"f/{fn[0] / fn[1]:g}"
    if cam.get("iso"):
        out["iso"] = f"ISO {cam['iso']}"
    if cam.get("body_serial"):
        out["body_serial"] = str(cam["body_serial"])
    return out


def _pil_camera(path):
    """EXIF of a JPEG / TIFF / PNG (via PIL) -> camera dict, {} if none."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            sub = ex.get_ifd(0x8769) if ex else {}
            cam = {"make": ex.get(271), "model": ex.get(272), "datetime": ex.get(306),
                   "captured": sub.get(36867), "lens": sub.get(42036), "iso": sub.get(34855),
                   "body_serial": sub.get(42033)}
            for k, tag in (("exposure", 33434), ("f_number", 33437)):
                v = sub.get(tag)
                if v is not None:
                    try:
                        cam[k] = (v.numerator, v.denominator)
                    except AttributeError:
                        cam[k] = (float(v), 1) if k == "f_number" else None
            return _camera_text({k: v for k, v in cam.items() if v})
    except Exception:
        return {}


def camera_line(c):
    return " · ".join(x for x in (c.get("camera"), c.get("lens"),
                                   f"captured {c['captured']}" if c.get("captured") else None,
                                   " ".join(v for v in (c.get("exposure"), c.get("f_number"), c.get("iso")) if v)
                                   or None) if x)


def _dims(path):
    """(width, height) of an image or video – None if it cannot be read."""
    low = path.lower()
    try:
        if low.endswith(RAW):
            return _tiff_info(path)[0]
        if low.endswith(".exr"):
            from ..image_io.readwrite import read_rgba_raw
            rgb, _a = read_rgba_raw(path)
            return int(rgb.shape[1]), int(rgb.shape[0])
        if low.endswith(_IMG):
            from PIL import Image
            with Image.open(path) as im:
                return im.size
        if low.endswith(_VID):
            from ..image_io.media import ffmpeg_path
            exe = ffmpeg_path()
            if exe:
                r = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True, timeout=60)
                m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", r.stderr or "")
                if m:
                    return int(m.group(1)), int(m.group(2))
    except Exception as e:
        print(f"{LOG} Audit External Step: size of {os.path.basename(path)} not read ({type(e).__name__}: {e})")
    return None


def _asset(path):
    sha = hashing.sha256_file(path, label=os.path.basename(path))[0]
    d = {"file": path, "role": "camera raw" if path.lower().endswith(RAW) else _role(path), "sha256": sha}
    wh = _dims(path)
    if wh:
        d["width"], d["height"] = wh
    try:  # 1.8.0: camera, lens, capture time and exposure of a photo (EXIF)
        cam = _tiff_info(path)[1] if path.lower().endswith(RAW) else (
            _pil_camera(path) if path.lower().endswith(_IMG) else {})
        if cam:
            d["camera"] = cam
    except Exception as e:
        print(f"{LOG} Audit External Step: no camera data from {os.path.basename(path)} ({type(e).__name__}: {e})")
    return d


def _picture(path):
    """A small picture of the file for the PDF (image, video frame, waveform) – None if not possible."""
    if not path or not os.path.isfile(path):
        return None
    try:
        from .chain_node import _still, _video_frame, _waveform
        low = path.lower()
        if low.endswith(_VID):
            return _video_frame(path, 1.0)
        if low.endswith(_AUD):
            return _waveform(path)
        if low.endswith(_IMG):
            return _still(path)
    except Exception as e:
        print(f"{LOG} Audit External Step: no picture ({type(e).__name__}: {e})")
    return None


_NONE = {"none", "-", "n/a", "na", "no", "keine", "nichts"}


def _exists(p):
    return os.path.isfile(p) or os.path.isdir(p) or bool(sequence_of(p))


def _paths(text):
    """File paths, one per line – empty lines and 'none' / '-' are skipped. A path broken over lines (a line break
    pasted into it) is joined again: a line that does not start with '/' or '~' continues the line before; a line
    starting with '/' continues it too when the joined path exists and the line before is a folder or does not
    exist."""
    lines = [clean_path(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if x and x.lower() not in _NONE]
    out = []
    for ln in lines:
        if out:
            prev = out[-1]
            cands = [prev + ln, prev.rstrip("/") + "/" + ln.lstrip("/")]
            if not ln.startswith(("/", "~")):
                out[-1] = next((c for c in cands if _exists(c)), cands[1])
                continue
            if (os.path.isdir(prev) or not _exists(prev)) and not _exists(ln):
                hit = next((c for c in cands if _exists(c)), None)
                if hit:
                    out[-1] = hit
                    continue
        out.append(ln)
    return [os.path.realpath(x) for x in out]


_SEQ = (".exr", ".png", ".tif", ".tiff", ".dpx", ".jpg", ".jpeg", ".webp")
_NUM = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")


def sequence_of(path):
    """A folder or a pattern (name.####.exr, name.%04d.exr) -> {'folder', 'frames' [file names], 'name', 'pattern',
    'first', 'last', 'missing'} – None if it is not a sequence."""
    p = path.rstrip("/")
    if os.path.isdir(p):
        folder, rx = p, None
    else:
        folder, base = os.path.dirname(p), os.path.basename(p)
        m = re.search(r"(#+|%0?(\d*)d)", base)
        if not m or not os.path.isdir(folder):
            return None
        pre, post = base[:m.start()], base[m.end():]
        rx = re.compile("^" + re.escape(pre) + r"(\d+)" + re.escape(post) + "$")
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(_SEQ) and (rx is None or rx.match(f)))
    if not names:
        return None
    nums = []
    for f in names:
        m = _NUM.match(f)
        if m:
            nums.append(int(m.group(2)))
    first_m = _NUM.match(names[0])
    pattern = (first_m.group(1) + "#" * len(first_m.group(2)) + first_m.group(3)) if first_m else names[0]
    first, last = (min(nums), max(nums)) if nums else (None, None)
    missing = (last - first + 1 - len(set(nums))) if nums else 0
    return {"folder": folder, "frames": names, "name": os.path.basename(folder), "pattern": pattern,
            "first": first, "last": last, "missing": missing}


def _seq_asset(seq, all_frames=False):
    """Asset for a sequence: pattern, frame range, size of the first frame and the SHA256 of the first / middle /
    last frame (all frames with all_frames – needed when the sequence is the result of a declared step)."""
    fr = seq["frames"]
    pick = fr if all_frames else list(dict.fromkeys([fr[0], fr[len(fr) // 2], fr[-1]]))
    hashes = {}
    for i, f in enumerate(pick):
        if all_frames and i % 50 == 0:
            print(f"{LOG} Audit External Step: hashing {seq['name']} frame {i + 1} / {len(pick)}")
        hashes[f] = hashing.sha256_file(os.path.join(seq["folder"], f), label=f)[0]
    d = {"file": os.path.join(seq["folder"], seq["pattern"]), "role": "image sequence", "path": seq["folder"],
         "files_in_folder": len(fr), "first_frame": seq["first"], "last_frame": seq["last"],
         "missing_frames": seq["missing"], "frame_sha256": hashes,
         "sha256": hashes[fr[len(fr) // 2]] if fr[len(fr) // 2] in hashes else next(iter(hashes.values()))}
    wh = _dims(os.path.join(seq["folder"], fr[0]))
    if wh:
        d["width"], d["height"] = wh
    return d


def _report_for(path):
    """The audit report of a file: the path itself if it is a report, else <name>.audit.json / <name>.json."""
    if CH.is_report_name(os.path.basename(path)) and CH.load_report(path) is not None:
        return path
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", path)
    for cand in (stem + ".audit.json", stem + ".json"):
        if os.path.isfile(cand) and CH.load_report(cand) is not None:
            return cand
    return None


class PertineoAuditExternalStep:
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("written",)
    OUTPUT_TOOLTIPS = ("Path of the file that was written (.audit.json of the external step, or .amend.json).",)
    FUNCTION = "run"
    DESCRIPTION = (
        "Declares what happened outside ComfyUI, so the Audit Chain has no gap there. mode 'external step': give the "
        "file that was made or edited outside ComfyUI (a photo, a Krita / Nuke / Photoshop edit) and – if it was made "
        "from other files – those source files; the node writes <name>.audit.json + .audit.txt next to the file with "
        "the SHA256 of the file and its sources and your declarations. The chain then links it like any other step. "
        "mode 'amend report': give an existing audit report (or the file next to it) and fill in what was left empty "
        "there: filled text fields REPLACE the report's value (empty = keep), source_files ADDS missing inputs (already "
        "listed ones are ignored); the node writes <report>.amend.json next to the report – the "
        "report itself is not changed, the chain shows the values as 'added later'. Declarations are your statements, "
        "not recorded by ComfyUI. Values the workflow did not record (resolution, CFG …) cannot be amended.")

    @classmethod
    def INPUT_TYPES(cls):
        def text(tip, placeholder, multiline=True):
            return ("STRING", {"default": "", "multiline": multiline, "placeholder": placeholder, "tooltip": tip})
        return {
            "required": {
                "mode": (MODES, {"tooltip": "external step = declare a file made / edited outside ComfyUI (writes "
                                            "<name>.audit.json next to it). amend report = add missing declarations to "
                                            "an existing audit report (writes <report>.amend.json, the report stays "
                                            "unchanged)."}),
                "file": text("external step: the file made or edited outside ComfyUI – exactly the file the next step "
                             "loaded. amend report: the .audit.json (or the file it belongs to). Paste the path (in "
                             "Nemo: Ctrl+C on the file, Ctrl+V here).",
                             "FILE: the edited / outside file, or the report to amend"),
                "source_files": text("external step: the files it was made from, one per line (e.g. the "
                                     "generated image before the Krita edit, the raw photo, the .kra / .psd / .nk). "
                                     "Each one is hashed – if an audited step wrote it, the chain links through. "
                                     "Empty = made from scratch (e.g. a photo) – then say where it comes from in "
                                     "origin_and_rights. amend report: ADDS input files the report did not record (e.g. "
                                     "the music of a video made before 1.8.0) – marked 'added later'. Give only the "
                                     "missing ones; files the report already lists are ignored, nothing is removed.",
                                     "external: source files · amend: only the MISSING inputs (added) – one per line"),
                "step": text("Name of the step in the pipeline, e.g. '01b – label retouch'. amend report: a filled "
                             "text field REPLACES the value in the report, an empty one keeps it.",
                             "step, e.g. 01b – label retouch (amend: empty = keep)", False),
                "purpose": text("Why this was done (workflow purpose). amend report: filled = replaces, empty = "
                                "keeps the report's value.", "purpose (amend: empty = keep)"),
                "software": text("Software used outside ComfyUI, with version if known (e.g. 'Krita 5.3', 'Nuke 16', "
                                 "'LightCraft'). Write 'none' if there was none – empty stays UNKNOWN / NOT VERIFIED. "
                                 "amend report: filled = replaces, empty = keeps the report's value.",
                                 "software, e.g. Krita 5.3 – or none (amend: empty = keep)", False),
                "changes": text("What was changed outside ComfyUI (retouch, paint-over, colour, crop, comp …). Write "
                                "'none' if nothing – empty stays UNKNOWN / NOT VERIFIED. amend report: filled = "
                                "replaces, empty = keeps the report's value.",
                                "changes outside ComfyUI – or none (amend: empty = keep)"),
                "origin_and_rights": text("Origin and rights of the file / its sources: who made it, licence, release "
                                          "(e.g. 'photograph by Julia Kietzmann, own rights', 'stock, Adobe Stock "
                                          "#123456, standard licence'). amend report: REPLACES the reference / origin "
                                          "sources of the report if filled, empty keeps them.",
                                          "origin and rights (amend: empty = keep)"),
            },
            "optional": {
                "author": ("STRING", {"default": "", "tooltip": "Who declares this (optional)."}),
                "notes": ("STRING", {"default": "", "multiline": True, "placeholder": "notes (optional)",
                                     "tooltip": "Anything else about this step (optional)."}),
                "write_txt": ("BOOLEAN", {"default": True,
                    "tooltip": "Also write the declaration as text (<name>.audit.txt / <report>.amend.txt). The "
                               ".json is always written – the Audit Chain reads it."}),
                "write_pdf": ("BOOLEAN", {"default": True,
                    "tooltip": "Also write it as PDF (<name>.audit.pdf / <report>.amend.pdf), with a picture of the "
                               "file."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # files on disk may have changed

    @classmethod
    def VALIDATE_INPUTS(cls, file=None):
        if file is None:
            return True
        p = clean_path(file)
        if not p:
            return "Audit External Step: enter the file (or the report to amend)."
        if not os.path.isfile(p) and not sequence_of(p):
            return f"Audit External Step: file not found: {p} (a sequence: give its folder or name.####.exr)"
        return True

    def run(self, mode, file, source_files, step, purpose, software, changes, origin_and_rights, author="",
            notes="", write_txt=True, write_pdf=True):
        path = os.path.realpath(clean_path(file))
        vals = {"step": step, "purpose": purpose, "software": software, "changes": changes,
                "origin_and_rights": origin_and_rights}
        vals = {k: v.strip() for k, v in vals.items() if isinstance(v, str) and v.strip()}
        if mode == MODES[1]:
            written, text, pdf = self._amend(path, vals, (author or "").strip(), (notes or "").strip(),
                                             source_files)
        else:
            written, text, pdf = self._external(path, source_files, vals, (author or "").strip(),
                                                (notes or "").strip())
        base = re.sub(r"\.json$", "", written)
        if write_txt:
            with open(base + ".txt", "w", encoding="utf-8", newline="\n") as f:
                f.write(text + "\n")
        if write_pdf:
            try:
                from .chain_pdf import write_simple_pdf
                write_simple_pdf(base + ".pdf", **pdf)
            except Exception:
                print(f"{LOG} Audit External Step: PDF not written")
                traceback.print_exc()
        print(f"{LOG} Audit External Step: {written}")
        ui = {}
        try:
            from ..image_io.save_media import _text_image
            ui = {"images": [_text_image(text, max_chars=100)]}
        except Exception:
            traceback.print_exc()
        return {"ui": ui, "result": (written,)}

    # ------------------------------------------------------------------
    def _external(self, path, source_files, vals, author, notes):
        if CH.is_report_name(os.path.basename(path)) and CH.load_report(path) is not None:
            raise ValueError(f"{LOG} Audit External Step: {os.path.basename(path)} is an audit report – for mode "
                             "'external step' give the file itself (or use mode 'amend report').")
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", path)
        rpath = stem + ".audit.json"
        if os.path.isfile(rpath):
            old = CH.load_report(rpath)
            if old is not None and old.get("kind") != CH.EXTERNAL:
                raise ValueError(f"{LOG} Audit External Step: {os.path.basename(rpath)} is a ComfyUI audit report – "
                                 "it is not replaced. Use mode 'amend report' to add declarations to it.")
        sources = []
        for sp in _paths(source_files):
            if os.path.isfile(sp):
                if sp != path:
                    sources.append(_asset(sp))
                continue
            sq = sequence_of(sp)
            if not sq:
                raise ValueError(f"{LOG} Audit External Step: source file not found: {sp}")
            sources.append(_seq_asset(sq))
        seq = None if os.path.isfile(path) else sequence_of(path)
        if seq:  # the result is an image sequence: the report goes next to its folder, every frame is hashed
            rpath = os.path.join(os.path.dirname(seq["folder"]), seq["name"] + ".audit.json")
            if os.path.isfile(rpath):
                old = CH.load_report(rpath)
                if old is not None and old.get("kind") != CH.EXTERNAL:
                    raise ValueError(f"{LOG} Audit External Step: {os.path.basename(rpath)} is a ComfyUI audit report "
                                     "– it is not replaced. Use mode 'amend report' to add declarations to it.")
            me = _seq_asset(seq, all_frames=True)
            st = os.stat(os.path.join(seq["folder"], seq["frames"][0]))
        else:
            me = _asset(path)
            st = os.stat(path)
        manual = {}
        for key, field in FIELDS:
            if key in vals:
                manual[field] = vals[key]
        if author:
            manual["author"] = author
        if notes:
            manual["notes"] = notes
        rep = {
            "schema": "pertineo-audit/1",
            "kind": CH.EXTERNAL,
            "generator": {"package": "comfyui-pertineo-vfx", "version": own_version(),
                          "node": "Pertineo Audit External Step"},
            "generated": _now(),
            "statement": "Declared by the user – work outside ComfyUI, not recorded by ComfyUI.",
            "manual": manual,
            "file": ({"name": seq["name"], "folder": os.path.dirname(seq["folder"]), "format": "image sequence",
                      "width": me.get("width"), "height": me.get("height"),
                      "saved": datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
                      "first_frame": seq["first"], "last_frame": seq["last"], "frame_count": len(seq["frames"]),
                      "missing_frames": seq["missing"],
                      "outputs": [{"kind": "image sequence", "path": f"{seq['name']}/{seq['pattern']}",
                                   "frames": len(seq["frames"]), "first_frame": seq["first"],
                                   "last_frame": seq["last"]}],
                      "checksum_list": {f"{seq['name']}/{k}": v for k, v in me["frame_sha256"].items()}}
                     if seq else
                     {"name": os.path.basename(path), "folder": os.path.dirname(path),
                      "format": os.path.splitext(path)[1].lstrip(".").lower(), "sha256": me["sha256"],
                      "width": me.get("width"), "height": me.get("height"),
                      "saved": datetime.datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds"),
                      "bytes": st.st_size, **({"camera": me["camera"]} if me.get("camera") else {})}),
            "input_assets": sources,
        }
        with open(rpath, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        L = [f"EXTERNAL STEP – {seq['name'] if seq else os.path.basename(path)}", "",
             f"{'Declared':<18}{rep['generated'][:16].replace('T', ' ')}" + (f" by {author}" if author else ""),
             f"{'File':<18}{me['file'] if seq else path}"]
        if seq:
            L.append(f"{'Frames':<18}{seq['first']}–{seq['last']} ({len(seq['frames'])} frames"
                     + (f", {seq['missing']} missing" if seq["missing"] else "") + ") · SHA256 of every frame recorded")
        else:
            L.append(f"{'SHA256':<18}{me['sha256']}")
        if me.get("width"):
            L.append(f"{'Size':<18}{me['width']}x{me['height']}")
        if me.get("camera"):
            L.append(f"{'Camera':<18}{camera_line(me['camera'])}")
        for key, field in FIELDS:
            L.append(f"{key.replace('_', ' ').capitalize():<18}{manual.get(field) or CH.UNK}")
        if notes:
            L.append(f"{'Notes':<18}{notes}")
        for a in sources:
            rng = (f"  frames {a['first_frame']}–{a['last_frame']} ({a['files_in_folder']})"
                   if a.get("role") == "image sequence" else f"  (sha256 {a['sha256'][:16]})")
            L.append(f"{'Source':<18}{a['file']}{rng}")
            if a.get("camera"):
                L.append(f"{'Camera':<18}{camera_line(a['camera'])}")
        if not sources:
            L.append(f"{'Source':<18}none – made from scratch (see origin and rights)")
        note = "Declared by the user – not recorded by ComfyUI. The Audit Chain links this file by its SHA256."
        L += ["", note]
        rows = [(lbl.strip(), val, (0.92, 0.60, 0.10) if val == CH.UNK else None) for lbl, val in
                ((x[:18], x[18:]) for x in L[2:-2])]
        pdf = {"kicker": "AUDIT · EXTERNAL STEP", "title": seq["name"] if seq else os.path.basename(path),
               "subtitle": " · ".join(x for x in (manual.get("step"), author, rep["generated"][:10]) if x),
               "picture": _picture(os.path.join(seq["folder"], seq["frames"][len(seq["frames"]) // 2]) if seq else path),
               "rows": rows, "note": note,
               "footer": f"External step · {os.path.basename(path)} · Pertineo VFX {own_version()} · declared by the "
                         "user, not recorded by ComfyUI",
               "author": author or None}
        return rpath, "\n".join(L), pdf

    # ------------------------------------------------------------------
    def _amend(self, path, vals, author, notes, source_files=""):
        rpath = _report_for(path)
        if not rpath:
            raise ValueError(f"{LOG} Audit External Step: no audit report found for {os.path.basename(path)} – "
                             "give the .audit.json itself.")
        extra = _paths(source_files)
        for sp in extra:
            if not os.path.isfile(sp):
                raise ValueError(f"{LOG} Audit External Step: input file not found: {sp}")
        if not vals and not notes and not extra:
            raise ValueError(f"{LOG} Audit External Step: nothing to amend – fill in at least one field.")
        apath = CH.amend_path(rpath)
        data = {}
        if os.path.isfile(apath):
            try:
                with open(apath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        now = _now()
        fields = data.get("fields") or {}
        history = data.get("history") or []
        changed = {}
        for key, field in FIELDS:
            if key in vals:
                if (fields.get(field) or {}).get("value") != vals[key]:
                    changed[field] = vals[key]
                fields[field] = {"value": vals[key], "date": now}
        if notes:
            fields["notes"] = {"value": notes, "date": now}
            changed["notes"] = notes
        inputs = data.get("inputs") or []
        known = {str(a.get("sha256")).lower() for a in inputs}
        for sp in extra:
            a = _asset(sp)
            if a["sha256"] not in known:
                a["added"] = now
                inputs.append(a)
                known.add(a["sha256"])
                changed.setdefault("inputs", []).append(os.path.basename(sp))
        history.append({"date": now, "author": author or None, "fields": changed})
        rep = CH.load_report(rpath) or {}
        out = {
            "schema": "pertineo-audit-amend/1",
            "statement": "Declarations added after the report was written – statements of the user, not recorded by "
                         "ComfyUI. The report itself is unchanged.",
            "report": os.path.basename(rpath),
            "report_sha256": hashing.sha256_file(rpath, label=os.path.basename(rpath))[0],
            "file": (rep.get("file") or {}).get("name"),
            "generator": {"package": "comfyui-pertineo-vfx", "version": own_version(),
                          "node": "Pertineo Audit External Step"},
            "fields": fields,
            "inputs": inputs,
            "history": history,
        }
        with open(apath, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        man = rep.get("manual") or {}
        L = [f"AMENDMENT – {os.path.basename(rpath)}", "", f"{'Written':<18}{apath}", ""]
        for field, v in fields.items():
            before = str(man.get(field) or "").strip() or "(empty)"
            L.append(f"{field:<26}{v['value']}")
            L.append(f"{'':<26}in the report: {before} · added {v['date'][:10]}")
        for a in inputs:
            L.append(f"{'input (added)':<26}{a['file']}")
            L.append(f"{'':<26}sha256 {a['sha256'][:16]} · added {str(a.get('added'))[:10]}")
        note = "The report itself is unchanged. The Audit Chain shows these values as 'added later'."
        L += ["", note]
        rows = [("Report", rpath, None), ("Amendment", apath, None)]
        for field, v in fields.items():
            before = str(man.get(field) or "").strip() or "(empty)"
            rows.append((_LABELS.get(field, field), f"{v['value']}  (in the report: {before} · added "
                                                   f"{v['date'][:10]})", None))
        for a in inputs:
            rows.append(("Input (added)", f"{a['file']}  (sha256 {a['sha256'][:16]} · added {str(a.get('added'))[:10]})",
                         None))
        fname = (rep.get("file") or {}).get("name")
        pic = _picture(os.path.join(os.path.dirname(rpath), fname)) if fname else None
        pdf = {"kicker": "AUDIT · AMENDMENT", "title": os.path.basename(rpath),
               "subtitle": " · ".join(x for x in (author, now[:10]) if x), "picture": pic, "rows": rows,
               "note": "Declarations added after the report was written – statements of the user, not recorded by "
                       "ComfyUI. " + note,
               "footer": f"Amendment · {os.path.basename(rpath)} · Pertineo VFX {own_version()}",
               "author": author or None}
        return apath, "\n".join(L), pdf
