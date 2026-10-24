# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Plain-text rendering of the audit data, laid out for people (the JSON is for machines).

Order: what to check first, the file, the models and their licenses, how it was generated,
the author's declarations, and the technical environment last."""

import re
import textwrap

from .collect import NOT_PROVIDED, UNKNOWN_INFO, base_model_mismatches

HEADER = "AUDIT REPORT"
WIDTH = 78
KEY = 18  # width of the label column
INDENT = "  "

MANUAL_FIELDS = [
    ("reference_image_sources", "Reference images"),
    ("external_software", "External software"),
    ("changes_outside_comfyui", "Changes outside ComfyUI"),
    ("notes", "Notes"),
]
ROLE_NAMES = {"checkpoint": "Checkpoint", "diffusion_model": "Diffusion model", "vae": "VAE",
              "text_encoder": "Text encoder", "lora": "LoRA", "controlnet": "ControlNet", "upscaler": "Upscaler",
              "clip_vision": "CLIP vision", "ip_adapter": "IP-Adapter", "style_model": "Style model", "vlm": "VLM",
              "matting_model": "Matting model"}
PARAM_LABELS = [("seed", "Seed"), ("steps", "Steps"), ("cfg", "CFG"), ("guidance", "Guidance"),
                ("sampler", "Sampler"), ("scheduler", "Scheduler"), ("denoise", "Denoise"), ("shift", "Shift")]


# --------------------------------------------------------------------------
def _title(text):
    return ["", "", f"{text.upper()} " + "─" * max(4, WIDTH - len(text) - 1), ""]


def _kv(label, value, indent=INDENT, key=KEY):
    """Label column + value, long values wrapped and indented under the value column."""
    pad = indent + " " * key
    first = f"{indent}{label:<{key}}" if len(label) < key else f"{indent}{label} "
    out = []
    for i, para in enumerate(str(value).splitlines() or [""]):
        # never break hashes or paths – a long token simply runs past the margin
        lines = textwrap.wrap(para, WIDTH - len(pad), break_long_words=False, break_on_hyphens=False) or [""]
        for j, ln in enumerate(lines):
            out.append((first if i == 0 and j == 0 else pad) + ln)
    return out


def _fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _manual(data, key):
    v = (data.get("manual") or {}).get(key)
    return v.strip() if isinstance(v, str) and v.strip() else None


def _size(d):
    if not d:
        return None
    s = f"{d['width']} x {d['height']}"
    if d.get("frames", 1) > 1:
        s += f"  ({d['frames']} images)"
    return s


def _date(iso):
    """2026-10-23T16:24:40+02:00 -> 2026-10-23 16:24:40 (UTC+02:00)"""
    if not isinstance(iso, str) or "T" not in iso:
        return iso or "?"
    d, t = iso.split("T", 1)
    for sign in ("+", "-"):
        if sign in t:
            t, tz = t.split(sign, 1)
            return f"{d} {t} (UTC{sign}{tz})"
    return f"{d} {t}"


_COVERED = ("No model_info entry for", "No model_info file", "An empty model_info.csv")  # already summarised under "To check"


def _unknown(v):
    return v == UNKNOWN_INFO or not v


# --------------------------------------------------------------------------
def _to_check(data):
    """Short list of open points for the reader – derived from the data, nothing new."""
    items = []
    mods = data.get("models") or []
    no_info = [m for m in mods if _unknown(m.get("license"))]
    if mods and not (data.get("options") or {}).get("model_info_file"):
        items.append("No model_info file – source and license are unknown for all models "
                     "(see README, section model_info.json).")
    elif no_info:
        items.append(f"{len(no_info)} of {len(mods)} model(s) without source / license – add them to the "
                     f"model_info file.")
    restricted = [m for m in mods if str(m.get("commercial_use", "")).lower() in ("no", "restricted", "unclear")]
    if restricted:
        items.append("Commercial use is not plainly allowed for: " +
                     ", ".join(m["file"].split("/")[-1] for m in restricted) + ".")
    for m in mods:
        if m.get("hash_mismatch"):
            items.append(f"{m['file'].split('/')[-1]}: {' and '.join(m['hash_mismatch'])} in the model_info "
                         f"file {'do' if len(m['hash_mismatch']) > 1 else 'does'} not match this file – check the entry.")
    for m in base_model_mismatches(mods):
        items.append(f"{m.get('name') or m['file'].split('/')[-1]} is made for {m['base_model']}, which is not "
                     f"the base model of this run.")
    missing_files = [m for m in mods if str(m.get("sha256", "")).startswith("not computed – file not found")]
    if missing_files:
        items.append(f"{len(missing_files)} model file(s) not found – no hash.")
    for s in data.get("sampling") or []:
        for k in (s.get("unresolved") or {}):
            items.append(f"{s['node_type']} #{s['node_id']}: {k} is computed in the graph, not a fixed value.")
    if data.get("api_nodes"):
        items.append("Cloud / API nodes were used – data left this machine.")
    empty = [label for key, label in [("workflow_purpose", "Purpose")] + MANUAL_FIELDS
             if key != "notes" and _manual(data, key) is None]
    if empty:
        items.append("Not filled in: " + ", ".join(empty) + ".")
    return items


def render(data):
    data = data or {}
    f = data.get("file") or {}
    author, company = _manual(data, "author"), _manual(data, "company")
    who = " – ".join(x for x in [author, company] if x) if not (author and company) else f"{author} ({company})"
    title = f"{HEADER} – {who}" if who else HEADER
    L = [title, "═" * WIDTH, ""]

    # ---- file ----------------------------------------------------------
    if f:
        batch = f" (image {f['batch_index'] + 1} of {f['batch_size']})" if f.get("batch_size", 1) > 1 else ""
        L += _kv("File", f"{f['name']}{batch}")
        L += _kv("Saved", _date(f.get("saved")))
        L += _kv("Image", f"{f['width']} x {f['height']} · {f.get('format', '?')}")
        L += _kv("Folder", f.get("folder", "?"))
    else:
        L += _kv("Collected", _date(data.get("generated")))
    if author:
        L += _kv("Author", author)
    if company:
        L += _kv("Company", company)
    L += _kv("Purpose", _manual(data, "workflow_purpose") or NOT_PROVIDED)

    # ---- to check ------------------------------------------------------
    todo = _to_check(data)
    L += _title("To check")
    if todo:
        for t in todo:
            L += _kv("•", t, key=2)
    else:
        L.append(INDENT + "Nothing open – all models documented, all fields filled in.")

    # ---- models --------------------------------------------------------
    mods = data.get("models") or []
    L += _title(f"Models ({len(mods)})")
    if not mods:
        L.append(INDENT + "No model files found in the executed graph.")
    for n, m in enumerate(mods, 1):
        role = ROLE_NAMES.get(m["role"], m["role"])
        L.append(f"{INDENT}{n}. {role}" + (f" – {m['name']}" if m.get("name") else ""))
        L += _kv("File", m["file"], indent=INDENT * 2)
        if m.get("version"):
            L += _kv("Version", m["version"], indent=INDENT * 2)
        if m.get("released"):
            L += _kv("Released", m["released"], indent=INDENT * 2)
        if m.get("base_model"):
            L += _kv("Base model", m["base_model"], indent=INDENT * 2)
        if m.get("trigger_words"):
            L += _kv("Trigger words", ", ".join(m["trigger_words"]), indent=INDENT * 2)
        for k in ("strength_model", "strength_clip", "strength", "lora_strength", "model_strength",
                  "clip_strength", "weight"):
            if k in m:
                L += _kv(k.replace("_", " ").capitalize(), _fmt(m[k]), indent=INDENT * 2)
        sha = m.get("sha256", "–")
        L += _kv("SHA256", sha, indent=INDENT * 2)
        if m.get("autov2"):
            L += _kv("AutoV2", m["autov2"], indent=INDENT * 2)
        if _unknown(m.get("source")) and _unknown(m.get("license")) and _unknown(m.get("commercial_use")):
            L += _kv("License", UNKNOWN_INFO, indent=INDENT * 2)
        else:
            L += _kv("Source", m.get("source", UNKNOWN_INFO), indent=INDENT * 2)
            L += _kv("License", m.get("license", UNKNOWN_INFO), indent=INDENT * 2)
            L += _kv("Commercial use", m.get("commercial_use", UNKNOWN_INFO), indent=INDENT * 2)
        if m.get("checked"):
            L += _kv("License checked", m["checked"], indent=INDENT * 2)
        if m.get("license_notes"):
            L += _kv("Notes", m["license_notes"], indent=INDENT * 2)
        L += _kv("Loaded by", f"{m['node_type']} #{m['node_id']}", indent=INDENT * 2)
        L.append("")
    info = (data.get("options") or {}).get("model_info_file")
    L.append(f"{INDENT}License data from: {info or 'no model_info file'}")
    own = (data.get("options") or {}).get("model_info_disclaimer")
    if own:
        L += textwrap.wrap(f"Note from the model_info file: {own}", WIDTH, initial_indent=INDENT,
                           subsequent_indent=INDENT)

    # ---- generation ----------------------------------------------------
    samp = data.get("sampling") or []
    L += _title("Generation")
    if not samp:
        L.append(INDENT + "No sampler node found in the executed graph.")
    for s in samp:
        title = f"  \"{s['title']}\"" if s.get("title") and s["title"] != s["node_type"] else ""
        L.append(f"{INDENT}{s['node_type']} #{s['node_id']}{title}")
        p = dict(s.get("params") or {})
        p.update({k: f"not fixed – {v}" for k, v in (s.get("unresolved") or {}).items()})
        shown = set()
        for k, label in PARAM_LABELS:
            if k in p:
                L += _kv(label, _fmt(p[k]), indent=INDENT * 2)
                shown.add(k)
        for k, v in p.items():
            if k not in shown:
                L += _kv(k.replace("_", " ").capitalize(), _fmt(v), indent=INDENT * 2)
        lat = s.get("latent") or {}
        if "width" in lat:
            extra = "".join(f", {k.replace('_size', '')} {_fmt(lat[k])}" for k in ("batch_size", "length", "frames")
                            if k in lat)
            L += _kv("Latent", f"{lat['width']} x {lat['height']}{extra}  ({lat.get('from', '?')})", indent=INDENT * 2)
        elif lat:
            L += _kv("Latent", lat.get("note", "?") + (f"  ({lat['from']})" if lat.get("from") else ""),
                     indent=INDENT * 2)
        for role in ("positive", "negative"):
            encs = (s.get("prompts") or {}).get(role) or []
            label = role.capitalize() + " prompt"
            if not encs:
                L += _kv(label, "–", indent=INDENT * 2)
            for e in encs:
                for k, t in e["texts"].items():
                    lab = label if k == "text" else f"{label} [{k}]"
                    L += _kv(lab, t if t.strip() else "(empty)", indent=INDENT * 2)
        L.append("")

    res = data.get("resolution") or {}
    gens = []
    for g in res.get("generation") or []:
        if "width" in g:
            gens.append(f"{g['width']} x {g['height']}" + ("  (encoded source image)" if g.get("same_as") else ""))
        else:
            gens.append(g.get("note", "?"))
    L += _kv("Source image", _size(res.get("source")) or "– not connected –")
    L += _kv("Generated at", " / ".join(dict.fromkeys(gens)) if gens else "–")
    L += _kv("Final image", _size(res.get("final")) or "– not connected –")
    v = data.get("video")
    if v:
        dur = f"{v['duration_sec']:.3f} s" if v.get("duration_sec") is not None else "?"
        L += _kv("Video", f"{v.get('frames') or '?'} frames · {_fmt(v['fps'])} fps · {dur}")

    # ---- declarations ----------------------------------------------------
    L += _title("Declared by the author")
    for key, label in MANUAL_FIELDS:
        L += _kv(label, _manual(data, key) or NOT_PROVIDED, key=26)

    # ---- environment ---------------------------------------------------
    env = data.get("environment") or {}
    L += _title("Software and hardware")
    comfy = env.get("comfyui", "?") + (f" (git {env['comfyui_commit']})" if env.get("comfyui_commit") else "")
    L += _kv("ComfyUI", comfy)
    L += _kv("Frontend", env.get("frontend", "?") + (f"  ({env['frontend_override']})"
                                                     if env.get("frontend_override") else ""))
    L += _kv("Python / torch", f"{env.get('python', '?')} / {env.get('torch', '?')} · CUDA {env.get('cuda', '?')}")
    L += _kv("GPU", env.get("gpu", "?"))
    L += _kv("OS", env.get("os", "?"))
    api = data.get("api_nodes") or []
    L += _kv("API nodes", ", ".join(f"{a['node_type']} #{a['node_id']}" for a in api) if api
             else "none (everything ran locally)")
    L.append("")
    L.append(f"{INDENT}Node packages used:")
    for g in data.get("custom_nodes") or []:
        types = ", ".join(t["type"] + (f" ×{t['count']}" if t["count"] > 1 else "") for t in g["node_types"])
        wv = g.get("workflow_version")
        if wv and re.fullmatch(r"[0-9a-f]{12,40}", wv):
            wv = wv[:7]
        L.append(f"{INDENT * 2}{g['package']}  {g['version']}" + (f"   (workflow saved with {wv})" if wv else ""))
        L += textwrap.wrap(types, WIDTH, initial_indent=INDENT * 3, subsequent_indent=INDENT * 3,
                           break_long_words=False)

    # ---- warnings / footer --------------------------------------------
    warns = [w for w in (data.get("warnings") or []) if not w.startswith(_COVERED)]
    if warns:
        L += _title("Warnings")
        for w in warns:
            L += _kv("•", w, key=2)
    g = data.get("graph") or {}
    L += ["", "─" * WIDTH]
    if data.get("disclaimer"):
        L += textwrap.wrap(data["disclaimer"], WIDTH) + [""]
    L += textwrap.wrap(
        f"Report by Pertineo VFX {data.get('generator', {}).get('version', '?')} · collected "
        f"{_date(data.get('generated'))} · {g.get('nodes_executed', '?')} of {g.get('nodes_in_prompt', '?')} "
        f"nodes contributed to the output; bypassed and muted nodes are not part of the run and not listed. "
        f"Machine-readable version: the .json with the same name (when write_json is on).", WIDTH)
    return "\n".join(L).rstrip() + "\n"
