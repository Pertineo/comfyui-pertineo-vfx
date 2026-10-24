# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Plain-text rendering of the audit data, laid out for people (the JSON is for machines).

Order: the file, the indicative triage, what to check first, the declared production context,
the models and their licenses, inputs, how it was generated, outputs, inactive nodes, guiding
principles, the author's declarations, and the technical environment last."""

import re
import textwrap

from . import context as K
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
U = "UNKNOWN"


def _u(v, note=None):
    """Values that are not known are always shown as UNKNOWN (never as '?', empty or 'unknown')."""
    if v is None or (isinstance(v, str) and (not v.strip() or v.strip() in ("?", "–", "-")
                                              or v.strip().lower().startswith("unknown"))):
        rest = v.strip()[7:].strip() if isinstance(v, str) and v.strip().lower().startswith("unknown") else ""
        return U + (f" {rest}" if rest else "") + (f" – {note}" if note else "")
    return v


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
    restricted = [m for m in mods if str(m.get("commercial_use", "")).lower() in ("no", "restricted", "unclear")
                  and m.get("separate_license") != "yes"]
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

    ctx = data.get("context") or {}
    for key, label in K.TEXT_FIELDS[:3]:
        if ctx.get(key):
            L += _kv(label.split(" / ")[0] if key != "shot_asset" else "Shot / asset", ctx[key])

    # ---- triage --------------------------------------------------------
    tri = data.get("triage")
    if tri:
        L += _title("Indicative triage")
        L.append(f"{INDENT}[ {tri['grade']} ]  {tri['label']}")
        L += _kv("Why", tri.get("reason", "?"))
        if tri.get("routes"):
            L += _kv("Open points", " · ".join(f"{r['route']} {r['open_points']}" for r in tri["routes"]))
        c = tri.get("counts") or {}
        L += _kv("Findings", " · ".join(f"{c[k]} {k}" for k in K.SEVERITIES if c.get(k)) or "none")
        rep = data.get("reproducibility") or {}
        if rep:
            L += _kv("Reproducibility", rep.get("status", "?"))
        L.append("")
        L.append(f"{INDENT}Scale:")
        for g in ("A", "B", "C", "D"):
            mark = ">" if g == tri["grade"] else " "
            L += textwrap.wrap(f"{K.TRIAGE[g]} – {K.TRIAGE_LEGEND[g]}", WIDTH,
                               initial_indent=f"{INDENT}{mark} {g}  ", subsequent_indent=INDENT + " " * 5)
        L.append("")
        L += textwrap.wrap(tri.get("note", K.TRIAGE_NOTE), WIDTH - 2, initial_indent=INDENT, subsequent_indent=INDENT)

    # ---- to check ------------------------------------------------------
    L += _title("To check")
    findings = data.get("findings")
    if findings:
        L += textwrap.wrap("Severity: critical > high > medium > low. Evidence: " + K.EVIDENCE_LEGEND, WIDTH - 2,
                           initial_indent=INDENT, subsequent_indent=INDENT)
        L.append("")
    if findings is not None:
        if not findings:
            L.append(INDENT + "Nothing open.")
        for fi in findings:
            route = K.ROUTES.get(fi.get("route"), fi.get("route", "?"))
            L += _kv("•", f"[{fi['severity'].upper()}] {fi['text']}", key=2)
            L += _kv("", f"→ {route} · {fi.get('evidence', '?')}", key=4)
    else:  # data from 1.1.x
        todo = _to_check(data)
        for t in todo:
            L += _kv("•", t, key=2)
        if not todo:
            L.append(INDENT + "Nothing open – all models documented, all fields filled in.")

    # ---- production context -------------------------------------------
    if tri:
        L += _title("Production context (declared)")
        if not ctx:
            L += textwrap.wrap("Not connected – add 'Pertineo Production Context' to declare purpose, rights and "
                               "consent.", WIDTH - 2, initial_indent=INDENT, subsequent_indent=INDENT)
        else:
            for key, label in K.TEXT_FIELDS[3:]:
                if ctx.get(key):
                    L += _kv(label, ctx[key], key=44)
            for key, label, _a, _d in K.CHOICE_FIELDS:
                v = ctx.get(key)
                if key == "execution_environment" and str(v).startswith("auto"):
                    v = f"{data.get('environment_declared', '?')} (derived from the workflow)"
                L += _kv(label, _u(v, "not declared"), key=44)
            for key, label in K.REVIEWER_FIELDS:
                L += _kv(label, ctx.get(key) or "– not recorded –", key=44)
            if ctx.get("context_notes"):
                L += _kv("Notes", ctx["context_notes"], key=44)

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
        sha = str(m.get("sha256") or "not computed")
        L += _kv("SHA256", sha if re.fullmatch(r"[0-9a-f]{64}", sha) else f"{U} – {sha}", indent=INDENT * 2)
        if m.get("autov2"):
            L += _kv("AutoV2", m["autov2"], indent=INDENT * 2)
        if _unknown(m.get("source")) and _unknown(m.get("license")) and _unknown(m.get("commercial_use")):
            L += _kv("License", UNKNOWN_INFO, indent=INDENT * 2)
        else:
            L += _kv("Source", m.get("source", UNKNOWN_INFO), indent=INDENT * 2)
            L += _kv("License", m.get("license", UNKNOWN_INFO), indent=INDENT * 2)
            cu = m.get("commercial_use", UNKNOWN_INFO)
            if str(cu).lower() not in ("yes", "no", "restricted", "unclear") and not _unknown(cu):
                cu = f"{U} – '{cu}' is not yes / no / restricted / unclear"
            L += _kv("Commercial use", cu, indent=INDENT * 2)
        lic_known = not _unknown(m.get("license"))
        if m.get("checked"):
            L += _kv("License checked", m["checked"], indent=INDENT * 2)
        elif lic_known and "findings" in data:
            L += _kv("License checked", U + " – date not recorded", indent=INDENT * 2)
        if m.get("separate_license"):
            L += _kv("Separate license", m["separate_license"] + (f" – {m['license_reference']}"
                                                                    if m.get("license_reference") else
                                                                    (" – reference not recorded"
                                                                     if m["separate_license"] == "yes" else "")),
                     indent=INDENT * 2)
        elif "findings" in data and str(m.get("commercial_use", "")).lower() != "yes":
            L += _kv("Separate license", U + " – not recorded", indent=INDENT * 2)
        for k, lab in (("creator", "Creator"), ("attribution", "Attribution"), ("training_data", "Training data")):
            if m.get(k):
                L += _kv(lab, m[k], indent=INDENT * 2)
        if m.get("family_rows"):
            L += _kv("Family row", ", ".join(m["family_rows"]), indent=INDENT * 2)
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

    # ---- input files -----------------------------------------------------
    if "input_assets" in data:
        ins = data.get("input_assets") or []
        L += _title(f"Input files ({len(ins)})")
        if not ins:
            L.append(INDENT + "No input files found in the executed graph.")
        for n, a in enumerate(ins, 1):
            L.append(f"{INDENT}{n}. {a['role']} – {a['file']}")
            L += _kv("Location", a.get("location", "?") + (f" · {a['files_in_folder']} file(s)"
                                                           if "files_in_folder" in a else ""), indent=INDENT * 2)
            ash = str(a.get("sha256") or "not computed")
            L += _kv("SHA256", ash if re.fullmatch(r"[0-9a-f]{64}", ash) else f"{U} – {ash}", indent=INDENT * 2)
            L += _kv("Used by", f"{a['node_type']} #{a['node_id']} ({a['input']})", indent=INDENT * 2)
        if ins:
            L.append(INDENT + "Rights of these inputs: see 'Reference images' and the production context.")

    # ---- generation ----------------------------------------------------
    samp = data.get("sampling") or []
    L += _title("Generation")
    if data.get("workflow_type"):
        L += _kv("Workflow type", ", ".join(data["workflow_type"]))
    if data.get("seed_mode"):
        L += _kv("Seed control", f"{data['seed_mode']['mode']} (frontend setting, #{data['seed_mode']['node_id']})")
    elif samp and "findings" in data:
        L += _kv("Seed control", U + " – not stored in the workflow")
    for cap in data.get("capabilities") or []:
        L += _kv("Capability", f"{cap['capability']} – {', '.join(cap['evidence'][:4])}")
    if data.get("workflow_type") or data.get("seed_mode") or data.get("capabilities"):
        L.append("")
    if not samp:
        L.append(INDENT + "No sampler node found in the executed graph.")
    for s in samp:
        title = f"  \"{s['title']}\"" if s.get("title") and s["title"] != s["node_type"] else ""
        L.append(f"{INDENT}{s['node_type']} #{s['node_id']}{title}")
        p = dict(s.get("params") or {})
        p.update({k: f"{U} – {v}" for k, v in (s.get("unresolved") or {}).items()})
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
                    if t.startswith("<linked from") or t.startswith("<linked"):
                        t = f"{U} – {t.strip('<>')}"
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

    notable = data.get("notable_settings") or []
    if notable:
        L.append("")
        L.append(f"{INDENT}Other settings:")
        for ns in notable:
            vals = ", ".join(f"{k} {_fmt(v)}" for k, v in ns["settings"].items())
            L += _kv(f"{ns['node_type']} #{ns['node_id']}", vals, indent=INDENT * 2, key=0)

    # ---- outputs ---------------------------------------------------------
    if data.get("output_nodes"):
        L += _title("Output")
        for o in data["output_nodes"]:
            vals = ", ".join(f"{k} {_fmt(v)}" for k, v in o["settings"].items()) or "–"
            L += _kv(f"{o['node_type']} #{o['node_id']}", vals, key=0)

    # ---- inactive nodes ----------------------------------------------------
    if data.get("inactive_nodes"):
        L += _title(f"Inactive nodes ({len(data['inactive_nodes'])}) – not part of the run")
        for x in data["inactive_nodes"]:
            t = f"  \"{x['title']}\"" if x.get("title") and x["title"] != x["node_type"] else ""
            L += _kv("•", f"{x['node_type']} #{x['node_id']}{t} – {x['state']}"
                     + (f" ({', '.join(x['model_files'])})" if x.get("model_files") else ""), key=2)

    # ---- principles ------------------------------------------------------
    if data.get("principles"):
        L += _title("Guiding principles (indicative)")
        L += textwrap.wrap(K.PRINCIPLES_NOTE, WIDTH - 2, initial_indent=INDENT, subsequent_indent=INDENT)
        L.append("")
        for pr in data["principles"]:
            L += _kv(f"{pr['id']} {pr['status'].upper()}", f"{pr['principle']} – {pr['basis']}", key=12)

    # ---- declarations ----------------------------------------------------
    L += _title("Declared by the author")
    for key, label in MANUAL_FIELDS:
        L += _kv(label, _manual(data, key) or NOT_PROVIDED, key=26)

    # ---- environment ---------------------------------------------------
    env = data.get("environment") or {}
    L += _title("Software and hardware")
    comfy = _u(env.get("comfyui")) + (f" (git {env['comfyui_commit']})" if env.get("comfyui_commit") else "")
    L += _kv("ComfyUI", comfy)
    L += _kv("Frontend", _u(env.get("frontend")) + (f"  ({env['frontend_override']})"
                                                    if env.get("frontend_override") else ""))
    L += _kv("Python / torch", f"{_u(env.get('python'))} / {_u(env.get('torch'))} · CUDA {_u(env.get('cuda'))}")
    L += _kv("GPU", _u(env.get("gpu")))
    L += _kv("OS", _u(env.get("os")))
    api = data.get("api_nodes") or []
    L += _kv("API nodes", ", ".join(f"{a['node_type']} #{a['node_id']}" for a in api) if api
             else "none (everything ran locally)")
    remote = [e for e in data.get("external_services") or [] if e["kind"] != "API node"]
    if remote:
        L += _kv("Remote inputs", ", ".join(e["service"] for e in remote))
    L.append("")
    L.append(f"{INDENT}Node packages used:")
    for g in data.get("custom_nodes") or []:
        types = ", ".join(t["type"] + (f" ×{t['count']}" if t["count"] > 1 else "") for t in g["node_types"])
        wv = g.get("workflow_version")
        if wv and re.fullmatch(r"[0-9a-f]{12,40}", wv):
            wv = wv[:7]
        L.append(f"{INDENT * 2}{_u(g['package'])}  {_u(g['version'])}" + (f"   (workflow saved with {wv})" if wv else ""))
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
        f"nodes contributed to the output; bypassed and muted nodes are not part of the run. "
        f"Machine-readable version: the .json with the same name (when write_json is on).", WIDTH)
    return "\n".join(L).rstrip() + "\n"
