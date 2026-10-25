# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Plain-text rendering of the audit data, laid out for people (the JSON is for machines).

Order: the file, the indicative triage, what to check first, the declared production context,
the models and their licenses, inputs, how it was generated and what the
prompt writers were given, outputs, inactive nodes, guiding
principles, the author's declarations, and the technical environment last."""

import re
import textwrap

from . import context as K
from .assess import USE_LEGEND, USES as USE_ORDER, WRITER_TEXT_WORDS
from .collect import NOT_PROVIDED, UNKNOWN_INFO, base_model_mismatches

HEADER = "AUDIT REPORT"
WIDTH = 78
KEY = 20  # width of the label column (longest label + at least 4 spaces: the PDF turns the gap into a dot leader)
INDENT = "  "

MANUAL_FIELDS = [
    ("reference_image_sources", "Reference images"),
    ("external_software", "External software"),
    ("changes_outside_comfyui", "Planned work outside ComfyUI"),
    ("notes", "Notes"),
]
OPTIONAL_FIELDS = {"notes", "changes_outside_comfyui"}
ROLE_NAMES = {"checkpoint": "Checkpoint", "diffusion_model": "Diffusion model", "vae": "VAE",
              "text_encoder": "Text encoder", "lora": "LoRA", "controlnet": "ControlNet", "upscaler": "Upscaler",
              "clip_vision": "CLIP vision", "ip_adapter": "IP-Adapter", "style_model": "Style model", "vlm": "VLM",
              "matting_model": "Matting model"}
U = K.UNKNOWN_LABEL


def _u(v, note=None):
    """Values that are not known are always shown as UNKNOWN / NOT VERIFIED (never as '?', empty or 'unknown')."""
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
    out = []
    if (key >= 8 and len(label) > key - 4) or (key and len(label) >= key):
        # a label longer than its column gets a line of its own – the value column stays aligned
        out.append(f"{indent}{label}")
        first = pad
    else:
        first = f"{indent}{label:<{key}}"
    for i, para in enumerate(str(value).splitlines() or [""]):
        # never break hashes or paths – a long token simply runs past the margin
        lines = textwrap.wrap(para, WIDTH - len(pad), break_long_words=False, break_on_hyphens=False) or [""]
        for j, ln in enumerate(lines):
            out.append((first if i == 0 and j == 0 else pad) + ln)
    return out


def _plain(label):
    """'PixaromaAIPrompt #109' -> 'PixaromaAIPrompt' – node numbers stay in the .json."""
    return re.sub(r"\s+#[\w:]+(?=$|[,)])", "", str(label))


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
    return not v or str(v).strip().upper().startswith("UNKNOWN")


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


def _wrap(text, indent):
    return textwrap.wrap(text, WIDTH, initial_indent=indent, subsequent_indent=indent, break_long_words=False,
                         break_on_hyphens=False)


def _quote(text):
    words = str(text).split()
    t = " ".join(words[:WRITER_TEXT_WORDS]) + (" …" if len(words) > WRITER_TEXT_WORDS else "")
    return "\u201c" + t + "\u201d"


# --------------------------------------------------------------------------
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
        if f.get("sha256"):
            L += _kv("SHA256", f["sha256"])
    else:
        L += _kv("Collected", _date(data.get("generated")))
    if author:
        L += _kv("Author", author)
    if company:
        L += _kv("Company", company)
    L += _kv("Purpose", _manual(data, "workflow_purpose") or NOT_PROVIDED)
    if _manual(data, "step"):
        L += _kv("Step", _manual(data, "step"))
    if _manual(data, "step_description"):
        L += _kv("Step description", _manual(data, "step_description"))

    ctx = data.get("context") or {}
    for key, label in K.TEXT_FIELDS[:3]:
        if ctx.get(key):
            L += _kv(label.split(" / ")[0] if key != "shot_asset" else "Shot / asset", ctx[key])

    # ---- triage --------------------------------------------------------
    tri = data.get("triage")
    def license_line(lic):
        parts = [f"{lic['documented']} commercial use yes", f"{lic['separate']} covered by separate license",
                 f"{lic['restricted']} restricted / unclear", f"{lic['unknown']} {U}",
                 f"{lic['conflicts']} conflict{'s' if lic['conflicts'] != 1 else ''}"]
        out = _kv("Model licenses", " · ".join(parts))
        cu = lic.get("commercial_use")
        if cu:
            n = cu.get("models", 0)
            sep = lic.get("separate", 0)
            txt = {"OK": "OK – all models allow it" + (f" ({sep} through a separate license)" if sep else ""),
                   "CHECK TERMS": f"CHECK TERMS for {n} model{'s' if n != 1 else ''} (e.g. revenue limits, "
                                  f"territories) – see To check",
                   "NOT PERMITTED": f"NOT PERMITTED for {n} model{'s' if n != 1 else ''} – see To check",
                   }.get(cu["status"], f"{cu['status']} for {n} model{'s' if n != 1 else ''} – see To check"
                         if n else cu["status"])
            out += _kv("Commercial use", txt)
        return out

    if tri and tri.get("grade") is None:
        L += _kv("Triage", tri.get("reason") or K.NOT_ASSESSED)
        if tri.get("licenses"):
            L += license_line(tri["licenses"])
        tri = None
    if tri:
        L += _title("Indicative triage")
        L.append(f"{INDENT}[ {tri['grade']} ]  {tri['label']}" + (" – APPROVAL ON FILE" if tri.get("covered") else ""))
        L += _kv("Why", tri.get("reason", "?"))
        for nd in tri.get("needs") or []:
            if nd.get("covered"):
                L += _kv("covered", f"{nd['need']} ({nd['covered']})", indent=INDENT * 2, key=12)
            else:
                L += _kv("OPEN", nd["need"], indent=INDENT * 2, key=12)
        if tri.get("routes"):
            L += _kv("Open findings", " · ".join(f"{r['route']} {r['open_points']}" for r in tri["routes"])
                     + "  (medium or higher)")
        c = tri.get("counts") or {}
        L += _kv("Findings", " · ".join(f"{c[k]} {k}" for k in K.SEVERITIES if c.get(k)) or "none")
        if tri.get("licenses"):
            L += license_line(tri["licenses"])
        rep = data.get("reproducibility") or {}
        if rep:
            L += _kv("Reproducibility", rep.get("status", "?"))
        L.append("")
        L += textwrap.wrap("Indicative – not an approval and not legal advice. Scale and terms: see LEGEND at the "
                           "end.", WIDTH - 2, initial_indent=INDENT, subsequent_indent=INDENT)

    # ---- to check ------------------------------------------------------
    L += _title("To check")
    findings = data.get("findings")
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
    if tri and ctx:
        L += _title("Production context (declared)")
        for key, label in K.TEXT_FIELDS[3:]:
            if ctx.get(key):
                L += _kv(label, ctx[key], key=49)
        for key, label, _a, _d in K.CHOICE_FIELDS:
            v = ctx.get(key)
            # consent / approval cannot be missing for something that is declared absent
            need = {"talent_consent": "talent_likeness_or_voice", "union_approval": "union_covered_work",
                    "input_rights_documented": "third_party_copyright"}.get(key)
            if need and ctx.get(need) == "no" and (v or "unknown") == "unknown":
                v = "n/a – none declared"
            if key == "execution_environment" and str(v).startswith("auto"):
                v = f"{data.get('environment_declared', '?')} (derived)"
            L += _kv(label, _u(v), key=49)
        for key, label in K.REVIEWER_FIELDS:
            L += _kv(label, ctx.get(key) or "not recorded", key=49)
        if ctx.get("context_notes"):
            L += _kv("Notes", ctx["context_notes"], key=49)

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
        if m.get("creator"):
            src = {"model_info": "model_info", "file metadata": "from the file"}.get(m.get("creator_source"), "")
            L += _kv("Creator", m["creator"] + (f"  ({src})" if src else ""), indent=INDENT * 2)
        fl = (m.get("file_metadata") or {}).get("license")
        if fl and (not lic_known or fl.lower().replace(" ", "") not in str(m.get("license", "")).lower().replace(" ", "")):
            L += _kv("License in file", f"{fl}  (stated in the file – not verified)", indent=INDENT * 2)
        for k, lab in (("attribution", "Attribution"), ("training_data", "Training data")):
            if m.get(k):
                L += _kv(lab, m[k], indent=INDENT * 2)
        if m.get("family_rows"):
            L += _kv("Family row", ", ".join(m["family_rows"]), indent=INDENT * 2)
        if m.get("license_notes"):
            L += _kv("Notes", m["license_notes"], indent=INDENT * 2)
        if m.get("used_for"):
            L += _kv("Used for", _plain(m["used_for"]), indent=INDENT * 2)
        L.append("")
    info = (data.get("options") or {}).get("model_info_file")
    # the file name says nothing to a reader – what matters is who provided the license data
    L += _wrap("License data: provided by the user (the person who ran the workflow) – not read from the "
               "workflow or the internet." if info else
               "License data: none – no license table found.", INDENT)
    own = (data.get("options") or {}).get("model_info_disclaimer")
    if own:
        L += textwrap.wrap(f"Note from the model_info file: {own}", WIDTH, initial_indent=INDENT,
                           subsequent_indent=INDENT)

    # ---- input files -----------------------------------------------------
    if "input_assets" in data:
        ins = data.get("input_assets") or []
        L += _title(f"Input files ({len(ins)})")
        if not ins:
            L.append(INDENT + "No input files found in the workflow.")
        uses = {}
        for a in ins:
            for u in a.get("used_as") or []:
                uses[u["use"]] = uses.get(u["use"], 0) + 1
        if uses:
            L += _kv("Summary", " · ".join(f"{uses[u]} {u}" for u in USE_ORDER if u in uses))
            L.append("")
        for n, a in enumerate(ins, 1):
            L.append(f"{INDENT}{a.get('index', n)}. {a['role']} – {a['file']}")
            if a.get("used_as"):
                L += _kv("Used as", "; ".join(u["use"] + (f" ({', '.join(_plain(v) for v in u['via'])})"
                                                          if u["via"] else "") for u in a["used_as"]),
                         indent=INDENT * 2)
            L += _kv("Location", a.get("location", "?") + (f" · {a['files_in_folder']} file(s)"
                                                           if "files_in_folder" in a else ""), indent=INDENT * 2)
            ash = str(a.get("sha256") or "not computed")
            L += _kv("SHA256", ash if re.fullmatch(r"[0-9a-f]{64}", ash) else f"{U} – {ash}", indent=INDENT * 2)
            if a.get("feeds") and not a.get("used_as"):  # data from 1.3.x
                L += _kv("Used by", ", ".join(_plain(x) for x in a["feeds"]), indent=INDENT * 2)
        if ins:
            L.append(INDENT + "Rights of these inputs: see 'Reference images' and the production context.")

    # ---- generation ----------------------------------------------------
    samp = data.get("sampling") or []
    L += _title("Generation")
    if data.get("workflow_type"):
        L += _kv("Workflow type", ", ".join(data["workflow_type"]))
    if data.get("seed_mode"):
        L += _kv("Seed control", data["seed_mode"]["mode"])
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
        L.append(f"{INDENT}{s['node_type']}{title}" if len(samp) == 1 else f"{INDENT}{s['node_type']} #{s['node_id']}{title}")
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
            L += _kv("Latent", f"{lat['width']} x {lat['height']}{extra}", indent=INDENT * 2)
        elif lat:
            note = lat.get("note", "?")
            if re.search(r"not followed|not resolved|too deep|missing node", note):
                note = f"{U} – size not recorded"
            L += _kv("Latent", note, indent=INDENT * 2)
        for role in ("positive", "negative"):
            encs = (s.get("prompts") or {}).get(role) or []
            label = role.capitalize() + " prompt"
            if not encs:
                L += _kv(label, "–", indent=INDENT * 2)
            for e in encs:
                if e.get("computed_by"):
                    see = " – see PROMPT GENERATION" if data.get("prompt_writers") else ""
                    L += _kv(f"{role.capitalize()} source", f"generated in this run ({_plain(e['computed_by'])}) – "
                             f"recorded as produced{see}", indent=INDENT * 2)
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
            note = g.get("note", "?")
            gens.append(f"{U} – size not recorded" if re.search(r"not followed|not resolved|too deep|missing node",
                                                               note) else note)
    no_src = ("not connected – no source image in this run; inputs are listed under INPUT FILES"
              if data.get("input_assets") else "– not connected –")
    L += _kv("Source image", _size(res.get("source")) or no_src)
    L += _kv("Generated at", " / ".join(dict.fromkeys(gens)) if gens else "–")
    L += _kv("Final image", _size(res.get("final")) or "– not connected –")
    v = data.get("video")
    if v:
        dur = f"{v['duration_sec']:.3f} s" if v.get("duration_sec") is not None else "?"
        L += _kv("Video", f"{v.get('frames') or '?'} frames · {_fmt(v['fps'])} fps · {dur}")
    rp = data.get("reproducibility") or {}
    if rp:
        L.append("")
        L += _kv("Reproducibility", rp.get("status", "?"))
        for note in (rp.get("rerun") or []):
            L += _kv("", "• " + note, key=KEY + 2)
        for item in (rp.get("missing") or []):
            L += _kv("", f"• not recorded: {item}", key=KEY + 2)

    notable = data.get("notable_settings") or []
    if notable:
        L.append("")
        L.append(f"{INDENT}Other settings:")
        for ns in notable:
            vals = ", ".join(f"{k} {_fmt(v)}" for k, v in ns["settings"].items())
            L.append(f"{INDENT * 2}{ns['node_type']} #{ns['node_id']}")
            L += _kv("", vals, indent=INDENT * 3, key=0)

    # ---- prompt generation ------------------------------------------------
    pws = data.get("prompt_writers") or []
    if pws:
        L += _title("Prompt generation")
        L += _wrap("What the prompt writers were given in this run – images, instructions, presets and linked "
                   "texts. What each one wrote is recorded as produced.", INDENT)
        L.append("")
        for n, w in enumerate(pws, 1):
            title = f"  \"{w['title']}\"" if w.get("title") and w["title"] != w["node_type"] else ""
            L.append(f"{INDENT}{n}. {w['node_type']}{title}")
            L += _kv("Writes", ", ".join(_plain(f) for f in w["feeds"]), indent=INDENT * 2)
            if w.get("models"):
                L += _kv("Model", ", ".join(m.split("/")[-1] for m in w["models"]), indent=INDENT * 2)
            for im in w.get("images") or []:
                if im["file_names"]:
                    val = ", ".join(f"{f} (input {i})" for f, i in zip(im["file_names"], im["files"]))
                elif im["produced_by"]:
                    val = "produced in this run by " + ", ".join(_plain(x) for x in im["produced_by"])
                else:
                    val = f"{U} – source not found"
                L += _kv(im["input"], val, indent=INDENT * 2)
            for t in w.get("texts") or []:
                if t.get("text") is None:
                    val = f"{U} – {t.get('note') or 'not recorded'}"
                elif t["text"] == "REDACTED":
                    val = "REDACTED (credential-like value)"
                else:
                    val = _quote(t["text"])
                    if t.get("from"):
                        val += f"  (from {_plain(t['from'])})"
                L += _kv(t["input"], val, indent=INDENT * 2)
            if w.get("settings"):
                L += _kv("Settings", " · ".join(f"{k} {_fmt(v)}" for k, v in w["settings"].items()),
                         indent=INDENT * 2)
            if w.get("output") and not any(f.startswith(("positive prompt", "negative prompt")) for f in w["feeds"]):
                L += _kv("Wrote", _quote(w["output"]), indent=INDENT * 2)
            L.append("")
        if any(len(str(t.get("text") or "").split()) > WRITER_TEXT_WORDS
               for w in pws for t in w.get("texts") or []) or \
                any(len(str(w.get("output") or "").split()) > WRITER_TEXT_WORDS for w in pws):
            L += _wrap(f"Texts longer than {WRITER_TEXT_WORDS} words are shortened here; the .json has them in "
                       f"full.", INDENT)

    # ---- outputs ---------------------------------------------------------
    if data.get("output_nodes"):
        L += _title("Output")
        for o in data["output_nodes"]:
            L.append(f"{INDENT}{o['node_type']}")
            for k, v in o["settings"].items():
                L += _kv(k, _fmt(v), indent=INDENT * 2)

    # ---- inactive nodes ----------------------------------------------------
    if data.get("inactive_nodes"):
        inact = data["inactive_nodes"]
        counts = {}
        for x in inact:
            counts[x["state"]] = counts.get(x["state"], 0) + 1
        L += _title("Inactive nodes – not part of the run")
        L += _kv("Summary", " · ".join(f"{n} {st}" for st, n in counts.items()) + " (full list in the .json)")
        with_models = [x for x in inact if x.get("model_files")]
        for x in with_models:
            L += _kv("•", f"{x['node_type']} – {x['state']}: {', '.join(x['model_files'])}", key=2)

    # ---- principles ------------------------------------------------------
    if data.get("principles"):
        L += _title("Guiding principles (indicative)")
        for pr in data["principles"]:
            status = U if pr["status"] == "unknown" else pr["status"].upper()
            L += _kv(pr["id"], pr["principle"], key=7)
            L += _kv("", f"→ {status} – {pr['basis']}", key=7)

    # ---- credits ---------------------------------------------------------
    if data.get("findings") is not None:
        L += _title("Credits")
        L.append(f"{INDENT}Models")
        seen = set()
        for m in mods:
            nm = m.get("name") or m["file"].split("/")[-1]
            key = (nm, m.get("creator"), m.get("license"))
            if key in seen:
                continue
            seen.add(key)
            src = {"model_info": "model_info", "file metadata": "file"}.get(m.get("creator_source"))
            who = (m["creator"] + (f" ({src})" if src else "")) if m.get("creator") else None
            lic = m.get("license") if not _unknown(m.get("license")) else None
            if who and lic:
                val = f"{who} · {lic}"
            elif who or lic:
                val = f"{who} · license {U}" if who else f"creator {U} · {lic}"
            else:
                val = f"{U} (creator, license)"
            L += _kv(nm, val, indent=INDENT * 2, key=33)
        L.append(f"{INDENT}Node packages")
        for g in data.get("custom_nodes") or []:
            c = g.get("credits") or {}
            if g["package"] == "unknown":
                L += _kv("package not determined", U, indent=INDENT * 2, key=33)
                continue
            who = c.get("publisher") or c.get("authors") or ""
            lic = c.get("license") or f"license {U}"
            repo = re.sub(r"^https?://", "", c.get("repository", "")) if c.get("repository") else ""
            label = f"{g['package']} {_u(g['version']).split(' (git')[0]}"
            L += _kv(label, " · ".join(x for x in (who, lic, repo) if x), indent=INDENT * 2, key=33)
        att = [(m.get("name") or m["file"].split("/")[-1], m["attribution"]) for m in mods
               if m.get("attribution") and str(m["attribution"]).strip().lower() not in ("no", "none", "not required")]
        L.append(f"{INDENT}Required attribution")
        if att:
            for nm, a in dict(att).items():
                L += _kv(nm, a, indent=INDENT * 2, key=33)
        else:
            L.append(f"{INDENT * 2}none recorded")
        L += textwrap.wrap("Source of each value: model_info = your table (preferred), file = stated in the model "
                           "file, node packages = their own pyproject / LICENSE / git files.", WIDTH - 2,
                           initial_indent=INDENT, subsequent_indent=INDENT)

    # ---- declarations ----------------------------------------------------
    L += _title("Declared by the author")
    for key, label in MANUAL_FIELDS:
        # notes and planned work are optional: empty is simply empty, not unknown
        L += _kv(label, _manual(data, key) or ("–" if key in OPTIONAL_FIELDS else NOT_PROVIDED), key=33)

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
        L += _kv(_u(g["package"]) if g["package"] != "unknown" else "package not determined",
                 _u(g["version"]) + (f" (workflow saved with {wv})" if wv else ""), indent=INDENT * 2, key=33)
        L += textwrap.wrap(types, WIDTH, initial_indent=INDENT * 3, subsequent_indent=INDENT * 3,
                           break_long_words=False)

    # ---- warnings / footer --------------------------------------------
    warns = [w for w in (data.get("warnings") or []) if not w.startswith(_COVERED)]
    if warns:
        L += _title("Warnings")
        for w in warns:
            L += _kv("•", w, key=2)
    # ---- legend ----------------------------------------------------------
    if data.get("findings") is not None:
        L += _title("Legend")
        if data.get("triage") and data["triage"].get("grade"):
            L.append(f"{INDENT}Grade – what has to happen before the material is used:")
            for g in K.GRADE_ORDER:
                mark = ">" if g == data["triage"]["grade"] else " "
                L += textwrap.wrap(f"{K.TRIAGE[g]} – {K.TRIAGE_LEGEND[g]}", WIDTH,
                                   initial_indent=f"{INDENT}{mark} {g}  ", subsequent_indent=INDENT + " " * 5)
            L.append("")
        present = {u["use"] for a in data.get("input_assets") or [] for u in a.get("used_as") or []}
        if present:
            L.append(f"{INDENT}Input use – how an input file reaches the result:")
            for u in [x for x in USE_ORDER if x in present]:
                L += textwrap.wrap(USE_LEGEND[u], WIDTH, initial_indent=f"{INDENT}  {u:<19}  ",
                                   subsequent_indent=INDENT + " " * 23)
            L.append("")
        graded = bool(data.get("triage") and data["triage"].get("grade"))
        # without a Production Context there is no grade: only what explains the findings is printed
        sev = K.SEVERITY_LEGEND if graded else "Severity (critical > high > medium > low) rates a single finding."
        for txt in (sev, "Evidence: " + K.EVIDENCE_LEGEND,
                    K.PRINCIPLES_NOTE if data.get("principles") else None, K.TRIAGE_NOTE if graded else None,
                    f"Assessment rules as of {data.get('rules_date') or K.RULES_DATE}."):
            if txt:
                L += textwrap.wrap(txt, WIDTH - 2, initial_indent=INDENT, subsequent_indent=INDENT)
                L.append("")

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
