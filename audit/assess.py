# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Second pass over the graph (input files, inactive nodes, settings, outputs, workflow type,
capabilities) and the assessment (reproducibility, findings, indicative triage, guiding principles).

Rules: everything is derived from the graph or from the user's declarations – nothing is guessed.
Each finding carries its evidence class (observed / derived / declared / external) and the department
it should go to."""

import os
import re

from . import context as K
from . import hashing

MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".exr", ".dpx", ".bmp", ".gif", ".hdr",
              ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mxf",
              ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
              ".txt", ".csv", ".srt", ".npz", ".npy", ".obj", ".glb", ".fbx", ".ply")
_INPUT_NAME_RE = re.compile(r"^(image|images|video|audio|mask|file|filename|file_path|path|folder|directory|"
                            r"image_path|video_path|audio_path|audio_file|image_file|video_file|input_path|"
                            r"source_folder|plate_folder|trimap_folder|sequence|text_file|caption_file|url)\d*$", re.I)
_SKIP_INPUT_RE = re.compile(r"output|save|prefix|model_info", re.I)
_SECRET_NAME_RE = re.compile(r"(^|[_-])(api[_-]?key|apikey|token|access[_-]?key|secret|secret[_-]?key|password|"
                             r"passwd|bearer|credentials?|auth[_-]?token|authorization)($|[_-])", re.I)
_SECRET_VALUE_RE = re.compile(r"^(sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
                              r"AIza[0-9A-Za-z_-]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|r8_[A-Za-z0-9]{20,})$")
_URL_RE = re.compile(r"^https?://", re.I)
SEED_MODES = ("fixed", "increment", "decrement", "randomize")


# --------------------------------------------------------------------------
# Workflow (UI format) access
# --------------------------------------------------------------------------
def workflow_nodes(extra_pnginfo):
    """All UI nodes incl. subgraph definitions: id -> node dict."""
    wf = (extra_pnginfo or {}).get("workflow") if isinstance(extra_pnginfo, dict) else None
    out = {}
    if not isinstance(wf, dict):
        return out
    lists = [wf.get("nodes") or []]
    for sg in ((wf.get("definitions") or {}).get("subgraphs") or []):
        lists.append(sg.get("nodes") or [])
    for nodes in lists:
        for n in nodes:
            if isinstance(n, dict) and "id" in n:
                out.setdefault(str(n["id"]), n)
    return out


def _mode_name(mode):
    return {2: "muted", 4: "bypassed"}.get(mode)


def inactive_nodes(graph, executed, wf_nodes, show):
    """Bypassed / muted nodes (from the workflow) and nodes that are in the prompt but did not
    contribute to any output. Listed for transparency – they are not part of the run."""
    out = []
    exe = set(executed)
    model_exts = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".sft", ".bin", ".onnx")
    for nid, n in wf_nodes.items():
        state = _mode_name(n.get("mode"))
        if not state:
            continue
        files = []
        for v in (n.get("widgets_values") or []) if isinstance(n.get("widgets_values"), list) else []:
            if isinstance(v, str) and v.lower().endswith(model_exts):
                files.append(show.path(v))
        out.append({"node_id": nid, "node_type": n.get("type", "?"), "title": n.get("title"), "state": state,
                    "model_files": files})
    for nid in graph.nodes:
        if nid in exe or _mode_name(wf_nodes.get(nid, {}).get("mode")) or \
                graph.ctype(nid).startswith(("PertineoCollectMetadata", "PertineoProductionContext")):
            continue
        out.append({"node_id": nid, "node_type": graph.ctype(nid), "title": graph.title(nid),
                    "state": "not connected to an output", "model_files": []})
    def key(x):
        return (x["state"], [(0, int(p), "") if p.isdigit() else (1, 0, p) for p in str(x["node_id"]).split(":")])
    return sorted(out, key=key)


def seed_node_ids(graph, samplers):
    """Sampler ids plus every node on the seed / noise link chain (seed widgets often sit on a
    separate seed or RandomNoise node)."""
    out = []
    for s in samplers:
        chain, stack = [], [s["node_id"]]
        for _ in range(12):
            if not stack:
                break
            nid = stack.pop()
            if nid in chain:
                continue
            chain.append(nid)
            ins = graph.inputs(nid)
            for k in ("seed", "noise_seed", "noise"):
                v = ins.get(k)
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    stack.append(v[0])
        # only the node that really holds the value counts: a sampler whose seed input is linked keeps a stale
        # control widget of its own ('randomize' …) that has no effect
        holders = [n for n in chain if not any(isinstance(graph.inputs(n).get(k), list)
                                               for k in ("seed", "noise_seed", "noise"))]
        out += [n for n in reversed(holders) if n not in out]
    return out


def seed_mode(wf_nodes, node_ids):
    """'fixed' / 'randomize' / … from the frontend widget next to a seed, for the first node that has one."""
    for nid in node_ids:
        n = wf_nodes.get(str(nid))
        if not n:
            continue
        wv0 = n.get("widgets_values")
        if "rgthree" in str(n.get("type", "")).lower() and "seed" in str(n.get("type", "")).lower() \
                and isinstance(wv0, list) and wv0 and isinstance(wv0[0], int):
            # Seed (rgthree): -1 = random each run, -2 / -3 = increment / decrement, any other value is fixed
            return {-1: "randomize", -2: "increment", -3: "decrement"}.get(wv0[0], "fixed"), nid
        named = n.get("widgets_values_named")
        if isinstance(named, dict) and named.get("control_after_generate") in SEED_MODES:
            return named["control_after_generate"], nid
        wv = n.get("widgets_values")
        if isinstance(wv, list):
            for i, v in enumerate(wv):
                if v in SEED_MODES and i > 0 and isinstance(wv[i - 1], (int, float)) and not isinstance(wv[i - 1], bool):
                    return v, nid
    return None, None


# --------------------------------------------------------------------------
# Input files
# --------------------------------------------------------------------------
def _resolve_input(value):
    """(full path or None, kind) for an input file / folder name."""
    v = str(value).strip().strip("\"'")
    if not v:
        return None, None
    try:
        import folder_paths

        p = folder_paths.get_annotated_filepath(v)
        if p and os.path.exists(p):
            return p, "folder" if os.path.isdir(p) else "file"
    except Exception:
        pass
    p = os.path.expanduser(v)
    if os.path.isabs(p) and os.path.exists(p):
        return p, "folder" if os.path.isdir(p) else "file"
    return None, None


def _input_role(ct, name, value):
    c, n = ct.lower(), name.lower()
    ext = os.path.splitext(str(value))[1].lower()
    if "mask" in c or "mask" in n:
        return "mask"
    if "trimap" in n or "trimap" in c:
        return "trimap"
    if ext in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mxf") or "video" in c or "video" in n:
        return "video"
    if ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac") or "audio" in c or "audio" in n:
        return "audio"
    if "sequence" in c or n in ("folder", "plate_folder", "source_folder", "directory"):
        return "image sequence / folder"
    if ext in (".txt", ".csv", ".srt") or "text" in n or "caption" in n:
        return "text"
    if n == "url":
        return "remote URL"
    return "image"


def _consumers(graph, executed, nid):
    from .collect import consumers

    return consumers(graph, executed, nid)


def input_assets(graph, executed, show, warn, compute_hashes=True):
    out = []
    try:
        import folder_paths

        input_dir = os.path.realpath(folder_paths.get_input_directory())
    except Exception:
        input_dir = None
    for nid in executed:
        ct = graph.ctype(nid)
        if ct.startswith(("PertineoCollectMetadata", "PertineoSaveImageMetadata", "PertineoModelInfo")):
            continue
        for name, raw in graph.inputs(nid).items():
            ok, val, _n, _c = graph.resolve(raw)
            if not ok or not isinstance(val, str) or not val.strip() or _SKIP_INPUT_RE.search(name):
                continue
            low = val.strip().lower()
            if not (_INPUT_NAME_RE.match(name) or low.endswith(MEDIA_EXTS) or _URL_RE.match(low)):
                continue
            if _URL_RE.match(low):
                out.append({"node_id": nid, "node_type": ct, "input": name, "file": val.strip(),
                            "role": "remote URL", "location": "remote", "sha256": "not computed (remote)"})
                continue
            full, kind = _resolve_input(val)
            if full is None and not low.endswith(MEDIA_EXTS):
                continue  # a plain text widget that happens to be called 'path' etc.
            entry = {"node_id": nid, "node_type": ct, "input": name, "role": _input_role(ct, name, val)}
            real = os.path.realpath(full) if full else None
            if real and input_dir and real.startswith(input_dir + os.sep):
                entry["file"] = "<input>/" + os.path.relpath(real, input_dir).replace(os.sep, "/")
                entry["location"] = "ComfyUI input folder"
            elif real:
                entry["file"] = show.path(full) if show.full else os.path.basename(real.rstrip(os.sep)) + (
                    "/" if kind == "folder" else "")
                entry["location"] = "network / mounted drive" if real.startswith(("/mnt/", "/media/", "/net/", "//", "\\\\")) \
                    else "local path"
            else:
                entry["file"] = show.path(val)
                entry["location"] = "not found"
            if show.full and full:
                entry["path"] = full
            if kind == "file" and compute_hashes:
                try:
                    sha, _cached = hashing.sha256_file(full, label=os.path.basename(full))
                    entry["sha256"] = sha
                except Exception as e:
                    entry["sha256"] = f"not computed – {type(e).__name__}"
                    warn(f"Input file of {graph.label(nid)} could not be hashed ({type(e).__name__})")
            elif kind == "file":
                entry["sha256"] = "not computed (compute_model_hashes is off)"
            elif kind == "folder":
                try:
                    n = sum(1 for f in os.listdir(full) if f.lower().endswith(MEDIA_EXTS))
                    entry["files_in_folder"] = n
                except OSError:
                    pass
                entry["sha256"] = "not computed (folder)"
            else:
                entry["sha256"] = "not computed – file not found"
                warn(f"Input file of {graph.label(nid)} not found: {show.path(val)}")
            feeds = [graph.label(c) for c in _consumers(graph, executed, nid)]
            if feeds:
                entry["feeds"] = feeds
            out.append(entry)
    return out


# --------------------------------------------------------------------------
# Settings, outputs, credentials
# --------------------------------------------------------------------------
_SETTINGS_RE = re.compile(r"controlnet|control_net|ipadapter|ip_adapter|upscale|scale|resize|mask|grow|blur|feather|"
                          r"crop|inpaint|pad|outpaint|detailer|interpolat|fps|frame|tile|denoise|stylemodel|"
                          r"instantid|pulid|reactor|faceswap|facerestore|colormatch|color_match|trimap|vitmatte", re.I)
_SETTING_SKIP = re.compile(r"image|images|pixels|samples|latent|model|clip|vae|conditioning|positive|negative|"
                           r"mask$|control_net$|text$|prompt", re.I)


def node_settings(graph, executed, show, warn):
    """All constant inputs of every executed node (JSON) + the notable ones (report).
    Secrets are redacted and reported."""
    full, notable, secrets = {}, [], []
    for nid in executed:
        ct = graph.ctype(nid)
        if ct.startswith(("PertineoCollectMetadata", "PertineoProductionContext")):
            continue  # their content is in the report already (author respects include_author)
        vals = {}
        for name, raw in graph.inputs(nid).items():
            ok, val, _n, _c = graph.resolve(raw)
            if not ok or not isinstance(val, (str, int, float, bool)) or isinstance(raw, dict):
                continue
            if isinstance(val, str) and (_SECRET_NAME_RE.search(name) or _SECRET_VALUE_RE.match(val.strip())):
                if val.strip():
                    secrets.append({"node_id": nid, "node_type": ct, "input": name})
                    val = "REDACTED"
            elif isinstance(val, str):
                val = show.text(val) if (os.sep in val and not show.full) else val
            vals[name] = val
        if vals:
            full[nid] = {"node_type": ct, "inputs": vals}
        if _SETTINGS_RE.search(ct):
            short = {k: v for k, v in vals.items()
                     if not _SETTING_SKIP.search(k) and not (isinstance(v, str) and len(v) > 60)}
            if short:
                notable.append({"node_id": nid, "node_type": ct, "settings": short})
    for s in secrets:
        warn(f"{s['node_type']} #{s['node_id']}: input '{s['input']}' looks like a credential – redacted in this "
             f"report. It is still stored in the workflow / PNG metadata.")
    return full, notable, secrets


_OUT_KEY_RE = re.compile(r"prefix|format|folder|path|quality|fps|frame_rate|codec|crf|bit|lossless|compress|"
                         r"pix_fmt|container|extension|embed|metadata", re.I)


def output_nodes(graph, executed, show):
    out = []
    for nid in executed:
        ct = graph.ctype(nid)
        if not graph.is_output_node(ct) or ct.startswith(("PertineoCollectMetadata", "PertineoModelInfo")):
            continue
        if not re.search(r"save|write|export|render", ct, re.I):
            continue
        settings = {}
        for name, raw in graph.inputs(nid).items():
            ok, val, _n, _c = graph.resolve(raw)
            if ok and isinstance(val, (str, int, float, bool)) and _OUT_KEY_RE.search(name):
                settings[name] = show.text(val) if isinstance(val, str) else val
        out.append({"node_id": nid, "node_type": ct, "settings": settings})
    return out


# --------------------------------------------------------------------------
# Workflow type and capabilities
# --------------------------------------------------------------------------
_TYPE_RULES = [
    ("inpainting", r"inpaint|setlatentnoisemask"),
    ("outpainting", r"outpaint"),
    ("ControlNet-guided", r"controlnetapply|controlnet_apply|apply ?controlnet|controlnetloader|union"),
    ("IP-Adapter / reference image", r"ipadapter|stylemodelapply|clipvisionencode|redux|reference"),
    ("identity / face conditioning", r"instantid|pulid|faceid|ecomid"),
    ("face swap", r"reactor|faceswap|face_swap|inswapper"),
    ("face restoration", r"facerestore|gfpgan|codeformer|gpen"),
    ("upscaling / super-resolution", r"upscale|supir|seedvr|esrgan|ultimatesd"),
    ("segmentation / masking", r"sam2|sam3|samloader|segment|rmbg|birefnet|grounding|mask"),
    ("depth / normal / pose / edge processing", r"depth|normalmap|normal_map|dsine|openpose|dwpose|canny|lineart|hedpre|pidinet|sdpose|moge"),
    ("matte extraction / roto", r"vitmatte|trimap|matte|alphagen"),
    ("frame interpolation", r"rife|film_?vfi|interpolat"),
    ("audio / voice / lip-sync", r"audio|voice|tts|lipsync|lip_sync|music|speech"),
    ("model training / fine-tuning", r"train(?!ing_data)|lora_training"),
    ("3D generation", r"trellis|hunyuan3d|pixal3d|triposr|tripo|meshgen|3dgen|stable_?fast_?3d"),
]
_VIDEO_RE = re.compile(r"video|wan\d|wanvace|ltxv|hunyuanvideo|mochi|cogvideo|animatediff|svd_img2vid", re.I)
_CAPABILITIES = [
    ("face swap", r"reactor|faceswap|face_swap|inswapper", "talent"),
    ("identity adapter", r"instantid|pulid|faceid|ecomid|identity", "talent"),
    ("face restoration (may alter identity)", r"facerestore|gfpgan|codeformer|gpen", "talent"),
    ("voice / lip-sync / talking head", r"tts|voice|lipsync|lip_sync|talkvid|talking|speech", "talent"),
    ("pose / performance transfer", r"openpose|dwpose|sdpose|anypose|pose", "talent"),
]


def classify(graph, executed, models, samplers, api_nodes):
    types = {graph.ctype(n) for n in executed}
    text = " ".join(types).lower() + " " + " ".join(m["file"].split("/")[-1].lower() for m in models)
    found = []
    is_video = bool(_VIDEO_RE.search(" ".join(types))) or any(
        isinstance((s.get("latent") or {}).get(k), int) and (s.get("latent") or {})[k] > 1
        for s in samplers for k in ("length", "frames"))
    for s in samplers:
        lat = (s.get("latent") or {})
        src = (lat.get("from") or "").lower()
        if "empty" in src:
            found.append("text-to-video" if is_video else "text-to-image")
        elif lat.get("same_as") or "vaeencode" in src or "encoded" in (lat.get("note") or ""):
            found.append("image/video-to-video" if is_video else "image-to-image")
    for label, pat in _TYPE_RULES:
        if re.search(pat, text):
            found.append(label)
    if any(m["role"] == "lora" for m in models):
        found.append("LoRA-assisted")
    if is_video and not any("video" in f for f in found):
        found.append("video generation")
    if len(samplers) > 1:
        found.append(f"multi-stage ({len(samplers)} sampler passes)")
    if api_nodes:
        found.append("external API / cloud-assisted")
    return list(dict.fromkeys(found))


def capabilities(graph, executed, models):
    out = []
    for label, pat, route in _CAPABILITIES:
        hits = [f"{graph.ctype(n)} #{n}" for n in executed if re.search(pat, graph.ctype(n), re.I)]
        hits += [m["file"].split("/")[-1] for m in models if re.search(pat, m["file"], re.I)]
        if hits:
            out.append({"capability": label, "evidence": hits, "route": route})
    return out


def external_services(api_nodes, assets, settings_full):
    out = [{"service": f"{a['node_type']} #{a['node_id']}", "kind": "API node",
            "note": "prompts / images may be sent to the provider"} for a in api_nodes]
    for a in assets:
        if a.get("role") == "remote URL":
            out.append({"service": a["file"], "kind": "remote URL", "note": f"{a['node_type']} #{a['node_id']}"})
    return out


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------
def reproducibility(data):
    missing = []
    if data.get("api_nodes"):
        missing.append("external API models and versions (not reproducible locally)")
    for m in data.get("models") or []:
        sha = str(m.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            missing.append(f"hash of {m['file'].split('/')[-1]}")
    for g in data.get("custom_nodes") or []:
        if g.get("version") in (None, "", "unknown"):
            missing.append(f"version of node package {g['package']}")
    for a in data.get("input_assets") or []:
        if not re.fullmatch(r"[0-9a-f]{64}", str(a.get("sha256", ""))):
            missing.append(f"hash of input {a['file']}")
    for s in data.get("sampling") or []:
        for k in (s.get("unresolved") or {}):
            missing.append(f"{k} of {s['node_type']} #{s['node_id']} (computed, not recorded)")
        for role in ("positive", "negative"):
            for e in (s.get("prompts") or {}).get(role) or []:
                if any(str(t).startswith("<linked from") for t in e["texts"].values()):
                    missing.append(f"resolved {role} prompt of {s['node_type']} #{s['node_id']}")
    if (data.get("environment") or {}).get("comfyui") in (None, "unknown"):
        missing.append("ComfyUI version")
    missing = list(dict.fromkeys(missing))
    # Things that change when the saved workflow is simply run again – the recorded values still allow a
    # reproduction (enter them as fixed values).
    rerun = []
    writers = []
    for s in data.get("sampling") or []:
        for role in ("positive", "negative"):
            for e in (s.get("prompts") or {}).get(role) or []:
                if e.get("computed_by") and e["computed_by"] not in writers:
                    writers.append(e["computed_by"])
    for w in writers:
        w = re.sub(r" #[\w:]+$", "", w)
        rerun.append(f"the prompt is written in each run by {w} – running the workflow "
                     f"again writes a new prompt "
                     f"(LLM / VLM output is not deterministic); the prompt recorded here reproduces this image")
    sm = (data.get("seed_mode") or {}).get("mode")
    if sm and sm != "fixed":
        rerun.append(f"seed control is '{sm}' – running the workflow again uses a new seed; the seed recorded here "
                     f"reproduces this image")
    if data.get("api_nodes") or any("file not found" in str(m.get("sha256", "")) for m in data.get("models") or []):
        status = "not reproducible"
    elif missing:
        status = "partially reproducible"
    elif rerun:
        status = "reproducible only with the recorded values"
    else:
        status = "reproducible"
    return {"status": status, "missing": missing, "rerun": rerun}


def _f(findings, text, severity, route, evidence, trigger=None, need=None, covered=None):
    """need: short name of what needs written approval (for the triage summary); covered: how it is
    already covered (e.g. 'studio / client approval on file') – None while it is open."""
    findings.append({"text": text, "severity": severity, "route": route, "evidence": evidence,
                     **({"trigger": trigger} if trigger else {}), **({"need": need} if need else {}),
                     **({"covered": covered} if covered else {})})


def assess(data, ctx):
    """Findings (with severity, department, evidence class), triggers, triage, guiding principles."""
    F = []
    mods = data.get("models") or []
    caps = data.get("capabilities") or []
    api = data.get("api_nodes") or []
    ctx = ctx or {}
    a = lambda k: (ctx.get(k) or "unknown")  # noqa: E731

    # ---- models / licenses (observed + model_info) -------------------------------
    # The weight of a license problem depends on whether the result is published (declared in the context).
    published = bool(ctx) and (a("in_final_cut") == "yes" or a("purpose") in ("final deliverable",
                                                                               "marketing / publicity"))
    internal_note = (" Commercial client work may count as commercial use even if nothing is published."
                     if ctx and not published else "")
    lic = {"documented": 0, "separate": 0, "restricted": 0, "unknown": 0, "conflicts": 0}
    unknown = [m for m in mods if m.get("info_matched_by") is None or m.get("license") in (None, "")
               or str(m.get("license", "")).startswith("UNKNOWN")]
    lic["unknown"] = len(unknown)
    if mods and unknown:
        names = ", ".join(m["file"].split("/")[-1] for m in unknown)
        if published:
            _f(F, f"{len(unknown)} of {len(mods)} model(s) with license {K.UNKNOWN_LABEL}, but the result is "
                  f"published: {names}. Check the licenses in writing before use.", "high", "legal",
               "external evidence required", trigger="indicated", need="model licenses (unknown)")
        else:
            _f(F, f"{len(unknown)} of {len(mods)} model(s) without source / license in the model_info file: {names}",
               "medium", "legal", "external evidence required")
    for m in mods:
        if m in unknown:
            continue
        cu = str(m.get("commercial_use", "")).lower()
        own = m.get("separate_license") == "yes"
        name = m.get("name") or m["file"].split("/")[-1]
        if cu in ("no", "restricted", "unclear") and own:
            lic["separate"] += 1
            ref = f" ({m['license_reference']})" if m.get("license_reference") else " (no reference given)"
            _f(F, f"{name}: commercial use '{cu}' under the public license – covered by a separate license{ref}. "
                  f"Keep the agreement on file.", "low", "legal", "declared")
        elif cu == "no":
            lic["conflicts"] += 1
            if published:
                _f(F, f"{name}: LICENSE CONFLICT – the license does not allow commercial use, no separate license "
                      f"is recorded, and the result is published. A studio or client approval cannot fix this – "
                      f"only a license from the rights holder, or a different model.", "critical", "legal",
                   "declared")
            else:
                _f(F, f"{name}: license does not allow commercial use and no separate license is recorded."
                      + internal_note, "high", "legal", "declared")
        elif cu in ("restricted", "unclear"):
            lic["restricted"] += 1
            _f(F, f"{name}: commercial use '{cu}' – check the license terms (e.g. revenue limits, territories) for "
                  f"this use." + internal_note, "high" if published else "medium", "legal", "declared")
        elif cu == "yes":
            lic["documented"] += 1
        else:
            lic["unknown"] += 1
            what = (f"license recorded, but commercial use is {K.UNKNOWN_LABEL} – add yes / no / restricted / "
                    f"unclear to the model_info file." if (not cu or cu.startswith("unknown")) else
                    f"commercial use '{m.get('commercial_use')}' is not one of yes / no / restricted / unclear – "
                    f"treated as {K.UNKNOWN_LABEL}.")
            _f(F, f"{name}: {what}", "high" if published else "medium", "legal", "external evidence required",
               trigger="indicated" if published else None, need=f"commercial use of {name}" if published else None)
    for m in mods:
        name = m.get("name") or m["file"].split("/")[-1]
        if m.get("hash_mismatch"):
            _f(F, f"{m['file'].split('/')[-1]}: {' and '.join(m['hash_mismatch'])} in the model_info file "
                  f"{'do' if len(m['hash_mismatch']) > 1 else 'does'} not match this file.", "medium", "docs", "observed")
        if str(m.get("sha256", "")).startswith("not computed – file not found"):
            _f(F, f"Model file not found: {m['file']} – no hash.", "medium", "vfx", "observed")
        if m.get("attribution") and str(m["attribution"]).strip().lower() not in ("no", "none", "not required"):
            _f(F, f"{name}: attribution required – {m['attribution']}", "low", "production", "declared")
    from .collect import base_model_mismatches  # local import, avoids a cycle at load time

    for m in base_model_mismatches(mods):
        _f(F, f"{m.get('name') or m['file'].split('/')[-1]} is made for {m['base_model']}, which is not the base "
              f"model of this run.", "low", "vfx", "derived")

    # ---- run (observed / derived) -------------------------------------------------
    for s in data.get("sampling") or []:
        for k in (s.get("unresolved") or {}):
            _f(F, f"{s['node_type']} #{s['node_id']}: {k} is computed in the graph and could not be recorded.",
               "low", "docs", "derived")
    if api:
        _f(F, "Cloud / API nodes were used – prompts or images may have left this machine: "
              + ", ".join(f"{x['node_type']} #{x['node_id']}" for x in api) + ". Check the provider's terms "
              "(storage, reuse, training).", "high", "security", "observed")
    for sec in data.get("secrets") or []:
        _f(F, f"{sec['node_type']} #{sec['node_id']}: credential-like value in '{sec['input']}' – redacted here, but "
              f"still stored in the workflow / PNG metadata. Remove it before sharing.", "high", "security",
           "observed")
    for cap in caps:
        talent = a("talent_likeness_or_voice")
        if talent == "no":
            sev, txt = "low", "declared: no talent likeness or voice in the inputs"
        elif talent == "yes" and a("talent_consent") == "yes":
            sev, txt = "medium", "talent involved, consent declared – keep it on file"
        else:
            sev, txt = "high", "check performer identity and consent"
        _f(F, f"Capability present: {cap['capability']} ({', '.join(cap['evidence'][:4])}) – {txt}.", sev, "talent",
           "derived", trigger=None if sev == "low" else "potential")
    for asset in data.get("input_assets") or []:
        if asset.get("location") == "network / mounted drive":
            _f(F, f"Input from a network / mounted drive: {asset['file']} – confirm the material may be used here.",
               "low", "security", "observed")
            break

    # ---- production context (declared) --------------------------------------------
    # Without a Production Context there is no triage at all (the report says 'not assessed');
    # the findings above (licenses, API, secrets, capabilities) apply anyway.
    if ctx:
        p = a("purpose")
        sa, la = a("studio_approval"), a("legal_approval")

        def approval(base):
            """(text, severity) for a written-approval trigger, depending on the declared studio approval."""
            if sa == "yes":
                return base + " – written studio / client approval on file (declared). Keep it with the project.", "low"
            if sa == "pending":
                return base + " – studio / client approval pending.", "high"
            return base + " – written approval is usually required.", "high"

        if p == "final deliverable" or a("in_final_cut") == "yes":
            t, sv = approval("Generated material is intended for the final / published result")
            _f(F, t, sv, "production", "declared", trigger="indicated", need="final / published result",
               covered="studio / client approval" if sa == "yes" else None)
        # story-critical only matters for what ends up in the result – a moodboard for a key scene is not a risk
        if a("story_critical") == "yes" and a("in_final_cut") == "yes":
            t, sv = approval("Generated elements are story-critical or prominently visible in the final result")
            _f(F, t, sv, "production", "declared", trigger="indicated", need="story-critical elements",
               covered="studio / client approval" if sa == "yes" else None)
        elif a("story_critical") == "yes" and a("in_final_cut") == "unknown":
            _f(F, "Generated elements are story-critical – unknown whether they appear in the final result. If "
                  "they do, written approval is usually required.", "medium", "production",
               "external evidence required", trigger="potential")
        if p == "marketing / publicity":
            _f(F, "Material for marketing / publicity – check rights and approvals for public use.",
               "medium" if (sa == "yes" and la == "yes") else "high", "production", "declared")
        if p == "internal production asset":
            fc = a("in_final_cut")
            if fc == "yes":
                _f(F, "Intermediate step – its output feeds the final result. Audit the final step as well; the "
                      "output SHA256 in this report links the two.", "low", "production", "declared")
            elif fc == "unknown":
                _f(F, "Internal production asset – unknown whether its output reaches the final result.", "medium",
                   "production", "external evidence required")
        if p == "unknown":
            _f(F, f"Purpose {K.UNKNOWN_LABEL}.", "medium", "production", "external evidence required")
        triggers = [
            ("unreleased_material", "Unreleased / proprietary production material entered into GenAI", "security"),
            ("personal_data", "Personal data entered into GenAI", "security"),
            ("talent_likeness_or_voice", "Talent likeness or voice used", "talent"),
            ("union_covered_work", "Union-covered performance or work affected", "talent"),
        ]
        # data that stays on a secured machine is not exposed to a GenAI provider – the risk there is the
        # client contract (is AI processing of their material allowed at all?), not data protection
        contained = (not api and data.get("environment_declared") in K.SECURE_ENVIRONMENTS
                     and a("external_tools_train") != "yes")
        for key, label, route in triggers:
            v = a(key)
            if v == "yes" and key in ("unreleased_material", "personal_data") and contained:
                what = "Unreleased / proprietary production material" if key == "unreleased_material" \
                    else "Personal data"
                extra = (" Personal data (e.g. faces) is still processed under data-protection law – keep the "
                         "legal basis on file." if key == "personal_data" else "")
                _f(F, f"{what} processed with GenAI – locally, no API / cloud node in the run, so nothing left "
                      f"this machine. Check that the client contract allows AI processing of the material.{extra}",
                   "low", "production", "declared")
            elif v == "yes":
                ok_consent = key in ("talent_likeness_or_voice", "union_covered_work") and \
                    a("talent_consent" if key == "talent_likeness_or_voice" else "union_approval") == "yes"
                _f(F, f"{label}{' – approval / consent declared, keep on file' if ok_consent else ''}.",
                   "medium" if ok_consent else "high", route, "declared", trigger="indicated",
                   need={"unreleased_material": "unreleased material sent to a cloud service",
                         "personal_data": "personal data sent to a cloud service",
                         "talent_likeness_or_voice": "talent likeness / voice",
                         "union_covered_work": "union / guild-covered work"}[key],
                   covered=("consent on file" if ok_consent else None))
            elif v == "unknown":
                _f(F, f"{K.UNKNOWN_LABEL}: {label.lower()}?", "medium", route, "external evidence required",
                   trigger="potential")
        tp = a("third_party_copyright")
        if tp == "yes":
            docs = a("input_rights_documented") == "yes" or la == "yes"
            _f(F, "Third-party copyrighted material in the inputs" + (" – rights documentation declared." if docs
               else " without documented rights."), "medium" if docs else "high", "legal", "declared",
               trigger=None if docs else "indicated", need=None if docs else "third-party material")
        elif tp == "unknown":
            _f(F, f"{K.UNKNOWN_LABEL}: third-party copyrighted material in the inputs?", "medium", "legal",
               "external evidence required")
        tm = a("trademarks_brands")
        published = a("in_final_cut") == "yes" or p in ("final deliverable", "marketing / publicity")
        if tm in ("yes", "unknown"):
            what = ("Trademarks / brands in the inputs or output" if tm == "yes" else
                    f"Trademarks / brands {K.UNKNOWN_LABEL} – names or designs may be close to existing brands")
            if la == "yes":
                _f(F, f"{what} – legal clearance declared, keep it on file.", "low", "legal", "declared")
            elif la == "pending":
                _f(F, f"{what} – legal review pending; clear it before publication.",
                   "high" if published else "medium", "legal", "declared",
                   trigger="indicated" if published else None,
                   need="trademark / brand clearance (legal review pending)" if published else None)
            else:
                _f(F, f"{what} – legal clearance not on file{' before publication' if published else ''}.",
                   "high" if published else "medium", "legal",
                   "declared" if tm == "yes" else "external evidence required",
                   trigger="indicated" if published else None,
                   need="trademark / brand clearance" if published else None)
        et = a("external_tools_train")
        if et == "yes":
            _f(F, "External tools may store, reuse or train on the data.", "high", "security", "declared",
               trigger="indicated", need="external tools that store / train on the data")
        elif et == "unknown" and api:
            _f(F, "Unknown whether the external services store, reuse or train on the data.", "high", "security",
               "external evidence required", trigger="potential")
        env = data.get("environment_declared")
        if env == "public / consumer cloud":
            _f(F, "Workflow ran in a public / consumer cloud.", "high", "security", "declared")
        for key, label in (("creative_reviewer", "creative review"), ("technical_reviewer", "technical / QC review")):
            if not (ctx.get(key) or "").strip():
                _f(F, f"No {label} recorded.", "low", "vfx", "external evidence required")
    empty = [lbl for key, lbl in (("workflow_purpose", "Purpose"), ("reference_image_sources", "Reference images"),
                                  ("external_software", "External software"))
             if not str((data.get("manual") or {}).get(key) or "").strip()]
    if empty:
        _f(F, "Not filled in: " + ", ".join(empty) + ".", "low", "docs", "external evidence required")

    order = {s: i for i, s in enumerate(K.SEVERITIES)}
    F.sort(key=lambda x: order[x["severity"]])
    for i, f in enumerate(F, 1):
        f["id"] = f"F{i:02d}"

    # ---- commercial use of the models (independent of the context) ----------------
    if not mods:
        commercial = {"status": "no model files", "models": 0}
    elif lic["conflicts"]:
        commercial = {"status": "NOT PERMITTED", "models": lic["conflicts"]}
    elif lic["unknown"]:
        commercial = {"status": K.UNKNOWN_LABEL, "models": lic["unknown"]}
    elif lic["restricted"]:
        commercial = {"status": "CHECK TERMS", "models": lic["restricted"]}
    else:
        commercial = {"status": "OK", "models": 0}
    lic["commercial_use"] = commercial

    # ---- triage ------------------------------------------------------------------
    sev = {s: sum(1 for f in F if f["severity"] == s) for s in K.SEVERITIES}
    indicated = [f for f in F if f.get("trigger") == "indicated"]
    if not ctx:
        return F, {"grade": None, "label": "NOT ASSESSED", "reason": K.NOT_ASSESSED, "counts": sev,
                   "licenses": lic, "note": K.TRIAGE_NOTE}, []
    if a("purpose") == "unknown" and a("in_final_cut") == "unknown":
        grade, why = "D", "purpose and final result undeclared"
    elif sev["critical"]:
        grade, why = "X", f"{sev['critical']} critical finding(s) – the use as declared is not permitted"
    elif indicated:
        n_cov = sum(1 for f in indicated if f.get("covered"))
        n_open = len(indicated) - n_cov
        grade = "C"
        why = (f"written approval needed for {len(indicated)} point{'s' if len(indicated) != 1 else ''}"
               + (" – all covered (declared)" if not n_open else f" – {n_open} open"))
    elif sev["high"] or sev["medium"] or a("purpose") not in K.LOW_RISK_PURPOSES or a("in_final_cut") != "no":
        grade, why = "B", "open points need review before use"
    else:
        grade, why = "A", "ideation / temporary use, no trigger, all models documented"
    routes = {}
    for f in F:
        if f["severity"] in ("critical", "high", "medium"):
            routes[f["route"]] = routes.get(f["route"], 0) + 1
    needs = []
    for f in indicated:
        nm = f.get("need") or f["text"].split(" – ")[0][:60]
        entry = {"need": nm, "covered": f.get("covered")}
        if entry not in needs:
            needs.append(entry)
    triage = {"grade": grade, "label": K.TRIAGE[grade], "reason": why, "counts": sev, "licenses": lic,
              "covered": bool(indicated) and grade == "C" and all(f.get("covered") for f in indicated),
              "needs": needs if grade in ("C", "X") else [],
              "routes": [{"route": K.ROUTES[r], "open_points": n} for r, n in
                         sorted(routes.items(), key=lambda x: -x[1])],
              "note": K.TRIAGE_NOTE}
    return F, triage, principles(data, ctx)


def principles(data, ctx):
    """Netflix-style guiding principles: yes / no / unknown with the reason. 'yes' only with evidence or
    an explicit declaration – never because nothing contrary was found."""
    a = lambda k: ((ctx or {}).get(k) or "unknown")  # noqa: E731
    api = bool(data.get("api_nodes"))
    caps = data.get("capabilities") or []
    out = []

    def add(pid, status, why):
        label = dict(K.PRINCIPLES)[pid]
        out.append({"id": pid, "principle": label, "status": status, "basis": why})

    if not ctx:
        return out
    tp, tm = a("third_party_copyright"), a("trademarks_brands")
    cleared = a("legal_approval") == "yes"
    if tp == "no" and (tm == "no" or (tm == "yes" and cleared)):
        add("P1", "yes", "declared: no third-party copyrighted material in the inputs"
            + (", brands legally cleared" if tm == "yes" else " and no brands") + " (output not inspected)")
    elif tp == "yes" and a("input_rights_documented") != "yes" and not cleared:
        add("P1", "no", "declared third-party material without documented rights")
    else:
        add("P1", "unknown", "inputs not fully declared; the output itself needs visual review")
    et = a("external_tools_train")
    if api and et == "yes":
        add("P2", "no", "external tools declared to store / reuse / train")
    elif api and et != "no":
        add("P2", "unknown", "API / cloud nodes in the run, their data terms are not declared")
    elif not api and et in ("no", "not applicable", "unknown"):
        add("P2", "yes", "no API / cloud nodes in the run (observed)" + ("" if et != "unknown" else
                                                                          "; external tools not declared"))
    else:
        add("P2", "yes", "declared: external tools do not store / reuse / train")
    env = data.get("environment_declared") or "unknown"
    if env in K.SECURE_ENVIRONMENTS:
        add("P3", "yes", f"{env}" + (" (derived: no API nodes)" if a("execution_environment").startswith("auto") else " (declared)"))
    elif env == "public / consumer cloud":
        add("P3", "no", "public / consumer cloud declared")
    else:
        add("P3", "unknown", f"environment: {env}")
    p, fc = a("purpose"), a("in_final_cut")
    if p in K.LOW_RISK_PURPOSES and fc == "no":
        add("P4", "yes", f"declared: {p}, not in the final result")
    elif p == "final deliverable" or fc == "yes":
        add("P4", "no", "declared as final deliverable / in the final result")
    else:
        add("P4", "unknown", f"purpose '{p}', final result '{fc}'")
    tl, un = a("talent_likeness_or_voice"), a("union_covered_work")
    if tl == "no" and un == "no":
        add("P5", "yes", "declared: no talent likeness / voice and no union-covered work"
            + (" – note: identity-related capability present" if caps else ""))
    elif (tl == "yes" and a("talent_consent") != "yes") or (un == "yes" and a("union_approval") != "yes"):
        add("P5", "no", "talent or union work declared without consent / approval")
    elif tl in ("yes", "no") and un in ("yes", "no"):
        add("P5", "yes", "consent / approval declared")
    else:
        add("P5", "unknown", "talent / union use not fully declared")
    return out


def effective_environment(ctx, api_nodes):
    env = (ctx or {}).get("execution_environment") or "auto (from workflow)"
    if env.startswith("auto"):
        return "mixed" if api_nodes else "local workstation"
    return env
