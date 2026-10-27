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
        named = n.get("widgets_values_named")
        if "rgthree" in str(n.get("type", "")).lower() and "seed" in str(n.get("type", "")).lower():
            # Seed (rgthree): -1 = random each run, -2 / -3 = increment / decrement, any other value is fixed.
            # 1.4.1: the named widget keeps what the user set (-1); widgets_values already holds the seed that
            # was rolled for this run – so the named value wins when the frontend provides it
            v = named.get("seed") if isinstance(named, dict) else None
            if not isinstance(v, int) or isinstance(v, bool):
                v = wv0[0] if isinstance(wv0, list) and wv0 and isinstance(wv0[0], int) else None
            if v is not None:
                return {-1: "randomize", -2: "increment", -3: "decrement"}.get(v, "fixed"), nid
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


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".exr")


def _image_size(path):
    """(width, height) from the file header, or None. Reads only the header."""
    try:
        if path.lower().endswith(".exr"):
            import OpenEXR

            with OpenEXR.File(path, header_only=True) as f:
                (x0, y0), (x1, y1) = f.header()["dataWindow"]
                return int(x1 - x0 + 1), int(y1 - y0 + 1)
        from PIL import Image

        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None


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
    """Every file / folder / URL a node of the prompt loads – also loaders that are not connected to an output
    (marked 'not used'), so nothing that was loaded into the workflow is missing from the report."""
    out = []
    try:
        import folder_paths

        configured = os.path.abspath(folder_paths.get_input_directory())
        input_dir = os.path.realpath(configured)
        default_dir = os.path.abspath(os.path.join(folder_paths.base_path, "input"))
    except Exception:
        configured = input_dir = default_dir = None
    # 1.4.1: say where the input folder really is – set with --input-directory, or linked elsewhere
    if input_dir is None:
        input_label = "ComfyUI input folder"
    elif configured != default_dir:
        input_label = "custom ComfyUI input folder (--input-directory)"
    elif input_dir != configured:
        input_label = "ComfyUI input folder (linked to another location)"
    else:
        input_label = "ComfyUI input folder"
    exe = set(executed)
    others = [n for n in graph.nodes if n not in exe]
    for nid in list(executed) + others:
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
                entry["location"] = input_label
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
            if kind == "file" and low.endswith(_IMAGE_EXTS):
                size = _image_size(full)
                if size:
                    entry["width"], entry["height"] = size
            if kind == "file" and compute_hashes:
                try:
                    sha, _cached = hashing.sha256_file(full, label=os.path.basename(full))
                    entry["sha256"] = sha
                except Exception as e:
                    entry["sha256"] = f"not computed – {type(e).__name__}"
                    if nid in exe:
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
                if nid in exe:
                    warn(f"Input file of {graph.label(nid)} not found: {show.path(val)}")
            feeds = [graph.label(c) for c in _consumers(graph, executed, nid)]
            if feeds:
                entry["feeds"] = feeds
            if nid in exe:
                entry["used_as"] = input_use(graph, executed, nid)
            else:
                entry["used_as"] = [{"use": "not used", "via": []}]
            out.append(entry)
    for i, e in enumerate(out, 1):
        e["index"] = i
    return out


def _unused(asset):
    return [u.get("use") for u in asset.get("used_as") or []] == ["not used"]


# --------------------------------------------------------------------------
# How an input is used (follows the graph downstream from the loader)
# --------------------------------------------------------------------------
USES = ("source", "guidance", "reference", "prompt reference", "preview only", "use not determined", "not used")
USE_LEGEND = {
    "source": "its pixels go into the result (img2img, inpaint, edit, video input, comp, matte plate)",
    "guidance": "steers structure or area (ControlNet image, pose, depth, mask) – its pixels are not in the result",
    "reference": "steers style or likeness (IP-Adapter, CLIP vision, style model, face ID)",
    "prompt reference": "read by a prompt writer (LLM / VLM) – it shapes the prompt, it is not in the image",
    "preview only": "only shown in a preview node – not used for the result",
    "use not determined": "connected, but the audit cannot tell how it is used",
    "not used": "loaded in the workflow, but not connected to an output – not part of the run",
}
_GUIDANCE_RE = re.compile(r"controlnet|control_net|t2i.?adapter|preprocess|canny|depth|openpose|dwpose|pose|lineart|"
                          r"softedge|\bhed\b|scribble|normal.?map|segment|\bsam\b|tile.?pre|aio", re.I)
_REFERENCE_RE = re.compile(r"ipadapter|ip_adapter|clip.?vision|style.?model|redux|instantid|pulid|faceid|photomaker|"
                           r"reactor|faceswap|face.?swap|infiniteyou|uno\b", re.I)
_SOURCE_RE = re.compile(r"vae.?encode|inpaint|img2vid|i2v|image.?to.?video|imagetovideo|encode.*video|"
                        r"reference.?latent|kontext|edit|vitmatte|trimap|composite|matte|rmbg|birefnet|inspyre|"
                        r"remove.?b(ack)?g(round)?|background|uncrop|stitch", re.I)  # uncrop / stitch: pasted back
_PREVIEW_RE = re.compile(r"preview|show|display|compare", re.I)
_MASK_INPUT_RE = re.compile(r"mask", re.I)


def _edges_from(graph, executed, nid):
    """(consumer, input name, output index) for every executed node that takes an output of `nid`."""
    out = []
    for c in executed:
        for name, v in graph.inputs(c).items():
            if isinstance(v, list) and len(v) == 2 and v[0] == nid:
                out.append((c, name, v[1]))
    return out


def _makes_conditioning(graph, nid):
    ct = graph.ctype(nid)
    c = graph.cls(ct)
    try:
        types = [str(t).upper() for t in (getattr(c, "RETURN_TYPES", None) or ())] if c is not None else None
    except Exception:
        types = None
    if types is None:
        return bool(re.search(r"encode|conditioning", ct, re.I))
    return "CONDITIONING" in types


def _is_generator(graph, nid):
    """A prompt writer (LLM / VLM / caption node): writes text, does not encode it into conditioning.
    It may take a text encoder as its language model (clip input) – that does not make it an encoder."""
    from .collect import _GENERATOR_RE, is_sampler

    return bool(_GENERATOR_RE.search(graph.ctype(nid))) and not is_sampler(graph, nid) \
        and not _makes_conditioning(graph, nid)


def _scalar_output(graph, nid, idx):
    """True for outputs that carry a number or text (frame count, folder name …), not the image itself."""
    c = graph.cls(graph.ctype(nid))
    try:
        types = getattr(c, "RETURN_TYPES", None) if c is not None else None
        t = str(types[idx]).upper() if types and idx < len(types) else ""
    except Exception:
        t = ""
    return t in ("INT", "FLOAT", "STRING", "BOOLEAN")


_IMAGE_INPUT_RE = re.compile(r"(^|[._])(image|images|img|pixels|reference|ref)(_?\d+)?$", re.I)


def _image_inputs(graph, nid):
    """Linked inputs of a node that take an image (image_1, image_2, images.image_1 …)."""
    return [k for k, v in graph.inputs(nid).items()
            if isinstance(v, list) and len(v) == 2 and _IMAGE_INPUT_RE.search(k.split(".")[-1])]


def input_use(graph, executed, nid):
    """[{use, via: [node labels]}] – how the output of a loader reaches the result. Pass-through nodes (resize,
    crop, colour …) are followed; the first node that consumes the image in a known way decides."""
    from .collect import is_sampler

    found = {}
    fallback = {}  # output nodes that pass the image on – used as "source" only if nothing further decides
    seen = set()
    stack = [(c, name, idx, 0, graph.output_name(nid, idx)) for c, name, idx in _edges_from(graph, executed, nid)
             if not _scalar_output(graph, nid, idx)]
    while stack:
        c, name, idx, depth, out_name = stack.pop()
        if (c, name) in seen or depth > 20:
            continue
        seen.add((c, name))
        ct = graph.ctype(c)
        ins = graph.inputs(c)
        use = None
        if ct.startswith("PertineoCollectMetadata"):
            # source_image only measures the resolution; output_image is passed through at the image output
            # (1.4.3) – follow it to whatever takes that output (e.g. Save Image + Metadata)
            if name == "output_image":
                for c2, n2, i2 in _edges_from(graph, executed, c):
                    if graph.output_name(c, i2) == "image":
                        stack.append((c2, n2, i2, depth + 1, "image"))
            continue
        if str(out_name).lower() == "mask" or _MASK_INPUT_RE.search(name):
            use = "guidance"
        elif _is_generator(graph, c):
            use = "prompt reference"
        elif _REFERENCE_RE.search(ct):
            use = "reference"
        elif _GUIDANCE_RE.search(ct) or name.lower() in ("control_image", "hint", "control_hint"):
            use = "guidance"
        elif _SOURCE_RE.search(ct) or name.lower() == "pixels":
            use = "source"
        elif "clip" in ins and any(k in ins for k in ("text", "prompt")):
            use = "source"  # text encoder of an edit model that reads the image (e.g. Qwen Image Edit)
        elif graph.is_output_node(ct) and not _PREVIEW_RE.search(ct) and any(
                not _scalar_output(graph, c, i2) for _c2, _n2, i2 in _edges_from(graph, executed, c)):
            # 1.4.3: an output node that also passes the image on (e.g. a crop node with its own preview) is a
            # step on the way, not the use – follow it; it only counts as source if nothing further decides
            fallback.setdefault(graph.label(c), None)
        elif graph.is_output_node(ct):
            use = "preview only" if _PREVIEW_RE.search(ct) else "source"
        elif is_sampler(graph, c):
            use = "source"
        if use:
            found.setdefault(use, [])
            lab = graph.label(c)
            if use != "preview only" and len(_image_inputs(graph, c)) > 1:
                lab += " · " + name.split(".")[-1]  # 1.4.1: which input of a multi-image node, e.g. image_1
            if lab not in found[use]:
                found[use].append(lab)
            if use != "preview only":
                continue
            # preview nodes can pass the image on (newer ComfyUI: PreviewImage has an images output) –
            # record the preview and keep following whatever takes its output
            for c2, n2, i2 in _edges_from(graph, executed, c):
                if not _scalar_output(graph, c, i2):
                    stack.append((c2, n2, i2, depth + 1, graph.output_name(c, i2)))
            continue
        nxt = _edges_from(graph, executed, c)
        if not nxt:
            found.setdefault("use not determined", []).append(graph.label(c))
        for c2, n2, i2 in nxt:
            if not _scalar_output(graph, c, i2):
                stack.append((c2, n2, i2, depth + 1, graph.output_name(c, i2)))
    if fallback and not any(u in found for u in ("source", "guidance", "reference", "prompt reference")):
        found.setdefault("source", []).extend(lab for lab in fallback if lab not in found.get("source", []))
    if len(found) > 1:
        found.pop("use not determined", None)
        if any(u in found for u in ("source", "guidance", "reference", "prompt reference")):
            found.pop("preview only", None)
    if not found:
        found = {"use not determined": []}
    return [{"use": u, "via": found[u]} for u in USES if u in found]


# --------------------------------------------------------------------------
# What a prompt writer (LLM / VLM node) was given
# --------------------------------------------------------------------------
_IMAGE_IN_RE = re.compile(r"^(image|images|img|pixels|input_image|frames|video|picture|photo)\d*$", re.I)
_SKIP_WRITER_RE = re.compile(r"model|quant|device|attention|dtype|precision|keep|unload|offload|memory|cache|gpu|"
                             r"cpu|vram|clip|vae|upload|^lora", re.I)
_WRITER_SETTING_KEYS = re.compile(r"seed|temperature|top_p|top_k|min_p|max_tokens|max_new_tokens|max_length|"
                                  r"repetition_penalty|num_beams|do_sample|length", re.I)
WRITER_TEXT_WORDS = 512  # PDF only; the .audit.txt and .audit.json keep every text in full


def _upstream_files(graph, link, assets_by_node, depth=0, seen=None):
    """Input files that reach a node through `link` (through resize, crop … nodes)."""
    seen = seen if seen is not None else set()
    if not (isinstance(link, list) and len(link) == 2) or depth > 20 or link[0] in seen:
        return [], []
    src = link[0]
    seen.add(src)
    if src in assets_by_node:
        return list(assets_by_node[src]), []
    files, made = [], []
    if _is_generator(graph, src):
        return [], [graph.label(src)]
    ups = [v for v in graph.inputs(src).values() if isinstance(v, list) and len(v) == 2]
    if not ups:
        made.append(graph.label(src))
    for v in ups:
        f, m = _upstream_files(graph, v, assets_by_node, depth + 1, seen)
        files += [x for x in f if x not in files]
        made += [x for x in m if x not in made]
    return files, made


def prompt_writers(graph, executed, sampling, assets, models, show):
    """Every LLM / VLM / prompt node that writes text used by a text encoder, with what it was given:
    images (traced back to the input files), instructions and presets, linked texts (resolved to what they
    held in this run), settings, and upstream writers. Nothing is guessed – unresolvable values are marked."""
    from .collect import is_link

    assets_by_node = {}
    for a in assets:
        assets_by_node.setdefault(a["node_id"], []).append(a)
    exe = set(executed)
    # writers that feed the prompts of a sampler (found by walking upstream from the text encoders)
    roots = []
    for s in sampling or []:
        for role in ("positive", "negative"):
            for e in (s.get("prompts") or {}).get(role) or []:
                if e.get("zeroed"):
                    continue  # zeroed-out conditioning: no prompt, even if the encoder behind it has one
                for k, raw in graph.inputs(e["node_id"]).items():
                    if not is_link(raw):
                        continue
                    for w in _writers_upstream(graph, raw):
                        roots.append((w, f"{role} prompt of {graph.label(s['node_id'])}"))
    out, order = {}, []

    def visit(wid, feeds, depth):
        if wid not in exe or depth > 6:
            return
        if wid in out:
            if feeds not in out[wid]["feeds"]:
                out[wid]["feeds"].append(feeds)
            return
        entry = {"node_id": wid, "node_type": graph.ctype(wid), "title": graph.title(wid), "feeds": [feeds],
                 "images": [], "texts": [], "settings": {}, "models": []}
        out[wid] = entry
        order.append(wid)
        for m in models or []:
            users = re.split(r",\s*", str(m.get("used_for") or "").split(" – ", 1)[-1])
            if m.get("node_id") == wid or graph.label(wid) in users:
                entry["models"].append(m["file"])
        ins = graph.inputs(wid)
        for name, raw in ins.items():
            if is_link(raw):
                src = raw[0]
                if src not in graph.nodes:
                    continue
                if _IMAGE_IN_RE.match(name) or str(graph.output_name(src, raw[1])).upper() in ("IMAGE", "MASK"):
                    files, made = _upstream_files(graph, raw, assets_by_node)
                    entry["images"].append({"input": name, "files": [f["index"] for f in files],
                                            "file_names": [f["file"] for f in files],
                                            "produced_by": made})
                    for w in made:
                        m = re.search(r"#([\w:]+)$", w)
                        if m and _is_generator(graph, m.group(1)):
                            visit(m.group(1), f"image of {graph.label(wid)}", depth + 1)
                    continue
                if _SKIP_WRITER_RE.search(name):
                    continue
                if _is_generator(graph, src):
                    ok, val, note, chain = graph.resolve(raw)
                    entry["texts"].append({"input": name, "from": graph.label(src), "written_by": graph.label(src),
                                           "text": val if ok and isinstance(val, str) else None,
                                           "note": None if ok else note})
                    visit(src, f"{name} of {graph.label(wid)}", depth + 1)
                    continue
                ok, val, note, chain = graph.resolve(raw)
                if ok and isinstance(val, str):
                    entry["texts"].append({"input": name, "from": " → ".join(c for c in chain
                                                                              if c != "value produced in this run"),
                                           "text": _redact(name, val, show)})
                    # a text joined from several pieces: list the writers behind it as well
                    for w in _writers_upstream(graph, raw):
                        visit(w, f"{name} of {graph.label(wid)}", depth + 1)
                elif ok and isinstance(val, (int, float, bool)):
                    entry["settings"][name] = val
                elif not ok:
                    entry["texts"].append({"input": name, "from": " → ".join(chain), "text": None, "note": note})
                continue
            if isinstance(raw, dict):
                continue
            if isinstance(raw, str):
                if _SKIP_WRITER_RE.search(name) or raw.lower().endswith((".gguf", ".safetensors", ".bin", ".pt",
                                                                         ".pth", ".ckpt")):
                    continue
                if raw.strip():
                    entry["texts"].append({"input": name, "from": None, "text": _redact(name, raw, show)})
            elif isinstance(raw, (int, float, bool)) and _WRITER_SETTING_KEYS.search(name):
                entry["settings"][name] = raw
        rv = graph.runtime_value(wid, 0)
        if isinstance(rv, str):
            entry["output"] = rv

    for w, feeds in roots:
        visit(w, feeds, 0)
    return [out[w] for w in order]


def _redact(name, val, show):
    if _SECRET_NAME_RE.search(name) or _SECRET_VALUE_RE.match(val.strip()):
        return "REDACTED"
    return show.text(val) if (os.sep in val and not show.full) else val


def _writers_upstream(graph, link, depth=0, seen=None):
    """Nearest prompt-writer nodes upstream of a text link."""
    seen = seen if seen is not None else set()
    if not (isinstance(link, list) and len(link) == 2) or depth > 15 or link[0] in seen or link[0] not in graph.nodes:
        return []
    src = link[0]
    seen.add(src)
    if _is_generator(graph, src):
        return [src]
    out = []
    for name, v in graph.inputs(src).items():
        if isinstance(v, list) and len(v) == 2 and not _IMAGE_IN_RE.match(name) and not _SKIP_WRITER_RE.search(name):
            out += [w for w in _writers_upstream(graph, v, depth + 1, seen) if w not in out]
    return out


# --------------------------------------------------------------------------
# Workflow outline from the groups of the workflow
# --------------------------------------------------------------------------
def _xy(v):
    if isinstance(v, dict):
        v = [v.get("0", v.get(0, 0)), v.get("1", v.get(1, 0))]
    try:
        return float(v[0]), float(v[1])
    except Exception:
        return None


def workflow_outline(extra_pnginfo, graph, executed):
    """The groups of the workflow (as the artist laid them out) with the nodes in each, in reading order
    (left to right, top to bottom). Nodes that are not part of the run are counted, not listed."""
    wf = (extra_pnginfo or {}).get("workflow") if isinstance(extra_pnginfo, dict) else None
    if not isinstance(wf, dict):
        return None
    groups = [g for g in (wf.get("groups") or []) if isinstance(g, dict) and isinstance(g.get("bounding"), list)
              and len(g["bounding"]) == 4]
    if not groups:
        return None
    exe_top = {}
    for n in executed:
        exe_top.setdefault(str(n).split(":")[0], []).append(n)
    prompt_top = {str(n).split(":")[0] for n in graph.nodes}
    members = {i: [] for i in range(len(groups))}
    ungrouped = []
    for n in wf.get("nodes") or []:
        if not isinstance(n, dict) or "id" not in n:
            continue
        nid = str(n["id"])
        state = _mode_name(n.get("mode"))
        if not state:
            if nid in exe_top:
                state = "run"
            elif nid in prompt_top:
                state = "not connected"
            else:
                continue  # notes, reroutes, primitives, Set/Get – not nodes that run
        p = _xy(n.get("pos"))
        if p is None:
            continue
        sz = _xy(n.get("size")) or (0.0, 0.0)
        cx, cy = p[0] + sz[0] / 2, p[1] + min(sz[1], 40) / 2
        best = None
        for i, g in enumerate(groups):
            x, y, w, h = [float(t) for t in g["bounding"]]
            if x <= cx <= x + w and y <= cy <= y + h and (best is None or w * h < best[1]):
                best = (i, w * h)
        item = {"node_id": nid, "node_type": n.get("type") or graph.ctype(nid), "title": n.get("title"),
                "state": state}
        (members[best[0]] if best else ungrouped).append(item)
    order = sorted(range(len(groups)), key=lambda i: (round(float(groups[i]["bounding"][0]) / 50),
                                                      float(groups[i]["bounding"][1])))
    out = []
    for i in order:
        mem = members[i]
        if not mem:
            continue
        out.append({"title": str(groups[i].get("title") or "").strip() or "(untitled group)",
                    "nodes": [m for m in mem if m["state"] == "run"],
                    "inactive": [m for m in mem if m["state"] != "run"]})
    return {"groups": out, "ungrouped": [m for m in ungrouped if m["state"] == "run"],
            "ungrouped_inactive": len([m for m in ungrouped if m["state"] != "run"])}


# --------------------------------------------------------------------------
# Workflow copy for the .json (1.4.1)
# --------------------------------------------------------------------------
def _secret_values(prompt, workflow):
    """Strings that look like credentials: values of secret-named inputs (API prompt, named widgets of the
    UI workflow) and values that match a known key format."""
    vals = set()
    for node in (prompt or {}).values():
        for name, v in ((node or {}).get("inputs") or {}).items() if isinstance(node, dict) else ():
            if isinstance(v, str) and v.strip() and (_SECRET_NAME_RE.search(name) or _SECRET_VALUE_RE.match(v.strip())):
                vals.add(v)

    def walk(o):
        if isinstance(o, dict):
            named = o.get("widgets_values_named")
            if isinstance(named, dict):
                for name, v in named.items():
                    if isinstance(v, str) and v.strip() and _SECRET_NAME_RE.search(str(name)):
                        vals.add(v)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(workflow)
    return vals


def redacted_copy(obj, secrets):
    """Deep copy of a workflow / prompt with every credential-looking string replaced by REDACTED."""
    if isinstance(obj, dict):
        return {k: redacted_copy(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redacted_copy(v, secrets) for v in obj]
    if isinstance(obj, str) and obj.strip() and (obj in secrets or _SECRET_VALUE_RE.match(obj.strip())):
        return "REDACTED"
    return obj


def workflow_for_json(prompt, extra_pnginfo):
    """{"workflow": UI workflow, "prompt": API prompt} with credentials redacted, or None."""
    wf = (extra_pnginfo or {}).get("workflow") if isinstance(extra_pnginfo, dict) else None
    if not wf and not prompt:
        return None
    secrets = _secret_values(prompt, wf)
    out = {"note": "The workflow as it was run. Credentials are redacted. To load it in ComfyUI, save the "
                   "'workflow' object as its own .json file."}
    if wf:
        out["workflow"] = redacted_copy(wf, secrets)
    if prompt:
        out["prompt"] = redacted_copy(prompt, secrets)
    out["redacted"] = len(secrets)
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


_COND_IMAGE_SKIP_RE = re.compile(r"controlnet|ipadapter|stylemodel|clipvision|redux|instantid|pulid", re.I)
_COND_KEYS = ("positive", "negative", "conditioning", "conditioning_1", "conditioning_2", "conditioning_to",
              "conditioning_from", "cond", "guider")


def _conditioning_has_image(graph, root, depth=0, seen=None):
    """True when an image is encoded into the conditioning itself (edit models such as Qwen Image Edit /
    TextEncodeQwenImage*, Kontext, ReferenceLatent) – the sampler then edits that image even though its latent
    is empty. ControlNet / IP-Adapter images are guidance, not this, and are reported separately."""
    from .collect import is_link
    seen = set() if seen is None else seen
    if not is_link(root) or depth > 12 or root[0] in seen or root[0] not in graph.nodes:
        return False
    nid = root[0]
    seen.add(nid)
    ct = graph.ctype(nid)
    ins = graph.inputs(nid)
    if not _COND_IMAGE_SKIP_RE.search(ct):
        if re.search(r"reference.?latent|kontext", ct, re.I) and is_link(ins.get("latent")):
            return True
        if "clip" in ins and any(is_link(v) and _IMAGE_IN_RE.match(str(k).split(".")[-1].replace("image_", "image"))
                                 for k, v in ins.items()):
            return True
    return any(_conditioning_has_image(graph, ins.get(k), depth + 1, seen) for k in _COND_KEYS)


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
        sins = graph.inputs(s.get("node_id"))
        if lat.get("image_based"):
            pass  # tiled refiners / upscalers (e.g. Ultimate SD Upscale) – covered by the rules below
        elif any(_conditioning_has_image(graph, sins.get(k)) for k in ("positive", "negative", "guider")):
            # 1.4.3: an empty latent with the image in the conditioning is an edit, not text-to-image
            found.append("image-conditioned video" if is_video else "image edit (image in the conditioning)")
        elif "empty" in src:
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
        if a.get("role") == "remote URL" and not _unused(a):
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
        if _unused(a):
            continue  # not part of the run
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
        if asset.get("location") == "network / mounted drive" and not _unused(asset):
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
