# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Collects audit information from the executed prompt graph.

Everything here is defensive: any problem becomes a warning in the report, never an
exception that stops the generation.
"""

import datetime
import importlib.metadata
import inspect
import json
import os
import platform
import re
import sys

from ..common import LOG
from . import hashing

SCHEMA = "pertineo-audit/1"
UNKNOWN_INFO = "UNKNOWN – add to model_info file"
NOT_PROVIDED = "– not provided –"
MODEL_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".sft", ".bin")
MODEL_INFO_DEFAULT = "model_info.json"
DISCLAIMER_KEY = "_disclaimer"
TOOL_DISCLAIMER = ("License and source information in this report comes from the model_info file of the person "
                   "who ran the workflow. It is not verified by Pertineo VFX, comes without warranty and is not "
                   "legal advice. The license published by each model's author always prevails.")
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int) \
        and not isinstance(v[1], bool)


def _id_key(nid):
    """Natural sort for node ids incl. subgraph ids like '12:5'."""
    return [int(p) if p.isdigit() else p for p in re.split(r"[:]", str(nid))]


def _read_toml_version(pyproject):
    try:
        import tomllib

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        v = data.get("project", {}).get("version")
        return str(v) if v else None
    except Exception:
        try:
            with open(pyproject, "r", encoding="utf-8") as f:
                m = re.search(r'^\s*version\s*=\s*"([^"]+)"', f.read(), re.M)
            return m.group(1) if m else None
        except Exception:
            return None


def own_version():
    return _read_toml_version(os.path.join(_PKG_DIR, "pyproject.toml")) or "unknown"


def _git_short_hash(repo_dir):
    """Reads the checked-out commit from .git without running git."""
    try:
        git = os.path.join(repo_dir, ".git")
        if os.path.isfile(git):  # worktree / submodule: "gitdir: ..."
            with open(git) as f:
                git = os.path.join(repo_dir, f.read().split(":", 1)[1].strip())
        with open(os.path.join(git, "HEAD")) as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(" ", 1)[1].strip()
        ref_file = os.path.join(git, ref)
        if os.path.isfile(ref_file):
            with open(ref_file) as f:
                return f.read().strip()[:7]
        with open(os.path.join(git, "packed-refs")) as f:
            for line in f:
                if line.strip().endswith(ref):
                    return line.split()[0][:7]
    except Exception:
        pass
    return None


class PathDisplay:
    """Turns absolute paths into shareable names unless full paths are wanted."""

    def __init__(self, full):
        self.full = full
        roots = []
        try:
            import folder_paths

            roots.append((folder_paths.models_dir, "<models>"))
            roots.append((folder_paths.base_path, "<ComfyUI>"))
            for name, (paths, _exts) in folder_paths.folder_names_and_paths.items():
                for p in paths:
                    roots.append((p, f"<{name}>"))
        except Exception:
            pass
        roots.append((os.path.expanduser("~"), "~"))
        # longest first, so the most specific root wins
        self.roots = sorted({(os.path.normpath(r), n) for r, n in roots if r}, key=lambda x: -len(x[0]))

    def path(self, p):
        if p is None or self.full:
            return p
        if not os.path.isabs(str(p)):
            return p
        return os.path.basename(str(p))

    def text(self, s):
        """Redacts absolute paths inside free text (warnings, error messages)."""
        if self.full or not s:
            return s
        s = str(s)
        for root, name in self.roots:
            s = s.replace(root + os.sep, name + "/").replace(root, name)
        return re.sub(r"(?<![\w<>~.])/(?:[^\s/'\"]+/)+([^\s/'\"]+)", r"\1", s)


# --------------------------------------------------------------------------
# Graph access and value resolution
# --------------------------------------------------------------------------
# Known nodes that only pass a widget value through: output index -> input name.
PASS_THROUGH = {
    "KSampler Config (rgthree)": ["steps_total", "refiner_step", "cfg", "sampler_name", "scheduler"],
    "Seed (rgthree)": ["seed"],
    "PrimitiveInt": ["value"],
    "PrimitiveFloat": ["value"],
    "PrimitiveString": ["value"],
    "PrimitiveStringMultiline": ["value"],
    "PrimitiveBoolean": ["value"],
}
OUTPUT_NAMES = {
    "KSampler Config (rgthree)": ["STEPS", "REFINER_STEP", "CFG", "SAMPLER", "SCHEDULER"],
}
# Node types that look like literal / config nodes (their outputs equal their widgets).
_LITERAL_RE = re.compile(r"primitive|literal|constant|config|settings|seed|^(int|float|string|boolean|bool|number|text)\b",
                         re.I)
_COMPUTED_RE = re.compile(r"random|math|convert|concat|join|replace|format|calc|expr|eval|op\b|to ", re.I)


class Graph:
    def __init__(self, prompt, class_mappings=None, execution_list=None):
        self.nodes = {str(k): v for k, v in (prompt or {}).items() if isinstance(v, dict)}
        self.mappings = class_mappings or {}
        self.execution_list = execution_list

    def runtime_value(self, nid, idx):
        """The value a node actually produced in this run (from ComfyUI's output cache), if it is a
        plain value (text / number / bool) and the node has already run. Otherwise None."""
        cache = getattr(self.execution_list, "output_cache", None)
        if cache is None or not hasattr(cache, "get_local"):
            return None
        try:
            entry = cache.get_local(nid)
            outs = getattr(entry, "outputs", None)
            if outs is None or idx >= len(outs):
                return None
            v = outs[idx]
            if isinstance(v, (list, tuple)):
                if len(v) != 1:
                    return None
                v = v[0]
            return v if isinstance(v, (str, int, float, bool)) else None
        except Exception:
            return None

    def _computed(self, src, idx, unresolved, chain):
        """Last resort for a computed value: what the node produced in this run, never a guess."""
        v = self.runtime_value(src, idx)
        if v is None:
            return False, None, unresolved, chain
        return True, v, None, chain + ["value produced in this run"]

    def ctype(self, nid):
        return self.nodes.get(str(nid), {}).get("class_type", "?")

    def inputs(self, nid):
        return self.nodes.get(str(nid), {}).get("inputs", {}) or {}

    def title(self, nid):
        return (self.nodes.get(str(nid), {}).get("_meta") or {}).get("title")

    def cls(self, ct):
        return self.mappings.get(ct)

    def output_names(self, ct):
        if ct in OUTPUT_NAMES:
            return OUTPUT_NAMES[ct]
        c = self.cls(ct)
        if c is None:
            return None
        try:
            names = getattr(c, "RETURN_NAMES", None) or getattr(c, "RETURN_TYPES", None)
            return [str(n) for n in names] if names else None
        except Exception:
            return None

    def output_name(self, nid, idx):
        names = self.output_names(self.ctype(nid))
        return names[idx] if names and 0 <= idx < len(names) else f"#{idx}"

    def is_output_node(self, ct):
        c = self.cls(ct)
        try:
            return bool(c is not None and getattr(c, "OUTPUT_NODE", False))
        except Exception:
            return False

    def executed_ids(self):
        """Nodes that contribute to an output node (what ComfyUI actually runs).
        Without class information every node in the prompt is used."""
        outs = [n for n in self.nodes if self.is_output_node(self.ctype(n))]
        if not outs:
            return sorted(self.nodes, key=_id_key)
        seen, stack = set(), list(outs)
        while stack:
            n = stack.pop()
            if n in seen or n not in self.nodes:
                continue
            seen.add(n)
            for v in self.inputs(n).values():
                if is_link(v):
                    stack.append(v[0])
        return sorted(seen, key=_id_key)

    def label(self, nid):
        return f"{self.ctype(nid)} #{nid}"

    # ----------------------------------------------------------------------
    def resolve(self, value, _depth=0, _chain=None):
        """Follows a link upstream to a constant widget value.
        Returns (ok, value, note, chain). Never invents a value: if the source is
        computed, ok is False and note says where the value comes from."""
        chain = list(_chain or [])
        if not is_link(value):
            return True, value, None, chain
        src, idx = value
        if src not in self.nodes:
            return False, None, f"linked from missing node #{src}", chain
        ct = self.ctype(src)
        ins = self.inputs(src)
        chain.append(self.label(src))
        unresolved = f"linked from {ct} #{src}, output {self.output_name(src, idx)}"
        if _depth > 12:
            return self._computed(src, idx, unresolved, chain)

        name = None
        if ct in PASS_THROUGH:
            names = PASS_THROUGH[ct]
            name = names[idx] if idx < len(names) else None
        elif {"on_true", "on_false", "boolean"} <= set(ins):  # e.g. Switch any [Crystools]
            ok, b, _n, chain2 = self.resolve(ins["boolean"], _depth + 1, chain)
            if ok and isinstance(b, bool):
                return self.resolve(ins["on_true" if b else "on_false"], _depth + 1, chain2)
            return self._computed(src, idx, unresolved, chain)
        elif _LITERAL_RE.search(ct) and not _COMPUTED_RE.search(ct):
            widget_inputs = {k: v for k, v in ins.items() if not is_link(v)}
            links = [k for k, v in ins.items() if is_link(v)]
            outs = self.output_names(ct)
            if len(ins) == 1 and idx == 0 and (outs is None or len(outs) == 1):
                name = next(iter(ins))
            elif not links and outs and idx < len(outs):
                out = outs[idx].lower()
                for k in widget_inputs:
                    kl = k.lower()
                    if kl == out or kl.startswith(out + "_") or out.startswith(kl + "_"):
                        name = k
                        break
        if name is None or name not in ins:
            return self._computed(src, idx, unresolved, chain)
        return self.resolve(ins[name], _depth + 1, chain)

    def value(self, nid, name):
        """(ok, value, note, chain) for one input of a node."""
        ins = self.inputs(nid)
        if name not in ins:
            return False, None, None, []
        return self.resolve(ins[name])


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def environment(warn):
    env = {}
    base = None
    try:
        import folder_paths

        base = folder_paths.base_path
    except Exception:
        pass
    try:
        import comfyui_version

        env["comfyui"] = comfyui_version.__version__
    except Exception:
        env["comfyui"] = (_read_toml_version(os.path.join(base, "pyproject.toml")) if base else None) or "unknown"
    if base:
        commit = _git_short_hash(base)
        if commit:
            env["comfyui_commit"] = commit
    try:
        env["frontend"] = importlib.metadata.version("comfyui-frontend-package")
    except Exception:
        env["frontend"] = "unknown (package comfyui-frontend-package not found)"
    for arg in sys.argv:
        if arg.startswith("--front-end-version"):
            env["frontend_override"] = arg
    env["python"] = platform.python_version()
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda or "none"
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            env["gpu"] = f"{props.name} ({props.total_memory / 1024 ** 3:.1f} GiB)"
        else:
            env["gpu"] = "none (CPU)"
    except Exception as e:
        warn(f"torch / GPU info not available ({type(e).__name__})")
    os_name = platform.system()
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', f.read(), re.M)
            if m:
                os_name = m.group(1)
    except Exception:
        if os_name == "Darwin":
            os_name = f"macOS {platform.mac_ver()[0]}"
        elif os_name == "Windows":
            os_name = f"Windows {platform.release()} ({platform.version()})"
    env["os"] = f"{os_name} · kernel {platform.release()}" if platform.system() == "Linux" else os_name
    env["pertineo_vfx"] = own_version()
    return env


# --------------------------------------------------------------------------
# Custom nodes / packages
# --------------------------------------------------------------------------
def workflow_package_info(extra_pnginfo):
    """node type -> {package, version, source} from the node properties saved in the workflow
    (cnr_id / ver, or aux_id for git installs). Includes subgraph definitions."""
    info = {}
    wf = (extra_pnginfo or {}).get("workflow") if isinstance(extra_pnginfo, dict) else None
    if not isinstance(wf, dict):
        return info
    node_lists = [wf.get("nodes") or []]
    for sg in ((wf.get("definitions") or {}).get("subgraphs") or []):
        node_lists.append(sg.get("nodes") or [])
    for nodes in node_lists:
        for n in nodes:
            if not isinstance(n, dict):
                continue
            ct = n.get("type")
            props = n.get("properties") or {}
            pkg = props.get("cnr_id") or props.get("aux_id")
            if not ct or not pkg or ct in info:
                continue
            info[ct] = {"package": str(pkg), "version": str(props.get("ver") or "unknown"),
                        "source": "workflow (cnr_id)" if props.get("cnr_id") else "workflow (aux_id)"}
    return info


def module_package_info(cls, comfy_version):
    """Fallback: package from the file the node class is defined in."""
    try:
        raw = os.path.abspath(inspect.getfile(cls))
    except Exception:
        return None
    # compare unresolved and resolved paths: custom node folders are often symlinks (dev checkouts)
    paths = list(dict.fromkeys([raw, os.path.realpath(raw)]))
    try:
        import folder_paths

        custom_dirs = []
        for p in folder_paths.get_folder_paths("custom_nodes"):
            custom_dirs += [os.path.abspath(p), os.path.realpath(p)]
        base = os.path.realpath(folder_paths.base_path)
    except Exception:
        custom_dirs, base = [], None
    for path in paths:
        for d in dict.fromkeys(custom_dirs):
            if path.startswith(d + os.sep):
                pkg = os.path.relpath(path, d).split(os.sep)[0]
                pkg_dir = os.path.join(d, pkg)
                ver = (_read_toml_version(os.path.join(pkg_dir, "pyproject.toml")) or _git_short_hash(pkg_dir)
                       or "unknown")
                return {"package": pkg, "version": ver, "source": "module path"}
    if os.path.realpath(raw).startswith(_PKG_DIR + os.sep) or raw.startswith(_PKG_DIR + os.sep):
        return {"package": "comfyui-pertineo-vfx", "version": own_version(), "source": "module path"}
    if base and os.path.realpath(raw).startswith(base + os.sep):
        return {"package": "comfy-core", "version": comfy_version, "source": "module path"}
    return None


def is_api_node(cls, ct):
    try:
        if getattr(cls, "API_NODE", False) is True:
            return True
    except Exception:
        pass
    try:
        if "comfy_api_nodes" in inspect.getfile(cls):
            return True
    except Exception:
        pass
    return False


def custom_nodes(graph, executed, extra_pnginfo, comfy_version, warn):
    wf_info = workflow_package_info(extra_pnginfo)
    groups, api = {}, []
    for nid in executed:
        ct = graph.ctype(nid)
        cls = graph.cls(ct)
        info = wf_info.get(ct)
        if info is None and cls is not None:
            info = module_package_info(cls, comfy_version)
        elif info is not None and info["source"].endswith("(aux_id)"):
            # git install: 'ver' is the commit the frontend saw; add the package version from its folder
            info = dict(info)
            commit = info["version"]
            if re.fullmatch(r"[0-9a-f]{12,40}", commit):
                commit = commit[:7]
            mod = module_package_info(cls, comfy_version) if cls is not None else None
            mod_ver = mod["version"] if mod else None
            if mod_ver and mod_ver != "unknown" and not re.fullmatch(r"[0-9a-f]{7,40}", mod_ver):
                info["version"] = f"{mod_ver} (git {commit})"
            elif commit != "unknown":
                info["version"] = f"git {commit}"
        if info is None:
            info = {"package": "unknown", "version": "unknown", "source": "not found"}
        key = (info["package"], info["version"])
        g = groups.setdefault(key, {"package": info["package"], "version": info["version"],
                                    "source": info["source"], "node_types": {}})
        g["node_types"][ct] = g["node_types"].get(ct, 0) + 1
        if (cls is not None and is_api_node(cls, ct)) or info["package"] == "comfy_api_nodes":
            api.append({"node_id": nid, "node_type": ct, "package": info["package"]})
    out = []
    for g in sorted(groups.values(), key=lambda g: (g["package"] != "comfy-core", g["package"].lower())):
        g["node_types"] = [{"type": t, "count": c} for t, c in sorted(g["node_types"].items())]
        out.append(g)
    if any(g["package"] == "unknown" for g in out):
        warn("Package of some node types could not be determined (see Custom Nodes: unknown).")
    return out, api


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
_FOLDER_BY_INPUT = [
    (r"^ckpt_name$|checkpoint", ["checkpoints"]),
    (r"^unet_name$|diffusion|^model_name$", ["diffusion_models", "checkpoints"]),
    (r"^vae_name$|vae", ["vae"]),
    (r"^clip_name\d*$|text_encoder|^t5|^llm|encoder", ["text_encoders"]),
    (r"lora", ["loras"]),
    (r"control_net|controlnet", ["controlnet"]),
    (r"clip_vision", ["clip_vision"]),
    (r"style_model", ["style_models"]),
    (r"upscale", ["upscale_models", "latent_upscale_models"]),
    (r"ipadapter|ip_adapter", ["ipadapter"]),
    (r"gligen", ["gligen"]),
    (r"hypernetwork", ["hypernetworks"]),
    (r"photomaker", ["photomaker"]),
    (r"patch", ["model_patches"]),
    (r"audio", ["audio_encoders"]),
]


def model_role(ct, input_name):
    c, i = ct.lower(), input_name.lower()
    if "lora" in i or "lora" in c:
        return "lora"
    if "control_net" in i or "controlnet" in c:
        return "controlnet"
    if "clip_vision" in i or "clipvision" in c:
        return "clip_vision"
    if "ipadapter" in i or "ipadapter" in c or "ip_adapter" in i:
        return "ip_adapter"
    if "upscale" in c or "upscale" in i:
        return "upscaler"
    if "style_model" in i:
        return "style_model"
    if i == "ckpt_name" or "checkpoint" in c:
        return "checkpoint"
    if i == "vae_name" or ("vae" in i and "vae" in c):
        return "vae"
    if re.match(r"clip_name\d*$", i) or "text_encoder" in i or "cliploader" in c or "textencoder" in c:
        return "text_encoder"
    if re.search(r"llm|vlm|florence|qwen.?vl|llava|joycaption|caption|gemma", c + " " + i):
        return "vlm"
    if i == "unet_name" or "unet" in c or "diffusion" in c or "diffusion" in i:
        return "diffusion_model"
    return f"model ({input_name})"


def _candidate_folders(input_name, role):
    i = input_name.lower()
    folders = []
    for pattern, names in _FOLDER_BY_INPUT:
        if re.search(pattern, i):
            folders += names
    role_map = {"lora": "loras", "vae": "vae", "text_encoder": "text_encoders", "checkpoint": "checkpoints",
                "diffusion_model": "diffusion_models", "controlnet": "controlnet", "upscaler": "upscale_models",
                "clip_vision": "clip_vision", "ip_adapter": "ipadapter", "style_model": "style_models"}
    if role in role_map:
        folders.insert(0, role_map[role])
    return list(dict.fromkeys(folders))


def find_model_file(name, input_name, role):
    """Returns (full path or None, folder key or None)."""
    if os.path.isabs(name):
        return (name, None) if os.path.isfile(name) else (None, None)
    try:
        import folder_paths
    except Exception:
        return None, None
    tried = _candidate_folders(input_name, role)
    rest = [k for k in folder_paths.folder_names_and_paths if k not in tried and k != "custom_nodes"]
    for folder in tried + rest:
        try:
            p = folder_paths.get_full_path(folder, name)
        except Exception:
            p = None
        if p:
            return p, folder
    return None, None


_STRENGTH_KEYS = ("strength_model", "strength_clip", "strength", "lora_strength", "model_strength", "clip_strength",
                  "weight", "strengthTwo")


def _model_refs(graph, nid, warn):
    """Yields (input name, file name, extra dict) for every model file referenced by a node."""
    ct = graph.ctype(nid)
    ins = graph.inputs(nid)
    if ct.startswith("PertineoModelInfo"):  # only reads model_info, loads nothing
        return
    for name, raw in ins.items():
        if isinstance(raw, dict):  # e.g. Power Lora Loader (rgthree): {"on": true, "lora": "...", "strength": 1}
            if raw.get("on") is False:
                continue
            for k, v in raw.items():
                if isinstance(v, str) and v.lower().endswith(MODEL_EXTS):
                    extra = {sk: raw[sk] for sk in _STRENGTH_KEYS if isinstance(raw.get(sk), (int, float))}
                    yield f"{name}.{k}", v, extra
            continue
        ok, val, _note, _chain = graph.resolve(raw)
        if not ok or not isinstance(val, str) or not val.lower().endswith(MODEL_EXTS):
            continue
        extra = {}
        if "lora" in name.lower() or "lora" in ct.lower():
            for sk in _STRENGTH_KEYS:
                if sk in ins:
                    sok, sval, snote, _ = graph.resolve(ins[sk])
                    extra[sk] = sval if sok else snote
        yield name, val, extra

    # Pertineo's own ViTMatte nodes reference a Hugging Face model folder, not a file
    if ct in ("PertineoViTMatteRefiner", "PertineoViTMatteSequence"):
        try:
            from ..matting.core import CUSTOM_MODEL, MODEL_REPOS, find_model_dir, resolve_custom_model

            ok, choice, _n, _c = graph.resolve(ins.get("model"))
            if ok and choice in MODEL_REPOS:
                d = find_model_dir(MODEL_REPOS[choice])
                yield "model", MODEL_REPOS[choice] + "/model.safetensors", {
                    "_path": os.path.join(d, "model.safetensors") if d else None, "repo": MODEL_REPOS[choice]}
            elif ok and choice == CUSTOM_MODEL:
                cok, cm, _n, _c = graph.resolve(ins.get("custom_model"))
                d = resolve_custom_model(cm) if cok and cm else None
                yield "custom_model", "custom ViTMatte/model.safetensors", {
                    "_path": os.path.join(d, "model.safetensors") if d else None}
        except Exception as e:
            warn(f"ViTMatte model of {graph.label(nid)} could not be resolved ({type(e).__name__}: {e})")


# Column order of new tables (what people read first comes first); the reader accepts any order.
CSV_COLUMNS = ["type", "base_model", "name", "trigger_words", "recommended", "commercial_use", "license", "checked",
               "file", "version", "source", "released", "autov2", "sha256", "notes"]
COLUMN_TITLES = {"type": "Type", "base_model": "Base model", "name": "Name", "trigger_words": "Trigger words",
                 "recommended": "Recommended", "commercial_use": "Commercial use", "license": "License",
                 "checked": "License checked", "file": "File", "version": "Version", "source": "Source",
                 "released": "Released", "autov2": "AutoV2", "sha256": "SHA256", "notes": "Notes"}
_HEADER_ALIASES = {
    "type": ("type", "model type", "kind", "category"),
    "base_model": ("base model", "basemodel", "base"),
    "name": ("name", "model name", "title"),
    "trigger_words": ("trigger words", "triggerwords", "trigger word", "trigger", "keywords", "trained words"),
    "recommended": ("recommended", "recommended settings", "settings"),
    "commercial_use": ("commercial use", "commercial", "commercial usage"),
    "license": ("license", "licence"),
    "checked": ("checked", "license checked", "licence checked", "last checked"),
    "file": ("file", "filename", "file name", "path", "model file"),
    "version": ("version", "model version"),
    "source": ("source", "url", "link", "download"),
    "released": ("released", "release date", "upload date", "uploaded", "published"),
    "autov2": ("autov2", "auto v2", "civitai hash"),
    "sha256": ("sha256", "sha 256", "sha-256", "sha", "hash"),
    "notes": ("notes", "note", "comment", "comments"),
}
_ALIAS_TO_KEY = {a: k for k, al in _HEADER_ALIASES.items() for a in al + (k, k.replace("_", " "))}
MODEL_INFO_CSV = "model_info.csv"
MODEL_INFO_ODS = "model_info.ods"
MODEL_INFO_XLSX = "model_info.xlsx"
_SHA_RE = re.compile(r"[0-9a-fA-F]{64}")


def _header_key(cell):
    c = re.sub(r"[\s_\-]+", " ", str(cell).strip().lower()).strip(" :*")
    return _ALIAS_TO_KEY.get(c)


def default_model_info_path(create=True):
    """<user>/pertineo_vfx/model_info.(ods|xlsx|csv|json), first one found. If none exists, an empty
    model_info.csv (header row only) is created so there is a file to fill in. Returns (path or None, note)."""
    base = hashing.pertineo_user_dir()
    names = [MODEL_INFO_ODS, MODEL_INFO_XLSX, MODEL_INFO_CSV, MODEL_INFO_DEFAULT]
    found = [n for n in names if os.path.isfile(os.path.join(base, n))]
    if found:
        note = (f"Several model_info files in <user>/pertineo_vfx ({', '.join(found)}) – using {found[0]}."
                if len(found) > 1 else None)
        return os.path.join(base, found[0]), note
    if not create:
        return None, None
    csv_p = os.path.join(base, MODEL_INFO_CSV)
    try:
        os.makedirs(base, exist_ok=True)
        import csv as _csv

        with open(csv_p, "x", encoding="utf-8", newline="") as f:
            _csv.writer(f, lineterminator="\n").writerow([COLUMN_TITLES[c] for c in CSV_COLUMNS])
        print(f"{LOG} Audit: created an empty model_info.csv in <ComfyUI user dir>/pertineo_vfx – "
              f"open it in LibreOffice Calc or Excel and add your models.")
        return csv_p, "An empty model_info.csv was created in <user>/pertineo_vfx – add your models there."
    except FileExistsError:
        return csv_p, None
    except Exception as e:
        return None, f"Could not create an empty model_info.csv ({type(e).__name__})."


# ---- spreadsheet readers (stdlib only) -------------------------------------
def _csv_rows(path):
    import csv as _csv

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()
    lines = text.splitlines()
    # pick the delimiter that turns the most lines into a recognisable header row
    best, best_score = ",", -1
    for d in (",", ";", "\t"):
        for row in _csv.reader(lines[:40], delimiter=d):
            score = sum(1 for c in row if _header_key(c))
            if score > best_score:
                best, best_score = d, score
    return list(_csv.reader(lines, delimiter=best))


def _ods_text(el, NS_TEXT):
    """Cell text incl. <text:s/> spaces, tabs and line breaks."""
    out = []

    def walk(e):
        tag = e.tag.split("}")[-1]
        if tag == "s":
            out.append(" " * int(e.get(f"{{{NS_TEXT}}}c", "1")))
        elif tag == "tab":
            out.append("\t")
        elif tag == "line-break":
            out.append("\n")
        elif e.text:
            out.append(e.text)
        for ch in e:
            walk(ch)
            if ch.tail:
                out.append(ch.tail)

    walk(el)
    return "".join(out)


def _ods_rows(path):
    """First sheet of an OpenDocument spreadsheet (LibreOffice .ods). Formatting is ignored."""
    import xml.etree.ElementTree as ET
    import zipfile

    T = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    TX = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    O = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    sheet = root.find(f".//{{{T}}}table")
    rows = []
    if sheet is None:
        return rows
    for row in sheet.iter(f"{{{T}}}table-row"):
        cells = []
        for cell in row:
            name = cell.tag.split("}")[-1]
            if name not in ("table-cell", "covered-table-cell"):
                continue
            rep = min(int(cell.get(f"{{{T}}}number-columns-repeated", "1")), 64)
            if cell.get(f"{{{O}}}value-type") == "date" and cell.get(f"{{{O}}}date-value"):
                val = cell.get(f"{{{O}}}date-value")[:10]
            else:
                val = "\n".join(_ods_text(p, TX) for p in cell.findall(f"{{{TX}}}p"))
            cells += [val] * rep
        while cells and not str(cells[-1]).strip():
            cells.pop()
        rep = int(row.get(f"{{{T}}}number-rows-repeated", "1"))
        rows += [cells] * (1 if not cells else min(rep, 10000))
    return rows


def _xlsx_rows(path):
    """First sheet of an Excel .xlsx. Formatting is ignored."""
    import xml.etree.ElementTree as ET
    import zipfile

    M = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{{{M}}}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{M}}}t")))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        first = wb.find(f".//{{{M}}}sheet")
        target = "xl/worksheets/sheet1.xml"
        if first is not None and "xl/_rels/workbook.xml.rels" in z.namelist():
            rid = first.get(f"{{{R}}}id")
            for rel in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")):
                if rel.get("Id") == rid:
                    t = rel.get("Target").lstrip("/")
                    target = t if t.startswith("xl/") else "xl/" + t
        sheet = ET.fromstring(z.read(target))
    rows = []
    for row in sheet.iter(f"{{{M}}}row"):
        cells = {}
        for c in row.findall(f"{{{M}}}c"):
            ref = re.match(r"([A-Z]+)", c.get("r", "A"))
            col = 0
            for ch in ref.group(1):
                col = col * 26 + ord(ch) - 64
            t = c.get("t")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{M}}}t"))
            else:
                v = c.find(f"{{{M}}}v")
                val = "" if v is None else (v.text or "")
                if t == "s" and val.isdigit() and int(val) < len(shared):
                    val = shared[int(val)]
            cells[col - 1] = val
        idx = int(row.get("r", len(rows) + 1)) - 1
        while len(rows) < idx:
            rows.append([])
        rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
    return rows


def _rows_to_info(rows):
    """Turns table rows into {"_disclaimer": ..., "models": {...}}.
    The header row is the first row with at least three known column names. Above it, a row starting with
    'Disclaimer' (or '_disclaimer') holds the disclaimer – in the same cell after a colon or in the next
    cell; other rows above the header are free notes. Below it, rows starting with '#' are ignored and
    a row counts as a model only if it has a 64-character SHA256 or a file name with an extension."""
    out = {"models": {}}
    header_idx, header = None, None
    for i, row in enumerate(rows):
        keys = [_header_key(c) for c in row]
        if sum(1 for k in keys if k) >= 3 and ("file" in keys or "sha256" in keys):
            header_idx, header = i, keys
            break

    def disclaimer_of(row):
        cells = [str(c).strip() for c in row if str(c).strip()]
        if not cells:
            return None
        m = re.match(r"_?disclaimer\s*:?\s*(.*)$", cells[0], re.I | re.S)
        if not m:
            return None
        return (m.group(1).strip() or (cells[1] if len(cells) > 1 else "")) or None

    for i, row in enumerate(rows):
        if header_idx is not None and i == header_idx:
            continue
        d = disclaimer_of(row)
        if d:
            out["_disclaimer"] = d
    if header is None:
        return out
    for row in rows[header_idx + 1:]:
        first = next((str(c).strip() for c in row if str(c).strip()), "")
        if not first or first.startswith("#") or disclaimer_of(row) or first.lower().startswith("_comment"):
            continue
        rec = {}
        for i, k in enumerate(header):
            if k and i < len(row) and str(row[i]).strip():
                rec[k] = str(row[i]).strip()
        sha = rec.get("sha256", "")
        if sha and not _SHA_RE.fullmatch(sha):
            sha = ""
            rec.pop("sha256")
        f = rec.get("file", "")
        if not sha and not (f and "." in f.rsplit("/", 1)[-1]):
            continue  # hint / example row without a model
        if sha:
            rec["sha256"] = sha.lower()
        for k in list(rec):
            if k not in ("notes", "trigger_words"):  # pasted text: no line breaks / double spaces
                rec[k] = re.sub(r"\s+", " ", rec[k]).strip()
        if "trigger_words" in rec:
            tw = rec["trigger_words"].strip()
            if tw.startswith("[") and tw.endswith("]"):  # a pasted list like ["a", "b"]
                tw = tw[1:-1]
            parts = tw.split(";") if ";" in tw else re.split(r"[,\n]", tw)
            rec["trigger_words"] = [re.sub(r"\s+", " ", t).strip(" ,").strip("\"'").strip()
                                    for t in parts if t.strip(" ,\"'")]
        out["models"][sha.lower() if sha else f] = rec
    return out


def read_model_info_table(path):
    ext = os.path.splitext(path)[1].lower()
    rows = _ods_rows(path) if ext == ".ods" else _xlsx_rows(path) if ext == ".xlsx" else _csv_rows(path)
    return _rows_to_info(rows)


def load_model_info(model_info_file, show, warn):
    """Returns (dict of entries, display name or None). Accepts .json and .csv."""
    text = (model_info_file or "").strip().strip("\"'")
    if not text:
        path, note = default_model_info_path()
        if note:
            warn(note)
        if path is None:
            warn("No model_info file – source and license are UNKNOWN for all models.")
            return {}, None
    else:
        path = os.path.expanduser(text)
        if text.lower().startswith("file://"):
            from urllib.parse import unquote, urlparse

            path = unquote(urlparse(text).path)
        if not os.path.isabs(path):
            path = os.path.join(hashing.pertineo_user_dir(), path)
    try:
        if path.lower().endswith((".csv", ".ods", ".xlsx", ".tsv", ".txt")):
            data = read_model_info_table(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except FileNotFoundError:
        warn(f"model_info file not found: {show.path(path)} – source and license are UNKNOWN.")
        return {}, show.path(path)
    except Exception as e:
        warn(f"model_info file unreadable ({show.path(path)}: {type(e).__name__}: {show.text(str(e))}) – "
             f"source and license are UNKNOWN.")
        return {}, show.path(path)
    raw = data
    if isinstance(data, dict) and isinstance(data.get("models"), dict):
        data = data["models"]
    if not isinstance(data, dict):
        warn("model_info file must contain a JSON object – ignored.")
        return {}, show.path(path)
    entries = {}
    for k, v in data.items():
        if isinstance(v, dict) and not str(k).startswith("_"):
            entries[str(k).strip()] = v
            entries.setdefault(str(k).strip().lower(), v)
    # secondary index from the optional "file" field ("loras/SDXL/x.safetensors"), so entries keyed
    # by SHA256 are also found when hashing is off; explicit keys always win
    for k, v in list(entries.items()):
        f = str(v.get("file") or "").strip().replace("\\", "/")
        if f:
            for alias in (f, f.split("/", 1)[1] if "/" in f else f, f.rsplit("/", 1)[-1]):
                entries.setdefault(alias, v)
                entries.setdefault(alias.lower(), v)
    disclaimer = raw.get("_disclaimer") if isinstance(raw, dict) else None
    if isinstance(disclaimer, str) and disclaimer.strip():
        entries[DISCLAIMER_KEY] = disclaimer.strip()  # never matches a model: not a dict, not a file name
    return entries, show.path(path)


def lookup_model_info(entries, sha256, rel_name):
    entries = {k: v for k, v in entries.items() if isinstance(v, dict)}
    keys = []
    if sha256:
        keys += [sha256.lower(), hashing.autov2(sha256), hashing.autov2(sha256).lower()]
    if rel_name:
        keys += [rel_name, rel_name.replace("\\", "/"), os.path.basename(rel_name), os.path.basename(rel_name).lower()]
    for k in keys:
        if k and k in entries:
            return entries[k], k
    return None, None


def models(graph, executed, compute_hashes, info_entries, show, warn, self_id=None):
    result = []
    for nid in executed:
        if nid == self_id:
            continue
        ct = graph.ctype(nid)
        for input_name, name, extra in _model_refs(graph, nid, warn):
            role = "matting_model" if ct.startswith("PertineoViTMatte") else model_role(ct, input_name)
            entry = {"role": role, "node_id": nid, "node_type": ct, "input": input_name}
            explicit = extra.pop("_path", "missing")
            if explicit != "missing":
                full, folder = explicit, "vitmatte"
                if full is None:
                    warn(f"{graph.label(nid)}: ViTMatte model folder not found (not downloaded yet?)")
            else:
                full, folder = find_model_file(name, input_name, role)
            if os.path.isabs(name):
                entry["file"] = show.path(name)
            else:
                entry["file"] = f"{folder}/{name}" if folder else name
            if show.full and full:
                entry["path"] = full
            entry.update({k: v for k, v in extra.items()})

            sha = None
            if full is None and explicit == "missing":
                entry["sha256"] = "not computed – file not found in the model folders"
                warn(f"{graph.label(nid)}: model file not found: {show.path(name)}")
            elif full and compute_hashes:
                try:
                    sha, cached = hashing.sha256_file(full, label=os.path.basename(name))
                    entry["sha256"] = sha
                    entry["autov2"] = hashing.autov2(sha)
                    entry["hash_from_cache"] = cached
                except Exception as e:
                    entry["sha256"] = f"not computed – {type(e).__name__}"
                    warn(f"{graph.label(nid)}: could not hash {show.path(name)} "
                         f"({type(e).__name__}: {show.text(str(e))})")
            elif full:
                entry["sha256"] = "not computed (compute_model_hashes is off)"
            if full:
                try:
                    entry["size_bytes"] = os.path.getsize(full)
                except OSError:
                    pass

            info, matched = lookup_model_info(info_entries, sha, name)
            if info is None:
                entry["source"] = entry["license"] = entry["commercial_use"] = UNKNOWN_INFO
                entry["license_notes"] = ""
                entry["info_matched_by"] = None
            else:
                entry["source"] = str(info.get("source") or UNKNOWN_INFO)
                entry["license"] = str(info.get("license") or UNKNOWN_INFO)
                cu = info.get("commercial_use")
                entry["commercial_use"] = UNKNOWN_INFO if cu in (None, "") else (
                    ("yes" if cu else "no") if isinstance(cu, bool) else str(cu))
                entry["license_notes"] = str(info.get("notes") or "")
                entry["info_matched_by"] = "sha256" if sha and matched in (sha.lower(),) else (
                    "autov2" if sha and matched and matched.lower() == hashing.autov2(sha).lower() else "file name")
                entry.update(info_extras(info))
                bad = []
                if sha and info.get("sha256") and str(info["sha256"]).strip().lower() != sha:
                    bad.append("SHA256")
                if sha and info.get("autov2") and str(info["autov2"]).strip().upper() != hashing.autov2(sha):
                    bad.append("AutoV2")
                if bad:
                    entry["hash_mismatch"] = bad
                    warn(f"{entry['file']}: {' and '.join(bad)} in the model_info file {'do' if len(bad) > 1 else 'does'} not match the file "
                         f"– wrong entry, or the file was replaced.")
            result.append(entry)
    unknown = sorted({m["file"] for m in result if m.get("info_matched_by") is None})
    if unknown and info_entries:
        warn(f"No model_info entry for: {', '.join(unknown)}")
    return result


INFO_EXTRA_KEYS = ("type", "name", "version", "released", "base_model", "trigger_words", "recommended", "checked")


def info_extras(info):
    """Optional descriptive fields of a model_info entry (copied as given, trigger words as a list)."""
    out = {}
    for k in INFO_EXTRA_KEYS:
        v = info.get(k)
        if v in (None, "", []):
            continue
        if k == "trigger_words":
            if isinstance(v, str):
                v = [t.strip() for t in v.split(",") if t.strip()]
            v = [str(t) for t in v if str(t).strip()]
            if not v:
                continue
        out[k] = v if k == "trigger_words" else str(v)
    return out


def base_model_mismatches(models):
    """LoRAs / ControlNets whose base_model differs from every main model of the run.
    Only compares entries where both sides are documented – nothing is guessed."""
    norm = lambda x: re.sub(r"[\s_.-]+", "", str(x).lower())  # noqa: E731
    mains = {norm(m["base_model"]) for m in models
             if m.get("base_model") and m["role"] in ("checkpoint", "diffusion_model")}
    if not mains:
        return []
    out = []
    for m in models:
        if m["role"] in ("lora", "controlnet", "ip_adapter") and m.get("base_model") \
                and norm(m["base_model"]) not in mains:
            out.append(m)
    return out


# --------------------------------------------------------------------------
# Sampling, prompts, latent size
# --------------------------------------------------------------------------
SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
                 "KSampler (Efficient)", "KSampler Adv. (Efficient)", "KSamplerAdvanced (WLSH)"}
PARAM_KEYS = ["seed", "noise_seed", "steps", "cfg", "guidance", "sampler_name", "scheduler", "denoise",
              "start_at_step", "end_at_step", "add_noise", "return_with_leftover_noise", "shift", "max_shift",
              "base_shift", "max_tokens"]
_DISPLAY_KEY = {"noise_seed": "seed", "sampler_name": "sampler"}
_TEXT_KEYS = ("text", "text_g", "text_l", "prompt", "t5xxl", "clip_l", "clip_g", "llama", "system_prompt",
              "user_prompt", "positive_prompt", "negative_prompt")
_TRACE_SKIP = {"model", "clip", "vae", "image", "images", "pixels", "samples", "latent", "latent_image",
               "control_net", "mask", "noise", "sigmas", "sampler", "clip_vision", "clip_vision_output",
               "style_model", "upscale_model", "ipadapter", "audio", "video"}


def is_sampler(graph, nid):
    ct = graph.ctype(nid)
    if ct in SAMPLER_TYPES:
        return True
    ins = graph.inputs(nid)
    has_seed = "seed" in ins or "noise_seed" in ins or "noise" in ins
    return has_seed and ("steps" in ins or "sigmas" in ins) and ("positive" in ins or "guider" in ins)


def _collect_params(graph, nid, params, unresolved, sources, keys, depth, follow):
    ins = graph.inputs(nid)
    for k in keys:
        if k not in ins or _DISPLAY_KEY.get(k, k) in params or _DISPLAY_KEY.get(k, k) in unresolved:
            continue
        ok, val, note, chain = graph.resolve(ins[k])
        dk = _DISPLAY_KEY.get(k, k)
        if ok:
            params[dk] = val
            if chain:
                sources[dk] = " → ".join(chain)
        else:
            unresolved[dk] = note
    if depth <= 0:
        return
    for name in follow:
        v = ins.get(name)
        if is_link(v) and v[0] in graph.nodes:
            _collect_params(graph, v[0], params, unresolved, sources, keys, depth - 1, follow)


def _text_encoder_texts(graph, nid):
    ins = graph.inputs(nid)
    ct = graph.ctype(nid)
    if not any(k in ins for k in _TEXT_KEYS):
        return None
    if not ("clip" in ins or re.search(r"encode|prompt|text|conditioning", ct, re.I)):
        return None
    texts = {}
    for k in _TEXT_KEYS:
        if k in ins:
            ok, val, note, _chain = graph.resolve(ins[k])
            if ok and isinstance(val, str):
                texts[k] = val
            elif not ok:
                texts[k] = f"<{note}>"
    return texts or None


def trace_prompts(graph, start, role, extra):
    """Follows a conditioning input upstream to the text encoders. `extra` collects
    values met on the way (e.g. guidance from FluxGuidance)."""
    found, seen, stack = [], set(), [(start, 0)]
    while stack:
        v, depth = stack.pop()
        if not is_link(v) or v[0] in seen or depth > 40:
            continue
        nid = v[0]
        seen.add(nid)
        if nid not in graph.nodes:
            continue
        texts = _text_encoder_texts(graph, nid)
        ins = graph.inputs(nid)
        if "guidance" in ins and "guidance" not in extra:
            ok, val, note, _c = graph.resolve(ins["guidance"])
            extra["guidance"] = val if ok else note
        if texts is not None:
            found.append({"node_id": nid, "node_type": graph.ctype(nid), "title": graph.title(nid), "texts": texts})
            continue
        if "positive" in ins and "negative" in ins:
            names = [role]
        else:
            names = [k for k in ins if k not in _TRACE_SKIP]
        for k in names:
            if is_link(ins.get(k)):
                stack.append((ins[k], depth + 1))
    return sorted(found, key=lambda f: _id_key(f["node_id"]))


def latent_info(graph, nid, source_link=None):
    ins = graph.inputs(nid)
    v = ins.get("latent_image", ins.get("latent"))
    for _ in range(8):
        if not is_link(v):
            return {"note": "not connected"}
        src = v[0]
        if src not in graph.nodes:
            return {"note": f"linked from missing node #{src}"}
        sins = graph.inputs(src)
        if "width" in sins and "height" in sins:
            w = graph.resolve(sins["width"])
            h = graph.resolve(sins["height"])
            info = {"from": graph.label(src)}
            if w[0] and h[0]:
                info["width"], info["height"] = w[1], h[1]
            else:
                info["note"] = w[2] or h[2]
            for k in ("batch_size", "length", "frames"):
                if k in sins:
                    r = graph.resolve(sins[k])
                    info[k] = r[1] if r[0] else r[2]
            return info
        if "pixels" in sins or "image" in sins:
            px = sins.get("pixels", sins.get("image"))
            if source_link is not None and px == source_link:
                return {"from": graph.label(src), "note": "encoded source image", "same_as": "source_image"}
            return {"from": graph.label(src), "note": "encoded from an image (size of that image)"}
        if is_sampler(graph, src):
            return {"from": graph.label(src), "note": "output of an earlier sampler (multi-pass)"}
        if {"on_true", "on_false", "boolean"} <= set(sins):  # boolean switch, e.g. Crystools
            ok, b, _n, _c = graph.resolve(sins["boolean"])
            if ok and isinstance(b, bool):
                v = sins["on_true" if b else "on_false"]
                continue
            return {"from": graph.label(src), "note": "switch with a computed boolean – latent source not resolved"}
        nxt = next((sins[k] for k in ("samples", "latent", "latent_image", "input") if is_link(sins.get(k))), None)
        if nxt is None:
            return {"from": graph.label(src), "note": "latent source not followed further"}
        v = nxt
    return {"note": "latent source too deep to follow"}


def sampling(graph, executed, self_id=None):
    out = []
    source_link = graph.inputs(self_id).get("source_image") if self_id else None
    for nid in executed:
        if not is_sampler(graph, nid):
            continue
        params, unresolved, sources = {}, {}, {}
        _collect_params(graph, nid, params, unresolved, sources, PARAM_KEYS, 3,
                        ("noise", "guider", "sampler", "sigmas"))
        # ModelSampling* shift lives on the model path
        _collect_params(graph, nid, params, unresolved, sources, ["shift", "max_shift", "base_shift"], 8,
                        ("model",))
        ins = graph.inputs(nid)
        prompts = {}
        if "positive" in ins or "negative" in ins:
            pos_root, neg_root = ins.get("positive"), ins.get("negative")
        else:  # SamplerCustomAdvanced: conditioning sits on the guider
            g = ins.get("guider")
            gins = graph.inputs(g[0]) if is_link(g) else {}
            pos_root = gins.get("positive", gins.get("conditioning"))
            neg_root = gins.get("negative")
        extra = {}
        prompts["positive"] = trace_prompts(graph, pos_root, "positive", extra) if pos_root is not None else []
        prompts["negative"] = trace_prompts(graph, neg_root, "negative", {}) if neg_root is not None else []
        if "guidance" in extra and "guidance" not in params:
            params["guidance"] = extra["guidance"]
        out.append({"node_id": nid, "node_type": graph.ctype(nid), "title": graph.title(nid),
                    "params": params, "unresolved": unresolved, "resolved_from": sources,
                    "latent": latent_info(graph, nid, source_link), "prompts": prompts})
    return out


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------
def image_info(t):
    try:
        b, h, w = int(t.shape[0]), int(t.shape[1]), int(t.shape[2])
        return {"width": w, "height": h, "frames": b}
    except Exception:
        return None


def resolution(source_image, output_image, fps, samplers):
    res = {"source": image_info(source_image) if source_image is not None else None,
           "final": image_info(output_image) if output_image is not None else None,
           "generation": []}
    for s in samplers:
        lat = dict(s.get("latent") or {})
        if lat.get("same_as") == "source_image" and res["source"]:
            lat["width"], lat["height"] = res["source"]["width"], res["source"]["height"]
        res["generation"].append(dict(lat, sampler=f"{s['node_type']} #{s['node_id']}"))
    video = None
    frames = (res["final"] or {}).get("frames") or 0
    if fps and fps > 0:
        video = {"frames": frames or None, "fps": fps,
                 "duration_sec": round(frames / fps, 3) if frames else None}
    return res, video


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def collect(prompt, extra_pnginfo, unique_id, manual, compute_model_hashes, model_info_file,
            include_full_paths, source_image=None, output_image=None, fps=0.0, execution_list=None):
    warnings = []
    show = PathDisplay(include_full_paths)

    def warn(msg):
        msg = show.text(msg)
        if msg not in warnings:
            warnings.append(msg)
            print(f"{LOG} Audit warning: {msg}")

    data = {
        "schema": SCHEMA,
        "generator": {"package": "comfyui-pertineo-vfx", "version": own_version(),
                      "node": "PertineoCollectMetadata", "node_id": unique_id},
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "manual": manual,
        "options": {"compute_model_hashes": bool(compute_model_hashes),
                    "include_full_paths": bool(include_full_paths)},
    }

    def section(name, fn, default):
        try:
            return fn()
        except Exception as e:
            warn(f"{name} could not be collected ({type(e).__name__}: {e})")
            return default

    try:
        import nodes as comfy_nodes

        mappings = comfy_nodes.NODE_CLASS_MAPPINGS
    except Exception:
        mappings = {}
    if not prompt:
        warn("No prompt graph available (hidden PROMPT input empty) – only manual fields and environment.")
    graph = Graph(prompt, mappings, execution_list)
    executed = section("Executed nodes", graph.executed_ids, sorted(graph.nodes, key=_id_key))
    data["graph"] = {"nodes_in_prompt": len(graph.nodes), "nodes_executed": len(executed)}

    data["environment"] = section("Environment", lambda: environment(warn), {})
    entries, info_name = section("model_info file", lambda: load_model_info(model_info_file, show, warn), ({}, None))
    data["options"]["model_info_file"] = info_name
    if entries.get(DISCLAIMER_KEY):
        data["options"]["model_info_disclaimer"] = entries[DISCLAIMER_KEY]
    data["disclaimer"] = TOOL_DISCLAIMER
    data["models"] = section("Models", lambda: models(graph, executed, compute_model_hashes, entries, show, warn,
                                                       self_id=unique_id), [])
    cn = section("Custom nodes", lambda: custom_nodes(graph, executed, extra_pnginfo,
                                                       data["environment"].get("comfyui", "unknown"), warn), ([], []))
    data["custom_nodes"], data["api_nodes"] = cn
    data["sampling"] = section("Sampling", lambda: sampling(graph, executed, unique_id), [])
    for s in data["sampling"]:
        for k, note in s["unresolved"].items():
            warn(f"{s['node_type']} #{s['node_id']}: {k} not resolvable to a constant ({note})")
    data["resolution"], data["video"] = section(
        "Resolution", lambda: resolution(source_image, output_image, fps, data["sampling"]), ({}, None))
    data["warnings"] = warnings
    return data
