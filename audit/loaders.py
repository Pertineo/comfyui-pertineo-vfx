# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Loader + Info nodes: load a model with ComfyUI's own loader and output what the model_info file says about it.

The inputs (file lists, weight_dtype, CLIP types …) and the loading itself come from the core loader at run time,
so new options in ComfyUI appear here without an update of this package."""

from ..common import LOG
from .nodes import _info_file_mtime, info_outputs

CATEGORY = "Pertineo/Loaders"
INFO_RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
INFO_RETURN_NAMES = ("name", "base_model", "trigger_words", "recommended", "license", "info")
INFO_TOOLTIPS = ("Name from your model_info file.", "Base model from model_info.",
                 "Trigger words from model_info (comma separated).", "Recommended settings from model_info.",
                 "License from model_info.", "Short info text: file, base model, license, commercial use, source.")

# (class suffix, label, core node, [(model input, model folder)], core return types, core return names)
LOADER_TYPES = [
    ("Checkpoint", "Checkpoint", "CheckpointLoaderSimple", [("ckpt_name", "checkpoints")],
     ("MODEL", "CLIP", "VAE"), ("model", "clip", "vae")),
    ("DiffusionModel", "Diffusion Model", "UNETLoader", [("unet_name", "diffusion_models")],
     ("MODEL",), ("model",)),
    ("VAE", "VAE", "VAELoader", [("vae_name", "vae")], ("VAE",), ("vae",)),
    ("TextEncoder", "Text Encoder", "CLIPLoader", [("clip_name", "text_encoders")], ("CLIP",), ("clip",)),
    ("DualTextEncoder", "Dual Text Encoder", "DualCLIPLoader",
     [("clip_name1", "text_encoders"), ("clip_name2", "text_encoders")], ("CLIP",), ("clip",)),
    ("ControlNet", "ControlNet", "ControlNetLoader", [("control_net_name", "controlnet")],
     ("CONTROL_NET",), ("control_net",)),
    ("Upscaler", "Upscaler", "UpscaleModelLoader", [("model_name", "upscale_models")],
     ("UPSCALE_MODEL",), ("upscale_model",)),
]

_INFO_INPUTS = {
    "model_info_file": ("STRING", {"advanced": True, "default": "",
        "placeholder": "empty = <ComfyUI user dir>/pertineo_vfx/model_info.ods / .xlsx / .csv / .json",
        "tooltip": "Same file as in Pertineo Collect Metadata. Empty = "
                   "<ComfyUI user dir>/pertineo_vfx/model_info.ods / .xlsx / .csv / .json."}),
    "lookup_by_hash": ("BOOLEAN", {"advanced": True, "default": True,
        "tooltip": "Find the entry by SHA256 (survives renaming; hash is cached). Off = by file name only."}),
}


def _core_class(name):
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"{LOG} The ComfyUI loader '{name}' is not available in this ComfyUI version.")
    return cls


def _core_inputs(name, model_keys):
    """INPUT_TYPES of the core loader (required + optional), or a minimal fallback with the file lists."""
    try:
        spec = _core_class(name).INPUT_TYPES()
        req = dict(spec.get("required") or {})
        opt = dict(spec.get("optional") or {})
        if all(k in req for k, _f in model_keys):
            return req, opt
    except Exception as e:
        print(f"{LOG} Loader + Info: inputs of {name} not readable ({type(e).__name__}: {e}) – using file lists.")
    import folder_paths
    req = {}
    for key, folder in model_keys:
        try:
            files = folder_paths.get_filename_list(folder)
        except Exception:
            files = []
        req[key] = (files or ["– no files in this model folder –"],)
    return req, {}


def _call_core(name, kwargs):
    """Runs the core loader – classic nodes (FUNCTION) and the newer node schema (execute) both work."""
    cls = _core_class(name)
    if hasattr(cls, "define_schema") and hasattr(cls, "execute"):  # ComfyUI node schema v3
        out = cls.execute(**kwargs)
        for attr in ("result", "args"):
            if hasattr(out, attr) and getattr(out, attr) is not None:
                return tuple(getattr(out, attr))
        return tuple(out)
    inst = cls()
    return tuple(getattr(inst, cls.FUNCTION)(**kwargs))


def _load_upscaler_fallback(model_name):
    """Same steps as ComfyUI's UpscaleModelLoader, used only if the core node cannot be called."""
    import comfy.utils
    import folder_paths
    from spandrel import ImageModelDescriptor, ModelLoader

    path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise TypeError("Upscale model must be a single-image model.")
    return (out,)


class _LoaderInfoBase:
    CATEGORY = CATEGORY
    FUNCTION = "run"
    CORE = None
    MODEL_KEYS = ()
    LABEL = None
    CORE_RETURNS = ()

    @classmethod
    def INPUT_TYPES(cls):
        req, opt = _core_inputs(cls.CORE, cls.MODEL_KEYS)
        opt = dict(opt)
        opt.update(_INFO_INPUTS)  # ours always last
        return {"required": req, "optional": opt}

    @classmethod
    def IS_CHANGED(cls, model_info_file="", **kwargs):
        # re-run when the model_info file was edited; the model itself is cached by ComfyUI as usual
        return _info_file_mtime(model_info_file)

    def run(self, model_info_file="", lookup_by_hash=True, **kwargs):
        try:
            loaded = _call_core(self.CORE, kwargs)
        except Exception:
            if self.CORE != "UpscaleModelLoader":
                raise
            loaded = _load_upscaler_fallback(kwargs["model_name"])
        infos = [info_outputs(folder, kwargs.get(key), model_info_file, lookup_by_hash,
                              self.LABEL if len(self.MODEL_KEYS) == 1 else f"{self.LABEL} {i + 1}")
                 for i, (key, folder) in enumerate(self.MODEL_KEYS) if kwargs.get(key)]
        if len(infos) == 1:
            info = infos[0]
        else:  # dual text encoder: values side by side, info texts one after the other
            info = tuple(" + ".join(x for x in col if x) for col in zip(*infos[:]))[:-1] + \
                ("\n\n".join(i[-1] for i in infos),)
        return tuple(loaded[:len(self.CORE_RETURNS)]) + tuple(info)


LOADER_CLASSES = {}
for _suffix, _label, _core, _keys, _rt, _rn in LOADER_TYPES:
    _cls = type(f"Pertineo{_suffix}LoaderInfo", (_LoaderInfoBase,), {
        "CORE": _core, "MODEL_KEYS": tuple(_keys), "LABEL": _label, "CORE_RETURNS": _rt,
        "RETURN_TYPES": _rt + INFO_RETURN_TYPES, "RETURN_NAMES": _rn + INFO_RETURN_NAMES,
        "OUTPUT_TOOLTIPS": tuple(f"As in the core loader ({_core})." for _ in _rt) + INFO_TOOLTIPS,
        "DESCRIPTION": (f"Loads a {_label} with ComfyUI's own loader ({_core} – same inputs, same result) and "
                        f"additionally outputs what your model_info file says about it: name, base model, trigger "
                        f"words, recommended settings, license and an info text. The file is chosen only once, so "
                        f"loader and info always match. A missing entry never stops the run."),
    })
    LOADER_CLASSES[_cls.__name__] = (_cls, f"Pertineo Load {_label} + Info")
