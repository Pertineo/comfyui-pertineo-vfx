# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Pertineo VFX – compositing-grade tools for ComfyUI.

Every node group is loaded on its own: if one group fails to import (e.g. a
missing dependency), the other groups still work and the error is logged.
"""

import importlib
import os
import traceback

# Let OpenCV read/write EXR as a fallback; must be set before cv2 is first used.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

from .common import LOG

NODE_GROUPS = [
    "image_io",
    "matting",
    "audit",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _group in NODE_GROUPS:
    try:
        _mod = importlib.import_module(f".{_group}.nodes", __name__)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception:
        print(f"{LOG} Could not load node group '{_group}':")
        traceback.print_exc()

print(f"{LOG} Loaded {len(NODE_CLASS_MAPPINGS)} nodes: " + ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values()))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
