# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""SHA256 of model files with a persistent cache (key: real path + size + mtime)."""

import datetime
import hashlib
import json
import os
import threading
import time

from ..common import LOG

CACHE_DIRNAME = "pertineo_vfx"
CACHE_FILENAME = "model_hash_cache.json"
_CHUNK = 16 * 1024 * 1024

_LOCK = threading.Lock()
_CACHE = None
_CACHE_PATH = None


def pertineo_user_dir():
    """<ComfyUI user directory>/pertineo_vfx (falls back to ~/.cache/pertineo_vfx outside ComfyUI)."""
    try:
        import folder_paths

        base = folder_paths.get_user_directory()
    except Exception:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, CACHE_DIRNAME)


def cache_path():
    return os.path.join(pertineo_user_dir(), CACHE_FILENAME)


def _load_cache():
    global _CACHE, _CACHE_PATH
    path = cache_path()
    if _CACHE is not None and _CACHE_PATH == path:
        return _CACHE
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"{LOG} Audit: hash cache unreadable, starting a new one ({type(e).__name__})")
        data = {}
    _CACHE, _CACHE_PATH = data, path
    return data


def _write_cache(cache):
    path = _CACHE_PATH or cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except Exception as e:
        print(f"{LOG} Audit: could not write hash cache ({type(e).__name__}: {e})")


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(f, "sha256").hexdigest()
        while True:
            chunk = f.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path, label=None):
    """Returns (sha256 hex lower case, from_cache). Raises OSError if the file cannot be read."""
    real = os.path.realpath(path)
    st = os.stat(real)
    with _LOCK:
        entry = _load_cache().get(real)
        if (isinstance(entry, dict) and entry.get("sha256") and entry.get("size") == st.st_size
                and entry.get("mtime_ns") == st.st_mtime_ns):
            return entry["sha256"], True

    name = label or os.path.basename(real)
    print(f"{LOG} Audit: hashing {name} ({st.st_size / 1024 ** 3:.2f} GB) ...")
    t0 = time.time()
    digest = _hash_file(real)
    print(f"{LOG} Audit: {name} hashed in {time.time() - t0:.1f} s (cached for next runs)")

    with _LOCK:
        cache = _load_cache()
        cache[real] = {
            "sha256": digest,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "hashed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        _write_cache(cache)
    return digest, False


def autov2(sha256):
    """Civitai AutoV2 short hash: first 10 hex characters of the SHA256, upper case."""
    return sha256[:10].upper() if sha256 else None
