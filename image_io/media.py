# Pertineo VFX for ComfyUI
# Copyright (C) 2026 Julia Kietzmann (Pertineo, https://pertineo.de)
# SPDX-License-Identifier: GPL-3.0-or-later
# This program comes with ABSOLUTELY NO WARRANTY. It is free software, and you are welcome to
# redistribute it under the terms of the GNU GPL v3 or later; see the LICENSE and NOTICE files.

"""Fast media writing for Save Media: image / mask sequences (PNG, TIFF, EXR, JPG) written in parallel,
video through ffmpeg (ProRes, H.264, H.265), versions (v001 …), date folders and SHA256 lists."""

import datetime
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..common import LOG
from .readwrite import EXR_HINT, HAS_OPENEXR, OpenEXR, clean_path, cv2

# --------------------------------------------------------------------------
# Choices (shown in the UI – append new values at the end, never rename)
# --------------------------------------------------------------------------
SEQ_FORMATS = ["png (16 bit)", "png (8 bit)", "exr (16 bit half)", "exr (32 bit float)", "tiff (16 bit)",
               "tiff (8 bit)", "tiff (32 bit float)", "jpg (8 bit)"]
COMPRESSION = ["balanced", "fast (bigger files)", "small (slow)"]
VIDEO_QUALITY = ["preview (H.264 8 bit, mp4)", "review (H.265 10 bit, mp4)", "ProRes 422 HQ (10 bit, mov)",
                 "ProRes 4444 (10 bit, alpha, mov)"]
VIDEO_OF = ["image", "mask", "image + mask (2 videos)"]
VIDEO_AUDIO = ["only if connected", "always (silent track if not connected)"]
# where the alpha of the image sequence / ProRes 4444 comes from
ALPHA_CHANNEL = ["from mask (RGB + mask = RGBA)", "from image (RGBA stays RGBA)", "none (always RGB)"]
VERSION_MODES = ["next free", "fixed", "off"]
DATE_FOLDERS = ["off", "YYYY-MM-DD", "YYYYMMDD", "YYMMDD"]
IF_EXISTS = ["error", "overwrite"]
ALPHA_MODES = ["straight", "premultiplied", "auto (EXR premultiplied, others straight)"]
VIDEO_ALPHA_MODES = ["straight", "premultiplied"]
VERSION_IN_NAME = ["replace existing (v01 -> v02)", "append (_v001)"]

_DATE_FMT = {"YYYY-MM-DD": "%Y-%m-%d", "YYYYMMDD": "%Y%m%d", "YYMMDD": "%y%m%d"}
_DATE_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{8}|\d{6})$")
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_GERMAN = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss", "ẞ": "SS"})
MASK_SUFFIX = "_mask"
DEFAULT_NAME = "render"


def worker_count():
    """Threads for writing frames. cv2 / zlib / OpenEXR release the GIL while encoding, so frames are
    compressed on several cores at once; more than 16 rarely helps (the disk becomes the limit)."""
    return max(2, min(16, os.cpu_count() or 4))


# --------------------------------------------------------------------------
# Names, folders, versions
# --------------------------------------------------------------------------
def output_root(output_folder, default_root):
    """Empty = ComfyUI output folder, relative = inside it, absolute allowed (like Save Image + Metadata)."""
    p = clean_path(output_folder)
    if not p:
        return os.path.normpath(default_root)
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(default_root, p))


def _is_date_dir(name):
    """True for a folder name like 2026-11-06, 20261106 or 261106 that is also a real date."""
    if not _DATE_DIR_RE.match(name):
        return False
    fmt = {10: "%Y-%m-%d", 8: "%Y%m%d", 6: "%y%m%d"}[len(name)]  # by length: strptime would read 999999 as 9999-9-9
    try:
        datetime.datetime.strptime(name, fmt)
        return True
    except ValueError:
        return False


def folder_root_for(folder, seq_name=None):
    """The folder to use as Output Name's output_folder, so a new version lands NEXT TO the loaded one.
    Sequence (seq_name given): if the frames sit in their own folder – named like the sequence, or with a version
    token (sh010_comp_v01/) – step out of it. Then, if that folder is a date folder, step out of that too (Output
    Name adds today's date folder itself).
    A file in ComfyUI's input folder (upload) gives the ComfyUI output folder instead – renders must never land in
    the input folder."""
    f = os.path.normpath(folder)
    try:
        import folder_paths
        inp = os.path.realpath(folder_paths.get_input_directory())
        if os.path.commonpath([os.path.realpath(f), inp]) == inp:
            return os.path.normpath(folder_paths.get_output_directory())
    except Exception:
        pass
    if seq_name is not None:
        leaf = os.path.basename(f)
        if leaf.lower() == str(seq_name).lower() or _VTOKEN_RE.search(leaf):
            f = os.path.dirname(f) or f
    if _is_date_dir(os.path.basename(f)):
        f = os.path.dirname(f) or f
    return f


def date_folder_name(mode, now=None):
    fmt = _DATE_FMT.get(mode)
    return (now or datetime.datetime.now()).strftime(fmt) if fmt else ""


def clean_name(text):
    """Pipeline-safe file name: ä -> ae, ß -> ss, accents dropped (é -> e), everything except letters, digits,
    '_', '-' and '.' becomes '_', repeated '_' collapse, no separators at the start or end.
    'Video_002 (1)_edit' -> 'Video_002_1_edit'."""
    t = str(text).translate(_GERMAN)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = _UNSAFE_RE.sub("_", t)
    t = re.sub(r"_{2,}", "_", t)
    t = re.sub(r"_?([.-])_?", r"\1", t)  # 'a_-_b' -> 'a-b', 'a_.b' -> 'a.b'
    return t.strip("._-")


def base_name(filename, name=None):
    """filename is a template: {name} = the linked name (e.g. from a loader). The result is cleaned to a
    pipeline-safe name (see clean_name)."""
    template = (filename or "").strip() or "{name}"
    linked = (name or "").strip()
    text = template.replace("{name}", linked or DEFAULT_NAME)
    # a pasted file name: drop the extension and a trailing frame number
    text = re.sub(r"(\.\d+)?\.(png|exr|tiff?|jpe?g|mov|mp4)$", "", text.strip(), flags=re.I)
    return clean_name(text) or DEFAULT_NAME


# A version token inside a name: v01, _v003, .V2, -v0004 – between separators (or at the start / end)
_VTOKEN_RE = re.compile(r"(?:^|(?<=[_.\-]))([vV])(\d+)(?=$|[_.\-])")


def version_slot(base, in_name=VERSION_IN_NAME[0]):
    """Where the version goes. 'replace existing': the last version token in the name (sh010_comp_v01 ->
    prefix 'sh010_comp_', 'v', 2 digits, suffix ''); otherwise, or without a token, '_v001' is appended."""
    if in_name.startswith("replace"):
        found = list(_VTOKEN_RE.finditer(base))
        if found:
            m = found[-1]
            return {"prefix": base[:m.start(1)], "vchar": m.group(1), "digits": len(m.group(2)),
                    "suffix": base[m.end(2):], "current": int(m.group(2)), "token": m.group(0), "base": base}
    return {"prefix": base + "_", "vchar": "v", "digits": 3, "suffix": "", "current": None, "token": None,
            "base": base}


def slot_text(slot, n):
    return f"{slot['vchar']}{int(n):0{slot['digits']}d}"


def existing_versions(root, slot):
    """All version numbers already used at this slot – files, sequence folders, mask folders, videos and audit
    files – in root and in date folders directly below it (a version continues across days)."""
    # after the version (and the rest of the name) only the end, an extension (.mov, .sha256, .audit.txt …) or the
    # mask suffix may follow, so shot_v002_comp_v001 is not counted as a version of 'shot'
    # "_01" = image number of a batch saved by Save Image + Metadata (name_v003_01.png)
    pat = re.compile(re.escape(slot["prefix"]) + r"[vV](\d+)" + re.escape(slot["suffix"]) +
                     r"(?:" + re.escape(MASK_SUFFIX) + r")?(?:_\d+)?(?:$|\.)", re.I)
    found = set()
    dirs = [root]
    try:
        dirs += [os.path.join(root, d) for d in os.listdir(root)
                 if _DATE_DIR_RE.match(d) and os.path.isdir(os.path.join(root, d))]
    except FileNotFoundError:
        return found
    for d in dirs:
        try:
            for entry in os.listdir(d):
                m = pat.match(entry)
                if m:
                    found.add(int(m.group(1)))
        except OSError:
            pass
    return found


def resolve_version(mode, number, root, base, in_name=VERSION_IN_NAME[0]):
    """Returns (full name, version text or '', number or None, note).
    next free = higher than every version already in the folder AND than the version in the name itself."""
    slot = version_slot(base, in_name)
    where = f" – replaces {slot['token']} in the name" if slot["token"] else ""
    if mode == "off":
        return base, (slot["token"] or ""), slot["current"], "off – name used as it is"
    if mode == "fixed":
        n = int(number)
        note = "fixed" + where
    else:
        used = existing_versions(root, slot)
        cands = set(used) | ({slot["current"]} if slot["current"] is not None else set())
        n = max(int(number), (max(cands) + 1) if cands else 1)
        hi = slot_text(slot, max(used)) if used else "none"
        note = f"next free (highest in folder: {hi}" + (f", in name: {slot['token']}" if slot["token"] else "") + ")"
        note += where
    ver = slot_text(slot, n)
    return slot["prefix"] + ver + slot["suffix"], ver, n, note


def plan_paths(root, date_mode, full, now=None):
    """All target paths of one Save Media run. The sequence folder sits next to the video and has its name."""
    folder = os.path.join(root, date_folder_name(date_mode, now)) if date_mode != "off" else root
    mask = full + MASK_SUFFIX
    return {
        "folder": folder, "base": full, "mask_base": mask,
        "seq_dir": os.path.join(folder, full), "mask_dir": os.path.join(folder, mask),
    }


# --------------------------------------------------------------------------
# Pixel conversion
# --------------------------------------------------------------------------
def srgb_to_linear(x, clamp=True):
    """sRGB transfer -> linear. Without clamp, values above 1 keep the same curve (and below 0 stay linear),
    so HDR values survive the conversion."""
    x = np.asarray(x, dtype=np.float32)
    if clamp:
        x = np.clip(x, 0.0, 1.0)
    out = np.where(x <= 0.04045, x / 12.92, np.power((np.abs(x) + 0.055) / 1.055, 2.4) * np.sign(x))
    return out.astype(np.float32)


def as_channels(arr):
    """HxW / HxWx1..4 float -> HxWx3 or HxWx4 float32 (C-contiguous)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    elif a.shape[-1] == 2:
        a = np.concatenate([np.repeat(a[..., :1], 3, axis=-1), a[..., 1:2]], axis=-1)
    elif a.shape[-1] > 4:
        a = a[..., :4]
    return np.ascontiguousarray(a)


def _to_int(a, bits):
    """float [0,1] -> uint8 / uint16 with one temporary copy (in place), to keep RAM low with many workers."""
    top = 65535.0 if bits == 16 else 255.0
    dtype = np.uint16 if bits == 16 else np.uint8
    t = np.multiply(a, top, dtype=np.float32)
    t += 0.5
    np.clip(t, 0, top, out=t)
    return t.astype(dtype)


def _bgr(a):
    """RGB(A) -> BGR(A) for OpenCV."""
    if a.shape[-1] == 4:
        return np.ascontiguousarray(a[..., [2, 1, 0, 3]])
    return np.ascontiguousarray(a[..., ::-1])


# --------------------------------------------------------------------------
# Frame writers
# --------------------------------------------------------------------------
def ext_for(fmt):
    return {"png": ".png", "exr": ".exr", "tif": ".tif", "jpg": ".jpg"}[fmt[:3]]


def _need_cv2(fmt):
    if cv2 is None:
        raise IOError(f"{LOG} Writing {fmt} needs OpenCV (cv2) in the Python environment of ComfyUI.")


def _exr_compression(level):
    name = {"fast (bigger files)": "ZIPS_COMPRESSION", "small (slow)": "PIZ_COMPRESSION"}.get(level, "ZIP_COMPRESSION")
    return name


def write_exr(path, channels, half, level):
    """channels: dict name -> HxW float. half = 16-bit float (HALF), else 32-bit float."""
    dtype = np.float16 if half else np.float32
    chans = {k: np.ascontiguousarray(v, dtype=dtype) for k, v in channels.items()}
    comp = _exr_compression(level)
    if HAS_OPENEXR:
        header = {"compression": getattr(OpenEXR, comp, OpenEXR.ZIP_COMPRESSION), "type": OpenEXR.scanlineimage}
        with OpenEXR.File(header, chans) as f:
            f.write(path)
        return
    if cv2 is not None:
        order = [k for k in ("B", "G", "R", "A") if k in chans]
        stack = np.dstack([chans[k].astype(np.float32) for k in order])
        cv_comp = getattr(cv2, "IMWRITE_EXR_COMPRESSION_" + comp.split("_")[0], None)
        params = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT]
        if cv_comp is not None:
            params += [cv2.IMWRITE_EXR_COMPRESSION, cv_comp]
        try:
            if cv2.imwrite(path, stack, params):
                return
        except cv2.error:
            pass
    raise IOError(f"Cannot write EXR: {path}. {EXR_HINT}")


def wants_premult(mode, kind):
    """auto: EXR premultiplied (what Nuke and every compositor expect), PNG / TIFF / ProRes straight."""
    return mode == "premultiplied" or (mode.startswith("auto") and kind == "exr")


def unpremultiply(rgb, alpha):
    a = alpha[..., None]
    return np.where(a > 1e-6, rgb / np.maximum(a, 1e-6), 0.0).astype(np.float32)


class FrameSource:
    """Frames of an image batch, optionally with a mask as alpha channel (replacing any alpha the images have),
    or with the alpha dropped (rgb_only). Indexable and iterable without copying the whole batch."""

    def __init__(self, images, mask=None, rgb_only=False):
        self.images, self.mask, self.rgb_only = images, mask, rgb_only

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        a = np.asarray(self.images[i], dtype=np.float32)
        if self.mask is None:
            return a[..., :3] if (self.rgb_only and a.ndim == 3 and a.shape[-1] == 4) else a
        m = np.asarray(self.mask[min(i, len(self.mask) - 1)], dtype=np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        rgb = as_channels(a)[..., :3]
        return np.concatenate([rgb, np.clip(m, 0, 1)[..., None]], axis=-1)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def write_frame(path, arr, fmt, level="balanced", clamp=True, exr_linear=True, is_mask=False,
                alpha_mode=ALPHA_MODES[0], input_premult=False):
    """Writes one frame. arr: HxW (mask) or HxWxC float in [0,1] (sRGB-encoded for images).
    Masks are written as stored (no color conversion); in EXR they go to R, G, B and A, like Save Matte Sequence.
    RGBA images: alpha_mode decides straight / premultiplied (auto = EXR premultiplied, others straight);
    input_premult = the incoming RGB is already multiplied by the alpha (it is un-premultiplied first).
    EXR is premultiplied in linear light (after the sRGB -> linear conversion), as it should be."""
    kind = fmt[:3]
    if kind == "exr":
        half = "half" in fmt
        if is_mask:
            m = np.asarray(arr, dtype=np.float32)
            if m.ndim == 3:
                m = m[..., 0]
            if clamp:
                m = np.clip(m, 0.0, 1.0)
            write_exr(path, {"R": m, "G": m, "B": m, "A": m}, half, level)
            return
        a = as_channels(arr)
        has_a = a.shape[-1] == 4
        al = (np.clip(a[..., 3], 0, 1) if clamp else a[..., 3]) if has_a else None
        rgb = a[..., :3]
        if has_a and input_premult:
            rgb = unpremultiply(rgb, al)
        rgb = srgb_to_linear(rgb, clamp) if exr_linear else (np.clip(rgb, 0, 1) if clamp else rgb)
        if has_a and wants_premult(alpha_mode, "exr"):
            rgb = rgb * al[..., None]
        chans = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
        if has_a:
            chans["A"] = al
        write_exr(path, chans, half, level)
        return

    _need_cv2(fmt)
    a = np.asarray(arr, dtype=np.float32)
    if is_mask and a.ndim == 3:
        a = a[..., 0]
    single = a.ndim == 2
    if not single:
        a = as_channels(a)
        if a.shape[-1] == 4 and kind != "jpg":  # JPG drops the alpha: RGB is written as it came in
            al = np.clip(a[..., 3], 0, 1)
            rgb = unpremultiply(a[..., :3], al) if input_premult else a[..., :3]
            if wants_premult(alpha_mode, kind):
                rgb = rgb * al[..., None]
            a = np.concatenate([rgb, al[..., None]], axis=-1)
    if kind == "tif" and "32" in fmt:
        data = (a if not clamp else np.clip(a, 0.0, 1.0)).astype(np.float32)
    else:
        bits = 16 if "16" in fmt else 8
        data = _to_int(a, bits)
    if not single:
        if kind == "jpg" and data.shape[-1] == 4:
            data = data[..., :3]
        data = _bgr(data)
    if kind == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, {"fast (bigger files)": 1, "small (slow)": 9}.get(level, 3)]
    elif kind == "tif":
        # 1 = none, 5 = LZW, 8 = Adobe Deflate (ZIP)
        params = [cv2.IMWRITE_TIFF_COMPRESSION, {"fast (bigger files)": 1, "small (slow)": 8}.get(level, 5)]
    else:
        params = [cv2.IMWRITE_JPEG_QUALITY, {"fast (bigger files)": 90, "small (slow)": 85}.get(level, 95)]
    if not cv2.imwrite(path, data, params):
        raise IOError(f"{LOG} Cannot write {path}")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sequence_parallel(jobs, hash_files=True, progress=None):
    """jobs: list of (path, callable that writes the file). Writes in parallel and returns {path: sha256 or None}.
    The first error stops the remaining frames and is raised."""
    results = {}
    if not jobs:
        return results
    stop = threading.Event()

    def run(job):
        path, fn = job
        if stop.is_set():
            return path, None
        fn()
        return path, (sha256_file(path) if hash_files else None)

    with ThreadPoolExecutor(max_workers=worker_count()) as ex:
        futures = [ex.submit(run, j) for j in jobs]
        done = 0
        try:
            for fut in as_completed(futures):
                path, digest = fut.result()
                results[path] = digest
                done += 1
                if progress:
                    progress(done, len(jobs))
        except BaseException:
            stop.set()
            for f in futures:
                f.cancel()
            raise
    return results


# --------------------------------------------------------------------------
# Video (ffmpeg)
# --------------------------------------------------------------------------
_ENCODERS = None


def ffmpeg_path():
    return shutil.which("ffmpeg")


def ffmpeg_encoders():
    global _ENCODERS
    if _ENCODERS is None:
        exe = ffmpeg_path()
        _ENCODERS = set()
        if exe:
            try:
                out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True,
                                     timeout=20).stdout
                _ENCODERS = {line.split()[1] for line in out.splitlines()
                             if len(line.split()) > 1 and line.startswith(" V")}
            except Exception:
                pass
    return _ENCODERS


def video_ext(quality):
    return ".mov" if quality.startswith("ProRes") else ".mp4"


_BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def _codec_args(quality, encoder):
    """ffmpeg output arguments and the output pixel format for one quality preset."""
    if quality.startswith("preview"):
        if encoder == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0",
                    "-movflags", "+faststart"], "yuv420p"
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart"], "yuv420p"
    if quality.startswith("review"):
        if encoder == "hevc_nvenc":
            return ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p6", "-rc", "vbr", "-cq", "18",
                    "-b:v", "0", "-tag:v", "hvc1", "-movflags", "+faststart"], "p010le"
        return ["-c:v", "libx265", "-preset", "medium", "-crf", "16", "-x265-params", "log-level=error",
                "-tag:v", "hvc1", "-movflags", "+faststart"], "yuv420p10le"
    if "422" in quality:
        return ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"], "yuv422p10le"
    return ["-c:v", "prores_ks", "-profile:v", "4", "-vendor", "apl0", "-alpha_bits", "16"], "yuv444p10le"


def _encoder_candidates(quality):
    enc = ffmpeg_encoders()
    if quality.startswith("preview"):
        return [e for e in ("h264_nvenc", "libx264") if e in enc] or ["libx264"]
    if quality.startswith("review"):
        return [e for e in ("hevc_nvenc", "libx265") if e in enc] or ["libx265"]
    return ["prores_ks"]


def _ffmeta_escape(text):
    out = str(text)
    for ch in ("\\", "=", ";", "#", "\n"):
        out = out.replace(ch, "\\" + ch)
    return out


def _metadata_file(metadata):
    """ffmetadata file for any length of text (a workflow is often larger than a command line allows)."""
    fd, path = tempfile.mkstemp(prefix="pertineo_meta_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for k, v in metadata.items():
            f.write(f"{_ffmeta_escape(k)}={_ffmeta_escape(v)}\n")
    return path


def audio_for_video(audio, n_frames, fps):
    """ComfyUI AUDIO -> (float32 [channels, samples], sample_rate) of exactly the video's length: longer audio is cut,
    shorter audio is padded with silence. None if no usable audio."""
    if not isinstance(audio, dict) or audio.get("waveform") is None:
        return None
    wav = audio["waveform"]
    wav = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav[None]
    rate = int(audio.get("sample_rate") or 48000)
    want = max(1, int(round(n_frames / float(fps) * rate)))
    if wav.shape[1] < want:
        wav = np.pad(wav, ((0, 0), (0, want - wav.shape[1])))
    return np.ascontiguousarray(wav[:, :want]), rate


def _audio_file(audio):
    """Raw float32 interleaved samples in a temp file (ffmpeg input -f f32le)."""
    wav, _rate = audio
    fd, path = tempfile.mkstemp(prefix="pertineo_audio_", suffix=".f32")
    with os.fdopen(fd, "wb") as f:
        f.write(np.ascontiguousarray(wav.T, dtype="<f4").tobytes())
    return path


def write_video(path, frames, fps, quality, is_mask=False, overwrite=False, metadata=None, audio=None,
                alpha_mode=ALPHA_MODES[0], input_premult=False):
    """frames: sequence of HxW (mask) or HxWxC float arrays in [0,1] (sRGB-encoded). Streams 16-bit frames to
    ffmpeg, so 10-bit codecs get real 10-bit values. NVENC is tried first where available, then the software
    encoder. metadata: dict of container tags (e.g. the workflow). audio: result of audio_for_video – muxed as
    24-bit PCM in .mov (lossless) or AAC 320 kbit/s in .mp4. Returns the encoder that was used."""
    exe = ffmpeg_path()
    if not exe:
        raise IOError(f"{LOG} Video output needs ffmpeg. Install it (Linux Mint / Ubuntu: sudo apt install ffmpeg) "
                      f"or turn write_video off.")
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    alpha = (not is_mask) and first.ndim == 3 and first.shape[-1] == 4 and "4444" in quality
    in_fmt = "gray16le" if is_mask else ("rgba64le" if alpha else "rgb48le")
    errors = []
    meta_path = _metadata_file(metadata) if metadata else None
    audio_path = _audio_file(audio) if audio is not None else None
    try:
        return _encode(exe, path, frames, fps, quality, is_mask, overwrite, meta_path, w, h, alpha, in_fmt, errors,
                       audio, audio_path, alpha_mode, input_premult)
    finally:
        for tmp in (meta_path, audio_path):
            if tmp:
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def _encode(exe, path, frames, fps, quality, is_mask, overwrite, meta_path, w, h, alpha, in_fmt, errors,
            audio=None, audio_path=None, alpha_mode=ALPHA_MODES[0], input_premult=False):
    for encoder in _encoder_candidates(quality):
        args, pix = _codec_args(quality, encoder)
        # custom tags (workflow, prompt) need use_metadata_tags in mov / mp4
        args = [a for a in args]
        if "-movflags" in args:
            i = args.index("-movflags")
            del args[i:i + 2]
        args += ["-movflags", ("+faststart" if path.lower().endswith(".mp4") else "") + "+use_metadata_tags"]
        if alpha:
            pix = "yuva444p10le"
        # 4:2:0 / 4:2:2 need even sizes: pad by one pixel (never scale)
        vf = []
        if pix not in ("yuv444p10le", "yuva444p10le") and (w % 2 or h % 2):
            vf.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        vf.append("scale=out_color_matrix=bt709:out_range=tv")
        vf.append(f"format={pix}")
        inputs = ["-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
        maps = ["-map", "0:v"]
        idx = 1
        if meta_path:
            inputs += ["-f", "ffmetadata", "-i", meta_path]
            maps += ["-map_metadata", str(idx)]
            idx += 1
        if audio_path:
            ch, rate = audio[0].shape[0], audio[1]
            inputs += ["-f", "f32le", "-ar", str(rate), "-ac", str(ch), "-i", audio_path]
            maps += ["-map", f"{idx}:a"]
            acodec = (["-c:a", "aac", "-b:a", "320k"] if path.lower().endswith(".mp4")
                      else ["-c:a", "pcm_s24le"])
        else:
            acodec = ["-an"]
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n"] + inputs + maps + [
               "-vf", ",".join(vf)] + acodec + args + _BT709 + [path]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr = b""
        try:
            for fr in frames:
                a = np.asarray(fr, dtype=np.float32)
                if is_mask:
                    if a.ndim == 3:
                        a = a[..., 0]
                    data = _to_int(a, 16)
                else:
                    a = as_channels(a)
                    if alpha:  # ProRes 4444: straight by default (auto), premultiplied on request
                        al = np.clip(a[..., 3], 0, 1)
                        rgb = unpremultiply(a[..., :3], al) if input_premult else a[..., :3]
                        if wants_premult(alpha_mode, "mov"):
                            rgb = rgb * al[..., None]
                        a = np.concatenate([rgb, al[..., None]], axis=-1)
                    data = _to_int(a if alpha else a[..., :3], 16)
                proc.stdin.write(np.ascontiguousarray(data, dtype="<u2").tobytes())
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        finally:
            stderr = proc.stderr.read()
            proc.wait()
        if proc.returncode == 0:
            return encoder
        msg = stderr.decode("utf-8", "replace").strip()
        errors.append(f"{encoder}: {msg[-400:] or 'exit code ' + str(proc.returncode)}")
        if "already exists" in msg:
            break
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        print(f"{LOG} Video: {encoder} failed, trying the next encoder …")
    raise IOError(f"{LOG} Video could not be written: {path}\n" + "\n".join(errors))


def write_preview_video(path, frames, fps, audio=None, is_mask=False, max_height=540):
    """Small H.264 / AAC mp4 for the player on the node (browser-playable): 8 bit, scaled down to max_height, fast.
    audio: result of audio_for_video or None."""
    exe = ffmpeg_path()
    if not exe:
        raise IOError("ffmpeg not found")
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    in_fmt = "gray" if is_mask else "rgb24"
    inputs = ["-f", "rawvideo", "-pix_fmt", in_fmt, "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-"]
    audio_path = _audio_file(audio) if audio is not None else None
    if audio_path:
        inputs += ["-f", "f32le", "-ar", str(audio[1]), "-ac", str(audio[0].shape[0]), "-i", audio_path]
    enc = "h264_nvenc" if "h264_nvenc" in ffmpeg_encoders() else "libx264"
    venc = (["-c:v", "h264_nvenc", "-preset", "p3", "-cq", "28"] if enc == "h264_nvenc"
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"])
    vf = f"scale=-2:'min({max_height},ih)':flags=bicubic,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p"
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y"] + inputs + ["-map", "0:v"] + \
          (["-map", "1:a", "-c:a", "aac", "-b:a", "160k"] if audio_path else ["-an"]) + \
          ["-vf", vf] + venc + ["-movflags", "+faststart", path]
    try:
        for attempt in (venc, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"]):
            cmd_try = cmd[:cmd.index(venc[0])] + attempt + cmd[cmd.index(venc[0]) + len(venc):]
            proc = subprocess.Popen(cmd_try, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for fr in frames:
                    a = np.asarray(fr, dtype=np.float32)
                    if is_mask and a.ndim == 3:
                        a = a[..., 0]
                    if not is_mask:
                        a = as_channels(a)[..., :3]
                    proc.stdin.write(np.ascontiguousarray(_to_int(a, 8)).tobytes())
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            err = proc.stderr.read()
            proc.wait()
            if proc.returncode == 0:
                return path
            if attempt[1] == "libx264":
                raise IOError(err.decode("utf-8", "replace")[-300:])
    finally:
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                pass


def write_checksums(path, digests, rel_to):
    """sha256sum format (hash, two spaces, relative path) – check later with: sha256sum -c <file>."""
    lines = [f"{d}  {os.path.relpath(p, rel_to).replace(os.sep, '/')}" for p, d in sorted(digests.items()) if d]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)
