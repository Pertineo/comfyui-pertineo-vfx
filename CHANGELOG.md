# Changelog

All notable changes to **Pertineo VFX for ComfyUI** (`comfyui-pertineo-vfx`).
Versions follow `major.minor.patch`: patch = bug fix, minor = new nodes / options (backwards compatible).

## 1.1.0 – 2026-10-24

### Added
- New node group **Audit** (`Pertineo/Audit`):
  - **Pertineo Collect Metadata** – collects an audit record of the run from the executed graph:
    environment (ComfyUI, frontend, Python, torch, CUDA, GPU, OS), all model files with SHA256 and
    AutoV2 (cached), source / license / commercial use from a user-maintained `model_info.json`
    (missing entries are marked UNKNOWN, never guessed), custom node packages with versions, API nodes,
    sampler settings with linked values resolved to constants (computed values such as joined prompts
    are taken from this run's output cache), prompts per sampler, resolutions,
    frames / fps / duration, five manual audit fields and optional author / company (with a switch
    to leave them out). Outputs a human-readable text and JSON.
  - **Pertineo Save Image + Metadata** – saves images (PNG 8 / 16 bit, EXR 32-bit float) and the
    `.txt` / `.json` report with guaranteed identical base names; one pair per batch image; never
    overwrites; embeds prompt, workflow and audit JSON in PNGs.
  - **Pertineo LoRA Loader + Info** – loads a LoRA and outputs its trigger words, base model and
    license info from `model_info.json`.
  - **Pertineo Model Info (LoRA / Checkpoint / Diffusion Model / VAE / Text Encoder / ControlNet /
    Upscaler)** – look up name, base model, trigger words, recommended settings and license of any
    model file without loading it.
- model_info as **spreadsheet (`.ods` with formatting, `.xlsx`, `.csv`)** or JSON, read with the
  standard library; header row found automatically, free column order; an empty `model_info.csv` is
  created in `<user>/pertineo_vfx/` on first use. Fields `name`, `version`,
  `released`, `base_model`, `trigger_words`, `recommended`, `checked`, `autov2`, `_disclaimer`.
  The report warns on a hash mismatch and when a LoRA's base model differs from the main model.
- Templates `model_info.example.ods` / `.csv` / `.json`; real `model_info.*` files are
  git- and registry-ignored.
- Every report states that license data comes from the user's own file and is given without warranty.
- `model_info.example.json` template and `example_workflows/audit_example.json`.
- README: Audit section, missing rows for Empty Audio and Frames for Model, development note.

### Notes
- These nodes were developed with AI assistance (Claude by Anthropic). Concept, VFX requirements,
  testing in production pipelines and maintenance are by Julia Kietzmann.
- No new dependencies. Existing Matting and I/O nodes are unchanged.

## 1.0.0 – 2026-10-12

- First release: Trimap from Matte, ViTMatte Tiled Refiner, ViTMatte Tiled (Sequence),
  Load Image Sequence, Save Matte Sequence, Empty Audio, Frames for Model.
