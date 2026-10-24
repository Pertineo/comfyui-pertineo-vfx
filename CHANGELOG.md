# Changelog

All notable changes to **Pertineo VFX for ComfyUI** (`comfyui-pertineo-vfx`).
Versions follow `major.minor.patch`: patch = bug fix, minor = new nodes / options (backwards compatible).

## 1.2.0 – 2026-10-24

### Added
- **Pertineo Production Context** (new node): declares production, project / episode, shot, vendor,
  purpose, final cut, story-critical, execution environment, unreleased material, personal data,
  talent likeness / voice, third-party material, trademarks, union-covered work, whether external
  tools store / train on the data, available approvals and consents, and reviewers. Connects to the
  new optional input `production_context` of Collect Metadata. Recorded as *declared*, never as verified.
- Audit report – **findings** with severity, responsible department (Legal / rights, Talent & consent,
  Data protection / IT security, Production / studio approval, VFX supervision / QC, Documentation)
  and evidence class (observed / derived / declared / external evidence required).
- Audit report – **indicative triage A–D** (low-risk candidate, review required, written approval
  indicated, indeterminate) and five **guiding principles** answered yes / no / unknown with the reason.
  Clearly marked as an orientation, not an approval and not legal advice.
- Audit report – input files with role and SHA256; bypassed, muted and unconnected nodes listed as
  *not part of the run*; seed control mode; settings of every executed node in the JSON (key ones such
  as ControlNet, IP-Adapter, upscale, mask in the text); save nodes; workflow type; identity-related
  capabilities (face swap, identity adapters, face restoration, voice / lip-sync, pose transfer);
  remote inputs; reproducibility status with what is missing.
- Credentials in node inputs (API keys, tokens) are redacted in the report and raise a critical finding.
- **PDF export**: optional `write_pdf` on Save Image + Metadata writes the report as A4 PDF with a
  thumbnail, without extra dependencies. Characters the standard PDF font cannot show (e.g. Chinese)
  become `?`, with a note in the PDF; `.txt` / `.json` keep the full text. The file counter also skips
  existing `.pdf` files.
- model_info: new columns **Separate license** (yes / no) and **License reference** – a model whose
  public license does not allow commercial use is shown as *covered by a separate license* instead of an
  open point. Further new columns: Creator, Attribution (listed as a finding), Training data.
- model_info: **family rows** – a pattern with `*` / `?` in `File` (e.g. `*minimax_h3*`) covers every
  matching model file. An entry for the exact file wins; empty cells are filled from the family row.
  Model Info and LoRA Loader + Info nodes use family rows as well.
- Templates (`model_info.example.ods` / `.csv` / `.json`) and the example workflow updated.

### Changed
- "To check" in the report now lists the findings with severity and department. Reports saved as JSON
  by 1.1.x still render as before.
- Everything that is not known is shown as **UNKNOWN** in the report: undeclared context answers, missing
  manual fields (`UNKNOWN – not provided`), missing license-check dates, unrecorded separate licenses,
  commercial-use values outside yes / no / restricted / unclear, hashes not computed, values computed in
  the graph that could not be recorded, unknown seed mode, package versions and environment values.
  A license without a commercial-use value is a finding of its own.
- The triage box shows the full A–D scale with the current grade marked; "To check" and the guiding
  principles explain severity, evidence classes and when YES / UNKNOWN is given.

All new inputs are optional and appended at the end; existing workflows keep working.

## 1.1.1 – 2026-10-24

### Fixed
- Audit report: node package versions now show the version that is **installed and ran**, not the
  version stored in the workflow when the node was added (that one can be outdated). If they differ,
  the report notes "workflow saved with …". Git checkouts show "x.y.z (git <commit>)", registry
  installs just "x.y.z".

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
