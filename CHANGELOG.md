# Changelog

All notable changes to **Pertineo VFX for ComfyUI** (`comfyui-pertineo-vfx`).
Versions follow `major.minor.patch`: patch = bug fix, minor = new nodes / options (backwards compatible).

## 1.4.2 – 2026-10-27

### Fixed
- **Models chosen by name were missing** (e.g. RMBG: `model = INSPYRENET`, files in `models/RMBG/INSPYRENET/`):
  the audit only recognised inputs that point to a model file. A text / choice value that is exactly the name of a
  model folder (in any model folder ComfyUI knows, incl. `extra_model_paths.yaml`, and one level below each models
  root) now counts as a model: its weight files are listed and hashed (role *Segmentation / matting model* for
  RMBG, BiRefNet, BEN …) and matched against model_info by hash or file pattern. ComfyUI's own folders (loras,
  checkpoints …) are never matched, so a value like `flux` cannot pull in a LoRA sub folder; a folder with more
  than 10 weight files is listed but not hashed.
- **PDF thumbnail of an image with alpha was black** where the image is transparent: it is now shown over a grey
  checkerboard (half-transparent areas stay visible). The saved PNG / EXR keeps its alpha as before.
- An empty manual field (e.g. *Reference images*) reads `UNKNOWN / NOT VERIFIED` instead of
  `UNKNOWN / NOT VERIFIED – not provided` – one label, the finding "Not filled in" already says the rest.
- An input that goes through a background-removal node is listed as *source (RMBG)* instead of
  *source (PertineoSaveImageMetadata)*.

## 1.4.1 – 2026-10-26

### Fixed
- **Positive prompt shown as negative prompt:** an encoder with a positive and a negative text field that feeds
  both sampler inputs (e.g. `TextEncodeQwenImage21`: `prompt` + `negative_prompt`, outputs *positive* /
  *negative*) showed both texts under both labels. Each side now shows only its own text, chosen by the output
  the link comes from.
- **Input wrongly listed as *preview only*:** when a preview node passes the image on (newer ComfyUI:
  `PreviewImage` has an `images` output), the audit stopped at the preview. It now follows the image further, so
  an image that reaches the model through a preview node is listed as *source* (or guidance / reference …).
- **Latent size UNKNOWN** when the latent is made by a node itself (e.g. the `latent` output of
  `TextEncodeQwenImage21`): the size is now read from the latent produced in this run and multiplied by the scale
  factor of the VAE that decodes it. If the factor cannot be read, the latent size is shown and the pixel size
  stays UNKNOWN – nothing is guessed.
- **Source image "not connected"** although inputs went into the result: the line now names the input files used as
  *source*, one per line. Connecting `source_image` to Collect Metadata is still optional.
- **Seed control "fixed" although rgthree Seed was set to random (-1):** the frontend stores the rolled seed in the
  workflow, the setting itself only in the named widget value. The setting is now read from there, so a random seed
  shows as *randomize* and the run as *reproducible only with the recorded values*.
- **Input location:** an input folder set with `--input-directory` is shown as *custom ComfyUI input folder*, an
  `input` folder that is a link to another drive as *ComfyUI input folder (linked to another location)*.
- Multi-image nodes: *Used as* names the input an image reaches (e.g. `source (TextEncodeQwenImage21 · image_1)`),
  and *Source image* lists the files in that order (`P1025544.png → image_1`) – which image is the base of an edit is up to the prompt, the
  report does not guess it.

### Added
- Every input image is listed with its **size** (read from the file header).
- INPUT FILES summary reads `2 × source · 1 × preview only`.
- **`<image>.workflow.json`** next to the report: the workflow that was run, credentials redacted – opens directly
  in ComfyUI (drag and drop). Written with `write_json` and `embed_workflow` on.

### Changed
- **Report files are named `<image>.audit.txt` / `.audit.json` / `.audit.pdf`** (before: `.txt` / `.json` / `.pdf`),
  so they stand out next to the image and the `.workflow.json`. The counter also skips numbers used by the old names.
- The **`.audit.txt` is complete:** every inactive node is listed by name, prompt texts are never shortened. The
  **PDF** lists up to five inactive nodes (loaders with model files always) and shortens texts over 512 words; when
  it shortens anything, its header says so and names the `.audit.txt`.
- The `.json` now carries the **workflow that was run** (`comfyui_workflow`: UI workflow and API prompt), with
  credentials redacted – the report and the graph it describes stay together, also without the PNG (needed for
  the audit chain), with the name and SHA256 of the `.workflow.json`. Follows `embed_workflow`: off = no workflow in
  the `.json` and no `.workflow.json`. Not in the `.txt` / `.pdf`.

## 1.4.0 – 2026-10-25

### Added
- **Every input file is listed with how it is used:** *source* (its pixels go into the result), *guidance*
  (ControlNet image, pose, depth, mask), *reference* (IP-Adapter, CLIP vision, style model, face ID), *prompt
  reference* (read by an LLM / VLM prompt writer – it shapes the prompt, it is not in the image), *preview only*,
  or *not used* (loaded, but not connected to an output). The section starts with a one-line summary; the legend
  explains each use. An image that only feeds a prompt writer is therefore no longer a bare "image" without context.
- New report section **PROMPT GENERATION**: for every LLM / VLM / prompt node that writes a prompt, what it was
  given in this run – images (traced back to the input file and its number), instructions, presets, linked texts
  (with the node they came from), settings such as seed and temperature, its model, and what an intermediate
  writer wrote. Chains of writers (e.g. QwenVL → PixaromaAIPrompt) are listed in order. Credentials are redacted.
  Texts longer than 512 words are shortened in the `.txt` / `.pdf`; the `.json` has them in full.
- Collect Metadata: optional inputs **`step`** and **`step_description`** (at the end, existing workflows are
  unchanged) – which step of the pipeline this workflow is, the same for every iteration of that step. Shown in the
  report header.
- The `.json` records the groups of the workflow (`workflow_outline`) with the nodes that ran in each – raw
  material for a later breakdown across steps. Not printed in the `.txt` / `.pdf`.

### Changed
- Prompt writers are recognised by more node names (QwenVL, LLaVA, MiniCPM, Moondream, InternVL, Molmo, Janus,
  BLIP, taggers / interrogators). A VLM node that loads its model itself is listed as *VLM – used for: writes
  the prompt – <that node>*.
- "Commercial use: OK" only mentions separate licenses when a model actually needs one
  ("OK – all models allow it (1 through a separate license)").
- The models section no longer prints the file name of the license table, only that the license data was
  provided by the user (the `.json` still records the file name).
- "Generated in this run (…)" names the node that actually produced the prompt, also when display or reroute
  nodes sit in between.

## 1.3.0 – 2026-10-25

### Added
- Save Image + Metadata writes the **SHA256 of the saved image** into `.txt` / `.json` / `.pdf`. A later workflow
  that loads the image lists the same hash under input files, which links multi-step work.
- Production Context: tooltips for every setting; evidence fields (input rights, talent consent, union approval,
  studio approval, legal approval) now offer `unknown / yes / no / pending / not applicable`, default `unknown`.
  Existing workflows keep their values.
- Studio and legal approval now count in the assessment: a declared written studio / client approval is noted at
  the written-approval triggers and next to grade C; legal clearance softens third-party and trademark findings.
- Trademarks / brands `yes` or `unknown` is a finding of its own; for published material without legal
  clearance it is a written-approval point (C).
- The triage box lists each point that needs written approval as *covered* (and by what) or *OPEN*, so the
  reason for grade C is visible at a glance.
- Intermediate steps (`internal production asset` with final result `yes`) get a note to audit the final step
  as well instead of a contradiction.
- Every report states the date of the rule set ("Rules as of 2026-10-25").
- **Model licenses count in the grade**, weighted by whether the result is published: a license that does not
  allow commercial use (without a separate license) in published material is a critical *license conflict*;
  an unknown license for published material is a written-approval trigger (C). The triage box shows a one-line
  license summary.
- "Commercial use" summary of the models – OK / CHECK TERMS / NOT PERMITTED / UNKNOWN / NOT VERIFIED (worst
  case wins) – also without a Production Context, as it depends on the model licenses only.
- New grade **X – BLOCKED** (scale A, B, C, D, X – D means "not assessable"): a critical finding means the use as declared is not permitted, and an approval
  cannot fix it. Critical always means X.
- README: design principle (simple, stable, record before appearance), every Production Context setting with its
  effect on the assessment, multi-step work, and what the assessment is based on (with sources and date).

### Changed
- Everything not known is shown as **`UNKNOWN / NOT VERIFIED`** instead of `UNKNOWN`. JSON values for scripts
  are unchanged (`"unknown"`).
- **Without a Production Context there is no triage:** the report says "Triage: not assessed – no Production
  Context connected" and leaves out the triage, production context and guiding principles sections (JSON:
  `triage.grade = null`, `principles = []`). Grade D now only means: context connected, but purpose and final
  result both unknown.
- Guiding principles are printed as principle + status on two lines.
- A credential found in the workflow is now a *high* finding (B) instead of critical – it endangers sharing,
  not the permission to use the image, and is fixed by removing it.
- The report explains the two scales: severity rates a single finding, the grade says what has to happen next.

### Fixed
- Prompts written in the run by an LLM / VLM node (e.g. a prompt generator that looks at a layout image) are
  marked with the node that wrote them, and reproducibility says so: running the workflow again writes a new
  prompt, the recorded prompt reproduces the image ("reproducible only with the recorded values"). A
  non-fixed seed control is treated the same way. The reproducibility notes are listed in the report.
- A model that only feeds a prompt generator is shown as VLM with "Used for: writes the prompt – …" instead of
  as a text encoder.
- Input files list which nodes they feed (e.g. the prompt generator).
- A negative conditioning made with ConditioningZeroOut is shown as "none – conditioning zeroed out" instead of
  repeating the positive prompt.
- Latent size from size nodes that output width / height (e.g. Pixaroma Sizes) is taken from the values they
  produced in the run.
- Seed control: a sampler whose seed input is linked no longer reports its own (unused) control widget; Seed
  (rgthree) is read correctly (-1 = randomize, other values = fixed).
- Output and "other settings" are listed one setting per line instead of one long line.
- PDF: lines wider than the page (long URLs, paths) are wrapped instead of cut off.
- PDF: copying / selecting text matches what is shown (explicit Courier glyph widths, every text column placed
  at its exact position, label and value joined by a light dot leader so viewers do not split them into
  separate columns – the label column is wide enough everywhere for that); a section heading is
  never left alone at the bottom of a page.
- The manual field `changes_outside_comfyui` is shown as "Planned work outside ComfyUI": work done before is
  an input of the workflow, work done later needs its own record. It no longer counts as "not filled in".
- Optional fields (notes, planned work) stay empty instead of UNKNOWN; "UNKNOWN / NOT VERIFIED" no longer
  repeats "add to model_info file" on every line (the findings still say where to add it).
- "Source image" says that there is no source image in this run and points to the input files.
- GGUF models of LLM / VLM nodes that list only the file name (e.g. QwenVL GGUF, files in sub folders of
  `models/LLM`) are found, hashed and matched against the model_info file. The vision projector
  (`mmproj*.gguf`) next to such a model is listed too, marked as derived – it is loaded automatically and does
  not appear in the workflow, but has a license of its own.

### Credits
- New **CREDITS** section: every model with creator and license, every node package with publisher, license
  and repository, and the attribution your licenses require.
- Node package credits are read from the installed packages themselves (`pyproject.toml`, LICENSE file,
  git remote) – nothing to maintain.
- Model creator: your model_info table is preferred; if it has none, the creator stated in the model file
  (safetensors `modelspec.author`, GGUF `general.author` / `general.organization`) is used. The report says
  where each value comes from. A license stated in the file is shown as "License in file … not verified" – it
  does not replace your table and does not change the grade.

### Readability
- Without a Production Context the LEGEND only explains severity and evidence (no grade scale, no triage note).
- Consent / approval fields show "n/a – none declared" when talent, union work or third-party material is
  declared absent, instead of UNKNOWN.
- The triage box is short: grade, reason, open points, model licenses, reproducibility. The scale, severity,
  evidence and principle explanations moved to a **LEGEND** section at the end of the report.
- Node numbers and node names are left out of the `.txt` / `.pdf` where they do not help a reviewer (latent
  source, seed node, "loaded by", "used by" numbers) – they remain in the `.json`. Findings that point to a node
  to fix (API node, credential) keep the number.
- Inactive nodes are summarised (counts, plus bypassed loaders with their model file); the full list is in the
  `.json`.
- A written studio / client approval on file turns the approval findings to *low*; the grade reads
  "C – APPROVAL ON FILE".
- Clearer labels, placeholders and tooltips: "final cut" also covers ads, packaging and print ("final result");
  the manual field for reference images asks for origin and rights, not the file name.

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
