# Pertineo VFX for ComfyUI

Compositing-grade tools for ComfyUI – built by a compositor for real VFX and post-production pipelines: full resolution, image sequences, high bit depth, no shortcuts.

| Group | Nodes |
|---|---|
| **Matting** | ViTMatte Tiled Refiner · ViTMatte Tiled (Sequence) · Trimap from Matte |
| **I/O** | Load Image Sequence · Save Matte Sequence (16-bit PNG / 32-bit EXR) · Empty Audio · Frames for Model |
| **Audit** | Collect Metadata · Save Image + Metadata · LoRA Loader + Info · Model Info (LoRA, Checkpoint, Diffusion Model, VAE, Text Encoder, ControlNet, Upscaler) |

More tools will follow. Each group loads independently – if one fails (e.g. a missing dependency), the others keep working.

---

## Matting – ViTMatte at full resolution

Production-grade matting with [ViTMatte](https://github.com/hustvl/ViTMatte) at **full resolution** – no downscaling, no tile seams. Refine coarse mattes from video models (e.g. an LTX alpha) or keyers into clean, detailed edges for hair, motion blur and semi-transparency – in 4K and beyond.

### Why

At 4K a single ViTMatte pass does not fit into VRAM, and most implementations quietly downscale the image. Splitting into tiles by hand leaves visible seams, because every tile sees a slightly different context.

These nodes compute ViTMatte on **overlapping tiles** and blend them with smooth, normalized weights. Every pixel is a weighted average of all tiles that cover it, so the result is seamless – even where neighbouring tiles disagree.

### Nodes

| Node | Purpose |
|---|---|
| **Pertineo Trimap from Matte** | Builds a trimap from a coarse matte: cut at a threshold, erode inwards for the solid core, dilate outwards for the unknown band. Takes the matte as `MASK` or as `IMAGE` (with channel select – e.g. a matte rendered as RGB sequence). Optional core mask (forced foreground) and garbage mask (forced background; `invert_garbage` for a roto drawn around the subject). The preview shows the band as an overlay on the plate. |
| **Pertineo ViTMatte Tiled Refiner** | Refines a matte batch inside a workflow (plate `IMAGE` + trimap `MASK` → `MASK`). The refiner stage behind a video-model alpha such as LTX. |
| **Pertineo ViTMatte Tiled (Sequence)** | Reads plate and trimap folders frame by frame (PNG / TIFF / EXR) and writes the matte sequence. For long, high-resolution shots: never holds the whole sequence in memory and can resume after an abort. |

### Typical workflows

**Inside ComfyUI (e.g. behind an LTX alpha):**

```
source plate (full res) ─────────────────┬──► ViTMatte Tiled Refiner ──► Save Matte Sequence
coarse alpha (LTX) ──► Trimap from Matte ─┘    (reference_image = plate)
```

**Hybrid with a compositing app (Nuke, Natron, …):**

1. Render plate and a hand-tuned trimap as image sequences.
2. Run **ViTMatte Tiled (Sequence)** on both folders.
3. Read the matte back into your comp.

Example workflows are in `example_workflows/`.

### Inputs and color

| | Expected |
|---|---|
| Plate | RGB. PNG / TIFF in sRGB (8 or 16 bit), or EXR in scene-linear (converted to sRGB automatically, can be turned off). |
| Trimap | Grayscale, written **raw / linear**: 0 = background, 0.5 = unknown, 1 = foreground. |
| Output | Matte, linear. Read PNG as raw / linear. EXR stores the matte in R, G, B and A. |

One sequence per folder; frames are paired by the number in the file name (`name.1001.png`).

**Setting paths:** besides a folder you can paste the path of any frame of the sequence (e.g. copy a file in your file manager and paste it into the field), a `file://` URI, or a pattern like `name.####.png`, `name.%04d.png` or `name.$F4.exr` – the node uses the folder. Paths are checked when you press *Run*: a missing folder, a folder without numbered images or plate/trimap sequences without common frame numbers are reported on the node before anything is computed. The console shows the frame count and range of both sequences.

**Trimap pitfalls**

- Writing the trimap in sRGB turns 0.5 into ≈ 0.73 – write it raw / linear.
- Dithering adds noise to the three values – turn it off.
- ViTMatte only refines the **unknown band**. Anything inside the white core stays hard. If hair lies inside a coarse matte, erode the core further.

### Parameters

| Parameter | Default | |
|---|---|---|
| `model` | base | Base = best quality, Small = faster, less VRAM, Custom = your own weights (see *Model license*). |
| `custom_model` | – | Only for `model = custom`: folder of your own ViTMatte weights. |
| `tile_size` | 1024 | Largest size that fits into VRAM = best quality (more context). VRAM grows quadratically: ≈ 1536 fits into 32 GB, 2048 does not. |
| `overlap` | 256 | Overlap between tiles in px. More = smoother transitions. |
| `flip_tta` | on | Second, horizontally flipped pass, averaged. Calmer edges, twice the time. |
| `clamp_known` | on | After matting: alpha = 0 / 1 where the trimap is black / white. |
| `quantize_trimap` | on | Before matting: round the trimap to 0 / 0.5 / 1. Turn off for soft trimaps (behaves like the Nuke Cattery node). |

Tiles without any unknown area are skipped. The console logs per frame how much of the trimap is unknown and how many tiles were computed – the fastest way to spot a broken trimap.

### Notes

- ViTMatte works frame by frame. Fine hair can flicker slightly over time; a light temporal smoothing of the unknown band in comp helps.
- Out of VRAM: lower `tile_size` in steps of 256 and restart ComfyUI.
- Long 4K shots as `IMAGE` batches need a lot of RAM (≈ 100 MB per frame). For very long shots use the Sequence node.

### Model license – please read

The node code is GPL licensed, but the **model weights are not part of this repository** and come with their own terms:

- The default models are the official ViTMatte weights by hustvl ([vitmatte-base-composition-1k](https://huggingface.co/hustvl/vitmatte-base-composition-1k), [vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k)). Hugging Face lists them as Apache-2.0.
- However, they were trained on **Composition-1k**, which is built from the **Adobe Deep Image Matting dataset**. That dataset's license restricts models trained on it to **non-commercial use**. Other projects using models trained on this data state the same restriction, e.g. [FBA Matting](https://github.com/Zikun97/FBA_Matting) and [LibreYOLO's port of the same ViTMatte weights](https://huggingface.co/LibreYOLO/LibreViTMattes-matte).
- This applies to these weights wherever they are used – in ComfyUI, in Nuke (Cattery) or elsewhere.

Whether and how such dataset terms bind end users is legally not settled, and this is not legal advice. **If you use the default weights in paid client work, clarify this with your client or a lawyer first.** The node prints a short reminder to the console whenever it loads the hustvl weights.

**Own weights:** set `model` to `custom` and point `custom_model` to a folder with ViTMatte weights in Hugging Face format (`config.json` + `model.safetensors`) – e.g. weights you trained on your own data or licensed for commercial use. `custom_model` accepts an absolute path or the name of a subfolder in your `vitmatte` model folder.

---

## I/O

| Node | Purpose |
|---|---|
| **Pertineo Load Image Sequence** | Loads a numbered sequence (PNG / TIFF 8 or 16 bit, EXR) in full precision, with first and last frame as real frame numbers (e.g. 1001–1100). Outputs `IMAGE`, a `MASK` from a chosen channel (red / green / blue / alpha / luminance, read raw), the first frame number, the frame count and the loaded `folder`. EXR images are converted to sRGB (optional); missing frames stop with an error, or are held / skipped. Shows the middle frame of the range as a preview on the node after loading. Only reloads when the files change. |
| **Pertineo Save Matte Sequence** | Writes a matte batch as a numbered sequence in 16-bit PNG or 32-bit float EXR, with a configurable start frame (default 1001). ComfyUI's standard Save Image only writes 8 bit. Connect `folder` from Load Image Sequence to `source_folder` to write next to the source automatically: folder `shot_v01` → `shot_v01_matte`; with an empty `filename_prefix` the files are named after the source as well (`shot_v01.1001.png` → `shot_v01_matte.1001.png`). Suffix configurable. |
| **Pertineo Empty Audio** | Silent `AUDIO` track whose length matches a sequence (`frame_count` / `fps`), for video models such as LTX that expect an audio input. |
| **Pertineo Frames for Model** | Central frame range: first and last frame are adjusted so the length fits the video model (LTX 8n+1, Wan 4n+1, MiniMax H3 17n+5 or a custom rule). Connect `first_frame` / `last_frame` to the loaders and Save Matte Sequence, `length` to the model and Empty Audio. |


---

## Audit – document how an image was made

Two nodes that record **what an image was made with** – for clients, license questions and your own traceability – and save image and report with the **same base name**.

```
… ──► VAE Decode ──┬──────────────────────────────► Save Image + Metadata ──► shot_001.png
                   └─► Collect Metadata ──text/json─┘                         shot_001.txt / .json
```

| Node | Purpose |
|---|---|
| **Pertineo Collect Metadata** | Reads the executed graph at run time (no widgets needed) and collects: environment (ComfyUI, frontend, Python, torch, CUDA, GPU, OS), every model file with **SHA256** and **AutoV2** (Civitai short hash), **source / license / commercial use** from your `model_info` table, custom node packages with versions, API nodes, sampler settings (seed, steps, cfg / guidance, sampler, scheduler, denoise, latent size), positive and negative prompts per sampler, and source / generation / final resolution (plus frames, fps and duration for video). Adds five manual audit fields. Outputs `metadata_text` and `metadata_json`. |
| **Pertineo Save Image + Metadata** | Saves the images plus `.txt` and/or `.json` with **guaranteed identical base names**: the counter skips every number where an image, `.txt` or `.json` already exists, so `name_004.png` can never sit next to `name_003.txt`. Every image of a batch gets its own pair (the report states the batch index). Never overwrites. PNG 8 bit, PNG 16 bit or EXR 32-bit float. Embeds prompt and workflow like the core Save Image node. |

| **Pertineo LoRA Loader + Info** | Loads a LoRA like the core LoRA loader and additionally outputs what your `model_info` table says about it: `trigger_words` (comma-separated, ready to join into a prompt), `base_model` and a short `info` text with license and source. The LoRA is chosen only once, so loader and info always match. `clip` is optional (model-only LoRAs). A missing entry never stops the run. |
| **Pertineo Model Info (…)** | Look up your own records without loading anything: one node per model type (LoRA, Checkpoint, Diffusion Model, VAE, Text Encoder, ControlNet, Upscaler), each with the same file list as the matching loader. Outputs `name`, `base_model`, `trigger_words`, `recommended`, `license` and an `info` text – connect `info` to *Preview Any* to read it. Useful for LoRAs loaded by other loaders and for recommended settings of a checkpoint. Not listed as a model in the audit report, as it loads nothing. |

An example is in `example_workflows/audit_example.json`.

### Example report (shortened)

The `.txt` is laid out for people: open points first, then the models and their licenses, how the image was generated, your own declarations, and the technical environment last. The `.json` holds the same content for machines.

```
AUDIT REPORT – Julia Kietzmann (Pertineo)
══════════════════════════════════════════════════════════════════════════════

  File              shot010_001.png (image 1 of 2)
  Saved             2026-10-23 16:24:40 (UTC+02:00)
  Image             6144 x 4096 · png (16 bit)
  Folder            <output>/job_x
  Author            Julia Kietzmann
  Company           Pertineo
  Purpose           Character insert, shot 010


TO CHECK ─────────────────────────────────────────────────────────────────────

  • 1 of 3 model(s) without source / license – add them to the model_info file.
  • Not filled in: Changes outside ComfyUI.


MODELS (3) ───────────────────────────────────────────────────────────────────

  1. Diffusion model
    File              diffusion_models/krea2.safetensors
    SHA256            9d5b02fa8990d1b776a54022fd672f6a5f2fdaa2ea151cca7a8a71e68b618b7d
    AutoV2            9D5B02FA89
    Source            https://huggingface.co/…
    License           …
    Commercial use    yes
    Loaded by         UNETLoader #17
  …


GENERATION ───────────────────────────────────────────────────────────────────

  KSampler #10
    Seed              0
    Steps             30
    CFG               1
    Sampler           res_multistep
    Scheduler         simple
    Denoise           1
    Latent            encoded source image  (VAEEncode #11)
    Positive prompt   …
    Negative prompt   …

  Source image      6144 x 4096
  Generated at      6144 x 4096  (encoded source image)
  Final image       6144 x 4096  (2 images)


DECLARED BY THE AUTHOR ───────────────────────────────────────────────────────

  Reference images          plate from client, delivered 2026-10-20
  External software         Nuke 16 (comp)
  Changes outside ComfyUI   – not provided –
  Notes                     – not provided –


SOFTWARE AND HARDWARE ────────────────────────────────────────────────────────

  ComfyUI           0.42.0 (git 1a82e6d)
  Frontend          1.57.1
  Python / torch    3.12.3 / 2.14.1+cu130 · CUDA 13.0
  GPU               NVIDIA GeForce RTX 5090 (31.4 GiB)
  OS                Linux Mint 22.3 · kernel 7.0.0-38-generic
  API nodes         none (everything ran locally)

  Node packages used:
    comfy-core  0.42.0
      CheckpointLoaderSimple, CLIPTextEncode ×2, KSampler, …
    rgthree-comfy  1.0.2610050029
      KSampler Config (rgthree)
```

### How values are found

- **Linked widgets** (e.g. steps and cfg from *KSampler Config (rgthree)*, a seed from a primitive or seed node, a value behind a boolean switch) are followed upstream to the constant. If a value is **computed** (text concatenation, math, an LLM, trigger words from LoRA Loader + Info …), the report shows the value that node actually produced **in this run**, taken from ComfyUI's output cache. For that, Collect Metadata has to run after those nodes – connect `output_image` (the final image), then it always runs last. If no run value is available, nothing is guessed – the report says `linked from <node type> #<id>, output <name>`.
- Only nodes that actually contribute to an output are listed. **Bypassed or muted nodes** are not part of the run and do not appear. Frontend-only nodes (KJNodes Set/Get, reroutes) are resolved by ComfyUI itself. Subgraph node ids (`12:5`) work.
- **Packages and versions** come from the node properties ComfyUI stores in the workflow (`cnr_id` / `ver`, or `aux_id` for git installs); otherwise from the folder the node is installed in.
- **Hashes** are cached in `<ComfyUI user dir>/pertineo_vfx/model_hash_cache.json` (key: path + size + modification time). The first run hashes large models once; later runs are instant. A changed or replaced file is hashed again.
- **Nothing stops the generation:** unreadable files, missing models or a broken `model_info` file become warnings in the report and the console.

### model_info – sources, licenses, trigger words

The nodes never look up or guess licenses. You keep a table with what you have verified, in the format you like:

| Format | Edit with | |
|---|---|---|
| **`model_info.ods`** | LibreOffice Calc | Recommended: keeps bold headers, colours, column widths. |
| `model_info.xlsx` | Excel / Calc | Same, for Excel users. |
| `model_info.csv` | any spreadsheet or text editor | Plain table, no formatting. |
| `model_info.json` | text editor | For scripts. `{"_disclaimer": "…", "models": {"<sha256 or file name>": {…}}}` |

- **Location:** leave `model_info_file` empty to use `<ComfyUI user dir>/pertineo_vfx/model_info.ods` (then `.xlsx`, `.csv`, `.json` – the first one found). **If none exists, an empty `model_info.csv` with just the header row is created there on the first run.** Open it in Calc and use *Save As* → `.ods` if you want formatting. You can also enter any path (relative paths are relative to that folder).
- **Templates** ship with the package: `model_info.example.ods` and `.csv` (empty, with headers and a hint row) and `model_info.example.json` (placeholder entries showing the JSON structure).
- **Layout:** the header row is found automatically (the first row with at least three known column names), so you can put a title, notes or a disclaimer above it. Column order is free; header names are matched loosely (`Base model`, `base_model`, `Trigger words`, `Filename`, `SHA256` …). Only the first sheet is read; formatting is ignored. Rows below the header count as a model only if they have a 64-character SHA256 or a file name with an extension – hint or example rows are skipped. Rows starting with `#` are ignored.
- **Disclaimer:** a row starting with `Disclaimer:` (text in the same cell or the next one) is your "no warranty" note.
- **Trigger words:** comma separated as on Civitai; use `;` only if a single trigger word contains a comma.
- **Spreadsheets:** import CSV columns as **Text** and turn off "detect scientific notation" – otherwise a hash like `12345E6789` may become a number.

Columns / fields per model – all optional, empty fields count as UNKNOWN:

| Field | Meaning | In the report |
|---|---|---|
| `file` | Path as in the report (`loras/SDXL/x.safetensors`). Also used to find the entry when hashing is off. | – |
| `type` | Model type, e.g. `checkpoint`, `lora`, `vae`, `text encoder`, `upscale model` – for your own sorting | JSON |
| `name` | Readable name | model heading |
| `sha256` | Identifies the model (CSV: first column; JSON: the key). Recommended. | – |
| `autov2` | Civitai short hash, to compare with the model page. A mismatch is listed under *To check*. | – |
| `version` | Model version, e.g. `v2.1` | yes |
| `released` | Release / upload date of that version | yes |
| `base_model` | e.g. `SDXL 1.0`, `Flux.1 dev`, `Krea2`. LoRAs / ControlNets whose base model differs from the main model of the run are listed under *To check*. | yes |
| `trigger_words` | List (or comma-separated text) of LoRA keywords – output by LoRA Loader + Info | yes |
| `recommended` | Free text, e.g. `strength 0.6–0.8` | JSON, LoRA info |
| `source` | Where you got it (URL) | yes |
| `license` | License as stated by the author | yes |
| `commercial_use` | `yes` / `no` / `restricted` / `unclear` | yes |
| `checked` | When you last checked the license | yes |
| `notes` | Anything else | yes |

- Identify models by **SHA256** (recommended – survives renaming; copy it from the first report) or by `file` (path relative to the model folder, or just the file name). If `sha256` / `autov2` is given and does not match the actual file, the report says so.
- **Disclaimer** (optional): your own "no warranty" note (JSON: `_disclaimer`). It is printed in every report that uses the file. Independently, every report ends with a fixed note that license data comes from the user's own file, is not verified by Pertineo VFX, comes without warranty and is not legal advice.
- Models without an entry are marked **`UNKNOWN – add to model_info file`** and listed under *To check*, so you see at once what is still missing.

### Sharing reports and images

- `include_full_paths` is **off** by default: reports contain only file names and sub folders relative to the model folder (`diffusion_models/3D/model.safetensors`), no local paths like `/mnt/…` or your home folder. Turn it on for purely internal records.
- `embed_workflow` writes the workflow into the PNG as the core Save Image node does. The workflow contains everything typed into any node – including local paths in other nodes' widgets. For anonymised sharing turn `embed_workflow` off and pass on the `.txt` / `.json`.
- The report documents what ComfyUI saw. Manual fields (reference images, external software, changes outside ComfyUI) are your statement – fill them in before delivering.

### Parameters

| Node | Parameter | Default | |
|---|---|---|---|
| Collect | `workflow_purpose`, `reference_image_sources`, `external_software`, `changes_outside_comfyui`, `notes` | empty | Manual audit fields. Empty = `– not provided –`. |
| Collect | `compute_model_hashes` | on | SHA256 + AutoV2 per model file (cached). |
| Collect | `model_info_file` | empty | Your license / source database, see above. |
| Collect | `include_full_paths` | off | Absolute paths in the report – only for internal use. |
| Collect | `source_image`, `output_image`, `fps` | – | Optional, for resolutions; `fps` > 0 adds frames and duration. |
| Collect | `author`, `company` | empty | Who is responsible for the image. Shown in the title (`AUDIT REPORT – Name (Company)`) and header. The footer always names the tool (`Report by Pertineo VFX`). |
| Collect | `include_author` | on | Off = author and company are left out of `.txt` and `.json` (they stay in the node). An embedded workflow still contains all widget values – also turn off `embed_workflow` if the name must not appear anywhere. |
| Save | `output_folder` | empty | Empty = ComfyUI output folder, relative = inside it, absolute allowed. |
| Save | `filename_prefix` | audit | Can be linked. A trailing image extension is removed; `sub/name` writes into a sub folder. |
| Save | `counter_digits` | 3 | `_001`. |
| Save | `format` | png (8 bit) | PNG 8 bit, PNG 16 bit, EXR 32-bit float (converted from sRGB to linear, no embedded workflow; needs OpenEXR). |
| Save | `embed_workflow` | on | PNG: prompt, workflow and audit JSON in the file. |
| Save | `write_txt` / `write_json` | on / off | Which sidecar files to write. |

Outputs of Save: `saved_basename` (name without extension, one per line for batches) and `saved_paths` (all written files).

Note: Collect Metadata runs on every queue (the graph and model files are invisible to ComfyUI's cache), so every run writes a new file pair.

---

## Development note

These nodes were developed with AI assistance (Claude by Anthropic). Concept, VFX requirements, testing in production pipelines and maintenance are by Julia Kietzmann.
