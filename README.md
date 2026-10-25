# Pertineo VFX for ComfyUI

Compositing-grade tools for ComfyUI – built by a compositor for real VFX and post-production pipelines: full resolution, image sequences, high bit depth, no shortcuts.

The nodes are **deliberately kept simple** – fewer moving parts, fewer ways to fail – so that they run stably and reliably in production. Where it matters, a correct and complete record takes priority over appearance.

| Group | Nodes |
|---|---|
| **Matting** | ViTMatte Tiled Refiner · ViTMatte Tiled (Sequence) · Trimap from Matte |
| **I/O** | Load Image Sequence · Save Matte Sequence (16-bit PNG / 32-bit EXR) · Empty Audio · Frames for Model |
| **Audit** | Collect Metadata · Production Context · Save Image + Metadata · LoRA Loader + Info · Model Info (LoRA, Checkpoint, Diffusion Model, VAE, Text Encoder, ControlNet, Upscaler) |

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

Nodes that record **what an image was made with** – for clients, license questions, studio GenAI reviews and your own traceability – and save image and report with the **same base name**.

```
… ──► VAE Decode ──┬──────────────────────────────► Save Image + Metadata ──► shot_001.png
                   └─► Collect Metadata ──text/json─┘                         shot_001.txt / .json / .pdf
Production Context ──┘
```

| Node | Purpose |
|---|---|
| **Pertineo Collect Metadata** | Reads the executed graph at run time (no widgets needed) and collects: environment (ComfyUI, frontend, Python, torch, CUDA, GPU, OS), every model file with **SHA256** and **AutoV2** (Civitai short hash), **source / license / commercial use** from your `model_info` table, custom node packages with versions, API nodes, sampler settings (seed, steps, cfg / guidance, sampler, scheduler, denoise, latent size), positive and negative prompts per sampler, and source / generation / final resolution (plus frames, fps and duration for video). In addition: input files with SHA256, bypassed / muted / unconnected nodes (listed as *not part of the run*), seed control mode, the settings of every executed node (credentials redacted), save nodes, workflow type, identity-related capabilities (face swap, identity adapters, voice / lip-sync …), reproducibility, and **findings with severity and the department they belong to**, an indicative **triage (A–D)** and five **guiding principles**. Adds five manual audit fields. Outputs `metadata_text` and `metadata_json`. |
| **Pertineo Production Context** | Declares the production context: production, project / episode, shot, purpose (ideation … final deliverable), final result, story-critical, execution environment, unreleased material, personal data, talent likeness / voice, third-party material, trademarks, union / guild-covered work, whether external tools store / train on the data, which approvals and consents are on file, and the reviewers. Connect to `production_context` of Collect Metadata. Everything here is recorded as **declared**, never as verified. Each setting is explained [below](#production-context--what-each-setting-means). |
| **Pertineo Save Image + Metadata** | Saves the images plus `.txt` and/or `.json` with **guaranteed identical base names**: the counter skips every number where an image, `.txt` or `.json` already exists, so `name_004.png` can never sit next to `name_003.txt`. Every image of a batch gets its own pair (the report states the batch index). Never overwrites. PNG 8 bit, PNG 16 bit or EXR 32-bit float. Embeds prompt and workflow like the core Save Image node. Optional **PDF** of the report (A4, with a thumbnail). Records the **SHA256 of the saved image** in the report, so a later workflow that loads it can be linked to this one. |
| **Pertineo LoRA Loader + Info** | Loads a LoRA like the core LoRA loader and additionally outputs what your `model_info` table says about it: `trigger_words` (comma-separated, ready to join into a prompt), `base_model` and a short `info` text with license and source. The LoRA is chosen only once, so loader and info always match. `clip` is optional (model-only LoRAs). A missing entry never stops the run. |
| **Pertineo Model Info (…)** | Look up your own records without loading anything: one node per model type (LoRA, Checkpoint, Diffusion Model, VAE, Text Encoder, ControlNet, Upscaler), each with the same file list as the matching loader. Outputs `name`, `base_model`, `trigger_words`, `recommended`, `license` and an `info` text – connect `info` to *Preview Any* to read it. Useful for LoRAs loaded by other loaders and for recommended settings of a checkpoint. Not listed as a model in the audit report, as it loads nothing. |

An example is in `example_workflows/audit_example.json`.

### Example report (shortened)

The `.txt` is laid out for people: the indicative triage and the open points first, then the declared production context, the models and their licenses, inputs, how the image was generated, outputs, inactive nodes, guiding principles, your own declarations, and the technical environment last. The `.json` holds the same content for machines (plus all node settings); the optional `.pdf` is the `.txt` on A4 pages.

```
AUDIT REPORT – Jane Doe (Example Studio)
══════════════════════════════════════════════════════════════════════════════

  File              shot010_001.png (image 1 of 2)
  Saved             2026-10-23 16:24:40 (UTC+02:00)
  Image             6144 x 4096 · png (16 bit)
  Folder            <output>/job_x
  SHA256            3f1c…(64 characters)…9a0e
  Author            Jane Doe
  Company           Example Studio
  Purpose           Character insert, shot 010
  Production        Example feature


INDICATIVE TRIAGE ────────────────────────────────────────────────────────────

  [ C ]  WRITTEN APPROVAL INDICATED
  Why               2 written-approval trigger(s)
  Open points       Production / studio approval 2 · Legal / rights 1
  Findings          2 high · 1 medium · 2 low
  Model licenses    5 commercial use yes · 1 covered by separate license · 0
                    restricted / unclear · 1 UNKNOWN / NOT VERIFIED · 0 conflicts
  Reproducibility   reproducible

  Indicative – not an approval and not legal advice. Scale and terms: see
  LEGEND at the end.


TO CHECK ─────────────────────────────────────────────────────────────────────

  • [HIGH] Generated material is intended for the final / published result –
    written approval is usually required.
      → Production / studio approval · declared
  • [MEDIUM] 1 of 3 model(s) without source / license in the model_info file:
    clip_l.safetensors
      → Legal / rights · external evidence required
  • [LOW] MiniMax H3: commercial use 'restricted' under the public license –
    covered by a separate license (agreement 2026-07). Keep the agreement on
    file.
      → Legal / rights · declared
  …


MODELS (3) ───────────────────────────────────────────────────────────────────

  1. Diffusion model – MiniMax H3
    File              diffusion_models/minimax_h3_t2i.safetensors
    SHA256            9d5b02fa8990d1b776a54022fd672f6a5f2fdaa2ea151cca7a8a71e68b618b7d
    AutoV2            9D5B02FA89
    Source            https://huggingface.co/…
    License           …
    Commercial use    restricted
    License checked   2026-10-24
    Separate license  yes – agreement 2026-07
    Family row        *minimax?h3*

  2. Text encoder
    File              text_encoders/clip_l.safetensors
    SHA256            7a3d14a7c65f670af67528e455e9e80011c10bc036d3d213c17375968aa78207
    License           UNKNOWN / NOT VERIFIED
  …


GENERATION ───────────────────────────────────────────────────────────────────

  Workflow type     image-to-image
  Seed control      fixed

  KSampler
    Seed              0
    Steps             30
    …


GUIDING PRINCIPLES (INDICATIVE) ──────────────────────────────────────────────

  P1  Output does not replicate unowned or copyrighted material
      → UNKNOWN / NOT VERIFIED – inputs not fully declared; the output itself
      needs visual review
  P2  Tools do not store, reuse or train on production data
      → YES – no API / cloud nodes in the run (observed)
  …


LEGEND ───────────────────────────────────────────────────────────────────────

  Grade – what has to happen before the material is used:
    A  LOW-RISK CANDIDATE – …
    B  REVIEW REQUIRED – …
  > C  WRITTEN APPROVAL INDICATED – …
    D  INDETERMINATE – …
    X  BLOCKED – …
  …
```

The report is meant to stay readable: node numbers, the full list of inactive nodes and all node settings are in the `.json`; the `.txt` / `.pdf` show what a reviewer needs. Scale, severity and evidence terms are explained once, in the **LEGEND** at the end of every report.

### Findings, triage and guiding principles

Every open point is a **finding** with a severity (`critical`, `high`, `medium`, `low`), the **department** it belongs to (Legal / rights, Talent & consent, Data protection / IT security, Production / studio approval, VFX supervision / QC, Documentation) and its evidence class: `observed` (seen in the graph), `derived` (concluded from it), `declared` (your statement in the model_info file or Production Context) or `external evidence required`.

Everything that is not known is shown as **`UNKNOWN / NOT VERIFIED`** – never as an empty field, a question mark or silently left out.

**Two scales, two questions.** The severity (`critical` > `high` > `medium` > `low`) rates **one finding**. The grade (A, B, C, D, X) answers **what has to happen before the material is used** – it is a next step, not just a risk level. That is why grade B can come from medium as well as from high findings.

The **indicative triage** sums this up – like a traffic light (the report prints this scale with the current grade marked):

| | | When |
|---|---|---|
| **A** | Low-risk candidate | Declared as ideation / temporary reference / R&D, not in the final result, no trigger, no open point of medium severity or higher. |
| **B** | Review required | Open points remain (unknown licenses, API nodes, undeclared answers, internal asset or marketing use …). |
| **C** | Written approval indicated | A typical trigger for written approval: final deliverable or final result, story-critical in the final result, unreleased material or personal data sent to a cloud service, talent likeness or voice, union / guild-covered work, third-party material without documented rights, external tools that train on the data, or a model license that is unknown for published material. The triage box lists every point that needs written approval and whether it is **covered** (e.g. by a written studio / client approval or consent on file – the findings then become *low*) or **OPEN**. When all are covered the grade reads **C – APPROVAL ON FILE**. Points the studio approval does not cover (talent without consent, trademark clearance before publication …) stay open. |
| **D** | Indeterminate | Production Context connected, but purpose and final result both unknown. |
| **X** | Blocked | A **critical** finding: the use as declared is not permitted – e.g. a model whose license does not allow commercial use, without a separate license, in published material. An approval cannot fix this; only a license from the rights holder or a change (e.g. another model) can. Critical always means X. |

**Model licenses count in the grade.** How much depends on whether the result is published (final result `yes`, purpose `final deliverable` or `marketing / publicity`):

| Model license | not published (moodboard, R&D, internal) | published |
|---|---|---|
| commercial use `no`, no separate license | high → B (note: commercial client work may count as commercial use even internally) | **critical → X** (license conflict) |
| license or commercial use `UNKNOWN / NOT VERIFIED` | medium → B | high, written check indicated → **C** |
| `restricted` / `unclear` | medium → B | high → B (check revenue limits, territories …) |
| `no` / `restricted` + separate license `yes` | low | low (keep the agreement on file) |
| `yes` | – | – |

The triage box also shows a one-line summary: `Model licenses  5 commercial use yes · 1 covered by separate license · 0 restricted / unclear · 1 UNKNOWN / NOT VERIFIED · 0 conflicts`.

Below it, **Commercial use** answers one question from the model licenses alone – would these models allow commercial use of the result? `OK` (all allow it, separate licenses included), `CHECK TERMS` (restricted / unclear, e.g. revenue limits or territories), `NOT PERMITTED` (a model says no and no separate license is recorded) or `UNKNOWN / NOT VERIFIED`. The worst case wins. This line is shown with and without a Production Context; it is not a grade.

**Without a Production Context** there is no triage: the report header says `Triage: not assessed – no Production Context connected`, and the triage, production context and guiding principles sections are left out. All other findings (licenses, API nodes, credentials, identity capabilities …) are listed as usual.

The five **guiding principles** (P1 output does not replicate unowned material, P2 tools do not store / train on production data, P3 secured environment, P4 temporary material, P5 no talent or union work replaced without consent) are answered `YES` / `NO` / `UNKNOWN / NOT VERIFIED` with the reason. `YES` needs evidence or an explicit declaration – never just the absence of a contrary finding.

**Triage and principles are an orientation for the people who decide** – derived only from your declarations and the graph. They are not an approval, not legal advice and not an official studio or vendor assessment. The report says so in every file.

### Production Context – what each setting means

Answer for **this workflow**. Everything is your declaration; leave `unknown` rather than guess – the report then shows `UNKNOWN / NOT VERIFIED`. "Final result" means the finished, published work: a film or episode, but equally an ad, packaging or print.

| Setting | Meaning | Effect on the assessment |
|---|---|---|
| `production_title`, `project_episode`, `shot_asset`, `vendor_department`, `audit_requester` | Who and what the work is for. | Shown in the report header; no effect on the triage. |
| `purpose` | What the output of this workflow is for. | `ideation / moodboard`, `temporary reference`, `research and development` are low risk (A possible). `internal production asset`: intermediate step – with final result `yes` a note to audit the final step as well. `marketing / publicity`: finding for public use. `final deliverable`: written-approval trigger (C). `unknown`: finding; with final result also unknown → D. |
| `in_final_cut` | Whether generated material – or anything derived from it – appears in the final result. | `yes` → written-approval trigger (C), P4 NO. `no` with a low-risk purpose → P4 YES. |
| `story_critical` | Generated elements carry the story or are prominent (key character, hero product, main set). | Only counts together with final result: `yes` + `yes` → C; `yes` + `unknown` → finding (B); `yes` + `no` → no finding (a moodboard for a key scene is not a risk). |
| `execution_environment` | Where the workflow ran. `auto` = local workstation, or `mixed` when API / cloud nodes are in the run. | Local workstation, private studio server, enterprise-secured cloud → P3 YES. Public / consumer cloud → finding, P3 NO. |
| `unreleased_material` | Unpublished footage, scripts or designs of the production in the inputs. | Locally (no API / cloud) → low finding "check that the client contract allows AI processing". Sent to a cloud service → C. |
| `personal_data` | Images, voices or data of identifiable people in the inputs. | As above; locally with the note that data-protection law still applies. |
| `talent_likeness_or_voice` | Face, body, performance or voice of a real performer used or changed. | `yes` → C (with consent on file: noted, still C). Also decides how identity-related nodes (face swap, lip-sync …) are rated. P5. |
| `third_party_copyright` | Material you do not own in the inputs (stock, stills, artwork by others). | `yes` without rights documentation or legal clearance → C, P1 NO. `unknown` → finding. |
| `trademarks_brands` | Third-party logos, brand names, products or packaging – in the inputs or likely in the output. | `yes` / `unknown` without legal clearance → legal finding; for published material it is a written-approval point (C, "trademark / brand clearance"); clearance on file → low. P1. |
| `union_covered_work` | Performance or work under a union or guild agreement (e.g. SAG-AFTRA actors, WGA writers). | `yes` → C; without union approval P5 NO. |
| `external_tools_train` | Whether an external service may store, reuse or train on inputs or outputs. | `yes` → C, P2 NO. `unknown` with API nodes in the run → finding. |
| `input_rights_documented`, `talent_consent`, `union_approval`, `studio_approval`, `legal_approval` | Is a **document** on file? `yes` only if you can show it (contract clause, e-mail, signed form); a verbal OK is `no` or `pending`. A job order alone ("design the label") is not a GenAI approval. | Soften the related findings: rights documentation / legal clearance for third-party material and trademarks, consent for talent, studio approval for the written-approval triggers (they become low, grade "C – approval on file"). `pending` keeps the finding open. |
| `creative_reviewer`, `technical_reviewer`, `rights_reviewer` | Who checked the result (name or role). | Empty creative / technical reviewer → low finding. |
| `context_notes` | Anything else – e.g. what is pending and with whom. | Printed in the report. |

The same explanations are in the tooltips of the node.

### Credits

Every report lists the credits of the run: each model with creator and license, each node package with publisher, license and repository, and the attribution your licenses require (column *Attribution* in the model_info table).

- **Node packages:** read from the installed package itself (`pyproject.toml`, LICENSE file, git remote). Nothing to maintain.
- **Models:** your model_info table is preferred – enter a creator there only when you want to state it explicitly. If the table has none, the creator stated in the model file is used (safetensors `modelspec.author`, GGUF `general.author` / `general.organization`). The report names the source: `(model_info)` or `(from the file)`.
- A license stated inside a model file is shown as *License in file … (not verified)*. It never replaces your table and does not change the grade – only your table counts as a checked statement.
- Node package licenses concern the code, not your images, so they are listed for transparency and do not affect the grade.

### Multi-step work

Audit **every step**. An intermediate step (e.g. a layout or an asset) is `internal production asset`; the step that makes the finished result is `final deliverable`. Save Image + Metadata writes the SHA256 of each saved image into its report. When the next workflow loads that image, the same SHA256 appears there under *Input files* – the two reports are linked without any extra bookkeeping.

**Steps and iterations are two different things.** `step` on Collect Metadata names the position in the pipeline (`01 – layout`, `02 – character insert`, `03 – upscale`) and stays the same for every iteration of that step. Each run of a step – a re-run, a variant, a take – is an iteration of its own: it is told apart by file name, time and the SHA256 of the saved image. Remarks on one iteration ("v03 – warmer light") go into `notes`. Which iteration went on is visible from the hashes: only its output SHA256 appears as an input of the next step. `step_description` is the breakdown line of the step.

### What the assessment is based on

- **The workflow itself** (observed): executed nodes, models and their hashes, input files, settings, API nodes.
- **Your declarations** (declared): the model_info table and the Production Context. They are recorded as stated – Pertineo VFX does not check them.
- **The rule set** – which answers raise which finding, grade and principle. It follows publicly documented industry practice:
  - the five guiding principles and the written-approval cases in Netflix's partner guidance [*Using Generative AI in Content Production*](https://partnerhelp.netflixstudios.com/hc/en-us/articles/43393929218323-Using-Generative-AI-in-Content-Production) (published August 2025),
  - the consent requirements for digital replicas and AI in the 2023 SAG-AFTRA TV / Theatrical agreement ([SAG-AFTRA: *Digital Replicas 101*](https://sagaftra.org/files/sa_documents/DigitalReplicas.pdf)),
  - general data-protection principles (data that leaves the machine vs. data that stays local),
  - and practical calibration in a VFX pipeline (e.g. moodboards and local processing are not treated like final deliverables).

**Rules as of 2026-10-25** (the date is also printed in every report). Studio guidelines, union agreements and laws change; the rule set does not update itself. It is an orientation, not a complete legal checklist and not legal advice – the governing documents of each production always prevail.

### How values are found

- **Linked widgets** (e.g. steps and cfg from *KSampler Config (rgthree)*, a seed from a primitive or seed node, a value behind a boolean switch) are followed upstream to the constant. If a value is **computed** (text concatenation, math, an LLM, trigger words from LoRA Loader + Info …), the report shows the value that node actually produced **in this run**, taken from ComfyUI's output cache. For that, Collect Metadata has to run after those nodes – connect `output_image` (the final image), then it always runs last. If no run value is available, nothing is guessed – the report says `linked from <node type> #<id>, output <name>`.
- Only nodes that actually contribute to an output count as part of the run. **Bypassed, muted and unconnected nodes** are summarised under *Inactive nodes – not part of the run* (counts, plus bypassed loaders with their model file); the full list is in the `.json`.
- **Input files** (images, videos, audio, sequence folders) are listed with SHA256 and **how they are used** – files in the ComfyUI input folder as `<input>/…`. Every loader in the workflow counts, also one that is not connected to an output (*not used*). The use is found by following the graph downstream from the loader (through resize, crop … nodes):

  | Use | Meaning |
  |---|---|
  | `source` | Its pixels go into the result – img2img, inpaint, edit, video input, comp, matte plate. |
  | `guidance` | Steers structure or area – ControlNet image, pose, depth, mask. Its pixels are not in the result. |
  | `reference` | Steers style or likeness – IP-Adapter, CLIP vision, style model, face ID. |
  | `prompt reference` | Read by a prompt writer (LLM / VLM) – it shapes the prompt, it is not in the image. |
  | `preview only` | Only shown in a preview node. |
  | `not used` | Loaded, but not connected to an output – not part of the run. |

  The rights to *every* input that is used matter, also to a prompt reference – declare them in `reference_image_sources`.
- **Prompts written by an LLM / VLM** in the run (e.g. a prompt generator that describes a layout image) are recorded as produced and marked with the node that wrote them. The section **Prompt generation** lists what each writer was given: images (with the number of the input file), instructions and presets, linked texts with the node they came from, settings such as seed and temperature, its model – and, for a chain of writers (e.g. a VLM that describes the image, then a prompt node that rewrites it), each step in order. Running the workflow again writes a new prompt, so the report says *reproducible only with the recorded values* – the same applies to a seed that is not fixed. A model that only feeds such a generator is listed as *VLM – used for: writes the prompt*.
- **Workflow groups** are recorded in the `.json` only (`workflow_outline`): the groups in reading order with the nodes that ran in each – raw material for a breakdown across steps, not part of the printed report.
- **Credentials:** inputs that look like an API key or token are replaced by `REDACTED` in the report and raise a high finding – they are still in the embedded workflow, remove them before sharing. Frontend-only nodes (KJNodes Set/Get, reroutes) are resolved by ComfyUI itself. Subgraph node ids (`12:5`) work.
- **Packages and versions:** the version that is **installed and ran** (pyproject of the package folder; git checkouts add the commit). The package name comes from the node properties in the workflow (`cnr_id` / `aux_id`); if the version stored there differs, the report adds "workflow saved with …".
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
- **Layout:** the header row is found automatically (the first row with at least three known column names), so you can put a title, notes or a disclaimer above it. Column order is free; header names are matched loosely (`Base model`, `base_model`, `Trigger words`, `Filename`, `SHA256` …). Only the first sheet is read; formatting is ignored. Rows below the header count as a model only if they have a 64-character SHA256, a file name with an extension or a file pattern (family row, see below) – hint or example rows are skipped. Rows starting with `#` are ignored.
- **Disclaimer:** a row starting with `Disclaimer:` (text in the same cell or the next one) is your "no warranty" note.
- **Trigger words:** comma separated as on Civitai; use `;` only if a single trigger word contains a comma.
- **Spreadsheets:** import CSV columns as **Text** and turn off "detect scientific notation" – otherwise a hash like `12345E6789` may become a number.

Columns / fields per model – all optional, empty fields count as `UNKNOWN / NOT VERIFIED`:

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
| `separate_license` | `yes` / `no` – you have your own agreement with the author (e.g. a commercial license or written approval). With `yes`, a model whose public license says commercial use `no` / `restricted` is shown as *covered by a separate license* instead of an open point. | yes |
| `license_reference` | Where that agreement is documented: contract or approval number, date | yes |
| `creator` | Author / publisher of the model | yes |
| `attribution` | Credit the license requires, if any – listed as a finding for Production | yes |
| `training_data` | Training data as stated by the author | yes |
| `notes` | Anything else | yes |

- Identify models by **SHA256** (recommended – survives renaming; copy it from the first report) or by `file` (path relative to the model folder, or just the file name). If `sha256` / `autov2` is given and does not match the actual file, the report says so.
- **Disclaimer** (optional): your own "no warranty" note (JSON: `_disclaimer`). It is printed in every report that uses the file. Independently, every report ends with a fixed note that license data comes from the user's own file, is not verified by Pertineo VFX, comes without warranty and is not legal advice.
- **Family rows:** one row can cover a whole model family. Put a pattern with `*` (any text) or `?` (one character) in `file`, e.g. `*minimax_h3*` – it matches every model file whose name or path contains `minimax_h3` (case-insensitive). A row for the exact file (SHA256 / file name) always wins; only the cells it leaves empty are taken from the family row. Per-file fields (name, version, released, hashes, trigger words, recommended) are never inherited. Typical use: license, source and separate license once in the family row, base model and trigger words per file. The report names the family row that was used.
- Models without an entry are marked **`UNKNOWN / NOT VERIFIED`** and listed under *To check*, so you see at once what is still missing.

### Sharing reports and images

- `include_full_paths` is **off** by default: reports contain only file names and sub folders relative to the model folder (`diffusion_models/3D/model.safetensors`), no local paths like `/mnt/…` or your home folder. Turn it on for purely internal records.
- `embed_workflow` writes the workflow into the PNG as the core Save Image node does. The workflow contains everything typed into any node – including local paths in other nodes' widgets. For anonymised sharing turn `embed_workflow` off and pass on the `.txt` / `.json`.
- The report documents what ComfyUI saw. Manual fields (reference images, external software, changes outside ComfyUI) and the Production Context are your statement – fill them in before delivering.

### Parameters

| Node | Parameter | Default | |
|---|---|---|---|
| Collect | `workflow_purpose`, `reference_image_sources`, `external_software`, `changes_outside_comfyui`, `notes` | empty | Manual audit fields. `reference_image_sources` = origin and **rights** of the inputs (not the file name – input files are listed automatically with their SHA256); `external_software` = which programs before / after ComfyUI; `changes_outside_comfyui` = **planned** work on the result after ComfyUI (e.g. comp in Nuke) – work already done belongs to its own record, an edited image that goes back into a workflow appears there as an input with its hash. Write `none` if there is nothing to declare; empty = `UNKNOWN / NOT VERIFIED – not provided` (`notes` and `changes_outside_comfyui` are optional – empty stays empty). |
| Collect | `compute_model_hashes` | on | SHA256 + AutoV2 per model file (cached). |
| Collect | `model_info_file` | empty | Your license / source database, see above. |
| Collect | `include_full_paths` | off | Absolute paths in the report – only for internal use. |
| Collect | `source_image`, `output_image`, `fps` | – | Optional, for resolutions; `fps` > 0 adds frames and duration. |
| Collect | `author`, `company` | empty | Who is responsible for the image. Shown in the title (`AUDIT REPORT – Name (Company)`) and header. The footer always names the tool (`Report by Pertineo VFX`). |
| Collect | `production_context` | – | Optional, from *Pertineo Production Context*. Without it the triage is D (indeterminate). |
| Collect | `step`, `step_description` | empty | Optional: which step of the pipeline this workflow is and what it does – the same for every iteration of the step (see *Multi-step work*). Shown in the report header. |
| Collect | `include_author` | on | Off = author and company are left out of `.txt` and `.json` (they stay in the node). An embedded workflow still contains all widget values – also turn off `embed_workflow` if the name must not appear anywhere. |
| Save | `output_folder` | empty | Empty = ComfyUI output folder, relative = inside it, absolute allowed. |
| Save | `filename_prefix` | audit | Can be linked. A trailing image extension is removed; `sub/name` writes into a sub folder. |
| Save | `counter_digits` | 3 | `_001`. |
| Save | `format` | png (8 bit) | PNG 8 bit, PNG 16 bit, EXR 32-bit float (converted from sRGB to linear, no embedded workflow; needs OpenEXR). |
| Save | `embed_workflow` | on | PNG: prompt, workflow and audit JSON in the file. |
| Save | `write_txt` / `write_json` | on / off | Which sidecar files to write. |
| Save | `write_pdf` | off | Also write the report as `.pdf` (A4, thumbnail of the image on page 1, no extra dependency). The PDF uses a standard font that only covers Western European characters: Chinese, Japanese, Cyrillic or emoji (e.g. in a prompt) become `?` and the PDF says so at the top – the `.txt` and `.json` keep the complete text. |
| Save | – | – | The report always contains the **SHA256 of the saved image** (not inside the PNG itself – a file cannot contain its own hash). |

Outputs of Save: `saved_basename` (name without extension, one per line for batches) and `saved_paths` (all written files).

Note: Collect Metadata runs on every queue (the graph and model files are invisible to ComfyUI's cache), so every run writes a new set of files.

---

## Development note

These nodes were developed with AI assistance (Claude by Anthropic). Concept, VFX requirements, testing in production pipelines and maintenance are by Julia Kietzmann.
