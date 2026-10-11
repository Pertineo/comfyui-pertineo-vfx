# Pertineo VFX for ComfyUI

Compositing-grade tools for ComfyUI – built by a compositor for real VFX and post-production pipelines: full resolution, image sequences, high bit depth, no shortcuts.

| Group | Nodes |
|---|---|
| **Matting** | ViTMatte Tiled Refiner · ViTMatte Tiled (Sequence) · Trimap from Matte |
| **I/O** | Load Image Sequence · Save Matte Sequence (16-bit PNG / 32-bit EXR) · Empty Audio · Frames for Model |

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