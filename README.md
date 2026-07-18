# Multi-Scale Turing Pattern Generator (AI GENERATED)

Python implementation of Jonathan McCabe's multi-scale Turing pattern algorithm, as described by [Jason Rampe](https://softologyblog.wordpress.com/2011/07/05/multi-scale-turing-patterns/).

## Quick Start

```bash
pip install -r requirements.txt
python generate.py
```

Outputs `output.png` (or whatever `output_path` is set to in `config.json`).

## Presets

Ready-to-run configs for distinct visual forms. All use `device: "cuda"` — change to `"cpu"` if needed.

| File | Command | Visual style |
|---|---|---|
| `config.json` | `python generate.py` | Baseline organic pattern (symmetry disabled by default) |
| `config.json` | `python generate.py --symmetrical` | Geometric mandala (8-fold + mirror) |
| `config-fractal.json` | `python generate.py --config config-fractal.json` | Fractal hierarchy, 7 scales, 2× ratio each |
| `config-cells.json` | `python generate.py --config config-cells.json` | Organic blobs / cell membranes |
| `config-maze.json` | `python generate.py --symmetrical --config config-maze.json` | Stripe labyrinth, 4-fold symmetric |
| `config-snowflake.json` | `python generate.py --symmetrical --config config-snowflake.json` | 6-fold + mirror, icy tones |
| `config-halos.json` | `python generate.py --config config-halos.json` | Ghosting / halos via negative weight |

**Key differences at a glance:**

| Preset | Scales | Symmetry | Blur | What drives the look |
|---|---|---|---|---|
| mandala | 5 | 8+mirror | gaussian | rotational enforcement |
| fractal | 7 (2× ratio) | none | gaussian | recursive self-similarity |
| cells | 3 (4× ratio) | none | gaussian | large inh/act radius gap |
| maze | 3 (1.5× ratio) | 4+mirror | box | tight ratio → stripes |
| snowflake | 4 (2× ratio) | 6+mirror | gaussian | hex symmetry + fine scales |
| halos | 4 (one negative weight) | none | gaussian | inverted scale contribution |

## How It Works

A 2D grid of float values is repeatedly updated using multiple "scales". Each scale defines a local (activator) and broader (inhibitor) neighborhood. Every cell each step is updated by whichever scale has the **smallest discrepancy** between its two neighborhood averages — this produces self-similar patterns at multiple spatial frequencies simultaneously. After each step the grid is renormalized to `[-1, +1]`.

## CLI

```
python generate.py [options]

Options:
  --config PATH        Config file to use (default: config.json)
  --output PATH        Override output_path from config
  --iterations N       Override iterations from config
  --seed N             Override seed from config
  --grayscale          Force grayscale output
  --save-frames        Save each iteration as a PNG frame
  --frames-dir PATH    Override frames_dir from config
  --symmetrical        Enable symmetry feature (disabled by default)
  --symmetry N         N-fold rotational symmetry (used only with --symmetrical)
  --mirror             Add mirror symmetry (used only with --symmetrical)
  --device cpu|cuda    Compute device (default: cpu)
```

## config.json Reference

### Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `width` | int | 800 | Output image width in pixels |
| `height` | int | 800 | Output image height in pixels |
| `iterations` | int | 300 | Number of simulation steps |
| `seed` | int or null | 42 | Random seed for reproducibility. `null` = random each run |
| `blur_method` | string | `"box"` | `"box"` (faster) or `"gaussian"` (smoother) |
| `color_mode` | bool | true | `true` = RGB output, `false` = grayscale |
| `color_lerp_alpha` | float | 0.01 | Color blend speed per step. Lower = slower color transitions |
| `output_path` | string | `"output.png"` | Final image output path |
| `save_frames` | bool | false | Save a PNG for every iteration (for video assembly) |
| `frames_dir` | string | `"frames"` | Directory to write per-step frames into |
| `symmetry` | int | 1 | N-fold rotational symmetry. `1` = off. `6`/`8`/`12` for mandalas |
| `mirror` | bool | false | Add mirror symmetry on top of rotational (doubles total folds) |
| `device` | string | `"cpu"` | `"cpu"` or `"cuda"` (requires CuPy) |
| `scales` | array | — | List of scale definitions (see below) |

### Scale definition

Each entry in `scales` controls one spatial frequency of the pattern.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `activator_radius` | int | yes | Radius of the smaller (local) blur neighborhood |
| `inhibitor_radius` | int | yes | Radius of the larger (broad) blur neighborhood. Should be > `activator_radius` |
| `small_amount` | float | yes | Increment magnitude applied each step (typically 0.01–0.05) |
| `weight` | float | no (1.0) | Multiplier applied to both blurred averages for this scale |
| `color` | [R, G, B] | no ([255,255,255]) | RGB color assigned to this scale in color mode |

### Tuning tips

- **More scales** → richer structure. 4–6 scales is typical.
- **Larger radii** → larger pattern features. Scale radii in proportion (e.g. 2× between scales).
- **`small_amount`** → larger values = more contrast but can produce harsh transitions.
- **`color_lerp_alpha`** → 0.005–0.02 for smooth color blending; higher values make colors react faster.
- **`blur_method: "gaussian"`** → softer, more organic edges; slower than `"box"`.
- **`iterations`** → patterns mature over time. 200–500 for a fully developed result.

### Recommended scale sets

**5-scale (Softology reference, good general purpose):**
```json
[
  {"activator_radius": 100, "inhibitor_radius": 200, "small_amount": 0.05},
  {"activator_radius": 20,  "inhibitor_radius": 40,  "small_amount": 0.04},
  {"activator_radius": 10,  "inhibitor_radius": 20,  "small_amount": 0.03},
  {"activator_radius": 5,   "inhibitor_radius": 10,  "small_amount": 0.02},
  {"activator_radius": 1,   "inhibitor_radius": 2,   "small_amount": 0.01}
]
```

**3-scale (fine detail focus):**
```json
[
  {"activator_radius": 10, "inhibitor_radius": 20, "small_amount": 0.03},
  {"activator_radius": 5,  "inhibitor_radius": 10, "small_amount": 0.02},
  {"activator_radius": 1,  "inhibitor_radius": 2,  "small_amount": 0.01}
]
```

## Making a Video from Frames

```bash
# Set "save_frames": true in config.json, then run:
python generate.py

# Assemble with ffmpeg (30 fps)
ffmpeg -r 30 -i frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p timelapse.mp4
```

Frame filenames are zero-padded to `len(str(iterations))` digits. Match `%0Nd` in the ffmpeg command:

| `iterations` | digits | ffmpeg pattern |
|---|---|---|
| 1–9 | 1 | `-i frames/frame_%01d.png` |
| 10–99 | 2 | `-i frames/frame_%02d.png` |
| 100–999 | 3 | `-i frames/frame_%03d.png` ← default (300 iters) |
| 1000–9999 | 4 | `-i frames/frame_%04d.png` |

**Quality presets:**
```bash
# Standard (good size/quality balance — default)
ffmpeg -r 30 -i frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p timelapse.mp4

# High quality (larger file, lower CRF = better)
ffmpeg -r 30 -i frames/frame_%03d.png -c:v libx264 -crf 18 -pix_fmt yuv420p timelapse.mp4

# Lossless
ffmpeg -r 30 -i frames/frame_%03d.png -c:v libx264 -crf 0 -pix_fmt yuv420p timelapse.mp4

# ProRes (for editing software)
ffmpeg -r 30 -i frames/frame_%03d.png -c:v prores_ks -pix_fmt yuva444p10le timelapse.mov
```

## GPU Acceleration (CuPy / CUDA)

All heavy computation (blurs, rotations, array ops) runs on the GPU when `device: "cuda"` is set. The grid lives on the GPU throughout simulation; only final image save transfers data back to CPU.

**Install CuPy for your CUDA version:**
```bash
# CUDA 13.x (RTX 3090 + CUDA 13.1)
pip install cupy-cuda13x

# CUDA 12.x
pip install cupy-cuda12x

# Verify
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) found')"
```

**Enable in config.json:**
```json
"device": "cuda"
```

**Or via CLI:**
```bash
python generate.py --device cuda
```

Falls back gracefully: if CuPy is not installed and `device` is omitted, runs on CPU automatically. If `device: "cuda"` is set and CuPy is missing, the script exits with an install hint.

## Mandala / Symmetric Patterns

Symmetry feature is opt-in. Add `--symmetrical` to activate it. Without `--symmetrical`, symmetry is off even if config has `symmetry` / `mirror` values.

With `--symmetrical` enabled, set `symmetry` to an integer ≥ 2 to enforce N-fold rotational symmetry around image centre after each step. Add `"mirror": true` for reflective symmetry on top (doubles total folds).

```json
"symmetry": 8,
"mirror": true
```

In `config.json`, `symmetry: 8` + `mirror: true` gives 16 symmetric copies averaged each step, but only when you run with `--symmetrical`.

| `symmetry` | `mirror` | total folds | looks like |
|---|---|---|---|
| 6 | false | 6 | snowflake / honeycomb |
| 6 | true | 12 | hex mandala |
| 8 | true | 16 | geometric mandala ← default |
| 12 | true | 24 | dense radial |
| 1 | — | 1 | no symmetry (organic) |

**CLI:**
```bash
python generate.py --symmetrical --symmetry 8 --mirror
python generate.py --symmetrical --config config-snowflake.json
python generate.py                        # symmetry off
```

> **Performance note:** each step runs `symmetry` (or `2×symmetry` with mirror) additional rotation operations. Symmetry 8+mirror on an 800×800 grid adds ~0.3 s/step. Reduce image size or use `blur_method: "box"` to speed up.

## References

- Jonathan McCabe — *Cyclic Symmetric Multi-Scale Turing Patterns*
- Jason Rampe — [Softology Blog: Multi-Scale Turing Patterns](https://softologyblog.wordpress.com/2011/07/05/multi-scale-turing-patterns/)
- Ricky Reusser — [Gallery](https://rreusser.github.io/notebooks/multi-scale-turing-pattern-gallery-1/)
