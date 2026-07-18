# AGENTS.md — Multi-Scale Turing Pattern Generator

Codebase context for LLM agents. Read this before editing.

## File Map

```
generate.py       — single-file Python script, all logic here
config.json       — runtime config (do not hardcode values from here in code)
requirements.txt  — numpy, scipy, Pillow, tqdm
README.md         — human docs
```

## generate.py Structure

| Symbol | Type | Purpose |
|--------|------|---------|
| `Scale` | dataclass | Holds one scale's params: `activator_radius`, `inhibitor_radius`, `small_amount`, `weight`, `color` |
| `_get_backend(device)` | fn | Returns `(xp, xp_ndimage)` — either `(cupy, cupyx.scipy.ndimage)` or `(numpy, scipy.ndimage)`. Called once at start of `run_simulation`. |
| `_blur(grid, radius, method, xp, xp_ndimage)` | fn | Wrapping blur via ndimage. `method="box"` → `uniform_filter`; `method="gaussian"` → `gaussian_filter`. `mode='wrap'` always. |
| `_enforce_symmetry(grid, n, mirror, xp, xp_ndimage)` | fn | Averages grid with N rotated copies (+ flipped copies if `mirror=True`). Uses `xp_ndimage.rotate`, `order=1`, `mode='nearest'`. |
| `run_simulation(...)` | fn | Main loop. Returns `(grid, color_grid)` as **numpy** arrays (transfers from GPU if needed). |
| `save_image(grid, color_grid, path, color_mode, quiet)` | fn | Converts arrays to PIL Image, saves PNG. `quiet=True` suppresses print. |
| `load_config(path)` | fn | JSON load, returns dict. |
| `build_scales(raw)` | fn | Validates + constructs `List[Scale]` from config list. Clamps colors to [0,255]. |
| `main()` | fn | argparse entry point. |

## Algorithm (McCabe/Rampe)

```
grid = uniform_random(H, W, [-1, +1])

for step in range(iterations):
    for each scale s:
        act[s] = blur(grid, s.activator_radius) * s.weight
        inh[s] = blur(grid, s.inhibitor_radius) * s.weight
        var[s] = |act[s] - inh[s]|

    best = argmin(var, axis=scales)          # per-cell winning scale

    increment[cell] = +s.small_amount  if act[best] > inh[best]
                      -s.small_amount  otherwise

    grid += increment
    grid = normalize(grid, [-1, +1])         # global min-max rescale

    if color_mode:
        color_grid += mask(best==s) * alpha * (scale_color[s] - color_grid)

    if symmetry > 1:
        grid = _enforce_symmetry(grid, symmetry, mirror)
        grid = normalize(grid, [-1, +1])
        if color_mode:
            color_grid[:,:,ch] = _enforce_symmetry(color_grid[:,:,ch], symmetry, mirror)  # per channel
            color_grid = clip(color_grid, 0, 1)

    if save_frames:
        save_image(grid, color_grid, frames_dir/frame_{step}.png, quiet=True)

return grid, color_grid
```

## config.json Keys

| Key | Type | Notes |
|-----|------||-------|
| `width`, `height` | int | Image dimensions |
| `iterations` | int | Simulation steps |
| `seed` | int or null | null = non-deterministic |
| `blur_method` | `"box"` or `"gaussian"` | box is faster |
| `color_mode` | bool | true = RGB, false = grayscale |
| `color_lerp_alpha` | float 0–1 | Color blend rate per step |
| `output_path` | string | Final image path |
| `device` | `"cpu"` or `"cuda"` | cuda requires CuPy installed |
| `save_frames` | bool | Write frame per step to `frames_dir` |
| `frames_dir` | string | Directory for frame PNGs |
| `symmetry` | int | N-fold rotational symmetry. 1 = off. 6/8/12 for mandalas |
| `mirror` | bool | Add mirror symmetry (doubles total folds) |
| `scales` | array | See scale schema below |

### Scale schema
```json
{
  "activator_radius": int,    // required — local blur radius (smaller)
  "inhibitor_radius": int,    // required — broad blur radius (larger)
  "small_amount": float,      // required — step increment magnitude
  "weight": float,            // optional, default 1.0
  "color": [R, G, B]          // optional, default [255,255,255]
}
```

## CLI Args

```
--config PATH       config file (default: config.json)
--output PATH       overrides output_path
--iterations N      overrides iterations
--seed N            overrides seed
--grayscale         forces color_mode=False
--save-frames       sets save_frames=True
--frames-dir PATH   overrides frames_dir
--symmetry N        overrides symmetry
--mirror            sets mirror=True
--device cpu|cuda   overrides device
```

## Key Invariants

- Grid always in `[-1, +1]` after each step (enforced by min-max normalization).
- `color_grid` is `None` when `color_mode=False`. Check before using.
- `_blur` returns `grid.copy()` unchanged when `radius <= 0`.
- Frame filenames are zero-padded to `len(str(iterations))` digits — ffmpeg-safe ordering.
- Edge handling is always `mode='wrap'` (seamless tiling).
- `build_scales` clamps RGB values to [0, 255] — never trust raw config values directly.

## Dependencies

```
numpy>=1.24        — grid operations
scipy>=1.11        — uniform_filter, gaussian_filter
Pillow>=10.0       — image saving
tqdm>=4.65         — progress bar
```

## Common Edit Patterns

**Add a new config key with a default:**
1. Add to `config.json` with a sensible default.
2. Read in `main()` with `cfg.get("key", default)`.
3. Pass as argument if needed by `run_simulation` or `save_image`.

**Add a new blur method:**
1. Add a branch in `_blur()`.
2. Update the validation check in `main()` that warns on unknown `blur_method`.

**Change output format (e.g. add TIFF support):**
- Only `save_image()` needs changing. PIL handles format from file extension.

**Add per-scale behavior (e.g. symmetry):**
- Add field to `Scale` dataclass.
- Read in `build_scales()`.
- Use in `run_simulation()` inner loop.
