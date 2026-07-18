"""
Multi-Scale Turing Pattern Generator
Based on: Jonathan McCabe's algorithm, described by Jason Rampe
  https://softologyblog.wordpress.com/2011/07/05/multi-scale-turing-patterns/
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import scipy.ndimage as _scipy_ndimage
from scipy.ndimage import gaussian_filter, rotate as ndimage_rotate, uniform_filter
from tqdm import trange

# ---------------------------------------------------------------------------
# GPU / CPU backend selection
# ---------------------------------------------------------------------------

try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _cp_ndimage
    _CUPY_AVAILABLE = True
except Exception:
    _cp = None  # type: ignore
    _cp_ndimage = None  # type: ignore
    _CUPY_AVAILABLE = False


def _get_backend(device: str):
    """Return (xp, xp_ndimage) array-module pair for the requested device.

    xp      — array module (cupy or numpy)
    xp_ndimage — ndimage module (cupyx.scipy.ndimage or scipy.ndimage)
    """
    if device == "cuda":
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy not found. Install with: pip install cupy-cuda13x\n"
                "Then verify: python -c 'import cupy; cupy.cuda.runtime.getDeviceCount()'"
            )
        return _cp, _cp_ndimage
    return np, _scipy_ndimage


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Scale:
    activator_radius: int    # smaller neighborhood (local avg)
    inhibitor_radius: int    # larger neighborhood (broad avg)
    small_amount: float      # per-step increment magnitude
    weight: float = 1.0      # multiplier applied to both blurred averages
    color: Tuple[int, int, int] = (255, 255, 255)  # RGB for color mode


# ---------------------------------------------------------------------------
# Blur helpers
# ---------------------------------------------------------------------------

def _blur(grid, radius: int, method: str, xp, xp_ndimage):
    """Blur `grid` with the given `radius` using the requested method.

    mode='wrap' ensures seamless tiling at edges.
    Works with both NumPy (CPU) and CuPy (GPU) arrays.
    """
    if radius <= 0:
        return grid.copy()

    if method == "gaussian":
        sigma = radius / 2.0
        return xp_ndimage.gaussian_filter(grid, sigma=sigma, mode="wrap")
    else:  # default: box
        size = 2 * radius + 1
        return xp_ndimage.uniform_filter(grid, size=size, mode="wrap")


def _enforce_symmetry(grid, n: int, mirror: bool, xp, xp_ndimage):
    """Average grid with all rotational (and optionally mirror) symmetric copies.

    Produces N-fold (or 2N-fold with mirror=True) cyclic symmetry around the
    image centre. Uses bilinear interpolation (order=1) for speed.
    Works with both NumPy (CPU) and CuPy (GPU) arrays.
    """
    copies = [grid]
    step_angle = 360.0 / n
    for k in range(1, n):
        copies.append(
            xp_ndimage.rotate(grid, step_angle * k, reshape=False, mode="nearest", order=1)
        )
    if mirror:
        flipped = xp.fliplr(grid)
        for k in range(n):
            copies.append(
                xp_ndimage.rotate(flipped, step_angle * k, reshape=False, mode="nearest", order=1)
            )
    return xp.mean(xp.stack(copies), axis=0)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    height: int,
    width: int,
    scales: List[Scale],
    iterations: int,
    blur_method: str,
    color_mode: bool,
    color_lerp_alpha: float,
    rng: np.random.Generator,
    save_frames: bool = False,
    frames_dir: str = "frames",
    symmetry: int = 1,
    mirror: bool = False,
    device: str = "cpu",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run the multi-scale Turing simulation.

    Returns
    -------
    grid       : (H, W) float64 numpy array in [-1, +1]  (always CPU)
    color_grid : (H, W, 3) float64 numpy array in [0, 1] or None
    """
    xp, xp_ndimage = _get_backend(device)

    # Init grid on CPU, then transfer to device
    grid_cpu = rng.uniform(-1.0, 1.0, (height, width))
    grid = xp.asarray(grid_cpu) if device == "cuda" else grid_cpu

    # Color grid: accumulate weighted scale colors per cell
    color_grid = None
    if color_mode:
        scale_colors = xp.array(
            [s.color for s in scales], dtype=xp.float64
        ) / 255.0  # shape (S, 3), values in [0, 1]
        color_grid = xp.zeros((height, width, 3), dtype=xp.float64)

    n_scales = len(scales)
    frame_digits = len(str(iterations))
    if save_frames:
        Path(frames_dir).mkdir(parents=True, exist_ok=True)

    for step in trange(iterations, desc="Iterating", unit="step"):
        # --- per-scale activator/inhibitor averages ---
        activators  = xp.empty((n_scales, height, width), dtype=xp.float64)
        inhibitors  = xp.empty((n_scales, height, width), dtype=xp.float64)
        variations  = xp.empty((n_scales, height, width), dtype=xp.float64)

        for i, sc in enumerate(scales):
            act = _blur(grid, sc.activator_radius, blur_method, xp, xp_ndimage) * sc.weight
            inh = _blur(grid, sc.inhibitor_radius, blur_method, xp, xp_ndimage) * sc.weight
            activators[i] = act
            inhibitors[i] = inh
            variations[i] = xp.abs(act - inh)

        # --- best scale per cell (argmin variation) ---
        best_scale_idx = xp.argmin(variations, axis=0)  # (H, W) int

        # --- build increment map ---
        increment = xp.empty((height, width), dtype=xp.float64)
        for i, sc in enumerate(scales):
            mask = best_scale_idx == i
            if not xp.any(mask):
                continue
            pos = activators[i] > inhibitors[i]
            increment[mask & pos]  =  sc.small_amount
            increment[mask & ~pos] = -sc.small_amount

        grid = grid + increment

        # --- normalize to [-1, +1] ---
        lo, hi = grid.min(), grid.max()
        if hi > lo:
            grid = (grid - lo) / (hi - lo) * 2.0 - 1.0

        # --- color update ---
        if color_mode and color_grid is not None:
            for i in range(n_scales):
                mask = (best_scale_idx == i)[:, :, xp.newaxis]  # (H, W, 1)
                target = scale_colors[i]                          # (3,)
                color_grid += mask * color_lerp_alpha * (target - color_grid)

        # --- symmetry enforcement ---
        if symmetry > 1:
            grid = _enforce_symmetry(grid, symmetry, mirror, xp, xp_ndimage)
            lo, hi = grid.min(), grid.max()
            if hi > lo:
                grid = (grid - lo) / (hi - lo) * 2.0 - 1.0
            if color_mode and color_grid is not None:
                for ch in range(3):
                    color_grid[:, :, ch] = _enforce_symmetry(
                        color_grid[:, :, ch], symmetry, mirror, xp, xp_ndimage
                    )
                color_grid = xp.clip(color_grid, 0.0, 1.0)

        # --- save frame ---
        if save_frames:
            frame_path = str(
                Path(frames_dir) / f"frame_{step:0{frame_digits}d}.png"
            )
            # Transfer to CPU for PIL save
            grid_cpu   = xp.asnumpy(grid)       if device == "cuda" else grid
            cg_cpu     = xp.asnumpy(color_grid) if (device == "cuda" and color_grid is not None) else color_grid
            save_image(grid_cpu, cg_cpu, frame_path, color_mode, quiet=True)

    # Transfer final arrays back to CPU
    grid_out = xp.asnumpy(grid) if device == "cuda" else grid
    cg_out   = xp.asnumpy(color_grid) if (device == "cuda" and color_grid is not None) else color_grid
    return grid_out, cg_out


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_image(
    grid: np.ndarray,
    color_grid: Optional[np.ndarray],
    output_path: str,
    color_mode: bool,
    quiet: bool = False,
) -> None:
    """Convert simulation grid to PIL Image and save."""
    # Intensity in [0, 1]
    intensity = (grid + 1.0) / 2.0  # (H, W)

    if color_mode and color_grid is not None:
        # Modulate accumulated colors by intensity
        rgb = np.clip(color_grid * intensity[:, :, np.newaxis], 0.0, 1.0)
        pixels = (rgb * 255).astype(np.uint8)
        img = Image.fromarray(pixels, mode="RGB")
    else:
        gray = np.clip(intensity * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(gray, mode="L")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    if not quiet:
        print(f"Saved → {output_path}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_scales(raw_scales: list) -> List[Scale]:
    scales = []
    for s in raw_scales:
        color_raw = s.get("color", [255, 255, 255])
        # Validate color values are in [0, 255]
        color = tuple(max(0, min(255, int(c))) for c in color_raw[:3])
        while len(color) < 3:
            color = (*color, 255)
        scales.append(Scale(
            activator_radius=int(s["activator_radius"]),
            inhibitor_radius=int(s["inhibitor_radius"]),
            small_amount=float(s["small_amount"]),
            weight=float(s.get("weight", 1.0)),
            color=color,
        ))
    return scales


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-scale Turing patterns (McCabe/Rampe algorithm)."
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to JSON config file (default: config.json)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output image path from config"
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override iteration count from config"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from config"
    )
    parser.add_argument(
        "--grayscale", action="store_true",
        help="Force grayscale output regardless of config"
    )
    parser.add_argument(
        "--save-frames", action="store_true", dest="save_frames",
        help="Save each iteration as a PNG frame for video assembly"
    )
    parser.add_argument(
        "--frames-dir", default=None, dest="frames_dir",
        help="Override frames output directory from config (default: frames/)"
    )
    parser.add_argument(
        "--symmetry", type=int, default=None,
        help="N-fold rotational symmetry (1=off, 6/8/12 for mandalas)"
    )
    parser.add_argument(
        "--mirror", action="store_true",
        help="Add mirror symmetry on top of rotational symmetry"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Compute device: 'cpu' (default) or 'cuda' (requires CuPy)"
    )
    args = parser.parse_args()

    # Load config
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in config: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    if args.output:
        cfg["output_path"] = args.output
    if args.iterations is not None:
        cfg["iterations"] = args.iterations
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.grayscale:
        cfg["color_mode"] = False
    if args.save_frames:
        cfg["save_frames"] = True
    if args.frames_dir:
        cfg["frames_dir"] = args.frames_dir
    if args.symmetry is not None:
        cfg["symmetry"] = args.symmetry
    if args.mirror:
        cfg["mirror"] = True
    if args.device is not None:
        cfg["device"] = args.device

    # Validate required fields
    for key in ("width", "height", "scales"):
        if key not in cfg:
            print(f"Error: config missing required key '{key}'", file=sys.stderr)
            sys.exit(1)

    if not cfg.get("scales"):
        print("Error: config 'scales' list is empty", file=sys.stderr)
        sys.exit(1)

    # Build params
    width         = int(cfg["width"])
    height        = int(cfg["height"])
    iterations    = int(cfg.get("iterations", 300))
    seed          = cfg.get("seed")
    blur_method   = cfg.get("blur_method", "box")
    color_mode    = bool(cfg.get("color_mode", False))
    lerp_alpha    = float(cfg.get("color_lerp_alpha", 0.01))
    output_path   = cfg.get("output_path", "output.png")
    save_frames   = bool(cfg.get("save_frames", False))
    frames_dir    = cfg.get("frames_dir", "frames")
    symmetry      = int(cfg.get("symmetry", 1))
    mirror        = bool(cfg.get("mirror", False))
    device        = cfg.get("device", "cpu")
    scales        = build_scales(cfg["scales"])

    if device == "cuda" and not _CUPY_AVAILABLE:
        print("Error: device='cuda' requested but CuPy is not installed.", file=sys.stderr)
        print("Install with: pip install cupy-cuda13x", file=sys.stderr)
        sys.exit(1)

    if blur_method not in ("box", "gaussian"):
        print(f"Warning: unknown blur_method '{blur_method}', using 'box'", file=sys.stderr)
        blur_method = "box"

    rng = np.random.default_rng(seed)

    sym_label = f"{symmetry}-fold" + ("+mirror" if mirror else "") if symmetry > 1 else "off"
    print(f"Grid: {width}×{height}  |  Scales: {len(scales)}  |  "
          f"Iterations: {iterations}  |  Blur: {blur_method}  |  "
          f"Color: {color_mode}  |  Symmetry: {sym_label}  |  "
          f"Device: {device.upper()}  |  Save frames: {save_frames}")
    if save_frames:
        print(f"Frames dir: {frames_dir}")

    grid, color_grid = run_simulation(
        height=height,
        width=width,
        scales=scales,
        iterations=iterations,
        blur_method=blur_method,
        color_mode=color_mode,
        color_lerp_alpha=lerp_alpha,
        rng=rng,
        save_frames=save_frames,
        frames_dir=frames_dir,
        symmetry=symmetry,
        mirror=mirror,
        device=device,
    )

    save_image(grid, color_grid, output_path, color_mode)


if __name__ == "__main__":
    main()
