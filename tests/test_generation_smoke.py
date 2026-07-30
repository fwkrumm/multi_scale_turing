import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from generate import Scale, build_scales, load_config, run_simulation, save_image


def test_load_config_reads_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"width": 4, "height": 3, "scales": []}', encoding="utf-8")

    assert load_config(str(config_path)) == {"width": 4, "height": 3, "scales": []}


def test_build_scales_clamps_colors_and_defaults():
    raw_scales = [
        {
            "activator_radius": 1,
            "inhibitor_radius": 3,
            "small_amount": 0.25,
            "weight": 1.5,
            "color": [400, -10, 50],
        }
    ]

    scales = build_scales(raw_scales)

    assert len(scales) == 1
    assert scales[0] == Scale(1, 3, 0.25, 1.5, (255, 0, 50))


def test_run_simulation_returns_expected_shapes_and_ranges():
    scales = [
        Scale(
            activator_radius=3,
            inhibitor_radius=6,
            small_amount=0.05,
            weight=1.0,
            color=(255, 255, 255),
        )
    ]

    rng = np.random.default_rng(42)
    grid, color_grid = run_simulation(
        height=32,
        width=32,
        scales=scales,
        iterations=3,
        blur_method="box",
        color_mode=True,
        color_lerp_alpha=0.01,
        rng=rng,
        save_frames=False,
        symmetry=1,
        mirror=False,
        device="cpu",
    )

    assert grid.shape == (32, 32)
    assert np.all(grid >= -1.0) and np.all(grid <= 1.0)
    assert color_grid is not None
    assert color_grid.shape == (32, 32, 3)
    assert np.all(color_grid >= 0.0) and np.all(color_grid <= 1.0)


def test_save_image_writes_rgb_file(tmp_path):
    grid = np.array([[-1.0, 1.0], [0.0, 0.5]], dtype=np.float64)
    color_grid = np.array(
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.5, 0.5, 0.5], [0.2, 0.4, 0.6]],
        ],
        dtype=np.float64,
    )
    output_path = tmp_path / "pattern.png"

    save_image(grid, color_grid, str(output_path), color_mode=True, quiet=True)

    assert output_path.exists()
    with Image.open(output_path) as img:
        assert img.mode == "RGB"
        assert img.size == (2, 2)


def test_ci_workflow_exists():
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), f"Missing workflow file: {workflow_path}"
