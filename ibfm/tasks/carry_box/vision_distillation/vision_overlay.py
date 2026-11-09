"""Show the student's camera inputs (depth + mask) in the MuJoCo viewer and the saved video.

Enabled with ``DISTILL_SHOW_VISION=1`` (``play_student.sh`` sets it).  The panel
is exactly what the student receives at the tick:

* depth (left): turbo colour map over the 0.2-4.0 m task range, black = no
  return (invalid), white = valid but beyond 4 m ("far"), magenta = outline of
  the latest mask, which lags the depth by the mask latency;
* mask (right): white = box, dark grey = colour FOV without box, dark blue =
  outside the colour FOV (unknown).
"""

from __future__ import annotations

import mujoco
import numpy as np

from vision_sensor import SENTINEL, VisionConfig, encode_frame


SCALE = 3
GAP_PX = 6
MARGIN_PX = 12


def _turbo_lut() -> np.ndarray:
    from matplotlib import colormaps

    return (colormaps["turbo"](np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)


class VisionPanel:
    def __init__(self, vision: VisionConfig, known: np.ndarray):
        self.vision, self.known = vision, known
        self.lut = _turbo_lut()
        self.image: np.ndarray | None = None

    def update(self, depth_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
        _mask_image, depth_image = encode_frame(depth_mm, mask, self.known, self.vision)
        normalised, valid, far, box = depth_image
        depth_rgb = self.lut[np.clip((normalised * 255).astype(int), 0, 255)]
        depth_rgb[valid == 0] = 0
        depth_rgb[far > 0] = 255
        box = box > 0
        outline = box & ~(
            np.roll(box, 1, 0) & np.roll(box, -1, 0) & np.roll(box, 1, 1) & np.roll(box, -1, 1)
        )
        depth_rgb[outline] = (255, 0, 255)
        mask_rgb = np.full(box.shape + (3,), (20, 20, 70), dtype=np.uint8)
        mask_rgb[self.known] = (45, 45, 45)
        mask_rgb[box] = 255
        upscale = lambda image: np.repeat(np.repeat(image, SCALE, axis=0), SCALE, axis=1)
        depth_rgb, mask_rgb = upscale(depth_rgb), upscale(mask_rgb)
        gap = np.full((depth_rgb.shape[0], GAP_PX, 3), 255, dtype=np.uint8)
        self.image = np.ascontiguousarray(np.concatenate((depth_rgb, gap, mask_rgb), axis=1))
        return self.image

    def composite(self, frame: np.ndarray) -> np.ndarray:
        """Paste the panel into the bottom-right corner of a video frame."""

        if self.image is None:
            return frame
        panel = self.image
        scale = min(1.0, (frame.shape[1] - 2 * MARGIN_PX) / panel.shape[1])
        if scale < 1.0:
            rows = (np.arange(int(panel.shape[0] * scale)) / scale).astype(int)
            columns = (np.arange(int(panel.shape[1] * scale)) / scale).astype(int)
            panel = panel[rows][:, columns]
        out = frame.copy()
        top = frame.shape[0] - panel.shape[0] - MARGIN_PX
        left = frame.shape[1] - panel.shape[1] - MARGIN_PX
        out[top : top + panel.shape[0], left : left + panel.shape[1]] = panel
        return out

    def show(self, viewer, texts: tuple[str, str]) -> None:
        if viewer is None or self.image is None:
            return
        height, width = self.image.shape[:2]
        viewport = viewer.viewport
        left = max(0, (viewport.width if viewport is not None else width) - width - MARGIN_PX)
        viewer.set_images((mujoco.MjrRect(left, MARGIN_PX, width, height), self.image))
        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, texts[0], texts[1]))


def describe(stats: np.ndarray, ages: tuple[float, ...], vision: VisionConfig, extra: dict[str, str]) -> tuple[str, str]:
    """Two text columns (labels, values) for the viewer overlay."""

    from vision_sensor import STATS_NAMES

    index = {name: i for i, name in enumerate(STATS_NAMES)}
    metres = lambda value: value * (vision.depth_max_m - vision.depth_min_m) + vision.depth_min_m

    def distance(name: str, valid: str) -> str:
        return f"{metres(stats[index[name]]):.2f} m" if stats[index[valid]] > 0 and stats[index[name]] != SENTINEL else "invalid"

    rows = [
        ("box detected", "yes" if stats[index["detected"]] > 0 else "no"),
        ("D435 mask median", distance("d435_median", "d435_valid")),
        *((("Depth Anything", distance("da_median", "da_valid")),) if "da_median" in index else ()),
        ("age depth/mask", " / ".join(f"{age * 1000:.0f} ms" for age in ages)),
        *extra.items(),
    ]
    return "\n".join(label for label, _ in rows), "\n".join(value for _, value in rows)
