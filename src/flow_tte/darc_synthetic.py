"""Versioned class-agnostic line cues for DARC controls."""

from __future__ import annotations

import math
from hashlib import sha256
from importlib import import_module
from typing import Callable, Final, NamedTuple, Protocol, Tuple, TypedDict, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import final, override

LINE_CUE_VERSION: Final = "darc-line-cue-v1"
_DARK_LIGHT_BOUNDARY: Final = 127.5


class _Cv2Like(Protocol):
    LINE_8: int
    line: Callable[
        [npt.NDArray[np.uint8], Tuple[int, int], Tuple[int, int], int, int, int],
        npt.NDArray[np.uint8],
    ]


class _PluginModule(Protocol):
    """Structurally typed boundary for a dynamically loaded Python module."""


def _load_cv2_plugin() -> _PluginModule:
    return import_module("cv2")


_CV2: Final = cast("_Cv2Like", _load_cv2_plugin())


class CueProfile(NamedTuple):
    name: str
    width: int
    length: int


THIN_W1_L32: Final = CueProfile("thin-w1-l32", 1, 32)
THIN_W2_L48: Final = CueProfile("thin-w2-l48", 2, 48)
BROAD_CONTROL_W16_L96: Final = CueProfile("broad-control-w16-l96", 16, 96)
LINE_CUE_PROFILES: Final = (THIN_W1_L32, THIN_W2_L48, BROAD_CONTROL_W16_L96)


class LineCueManifest(TypedDict):
    version: str
    profile_name: str
    width: int
    length: int
    seed: int
    start_xy: Tuple[int, int]
    end_xy: Tuple[int, int]
    color_rgb: Tuple[int, int, int]
    line_type: str


@final
class DarcSyntheticError(ValueError):
    __slots__ = ("field", "reason")

    field: str
    reason: str

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(str(self))

    @override
    def __str__(self) -> str:
        return f"Invalid {self.field}: {self.reason}"


class LineCueMetadata(NamedTuple):
    version: str
    profile_name: str
    width: int
    length: int
    seed: int
    start_xy: Tuple[int, int]
    end_xy: Tuple[int, int]
    color_rgb: Tuple[int, int, int]
    line_type: str

    def to_manifest(self) -> LineCueManifest:
        return {
            "version": self.version,
            "profile_name": self.profile_name,
            "width": self.width,
            "length": self.length,
            "seed": self.seed,
            "start_xy": self.start_xy,
            "end_xy": self.end_xy,
            "color_rgb": self.color_rgb,
            "line_type": self.line_type,
        }


class SyntheticCue(NamedTuple):
    image: npt.NDArray[np.uint8]
    mask: npt.NDArray[np.uint8]
    metadata: LineCueMetadata


def insert_line_cue(
    image: npt.NDArray[np.uint8],
    profile: CueProfile,
    seed: int,
) -> SyntheticCue:
    """Insert a deterministic black-or-white LINE_8 cue without mutating the input."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise DarcSyntheticError("image", "must be an HxWx3 uint8 RGB array")
    if profile not in LINE_CUE_PROFILES:
        raise DarcSyntheticError("profile", "must be a registered line-cue profile")
    if seed < 0:
        raise DarcSyntheticError("seed", "must be non-negative")

    shape = cast("Tuple[int, int, int]", tuple(image.shape))
    height, width, _ = shape
    margin = math.ceil(profile.length / 2.0) + math.ceil(profile.width / 2.0) + 1
    if height <= 2 * margin or width <= 2 * margin:
        raise DarcSyntheticError(
            "image",
            f"must be larger than {2 * margin} pixels in both spatial dimensions",
        )

    digest = sha256(f"{LINE_CUE_VERSION}\0{seed}".encode()).digest()
    angle = int.from_bytes(digest[:8], "big") / float(2**64) * math.pi
    center_x = margin + int.from_bytes(digest[8:16], "big") % (width - 2 * margin)
    center_y = margin + int.from_bytes(digest[16:24], "big") % (height - 2 * margin)
    direction_x = math.cos(angle)
    direction_y = math.sin(angle)
    dominant_direction = max(abs(direction_x), abs(direction_y))
    delta_x = round((profile.length - 1) * direction_x / dominant_direction)
    delta_y = round((profile.length - 1) * direction_y / dominant_direction)
    start_xy = (center_x - delta_x // 2, center_y - delta_y // 2)
    end_xy = (start_xy[0] + delta_x, start_xy[1] + delta_y)

    mask: npt.NDArray[np.uint8] = np.zeros((height, width), dtype=np.uint8)
    _CV2.line(mask, start_xy, end_xy, 1, profile.width, _CV2.LINE_8)
    selected = cast("npt.NDArray[np.bool_]", mask == 1)
    pixels = cast(
        "npt.NDArray[np.float64]",
        image[selected].astype(np.float64, copy=False),
    )
    luminance_values: npt.NDArray[np.float64] = (
        0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    )
    luminance = float(cast("np.float64", luminance_values.mean()))
    color_rgb = (0, 0, 0) if luminance >= _DARK_LIGHT_BOUNDARY else (255, 255, 255)
    output = cast("npt.NDArray[np.uint8]", image.copy())
    output[selected] = np.asarray(color_rgb, dtype=np.uint8)
    output.setflags(write=False)
    mask.setflags(write=False)
    metadata = LineCueMetadata(
        version=LINE_CUE_VERSION,
        profile_name=profile.name,
        width=profile.width,
        length=profile.length,
        seed=seed,
        start_xy=start_xy,
        end_xy=end_xy,
        color_rgb=color_rgb,
        line_type="LINE_8",
    )
    return SyntheticCue(image=output, mask=mask, metadata=metadata)
