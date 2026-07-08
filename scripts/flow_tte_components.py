# /// script
# requires-python = ">=3.8"
# dependencies = ["numpy"]
# ///
# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class ComponentSummary:
    component_count: int
    positive_area: float
    largest_component_share: float
    mean_component_area: float


def summarize_components(mask: BoolArray) -> ComponentSummary:
    cv2_summary = _summarize_components_cv2(mask)
    if cv2_summary is not None:
        return cv2_summary
    return _summarize_components_flood_fill(mask)


def _summarize_components_cv2(mask: BoolArray) -> ComponentSummary | None:
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return None
    positive_count = int(np.count_nonzero(mask))
    total_count = int(mask.size)
    if positive_count == 0:
        return ComponentSummary(
            component_count=0,
            positive_area=0.0 if total_count == 0 else float(positive_count / total_count),
            largest_component_share=0.0,
            mean_component_area=0.0,
        )
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8, copy=False),
        connectivity=8,
    )
    component_areas = np.asarray(stats[1:component_count, cv2.CC_STAT_AREA], dtype=np.float32)
    largest_area = float(np.max(component_areas)) if component_areas.size else 0.0
    return ComponentSummary(
        component_count=max(0, int(component_count) - 1),
        positive_area=0.0 if total_count == 0 else float(positive_count / total_count),
        largest_component_share=0.0 if positive_count == 0 else largest_area / positive_count,
        mean_component_area=(
            0.0 if component_areas.size == 0 else float(np.mean(component_areas))
        ),
    )


def _summarize_components_flood_fill(mask: BoolArray) -> ComponentSummary:
    visited = np.zeros(mask.shape, dtype=np.bool_)
    component_areas = []
    for y_index, x_index in np.argwhere(mask):
        y = int(y_index)
        x = int(x_index)
        if visited[y, x]:
            continue
        component_areas.append(_flood_component(mask, visited, y, x))
    positive_count = int(np.count_nonzero(mask))
    total_count = int(mask.size)
    largest_area = max(component_areas, default=0)
    return ComponentSummary(
        component_count=len(component_areas),
        positive_area=0.0 if total_count == 0 else float(positive_count / total_count),
        largest_component_share=(
            0.0 if positive_count == 0 else float(largest_area / positive_count)
        ),
        mean_component_area=0.0 if not component_areas else float(np.mean(component_areas)),
    )


def _flood_component(mask: BoolArray, visited: BoolArray, y: int, x: int) -> int:
    stack = [(y, x)]
    visited[y, x] = True
    area = 0
    while stack:
        current_y, current_x = stack.pop()
        area += 1
        for next_y in range(current_y - 1, current_y + 2):
            for next_x in range(current_x - 1, current_x + 2):
                if (
                    next_y < 0
                    or next_x < 0
                    or next_y >= mask.shape[0]
                    or next_x >= mask.shape[1]
                    or visited[next_y, next_x]
                    or not mask[next_y, next_x]
                ):
                    continue
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))
    return area
