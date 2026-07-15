from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from flow_tte.darc_gate1_provenance import JsonValue
from flow_tte.darc_synthetic import SyntheticCue


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class SourceSampleRecord:
    seed: int
    fold: int
    source_path: str
    clean_selection: Tuple[str, ...]
    cues: Tuple[Dict[str, JsonValue], ...]


def cue_record(cue: SyntheticCue, selected_support_ids: Sequence[str]) -> Dict[str, JsonValue]:
    """Persist enough cue provenance to reproduce the exact rasterized mask."""
    metadata = cue.metadata.to_manifest()
    return {
        "metadata": {
            "version": metadata["version"],
            "profile_name": metadata["profile_name"],
            "width": metadata["width"],
            "length": metadata["length"],
            "seed": metadata["seed"],
            "start_xy": list(metadata["start_xy"]),
            "end_xy": list(metadata["end_xy"]),
            "color_rgb": list(metadata["color_rgb"]),
            "line_type": metadata["line_type"],
        },
        "mask_sha256": hashlib.sha256(np.ascontiguousarray(cue.mask).tobytes()).hexdigest(),
        "mask_shape": list(cue.mask.shape),
        "mask_pixel_count": int(np.count_nonzero(cue.mask)),
        "selected_top5": list(selected_support_ids),
    }


def sample_record(
    object_name: str,
    data_root: Path,
    sample: SourceSampleRecord,
) -> Dict[str, JsonValue]:
    source = Path(sample.source_path)
    return {
        "object": object_name,
        "seed": sample.seed,
        "fold": sample.fold,
        "source": source.relative_to(data_root).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "clean_selected_top5": list(sample.clean_selection),
        "cues": list(sample.cues),
    }
