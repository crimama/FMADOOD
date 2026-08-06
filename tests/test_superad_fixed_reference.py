from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmad.methods.superad import load_fixed_reference_paths


def test_load_fixed_reference_paths_preserves_manifest_order(tmp_path: Path) -> None:
    train_dir = tmp_path / "can" / "train" / "good"
    train_dir.mkdir(parents=True)
    train_paths = []
    for name in ("000_regular.png", "001_regular.png", "002_regular.png"):
        path = train_dir / name
        path.touch()
        train_paths.append(str(path))
    manifest = tmp_path / "support.json"
    manifest.write_text(
        json.dumps(
            {
                "can": [
                    "/different/root/can/train/good/002_regular.png",
                    "/different/root/can/train/good/000_regular.png",
                ],
            }
        ),
        encoding="utf-8",
    )

    selected, digest = load_fixed_reference_paths(
        manifest,
        "can",
        train_paths,
        expected_count=2,
    )

    assert [Path(path).name for path in selected] == [
        "002_regular.png",
        "000_regular.png",
    ]
    assert len(digest) == 64


def test_load_fixed_reference_paths_rejects_missing_training_member(
    tmp_path: Path,
) -> None:
    train_dir = tmp_path / "can" / "train" / "good"
    train_dir.mkdir(parents=True)
    manifest = tmp_path / "support.json"
    manifest.write_text(
        json.dumps({"can": ["/different/root/can/train/good/999_regular.png"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not present in train/good"):
        load_fixed_reference_paths(
            manifest,
            "can",
            [str(train_dir / "000_regular.png")],
            expected_count=1,
        )


def test_load_fixed_reference_paths_rejects_same_basename_from_wrong_split(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "can" / "train" / "good" / "000_regular.png"
    train_path.parent.mkdir(parents=True)
    manifest = tmp_path / "support.json"
    manifest.write_text(
        json.dumps({"can": ["/different/root/can/test_public/good/000_regular.png"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside can/train/good"):
        load_fixed_reference_paths(
            manifest,
            "can",
            [str(train_path)],
            expected_count=1,
        )
