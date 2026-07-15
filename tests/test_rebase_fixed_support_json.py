from __future__ import annotations

from pathlib import Path

from scripts.rebase_fixed_support_json import rebase


def test_rebase_preserves_order_and_basename(tmp_path: Path) -> None:
    root = tmp_path / "data"
    names = ("052_regular.png", "388_regular.png")
    for name in names:
        path = root / "can" / "train" / "good" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    result = rebase(
        {
            "can": [f"/old/root/can/train/good/{name}" for name in names],
            "fabric": ["/old/root/fabric/train/good/not_synced.png"],
        },
        root,
        ("can",),
    )

    assert [Path(path).name for path in result["can"]] == list(names)
