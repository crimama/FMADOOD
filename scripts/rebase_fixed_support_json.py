"""Rebase absolute fixed-support paths while preserving object, order, and basename."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def rebase(
    payload: dict[str, list[str]],
    data_root: Path,
    objects: tuple[str, ...] | None = None,
) -> dict[str, list[str]]:
    rebased: dict[str, list[str]] = {}
    selected = tuple(payload) if objects is None else objects
    for object_name in selected:
        paths = payload[object_name]
        candidates = []
        for original in paths:
            path = data_root / object_name / "train" / "good" / Path(original).name
            if not path.is_file():
                raise FileNotFoundError(f"Rebased support file not found: {path}")
            candidates.append(str(path))
        if len(candidates) != len(set(candidates)):
            raise RuntimeError(f"Duplicate rebased support path for {object_name}")
        rebased[object_name] = candidates
    return rebased


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objects", default="")
    args = parser.parse_args(list(argv))
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            rebase(
                payload,
                Path(args.data_root),
                tuple(part for part in args.objects.replace(",", " ").split() if part) or None,
            ),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
