#!/usr/bin/env python3
"""Source-checkout entry point; installed environments can use `fmad-run`."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fmad.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
