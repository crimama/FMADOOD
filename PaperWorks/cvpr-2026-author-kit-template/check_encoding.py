#!/usr/bin/env python3
"""Fail fast when a manuscript source has invalid UTF-8 or U+FFFD replacement characters."""
import sys

bad = 0
for path in sys.argv[1:]:
    with open(path, "rb") as handle:
        data = handle.read()
    for number, raw in enumerate(data.split(b"\n"), 1):
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            print(f"{path}:{number}: invalid UTF-8 ({error.reason})", file=sys.stderr)
            bad += 1
            continue
        if "\ufffd" in line:
            column = line.index("\ufffd")
            context = line[max(0, column - 20):column + 5]
            print(f"{path}:{number}: broken character near '{context}'", file=sys.stderr)
            bad += 1
if bad:
    sys.exit(f"check_encoding: {bad} broken line(s); fix them before running LaTeX")
