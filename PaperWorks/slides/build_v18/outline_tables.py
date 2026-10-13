"""Parse numeric tables out of Outline_v18.md by caption so charts never re-type numbers."""
import re
from pathlib import Path

OUTLINE = Path('/home/hun/Volume/RESEARCH/FSAD-TTA/PaperWorks/outlines/Outline_v18.md')
TEXT = OUTLINE.read_text(encoding='utf-8')
LINES = TEXT.splitlines()


def _clean(cell: str) -> str:
    cell = re.sub(r'<u>|</u>|\*\*|\\\*|`', '', cell)
    return cell.replace('\\', '').strip()


def table(caption_start: str):
    """Rows (list of cleaned cells) of the first markdown table after the line starting with caption_start."""
    start = next(i for i, l in enumerate(LINES) if l.startswith(caption_start))
    i = start + 1
    while not LINES[i].lstrip().startswith('|'):
        i += 1
    rows = []
    while i < len(LINES) and LINES[i].lstrip().startswith('|'):
        cells = [_clean(c) for c in LINES[i].strip().strip('|').split('|')]
        if not all(re.fullmatch(r':?-+:?', c) or c == '' for c in cells):
            rows.append(cells)
        i += 1
    return rows


def num(cell: str):
    """'56.03 ± 2.99' -> 56.03, '54.81*' -> 54.81, '—' -> None, '−0.40' -> -0.40."""
    cell = cell.replace('−', '-')
    m = re.search(r'[-+]?\d+(?:\.\d+)?', cell)
    return float(m.group()) if m and '—' not in cell else None


def pair(cell: str):
    """'92.76 / 96.77' -> (92.76, 96.77); '— / —' -> (None, None)."""
    parts = [p.strip() for p in cell.split('/')]
    return tuple(num(p) for p in parts)


def table_raw(caption_start: str):
    """Like table(), but cells keep their markdown markers (** bold, <u> underline) for emphasis."""
    start = next(i for i, l in enumerate(LINES) if l.startswith(caption_start))
    i = start + 1
    while not LINES[i].lstrip().startswith('|'):
        i += 1
    rows = []
    while i < len(LINES) and LINES[i].lstrip().startswith('|'):
        cells = [c.strip() for c in LINES[i].strip().strip('|').split('|')]
        if not all(re.fullmatch(r':?-+:?', c) or c == '' for c in cells):
            rows.append(cells)
        i += 1
    return rows
