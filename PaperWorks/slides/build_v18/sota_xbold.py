"""Make best-value (bold) cells of result tables ExtraBold, in place. Header row and first column are left as is.
usage: python3 sota_xbold.py in.pptx out.pptx"""
import sys
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Pt

XBOLD = 'NanumSquareOTF_ac ExtraBold'
RESULT_SECTIONS = ('03.', '04.')
BUMP_PT = 1  # best values one point larger than the rest of the table


def section_of(slide):
    for sh in slide.shapes:
        if sh.name == 'SectionLabel' and sh.has_text_frame:
            return sh.text_frame.text.strip()
    return ''


prs = Presentation(sys.argv[1])
count = 0
for i, s in enumerate(prs.slides, 1):
    if not section_of(s).startswith(RESULT_SECTIONS):
        continue
    for sh in s.shapes:
        if not (getattr(sh, 'has_table', False) and sh.has_table):
            continue
        t = sh.table
        for r in range(1, len(t.rows)):
            for c in range(1, len(t.columns)):
                for p in t.cell(r, c).text_frame.paragraphs:
                    for run in p.runs:
                        if run.font.bold and run.text.strip():
                            rpr = run._r.get_or_add_rPr()
                            for tag in ('a:latin', 'a:ea', 'a:cs'):
                                el = rpr.find(qn(tag))
                                if el is not None:
                                    el.set('typeface', XBOLD)
                            if run.font.size is not None:
                                run.font.size = Pt(run.font.size.pt + BUMP_PT)
                            count += 1
prs.save(sys.argv[2])
print('ExtraBold runs:', count)
