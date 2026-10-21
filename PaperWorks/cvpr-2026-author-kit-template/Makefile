export max_print_line ?= 10000
export TEXMFVAR ?= /tmp/texmf-var
export TEXMFCACHE ?= /tmp/texmf-cache
export XDG_CACHE_HOME ?= /tmp/cache

PDFLATEX ?= lualatex
BIBTEX ?= bibtex
LATEX_FLAGS ?= -interaction=nonstopmode -halt-on-error -file-line-error
export PDFLATEX BIBTEX LATEX_FLAGS

SOURCES := main.tex preamble.tex main.bib $(wildcard sec/*.tex)

.PHONY: all rebuttal clean check-encoding

all: main.pdf

check-encoding:
	python3 check_encoding.py $(SOURCES)

main.pdf: main.tex preamble.tex cvpr.sty ieeenat_fullname.bst main.bib $(wildcard sec/*.tex) $(wildcard figures/*)
	python3 check_encoding.py $(SOURCES)
	sh build_pdf.sh main

rebuttal: rebuttal.pdf

rebuttal.pdf: rebuttal.tex preamble.tex cvpr.sty ieeenat_fullname.bst main.bib
	sh build_pdf.sh rebuttal

clean:
	rm -f main.aux main.bbl main.blg main.brf main.log main.out main.fls main.fdb_latexmk main.synctex.gz main.pdf
	rm -f rebuttal.aux rebuttal.bbl rebuttal.blg rebuttal.brf rebuttal.log rebuttal.out rebuttal.fls rebuttal.fdb_latexmk rebuttal.synctex.gz rebuttal.pdf
