PDFLATEX ?= lualatex
BIBTEX ?= bibtex
LATEX_FLAGS ?= -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all rebuttal clean

all: main.pdf

main.pdf: main.tex preamble.tex cvpr.sty ieeenat_fullname.bst main.bib $(wildcard sec/*.tex) $(wildcard figures/*)
	$(PDFLATEX) $(LATEX_FLAGS) main.tex
	$(BIBTEX) main
	$(PDFLATEX) $(LATEX_FLAGS) main.tex
	$(PDFLATEX) $(LATEX_FLAGS) main.tex
	$(PDFLATEX) $(LATEX_FLAGS) main.tex

rebuttal: rebuttal.pdf

rebuttal.pdf: rebuttal.tex preamble.tex cvpr.sty ieeenat_fullname.bst main.bib
	$(PDFLATEX) $(LATEX_FLAGS) rebuttal.tex
	$(BIBTEX) rebuttal
	$(PDFLATEX) $(LATEX_FLAGS) rebuttal.tex
	$(PDFLATEX) $(LATEX_FLAGS) rebuttal.tex
	$(PDFLATEX) $(LATEX_FLAGS) rebuttal.tex

clean:
	rm -f main.aux main.bbl main.blg main.brf main.log main.out main.fls main.fdb_latexmk main.synctex.gz main.pdf
	rm -f rebuttal.aux rebuttal.bbl rebuttal.blg rebuttal.brf rebuttal.log rebuttal.out rebuttal.fls rebuttal.fdb_latexmk rebuttal.synctex.gz rebuttal.pdf
