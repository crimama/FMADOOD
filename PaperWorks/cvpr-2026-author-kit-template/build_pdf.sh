#!/bin/sh
# Build <name>.pdf in a private temporary directory, then replace <name>.pdf atomically.
# Concurrent builds (editor auto-compile + make) therefore never share .aux/.bbl files.
set -eu
name="$1"
root=$(pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/latex-$name.XXXXXX")
latex=${PDFLATEX:-lualatex}
bib=${BIBTEX:-bibtex}
flags=${LATEX_FLAGS:--interaction=nonstopmode -halt-on-error -file-line-error}

run_latex() {
    if ! $latex $flags -output-directory="$out" "$name.tex" >"$out/pass.out" 2>&1; then
        cp "$out/$name.log" "$name.log" 2>/dev/null || true
        tail -n 30 "$out/pass.out"
        echo "LaTeX failed: see $root/$name.log (build directory kept at $out)" >&2
        exit 1
    fi
}

run_latex
if ! (cd "$out" && BIBINPUTS="$root:" BSTINPUTS="$root:" $bib "$name") >"$out/bibtex.out" 2>&1; then
    cat "$out/bibtex.out"
    echo "BibTeX failed (build directory kept at $out)" >&2
    exit 1
fi
run_latex
run_latex
run_latex
cp "$out/$name.log" "$name.log"
cp "$out/$name.pdf" ".$name.pdf.tmp"
mv ".$name.pdf.tmp" "$name.pdf"
rm -rf "$out"
echo "Wrote $name.pdf"
