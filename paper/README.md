# arXiv Submission Package

This directory is the uploadable manuscript source for arXiv.

Build locally:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If `latexmk` is installed, this is simpler:

```bash
cd paper
latexmk -pdf main.tex
```

Upload `main.tex`, `references.bib`, and any required figures as the arXiv source bundle.
