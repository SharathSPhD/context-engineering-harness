The Pratyaksa Context-Engineering Harness — arXiv submission
============================================================

Compile with tectonic (recommended; one command, no .bbl management):

    tectonic -X compile main.tex

Or with a vanilla TeX Live install:

    pdflatex main.tex
    bibtex   main
    pdflatex main.tex
    pdflatex main.tex

Files
-----
  main.tex                         — driver
  references.bib                   — 246 BibTeX entries (plainnat)
  sections/00..12_*.tex            — frontmatter + 12 main sections
  appendices/A..F_*.tex            — 6 appendices
  figures/F01..F13_*.png           — 13 PNG figures (raster, 300 dpi)
  generated/figures.tex            — \includegraphics wrappers for all PNGs
  generated/tables.tex             — Markdown tables converted to LaTeX

The .tex under sections/ and appendices/ are deterministic Pandoc output
from the corresponding .md files in the public source repository
(https://github.com/SharathSPhD/pratyaksha-context-eng-harness, paper/).
Editing the .tex directly is supported but the source-of-truth lives in
the Markdown drafts under paper/sections and paper/appendices.

Source code, MCP plugin, experiment runners, and the full test suite
referenced throughout the paper are at the URL above.

Reproducibility manifest: paper/appendices/C_reproducibility.tex.

License: MIT (code), CC-BY-4.0 (text + figures).
