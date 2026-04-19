#!/usr/bin/env bash
# Build the v2.0.0 release artifacts:
#   1. arxiv-submission.tar.gz  (LaTeX sources, figures, tables, .bib, README)
#   2. pratyaksha-context-eng-harness-v2.0.0.zip (the shipped plugin tree)
#
# Run from the worktree root:
#     bash release/build_release.sh
#
# Outputs land in `release/`. This script is deterministic apart from the
# embedded mtimes; reproducible-build flags are documented inline below.

set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$(pwd)"
REL="$ROOT/release"
VERSION="v2.0.0"

# --------------------------------------------------------------------------
# 1. arXiv submission tarball
# --------------------------------------------------------------------------
echo "=== Building arxiv-submission.tar.gz ==="
ARXIV="$REL/arxiv-submission"
rm -rf "$ARXIV"
mkdir -p "$ARXIV/sections" "$ARXIV/appendices" "$ARXIV/figures" \
         "$ARXIV/figures_tikz" "$ARXIV/tables"

# Ensure paper has been freshly built (so .tex are in sync with .md).
(cd paper && bash build.sh >/dev/null 2>&1)

# Top-level
cp paper/main.tex          "$ARXIV/"
cp paper/references.bib    "$ARXIV/"

# Section + appendix .tex (NOT the .md sources — arXiv wants TeX only)
cp paper/sections/*.tex      "$ARXIV/sections/"
cp paper/appendices/*.tex    "$ARXIV/appendices/"

# Figures (PNG raster), TikZ concept figures, and per-table .tex wrappers.
cp paper/figures/*.png         "$ARXIV/figures/"
cp paper/figures_tikz/*.tex    "$ARXIV/figures_tikz/"
cp paper/tables/*.tex          "$ARXIV/tables/"

# arXiv-style readme so reviewers know how to compile.
cat > "$ARXIV/00README.txt" <<'TXT'
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
  references.bib                   — BibTeX entries (plainnat)
  sections/00..12_*.tex            — frontmatter + 12 main sections
  appendices/A..F_*.tex            — 6 appendices
  figures/F01..F13_*.png           — 13 PNG figures (raster, 300 dpi)
  figures_tikz/fig{1..6}_*.tex     — 6 TikZ concept/flow diagrams (vector)
  tables/T1..T7_*.tex              — 7 result tables (longtable)

The .tex under sections/ and appendices/ are deterministic Pandoc output
from the corresponding .md files in the public source repository
(https://github.com/SharathSPhD/pratyaksha-context-eng-harness, paper/).
Editing the .tex directly is supported but the source-of-truth lives in
the Markdown drafts under paper/sections and paper/appendices.

Source code, MCP plugin, experiment runners, and the full test suite
referenced throughout the paper are at the URL above.

Reproducibility manifest: paper/appendices/C_reproducibility.tex.

License: MIT (code), CC-BY-4.0 (text + figures).
TXT

# Strip macOS resource forks if any leaked in.
find "$ARXIV" -name '._*' -delete

(
    cd "$REL"
    # --owner=0 --group=0 keeps the tarball reproducible across machines;
    # GNU tar accepts these on macOS via gtar (BSD tar emits an unknown
    # option warning but still produces a valid archive — we tolerate it).
    if command -v gtar >/dev/null 2>&1; then
        gtar --owner=0 --group=0 --sort=name -czf arxiv-submission.tar.gz arxiv-submission/
    else
        tar -czf arxiv-submission.tar.gz arxiv-submission/
    fi
)

echo "    → $REL/arxiv-submission.tar.gz"
echo "    size: $(du -h "$REL/arxiv-submission.tar.gz" | cut -f1)"

# Quick sanity-check: extract into a tmp dir and verify it compiles.
echo "=== Verifying arxiv-submission.tar.gz compiles ==="
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$REL/arxiv-submission.tar.gz" -C "$TMP"
(
    cd "$TMP/arxiv-submission"
    tectonic -X compile main.tex 2>&1 | tail -3
)
PAGES="$(pdfinfo "$TMP/arxiv-submission/main.pdf" 2>/dev/null | awk '/Pages/ {print $2}' || echo '?')"
echo "    → re-built PDF has ${PAGES} pages"

# --------------------------------------------------------------------------
# 2. Plugin zip (the shipped tree only — no experiments/, no docs/)
# --------------------------------------------------------------------------
echo "=== Building pratyaksha-context-eng-harness-${VERSION}.zip ==="
PLUGIN_ZIP="$REL/pratyaksha-context-eng-harness-${VERSION}.zip"
rm -f "$PLUGIN_ZIP"

(
    cd plugin
    # -X: strip OS-specific extras for reproducibility.
    # The shipped tree is self-contained: marketplace.json, .claude-plugin/,
    # .mcp.json, README.md, LICENSE, agents/, commands/, hooks/, mcp/, skills/.
    zip -r -X "$PLUGIN_ZIP" pratyaksha-context-eng-harness \
        -x '*/__pycache__/*' '*/.DS_Store' '*/._*' '*/*.pyc'
) >/dev/null

echo "    → $PLUGIN_ZIP"
echo "    size: $(du -h "$PLUGIN_ZIP" | cut -f1)"
echo "    contents:"
unzip -l "$PLUGIN_ZIP" | tail -5

# --------------------------------------------------------------------------
# 3. SHA256 manifest for both
# --------------------------------------------------------------------------
echo "=== SHA256 manifest ==="
(
    cd "$REL"
    shasum -a 256 arxiv-submission.tar.gz "pratyaksha-context-eng-harness-${VERSION}.zip" \
        | tee SHA256SUMS
)

echo
echo "=== Done. Artifacts in $REL/ ==="
ls -la "$REL/"*.tar.gz "$REL/"*.zip "$REL/SHA256SUMS"
