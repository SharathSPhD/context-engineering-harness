#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Pratyaksha preprint build (arxiv-clean, single column, inline figures/tables).
# Pandoc converts each markdown section/appendix to TeX; main.tex \input's them.
# Figures and tables are placed inline in the source markdown via \begin{figure}
# and \input{tables/...}.  No appendix-style figure/table dump.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p sections appendices tables

PANDOC_OPTS=(
    --from "markdown+raw_tex+pipe_tables+grid_tables+tex_math_dollars+yaml_metadata_block+fenced_code_blocks+fenced_code_attributes"
    --to latex
    --top-level-division=section
    --wrap=preserve
    --syntax-highlighting=idiomatic
)

echo "=== Converting sections ==="
for md in sections/*.md; do
    out="${md%.md}.tex"
    pandoc "${PANDOC_OPTS[@]}" -o "$out" "$md"
    echo "  $md -> $out"
done

echo "=== Converting appendices ==="
for md in appendices/*.md; do
    out="${md%.md}.tex"
    pandoc "${PANDOC_OPTS[@]}" -o "$out" "$md"
    echo "  $md -> $out"
done

# Per-table TeX (each table individually \input'd from a section). We keep a
# named .tex per table so sections can place them inline at first reference.
echo "=== Converting tables (per-table .tex) ==="
for md in tables/*.md; do
    out="${md%.md}.tex"
    pandoc --from "markdown+pipe_tables+grid_tables" --to latex -o "$out" "$md"
    echo "  $md -> $out"
done

echo "=== Compiling with tectonic ==="
tectonic -X compile main.tex --keep-logs --keep-intermediates 2>&1 | tail -40

echo
echo "=== Done. PDF -> $(pwd)/main.pdf ==="
ls -la main.pdf 2>/dev/null || echo "PDF NOT FOUND"

# Surface any remaining hbox warnings for the operator
if [[ -f main.log ]]; then
    overfull=$(grep -c '^Overfull \\hbox' main.log || true)
    underfull=$(grep -c '^Underfull \\hbox' main.log || true)
    echo "    overfull hboxes : ${overfull}"
    echo "    underfull hboxes: ${underfull}"
fi
