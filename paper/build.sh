#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p sections appendices generated

PANDOC_OPTS=(
    --from "markdown+raw_tex+pipe_tables+grid_tables+tex_math_dollars+yaml_metadata_block"
    --to latex
    --top-level-division=section
    --wrap=preserve
    --no-highlight
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

echo "=== Generating figures.tex ==="
{
    for png in figures/*.png; do
        name="$(basename "$png" .png)"
        cat <<EOF
\\begin{figure}[ht]
\\centering
\\includegraphics[width=0.85\\linewidth]{${name}}
\\caption{${name//_/\\_}.}
\\label{fig:${name}}
\\end{figure}

EOF
    done
} > generated/figures.tex

echo "=== Generating tables.tex ==="
{
    for md in tables/*.md; do
        name="$(basename "$md" .md)"
        echo "\\subsection*{${name//_/\\_}}"
        pandoc --from "markdown+pipe_tables" --to latex "$md"
        echo ""
    done
} > generated/tables.tex

echo "=== Compiling with tectonic ==="
tectonic -X compile main.tex --keep-logs --keep-intermediates 2>&1 | tail -30

echo
echo "=== Done. PDF -> $(pwd)/main.pdf ==="
ls -la main.pdf 2>/dev/null || echo "PDF NOT FOUND"
