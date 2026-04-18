# Ship checklist — Pratyakṣa Context-Engineering Harness v2.0.0

This is the human-facing manifest that turns the v2 worktree into a public
release. Everything that can be automated has been automated; the remaining
steps require credentials only you (`SharathSPhD`) can supply.

Run from the worktree root:

```bash
cd /Users/sharath/Library/CloudStorage/OneDrive-Personal/wsl_projects/context/.worktrees/v2
```

## 0. Pre-flight (already done in this commit)

- [x] All 502 tests pass, 2 skipped, 0 failed (`uv run pytest`)
- [x] `ruff check experiments/v2/ plugin/` is clean
- [x] Paper rebuilds to a 92-page PDF (`bash paper/build.sh`)
- [x] `release/build_release.sh` produced reproducible artifacts:
  - `release/arxiv-submission.tar.gz` (488 KB)
  - `release/pratyaksha-context-eng-harness-v2.0.0.zip` (48 KB)
  - `release/SHA256SUMS`
- [x] Six review documents committed under `docs/REVIEW*.md`
- [x] All A1–A12 cross-reviewer must-fixes addressed
- [x] All B1–B3 critical code findings addressed (with regression tests)

## 1. Tag v2.0.0

```bash
# Verify the commit you want to tag is HEAD on this worktree.
git log -1 --oneline

# Create an annotated tag with the release notes inline.
git tag -a v2.0.0 -m "v2.0.0 — Pratyakṣa Context-Engineering Harness

First public release. Self-contained Claude Code plugin (15 MCP tools,
3 skills, 3 agents, 4 commands, 3 lifecycle hooks) plus a 92-page
preprint validating it on six L1 benchmarks (RULER, HELMET, NoCha,
HaluEval, TruthfulQA, FACTS-Grounding), one P6-B live case study on
three real GitHub issues, and one P6-C SWE-bench Verified A/B over
synthetic research trails.

Validation summary
  L1: 6/7 hypotheses confirmed (Stouffer combined Z, k=7 effective).
  L2: 100 % accuracy on three live GitHub issues with the plugin
      vs 0 % without (case study; not statistically significant on
      its own — see §9).
  L3: 100 % target-path-hit rate on the SWE-bench A/B in 6/6 (model,
      seed) cells (720/720 paired runs, p=0.03125 per cell).

Reviewer scores
  6 reviewer passes (code, kieran-strict, adversarial, paper coherence,
  paper feasibility, paper scope) → consolidated to 12 cross-reviewer
  must-fixes (A1–A12), 3 critical code fixes (B1–B3), 6 polish items
  (C), 9 paper-only edits (D). All addressed in commit 491f237.

Artifacts
  release/arxiv-submission.tar.gz
  release/pratyaksha-context-eng-harness-v2.0.0.zip
  release/SHA256SUMS"
```

## 2. Push the worktree branch + tag to your GitHub

The worktree currently lives on the `v2` branch. To publish under
`SharathSPhD/pratyaksha-context-eng-harness`:

```bash
# Option A — split the plugin into its own repo (recommended):
#   1. Create a new empty repo on github.com under SharathSPhD,
#      named `pratyaksha-context-eng-harness`, no README, no LICENSE.
#   2. Push only the plugin tree as the root of that repo.
git subtree push --prefix=plugin/pratyaksha-context-eng-harness \
    https://github.com/SharathSPhD/pratyaksha-context-eng-harness.git main

# Option B — keep everything in the parent context repo and just tag it.
git push origin v2 v2.0.0
```

Option A is what the paper's title page and the plugin's `marketplace.json`
already point to (`https://github.com/SharathSPhD/pratyaksha-context-eng-harness`),
so I recommend it. After Option A, also push the experiment harness + paper
sources to a sibling repo or as a subdirectory — those are not user-facing
plugin artefacts but are needed for paper reproducibility.

## 3. Create the GitHub release

```bash
gh release create v2.0.0 \
    --repo SharathSPhD/pratyaksha-context-eng-harness \
    --title "v2.0.0 — first public release" \
    --notes-file release/RELEASE_NOTES.md \
    release/arxiv-submission.tar.gz \
    release/pratyaksha-context-eng-harness-v2.0.0.zip \
    release/SHA256SUMS \
    paper/main.pdf
```

(Generate `release/RELEASE_NOTES.md` from the tag annotation if you want
GitHub-flavoured Markdown; the tag message is already in the right voice.)

## 4. Submit the paper to arXiv

1. Log in at https://arxiv.org/submit
2. Pick category: **cs.CL** (primary), with **cs.AI** and **cs.SE** as
   cross-lists.
3. Upload `release/arxiv-submission.tar.gz`.
4. arXiv will compile it with their TeX Live; the bundle has been verified
   to compile end-to-end with `tectonic -X compile main.tex`. If arXiv's
   pdflatex disagrees, the most likely culprit is `\usepackage{xcolor}`
   ordering vs `\usepackage{hyperref}` — `main.tex` already loads them in
   the correct (xcolor before hyperref) order.
5. Title, author, abstract are pre-filled in the metadata once arXiv
   parses `main.tex`. Verify them against `paper/sections/00_frontmatter.md`.
6. License: **CC-BY 4.0** (matches what the paper claims).

## 5. Submit the plugin to the official Anthropic marketplace (optional)

This step is optional; the plugin is fully usable from the SharathSPhD repo
above. If you want to attempt the official marketplace:

```bash
git clone https://github.com/anthropics/claude-plugins-official.git
cd claude-plugins-official
git checkout -b add-pratyaksha-context-eng-harness

# Drop in your marketplace entry (the one that already lives at
# plugin/pratyaksha-context-eng-harness/marketplace.json in your repo).
mkdir -p plugins/pratyaksha-context-eng-harness
cp -r path/to/your/repo/plugin/pratyaksha-context-eng-harness/* \
      plugins/pratyaksha-context-eng-harness/

git add plugins/pratyaksha-context-eng-harness
git commit -m "Add pratyaksha-context-eng-harness plugin v2.0.0"
gh pr create \
    --repo anthropics/claude-plugins-official \
    --title "Add pratyaksha-context-eng-harness v2.0.0" \
    --body "$(cat <<'EOF'
This PR submits the Pratyakṣa Context-Engineering Harness — a self-
contained Claude Code plugin that adds Avacchedaka-typed retrieval,
sublation-based conflict resolution, witness-consciousness invariants,
event-boundary compaction, and Khyātivāda hallucination classification.

15 MCP tools, 3 skills, 3 agents, 4 commands, 3 lifecycle hooks. No
runtime dependencies on attractor-flow or ralph-loop (those were
development-time orchestration tools, not part of the shipped plugin).

Full validation: 92-page preprint at https://arxiv.org/abs/<TBD>
Reproducibility manifest: paper Appendix C.
Test suite: 502 passing, 0 failing.
EOF
)"
```

If that PR is rejected for any reason, you have the SharathSPhD-hosted
copy as the canonical install path; the marketplace entry already
documents that as the source of truth.

## 6. Verify install after publishing

```bash
# In a fresh Claude Code session:
/plugin install pratyaksha-context-eng-harness@SharathSPhD
# or, if you used the official marketplace:
/plugin install pratyaksha-context-eng-harness

# Then trigger the smoke test:
/context-status
/budget
/compact-now
/sublate
```

All four commands should return non-error JSON.

## 7. Post-release cleanup (optional)

The dev-time worktree at `.worktrees/v2/` can be archived; everything that
mattered for the public release is now in:

* `pratyaksha-context-eng-harness` GitHub repo (plugin + paper sources)
* arXiv preprint
* `release/*.tar.gz` + `release/*.zip` (artifact backups)

The parent `context/` repo's `v0` lineage and the dev-time `triz-engine`,
`attractor-flow`, `ralph-loop` plugins remain untouched — they're
deliberately not part of v2.

---

If you want me to perform any of steps 1–7 for you, ask and I will
execute the commands inline (modulo: I can run `git tag` locally; I cannot
push to your GitHub or submit to arXiv without your credentials).
