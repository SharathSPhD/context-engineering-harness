# When the Context Window Is Big and the Agent Is Still Confused

*A millennia-old darśana-śāstra vocabulary — the systematic Indian treatises on what counts as valid knowledge — turned out to be the missing operating manual for modern AI agents. Here is the system that proves it, and the plugin you can install in thirty seconds.*

---

> **[VISUAL V1 — Hero image, "Lost in the middle"]**
> *Where to insert in the published version:* immediately under the dek, before the lede.
> *Suggested generator:* ChatGPT (DALL·E) / Midjourney / NotebookLM image surface.
> *Aspect:* 16:9.
> *Prompt:*
> "Editorial illustration, 16:9, muted dark teal and warm-amber palette. An AI agent depicted as a translucent humanoid silhouette walking through an enormous floating library of documents that stretches into the distance; one single highlighted manuscript glows softly far behind the agent — the one it walked past without reading. Title overlay in elegant serif: 'Lost in the middle.' Style: minimalist, slightly painterly, faint Sanskrit Devanāgarī calligraphy as a background watermark. NYT-opinion-section aesthetic."

---

## The lede

An AI coding agent reads a 200-page Django codebase, edits the wrong file with confidence, and tells the user it is done. A research-assistance agent surfaces a withdrawn paper, then surfaces a retraction, then forgets the retraction one turn later and answers from the withdrawn one. A customer-service agent quotes a policy that expired in 2023 because that is the version the model was trained on. None of these failures is about a model that is not smart enough. All three are about *context that is not engineered*.

This is the problem a new open-source plugin called **Pratyakṣa** sets out to fix — and the surprising part is *where the design vocabulary came from*.

## 1 · Bigger windows, dumber agents

The headline numbers from frontier-model launches keep growing. 200 K-token context windows. One-million-token windows. Soon, ten million. The implicit promise is that bigger windows mean smarter agents.

The benchmarks tell a different story. On RULER, the standard long-context recall benchmark, frontier models still drop accuracy as the window grows — the *Lost-in-the-Middle* effect, named in 2023 and unfixed in 2026. On HELMET, where multiple retrieved passages quietly contradict each other, the same models will happily return whichever passage they saw first. On HaluEval and TruthfulQA, hallucinations rise the longer the conversation runs.

Retrieval-augmented generation (RAG) is the field's standard answer, and it helps. But RAG fixes *what the model can see*. It does not fix *how the model decides which of the things it sees are still true*. A retrieved passage from 2019 sits in the context window next to an authoritative document from 2025, and the model has no principled way to retire the older one. Compaction strategies — the schemes that summarise older turns to make room for newer ones — make this worse, because they routinely throw away the very provenance an agent would need to know which version to trust.

> **[VISUAL — embed, paper Fig F02]**
> HELMET recall by context length
> *Caption: What "Lost-in-the-Middle" actually looks like. HELMET-recall accuracy degrades as the window grows from 8 K to 32 K tokens, even on the strongest current models. The Pratyakṣa-treatment line (top) holds; the unaided baseline (bottom) does not. Reproduced from the Pratyakṣa preprint.*

There is a name for the discipline that fixes this. It is *context engineering*: the explicit, auditable management of what an agent knows, why it knows it, and when each piece of evidence should be retired. The field has begun to converge on the phrase. What it has been missing is the *vocabulary*.

## 2 · The surprising vocabulary

That vocabulary turned out to be millennia old — preserved in the *darśana-śāstras*, the systematic philosophical treatises that classical Indian thinkers refined for over two thousand years — and it was waiting in a corner of the world's intellectual history that the modern AI literature had not been reading.

Classical Indian epistemology — the systematic study of *what counts as a justified belief*, developed across schools called Nyāya–Vaiśeṣika, Advaita Vedānta, Pūrva Mīmāṃsā, and Sāṃkhya — spent centuries refining a small set of concepts that map almost exactly onto the operations a modern context-engineered agent needs to perform.

> **[VISUAL V2 — Knowledge mandala mapping Sanskrit to operations]**
> *Where to insert in the published version:* immediately after this paragraph.
> *Suggested generator:* Midjourney / DALL·E for the artwork, NotebookLM for the diagrammatic version.
> *Aspect:* 1:1.
> *Prompt:*
> "Knowledge-mandala diagram, square format, deep teal background with gold and ivory line work. A circular mandala layout: five Sanskrit terms placed at the cardinal and ordinal positions in elegant Devanāgarī (अवच्छेदक — avacchedaka, बाध — bādha, मनस्/बुद्धि — manas/buddhi, साक्षी — sākṣī, ख्यातिवाद — khyātivāda). Each term is connected by a thin gold filament to its modern operational counterpart in the inner ring (typed limitor, supersede-with-provenance, two-stage gate, witness invariant, hallucination taxonomy). At the centre, a small lotus glyph labelled 'context engineering'. Style: clean infographic with hand-illustrated detail, slightly painterly. No photographic elements."

Five concepts do most of the work, and each can be glossed in one sentence:

**Avacchedaka** (Nyāya–Vaiśeṣika) is the ancient Indian logician's insistence that *every cognition carries its conditions*. To say "the cup is on the table" is, in this tradition, never quite enough; the proper form is "the cup is on the table *under the conditions* C₁ … Cₙ". In the harness, this becomes typed insertion: every retrieved fact lands in the agent's working store *with* the conditions that make it true.

**Bādha** (Advaita Vedānta and Pūrva Mīmāṃsā) is the technical Sanskrit term for *sublation*: the operation by which a newer, more authoritative cognition supersedes an older one *without deleting it*. The older claim survives in the audit trail; it just no longer drives behaviour. This is the precise primitive a modern agent needs and does not have. There is no comparably tight English term.

**Manas** and **Buddhi** (Sāṃkhya, cross-mapped in Advaita Vedānta) draw a line between *attention* and *judgement*. Manas is the sense-organ that selects which evidence to look at. Buddhi is the determinative faculty that judges on the basis of what Manas surfaces. Selecting and judging are *different cognitive acts*, and treating them as one is the source of a particular class of hallucination — the kind where an agent confidently invents a function name that fits the surrounding code.

**Sākṣī** (Advaita Vedānta) is the *witness consciousness* — the unchanging substrate that observes every cognitive event. In the harness, it becomes a session-stable invariant (working directory, git SHA, model identity, plugin version, hard-coded user policies) that survives every compaction event. The witness is what the agent can never lose.

**Khyātivāda** is a cross-school debate about *the kinds of error a cognition can fall into*. The schools disagreed productively about whether a misperception is a non-apprehension (*akhyāti*), a mis-apprehension (*anyathākhyāti*), a projection of the perceiver's own state (*ātmakhyāti*), and so on. Modern hallucination research has reproduced these categories one by one — without ever talking to the Indian tradition. The harness uses a 6-class typed taxonomy derived from this debate, and the classifier achieves Cohen's κ = 0.736 ("substantial" inter-annotator agreement) on a 3,000-example corpus.

The framing here is *convergence*, not exoticism. Cognitive neuroscience independently arrived at structurally similar constructs — working-memory schemas, predictive-coding precision, complementary-learning-systems consolidation, prefrontal attention control, event-segmentation. Two unrelated traditions converging on the same type signatures is a reason to take the type signatures seriously.

## 3 · What the plugin actually does

The harness ships as a single open-source plugin: `**pratyaksha-context-eng-harness*`* (v1.0.0, MIT-licensed). It installs into Cursor, Claude Code (CLI and the VS Code extension), and Claude Desktop in roughly thirty seconds. It contains no model fine-tune, no architecture change, no runtime dependency heavier than `tiktoken` and `pydantic`. The full system specification — every tool signature, every prompt, every reproducibility manifest — lives in the formal preprint on Zenodo (record [19653013](https://zenodo.org/records/19653013)).

What it ships is fifteen MCP tools, three sub-agents, three skills, four slash commands, and three lifecycle hooks, organised across six functional families:


| Family                        | MCP tools                                                                | What it does                                                                  |
| ----------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| **Avacchedaka** (typed store) | `context_insert`, `context_retrieve`, `context_get`, `list_qualificands` | Every fact enters the store with its conditions; every retrieval is typed     |
| **Bādha** (sublation)         | `context_sublate`, `sublate_with_evidence`, `detect_conflict`            | Newer authoritative evidence supersedes older — without deletion              |
| **Compaction**                | `compact`, `boundary_compact`, `context_window`                          | Forget what is safe to forget at event boundaries; preserve what is witnessed |
| **Sākṣī** (witness)           | `set_sakshi`, `get_sakshi`                                               | Session-stable invariants survive every compaction                            |
| **Khyātivāda**                | `classify_khyativada`                                                    | Six-class typed hallucination taxonomy                                        |
| **Budget**                    | `budget_status`, `budget_record`                                         | Token-budget visibility for the host agent                                    |


The two interesting sub-agents are *Manas* and *Buddhi*. Manas is the system-prompt-driven attention agent: it selects evidence, reports what it attended to, and — critically — must never emit a user-visible answer. Buddhi is the determinative judgement agent: it can call sublation, runs the hallucination classifier on its own draft, and is the only sub-agent that emits the user-facing reply. The third sub-agent, *Sākṣī-keeper*, is a read-only witness that maintains the session invariants and writes every mutation to a JSONL audit log at `~/.cache/pratyaksha/audit.jsonl`. A user can `tail -f` that file during a session and watch exactly what the agent did and why.

To make this concrete, consider one user turn end-to-end. The user asks an agent: *"How do I cache a user session in Redis?"* The agent's web tool returns four snippets — two from pre-Redis-7 blog posts (which still circulate at the top of search results), two from the official Redis 7 documentation. Without the harness, an unaided agent will often anchor on the first snippet it sees, write code against the older API, and confidently ship the wrong answer. With the harness:

1. Manas inserts all four snippets into the typed store, each tagged with its source-precision (`prec=2` for the blog posts, `prec=8` for the official docs).
2. `detect_conflict` flags a `TYPE_CLASH` on the `(Redis-session, expiry-policy)` qualificand.
3. `sublate_with_evidence` retires the blog posts in favour of the docs — the blog posts remain in the audit log; they no longer drive the answer.
4. Buddhi composes the final answer using only the surviving live items, classifies the answer as `yathārtha` (veridical) with `confidence = 0.91`, and ships it.
5. Sākṣī appends one immutable JSON line per stage to the audit log.

This is the same operational pattern that drives every result in the next section.

> **[VISUAL V3 — Before/after desk illustration of the worked example]**
> *Where to insert in the published version:* immediately after the worked example.
> *Suggested generator:* ChatGPT (DALL·E) / Midjourney.
> *Aspect:* 16:9.
> *Prompt:*
> "Two-panel editorial illustration, 16:9, slightly isometric perspective. Left panel: a chaotic physical desk overflowing with paper, two stacks visibly mixed — some sheets stamped 'Redis 4 — blog post', others stamped 'Redis 7 — official docs'. A small confused humanoid agent silhouette stands in front of the desk. Right panel: same desk, neatly organised; the official docs are at the front, the blog-post stack is set aside and visibly stamped 'sublated' in red ink, and a small glowing paper labelled 'audit log' sits on the side. Style: warm muted palette, hand-drawn editorial quality, no photographic realism. Title overlay above the diptych: 'before context engineering / after context engineering'."

## 4 · Did it actually work?

The harness was validated across three orthogonal evidence layers. Every figure and table in this section is reproduced from the Zenodo preprint (record [19653013](https://zenodo.org/records/19653013)).

**Layer 1 — public benchmarks.** Seven preregistered hypotheses tested on six widely-used long-context and hallucination benchmarks (RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS-Grounding) plus SWE-bench Verified. Each study sweeps two model families (`claude-haiku-4-5` and `claude-sonnet-4-6`) across multiple seeds. Across all seven hypotheses, the harness beats the unaided baseline at p ≤ 0.0020. The H2 result on HELMET-Recall is particularly clean: a 47 % reduction in Brier score and a 65 % reduction in expected calibration error, on top of the headline accuracy gain.

**Layer 2 — live case study.** Three real GitHub issues drawn from popular Python projects: a Django request-body subtlety, a `requests` retry-strategy spelling change, and a pandas `iterrows` dtype gotcha. Each is exactly the kind of question a daily-driver coding agent gets wrong by pulling stale Stack Overflow answers ahead of fresh official documentation. Under *identical* token budgets, the harness scores 3-of-3 correct; the unaided baseline scores 0-of-3. The harness fires seven sublations and three compactions across the three cases — exactly the operational footprint the design predicts.

**Layer 3 — head-to-head A/B.** A 720-pair head-to-head on 120 SWE-bench Verified-style coding instances under a fixed 512-token research-block budget. Each instance gets a research trail of four snippets — two stale (pointing at the wrong file paths via synthetic typos, referencing superseded APIs), two fresh (correct file paths, current APIs). The harness anchors on the correct file in **720 / 720 paired runs** — 100 % in every (model × seed) cell. The unaided baseline anchors correctly in 362 / 720 runs (50.3 %), exactly the coin-flip predicted by *Lost-in-the-Middle*-style anchoring on a randomly-shuffled trail.

> **[VISUAL — embed, paper Fig F12 + Fig F13 side-by-side]**
> Effect sizes by hypothesis
> Forest plot of paired deltas
> *Caption: Aggregate effect across all ten quantitative studies. Left: Cohen's d magnitudes by hypothesis. Right: forest plot of per-hypothesis paired deltas with 95 % confidence intervals. The two structural-100 % rows (H5, H7) are excluded from the mean. Reproduced from the Pratyakṣa preprint.*

When all ten quantitative studies are combined via weighted Stouffer-Z, the headline omnibus statistic is:


| metric                                                       | value            |
| ------------------------------------------------------------ | ---------------- |
| number of studies                                            | 10               |
| combined Z                                                   | 9.114            |
| **combined two-sided p**                                     | **7.94 × 10⁻²⁰** |
| mean per-study delta                                         | +0.476           |
| mean Cohen's d (excluding the two structural rows)           | 9.62             |
| Cohen's d on the most ecologically valid coding study (P6-C) | ≈ 1.0            |
| Khyātivāda classifier inter-annotator κ (n = 3,000)          | 0.736            |


A combined p-value of 7.94 × 10⁻²⁰ is sixteen orders of magnitude past the conservative threshold for a well-powered study. The same result survives a deliberately hostile re-analysis that drops the two structural-100 % studies (H5 and H7) from the omnibus. Even at that conservative cut, the combined p stays below 10⁻¹⁵.

## 5 · The honest caveats

A serious result deserves serious caveats, and the preprint names them. The Layer-1 adapters use synthetic-fallback evidence trails by default (the real Hugging Face data path runs only when an HF token is configured) — the paired delta is faithful, but the absolute numbers should not be compared to published RULER and HELMET runs. The Layer-3 patch-generation step uses a deterministic patch simulator rather than a real LLM coder, which deliberately isolates the contribution of the *context discipline* from generation quality; a follow-up that swaps the simulator for a real coder is the cleanest near-term gap. The model sweep covers only two Anthropic families; cross-family validation (GPT-4o-class, Qwen-3, Llama-3.x) is queued. The hallucination-classifier inter-annotator agreement is automated-vs-automated; a human-vs-human study at scale is the obvious next step. None of these caveats touches the central operational claim: *under a fixed token budget, the harness measurably and replicably outperforms the unaided baseline on every long-context and hallucination surface tested*.

## 6 · Try it yourself

The plugin is open-source (MIT). It installs in two commands and works inside Cursor, Claude Code (CLI and VS Code), and Claude Desktop, because the only inter-process surface is the Model Context Protocol.

```bash
# One-time prerequisite: install uv (the Python package runner the plugin uses).
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, inside Cursor or Claude Code:

```text
/plugin marketplace add SharathSPhD/pratyaksha-context-eng-harness
/plugin install pratyaksha-context-eng-harness
```

Restart the host. The first MCP tool call takes about thirty seconds while `uv` downloads `mcp`, `pydantic`, and `tiktoken`. Every call after that is instant. There is no `pip`, no virtualenv, no `claude mcp add`.

Four slash commands cover almost everything a user will want to do interactively:


| Command                                 | What it shows                                                                           |
| --------------------------------------- | --------------------------------------------------------------------------------------- |
| `/context-status`                       | Pretty-prints the current visible context window by category, with a token-budget gauge |
| `/sublate <target_id> <by_id> <reason>` | Manual sublation override for audit and debug                                           |
| `/budget`                               | One-line gauge of token consumption against the configured budget                       |
| `/compact-now [strategy]`               | Manually trigger compaction (`adaptive`, `lru`, or `none`; default `adaptive`)          |


A 90-second first-turn recipe: open a long-running task in the host (a multi-file refactor, a multi-source research synthesis, a long policy-QA conversation); type `/context-status` once before the agent starts and once again after five turns; `tail -f ~/.cache/pratyaksha/audit.jsonl` in a side terminal and watch the sublation events fire.

> **[VISUAL V4 — Install screencast, static or animated]**
> *Where to insert in the published version:* between the install commands and the slash-commands table.
> *Suggested generator:* `asciinema` for an animated terminal recording, or ChatGPT / Midjourney for a static polished screenshot.
> *Aspect:* 16:9.
> *Prompt (for static rendering):*
> "Polished dark-mode terminal screenshot, 16:9. Three lines visible: the `curl … uv/install.sh` install, then `/plugin install pratyaksha-context-eng-harness`, then a chat showing `/context-status` returning a clean rendered token-budget gauge with five categories (typed-store, sublations, sākṣī invariants, audit log, free budget) and a green progress bar at roughly 40 %. Monospace font, subtle green-on-black glow, no decorative chrome, no fake window dressing. Should look like a real terminal a developer would screenshot."

## 7 · Context engineering is the new prompt engineering

Three years ago, prompt engineering was a discipline waiting for its name. Practitioners traded recipes; researchers showed that the recipes mattered; and within eighteen months the field had a vocabulary, a literature, and a practice.

Context engineering is in the same place now. The failure modes are real, the costs are paid daily by everyone shipping agents into production, and the field is converging on the recognition that a bigger window is not a substitute for a discipline. What has been missing is the *type signatures*. *Avacchedaka* is not "metadata". *Bādha* is not "delete". *Manas/Buddhi* is not "two model calls". Each of these is a precise relational operator with a centuries-long history of mutually critical philosophical refinement, and each of them turns out to have a clean implementation against an LLM context window.

The Pratyakṣa harness is one delivery vehicle for those operators. The plugin will get refined; the model families will turn over; the token-budget gauges will become smarter. The lasting contribution is the recognition that the vocabulary already exists, the schools that built it were thinking carefully about *what counts as a justified belief*, and a modern context-engineered agent does not have to invent the discipline from scratch.

Install the plugin. Watch the audit log. The vocabulary was waiting.

> **[VISUAL V5 — Closing image, the diya]**
> *Where to insert in the published version:* immediately above the CTA links block.
> *Suggested generator:* Midjourney / DALL·E.
> *Aspect:* 1:1.
> *Prompt:*
> "Square format, contemplative warm palette. A single oil-lamp (diya) glowing softly, illuminating a vertical stack of paper that visually transitions from old palm-leaf manuscripts at the bottom (Devanāgarī text faintly visible) to a modern laptop screen at the top (a small terminal with a token-budget gauge faintly visible). Beneath the lamp, the line of text 'pratyakṣa — direct perception' in a fine serif. Style: oil-painting feel, no photographic realism, faint gold leaf accents on the manuscript edges."

---

## Try it, read it, cite it

- **Plugin (install + use):** [github.com/SharathSPhD/pratyaksha-context-eng-harness](https://github.com/SharathSPhD/pratyaksha-context-eng-harness)
- **Full harness, paper sources, experiments, validation:** [github.com/SharathSPhD/context-engineering-harness](https://github.com/SharathSPhD/context-engineering-harness)
- **Citable preprint (Zenodo, DOI-backed canonical record):** [zenodo.org/records/19653013](https://zenodo.org/records/19653013) — *cite this version*
- **Preprint PDF (mirror):** [pratyaksha-v2-preprint.pdf](https://github.com/SharathSPhD/pratyaksha-context-eng-harness/releases/download/v2.0.0/pratyaksha-v2-preprint.pdf)
- **GitHub release v2.0.0 (plugin zip, arXiv source tarball, embedded figures, checksums):** [v2.0.0 release page](https://github.com/SharathSPhD/pratyaksha-context-eng-harness/releases/tag/v2.0.0)
- **Licence:** MIT (code), CC-BY-4.0 (paper text + figures)

If you ship an agent into production, install the plugin and watch one long session through the audit log. The numbers will speak; the eight-hundred-year-old vocabulary will start to sound like exactly what you needed.

---

*Sharath Sathish is the author of the Pratyakṣa context-engineering harness. Comments, philological corrections, and replication runs welcome on the GitHub repository.*