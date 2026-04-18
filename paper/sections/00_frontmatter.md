# The Pratyakṣa Context-Engineering Harness: A Vedic-Epistemology-Grounded Plugin for Long-Context, Hallucination-Resistant LLM Coding Agents

**Sharath Phulari**
Independent Researcher
`SharathSPhD@github`

**Preprint version:** 1.0 — 2026-04-18
**Code:** `https://github.com/SharathSPhD/pratyaksha-context-eng-harness`
**Plugin marketplace:** `https://github.com/SharathSPhD/pratyaksha-context-eng-harness/blob/main/.claude-plugin/marketplace.json`
**Reproducibility manifest:** `experiments/results/p7/_index.json`, `experiments/results/p7/_summary.md`

## Abstract

We present **Pratyakṣa**, a context-engineering harness for long-context, hallucination-resistant Large Language Model (LLM) coding agents, packaged as a Cursor- and Claude-Code-compatible plugin (15 Model Context Protocol [MCP] tools, 3 skills, 3 agents, 4 commands, 3 lifecycle hooks). The harness operationalizes seven constructs drawn from classical Indian (Vedic) epistemology — *pratyakṣa* (direct perception), *avacchedaka* (typed limitor conditions), *bādha* (sublation), *buddhi/manas* (judging vs. attending faculties), *sākṣī* (witness), *khyātivāda* (a six-class taxonomy of cognitive error), and *adaptive forgetting* (saṃskāra/vāsanā pruning) — into runtime mechanisms for an LLM agent's working context. We validate the harness across three orthogonal evidence layers: **(L1)** seven preregistered hypotheses (H1–H7) on five public benchmarks (RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS-Grounding) and SWE-bench Verified, with multi-seed multi-model paired permutation tests; **(L2)** a deterministic, reproducible live case study (P6-B) on three real GitHub issues spanning Django, Requests, and pandas; and **(L3)** a head-to-head A/B test on 120 SWE-bench Verified instances (P6-C, 720 paired runs across 2 models × 3 seeds × 120 issues), under a fixed 8 K-token research-block budget. Across **10 quantitative studies**, the harness produces a Stouffer-combined Z = **9.114** (two-sided p = **7.94 × 10⁻²⁰**), with mean per-study delta **+0.476** in the harness's favour and 100% target-path-hit rate on SWE-bench Verified versus 50.3% for the budgeted baseline. The Khyātivāda 6-class hallucination annotator achieves Cohen's κ = **0.736** ("substantial") on n = 3,000 jointly annotated examples. We argue that the harness's distinctive contribution is *not* a new model architecture but a **typed, witness-tracked, sublation-aware context discipline** that any LLM-based agent can adopt today via a drop-in plugin, and that classical Indian epistemology supplies exactly the vocabulary modern agent frameworks have been groping toward. The harness, the plugin, and the full reproducibility manifest are open-sourced.

**Keywords:** context engineering, long-context LLMs, retrieval-augmented generation, hallucination, Vedic epistemology, Nyāya, Advaita Vedānta, sublation, agentic coding, SWE-bench Verified, MCP plugin, Claude Code, Cursor.

---

## Plain-language summary (1 paragraph)

LLM coding agents fail in predictable ways when their context window grows: they latch onto stale information, ignore newer evidence in the middle of the prompt, and confidently emit hallucinations whose error type they cannot name. We propose a **plugin** — `pratyaksha-context-eng-harness` — that gives the agent a small, principled set of operations on its own context: every claim is stamped with *who said it, under what conditions, and with what precision*; newer authoritative claims **sublate** older ones (in the technical Vedānta sense, not "delete" but "supersede with provenance"); a **two-stage** Buddhi/Manas reasoning gate forces the model to *justify* what it attended to before answering; and a **six-class** error taxonomy (drawn from classical Indian *khyātivāda*) lets us *diagnose*, not merely count, hallucinations. Across 10 large studies the plugin beats the unaided baseline at p < 10⁻¹⁹.
