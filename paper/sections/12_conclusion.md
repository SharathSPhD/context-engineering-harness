# 12 · Conclusions and Future Work

## 12.1 What we did

We presented **Pratyakṣa**, a context-engineering system for long-context, hallucination-resistant agentic AI, packaged as a Cursor- and Claude-Code-compatible plugin (15 MCP tools, 3 skills, 3 agents, 4 commands, 3 lifecycle hooks, MIT-licensed, installable in 30 seconds). The system operationalises seven constructs from classical Indian epistemology — *pratyakṣa, avacchedaka, bādha, buddhi/manas, sākṣī, khyātivāda, saṃskāra/vāsanā* — drawn from four schools (**Nyāya–Vaiśeṣika, Advaita Vedānta, Pūrva Mīmāṃsā, Sāṃkhya**), into runtime mechanisms on the agent's working context. We validated across three orthogonal evidence layers: seven preregistered hypotheses on six public long-context and hallucination benchmarks for L1 (paired multi-seed multi-model), a deterministic live case study on three real GitHub issues (L2), and a 720-pair head-to-head A/B on 120 SWE-bench Verified instances under a fixed token budget (L3). Combining the 10 quantitative studies via weighted Stouffer-Z gives a two-sided $p$ of **7.94 × 10⁻²⁰** in the system's favour, with mean per-study delta **+0.476** and Cohen's $d \approx \mathbf{1.0}$ on the most ecologically valid study. The Khyātivāda 6-class hallucination annotator achieves Cohen's $\kappa = \mathbf{0.736}$ ("substantial") on $n = 3{,}000$ examples.

## 12.2 What we claim

We claim the *modest* version of three things:

1. **Classical Indian epistemology supplies usable type signatures for context engineering.** *Avacchedaka* (Nyāya–Vaiśeṣika), *bādha* (Advaita Vedānta + Pūrva Mīmāṃsā), *buddhi*/*manas* (Sāṃkhya, cross-mapped in Advaita), *sākṣī* (Advaita Vedānta), and *khyātivāda* (cross-school) are not metaphors; they admit precise runtime operationalisations on an LLM context window, and those operationalisations measurably improve downstream agent behaviour across general agentic-AI workloads.

2. **A drop-in plugin is the right delivery vehicle.** No model fine-tune, no client patch, no infrastructure dependency heavier than `tiktoken` and `pydantic`. The system installs in 30 seconds onto Cursor, Claude Code (CLI / VS Code), or Claude Desktop and is hot-swappable across all four hosts because the only inter-process surface is MCP.

3. **The system improves performance under fixed token budgets**, *not* by buying more context but by using the same budget more disciplinedly. This is the orthogonal axis to the architectural-extensions literature \citep{chen2023pi, peng2023yarn, ding2024longrope, gu2023mamba, lieber2024jamba} and the compression-extensions literature \citep{jiang2023llmlingua, jiang2024longllmlingua, chevalier2023autocompressors}.

## 12.3 What we do *not* claim

We do *not* claim a new model architecture, a new pre-training regime, a new positional encoding, or a new attention kernel. We do *not* claim that the historical Sanskrit authors had LLMs in mind or that our operationalisations are textually faithful in a way a Sanskrit philologist would require. We do *not* claim our synthetic-fallback adapters reproduce the absolute numbers of published RULER / HELMET runs — only that the *paired delta in the system's favour* is a faithful estimate. We do *not* claim that the L3 PatchSimulator's gains will fully transfer to a real LLM coder; Section 11 names this as the cleanest near-term gap.

## 12.4 Future work

Eight follow-ups are well-defined enough to scope today:

1. **Cross-family LLM sweep.** Run the L1+L3 pipeline against GPT-4o-class, Qwen-3-72B, Llama-3.x. The harness should not change; the deltas may.

2. **Real-LLM-coder P6-C.** Replace the deterministic PatchSimulator with a real Claude/GPT/Qwen coder and remeasure target-path-hit-rate, full pass-rate, and SWE-bench Verified pass@1.

3. **Human-vs-human Khyātivāda IAA at n ≥ 1,000.** Promote the κ = 0.736 from automated-vs-automated to human-validated, with at least three annotators.

4. **Live-deployment telemetry.** Ship the plugin to a small pilot group of Cursor / Claude Code users and measure (consensually) the rate of `sublate_with_evidence` events, the post-Buddhi `khyati_class` distribution, and per-session token-budget gauge curves.

5. **Non-coding agentic surfaces at scale.** Although the L1 evidence already exercises general long-context and hallucination tasks, a systematic application to research-assistance, document QA, multi-tool orchestration, customer-service triage, and legal-document review agents — each with a per-domain analogue of "target-path-hit-rate" — is the cleanest cross-domain validation.

6. **Philological refinement.** Engage Sanskritists to refine the avacchedaka, bādha, and khyātivāda operationalizations. Some commitments — e.g. our 4-tuple representation of avacchedaka — likely under-respect the Navya-Nyāya tradition's richer relational logic \citep{matilal1968navyanyaya, ingalls1951materials, gangesa14tattvacintamani}.

7. **Cognitive-neuroscience cross-checks (aspirational).** The complementary-learning-systems prediction \citep{mcclelland1995cls, kumaran2016cls} is *in principle* testable — witness-protected items might show consolidation curves that differ from non-protected items — but we have not measured this and treat it as a long-horizon research programme, not a near-term deliverable.

8. **A shared runtime contract for context-engineering plugins (aspirational).** Following Cognition's *Don't Build Multi-Agents* \citep{yan2025dontbuild} and Anthropic's context-engineering position \citep{anthropic2025contexteng}, an *industry-wide* MCP-side schema for typed limitor + sublation + witness-tracking could let host platforms interoperate; we propose `pratyaksha-context-eng-harness` as a **candidate** reference point and welcome co-design, while acknowledging that standardisation is uncertain and politically contingent.

## 12.5 A closing methodological note

The work began with TRIZ (Section 3) — a single contradiction, a contradiction-matrix lookup, a chosen inventive principle (#10 Preliminary Action), and the recognition that the principle required a vocabulary the modern LLM literature did not have. That vocabulary was sitting, fully developed, in classical Indian epistemology. The harness is the operationalization of that recognition. We retain the audit trail — `docs/triz-origin-session.md`, `docs/v0_retrospective.md`, the `cost_ledger.db`, the plugin audit log at `~/.cache/pratyaksha/audit.jsonl` — because the kind of provenance we ask the *agent* to maintain is the kind of provenance we maintained for ourselves while building it. *Pratyakṣa is its own first user.*

The shipped plugin, the full reproducibility manifest, the 246-entry bibliography, the 13 figures, and the 7 tables are released at `github.com/SharathSPhD/pratyaksha-context-eng-harness` under MIT licence. We hope it is useful.
