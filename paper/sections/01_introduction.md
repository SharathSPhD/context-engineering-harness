# 1 · Introduction

## 1.1 The context-engineering moment

By late 2025 the LLM community converged on a once-fringe phrase. Tobi Lütke wrote on 18 June 2025 that "context engineering" had eclipsed "prompt engineering" as the load-bearing skill in LLM application work \citep{lutke2025contexteng}; Karpathy amplified this a week later \citep{karpathy2025contexteng}; Anthropic's engineering team published their own position piece — *Effective Context Engineering for AI Agents* — endorsing the same shift \citep{anthropic2025contexteng}; and Cognition AI, builders of Devin, opened the field's most-cited essay of the year — *Don't Build Multi-Agents* — with the line, "if there's one principle to take away [from this post], it's that **context engineering is everything**" \citep{yan2025dontbuild}. Across these four near-simultaneous statements, the term was not introduced as marketing; it named a discipline the field had been groping toward without a vocabulary.

The empirical urgency for that vocabulary was already overwhelming. *Lost in the Middle* showed that even strong models cannot retrieve facts placed mid-prompt \citep{liu2023lostmiddle}. *RULER* \citep{hsieh2024ruler}, *HELMET* \citep{yen2024helmet}, *NoCha* \citep{karpinska2024nocha}, *LongBench* \citep{bai2024longbench} and *SCBench* \citep{wu2024scbench} systematically demonstrated that the *advertised* context window of frontier LLMs — 200 K, 1 M, 2 M tokens — overstates the *effective* window, often by a full order of magnitude. Chroma's 2025 *Context Rot* report quantified the failure mode: as input length grew from 1 K to 100 K tokens, exact-match accuracy on a held-fixed needle task degraded by 30–80 percentage points across every model tested, even when the model had been benchmarked as 1 M-context-capable \citep{hong2025contextrot}. Hallucination remained the largest single source of agentic-coding failure on SWE-bench-class workloads \citep{jimenez2024swebench, openai2024sweverified}, and surveys of hallucination types \citep{ji2023hallusurvey, huang2024hallucinationsurvey} agreed on one point: the field had no agreed-upon *taxonomy* into which to classify and respond to a particular hallucination at runtime.

This is the moment at which *context engineering* needed not slogans but a typed object model.

## 1.2 The argument of this paper

We claim three things:

1. **Classical Indian (Vedic) epistemology already supplies the vocabulary** that modern LLM context engineering has been improvising. The Nyāya school's *avacchedaka* (qualifier-conditioned cognition) is precisely the typed limitor we want on every retrieved chunk; Advaita Vedānta's *bādha* (sublation) is precisely the supersede-with-provenance operation we want on KV-cache contents; the *antaḥkaraṇa* duality of *manas* (attentional sense-organ) and *buddhi* (judging intellect) maps cleanly onto the two-stage attend-then-judge agentic loop; *sākṣī* (the witness) supplies a stable, model-invariant reference frame for cross-turn provenance; and the six-class *khyātivāda* error taxonomy — *anyathākhyāti*, *ātmakhyāti*, *anirvacanīyakhyāti*, *asatkhyāti*, *viparītakhyāti*, *akhyāti* — gives us, for the first time, a *principled* hallucination ontology that distinguishes *misattribution* from *projection* from *unspecifiability* \citep{matilal1986perception, ganeri2001philosophy, phillips2012epistemology, ram2007error}.

2. **A small, drop-in plugin** — `pratyaksha-context-eng-harness` — can operationalize all of the above on top of *unmodified* frontier LLMs (Claude 3.5/4.x, GPT-4-class, Qwen-3) via the Model Context Protocol \citep{anthropic2025mcp}. The plugin ships 15 MCP tools, 3 skills, 3 sub-agents, 4 slash commands, and 3 lifecycle hooks; we install it into Claude Code and Cursor with a single `/plugin install` and observe end-to-end statistical improvements without changing the underlying model.

3. **The improvements are real, large, and reproducible.** Across 10 preregistered studies — H1–H7 on public long-context and hallucination benchmarks, plus a deterministic live case study (P6-B) and a 720-pair SWE-bench Verified A/B (P6-C) — the harness produces a Stouffer-combined two-sided p = **7.94 × 10⁻²⁰** in its favour, with mean per-study delta **+0.476**. On the L3 study (SWE-bench Verified *instance set*, synthetic research trails, fixed 8 K-token research budget with `--research-block-budget 8192`, paired by issue-id × model × seed, patch generation deterministically anchored on the first plausible file path of the research block), the harness anchors the stub patch on the correct file in **100%** of cases versus **50.3%** for the budgeted baseline; the optional Docker scorer agreement is reported on a 30-instance sub-sample (κ = 0.97).

## 1.3 Why now, and why a plugin

Three converging facts make this work tractable today:

- **Plugin surfaces have stabilised.** The Model Context Protocol \citep{anthropic2025mcp}, Claude Code's plugin format \citep{anthropic2025claudecode}, and Cursor's plugin marketplace \citep{cursor2025plugins} now share enough conventions that one tarball can ship to all three host environments. We exploit this aggressively: the plugin's `marketplace.json` resolves identically in Cursor, Claude Code (CLI and VS Code extension), and the Claude Desktop app, with no fork.

- **MCP makes "give the agent a typed memory" a one-tool addition.** Before MCP, every memory layer required a bespoke fork of the agent harness (MemGPT \citep{packer2023memgpt}, CoALA-style architectures \citep{sumers2024coala}, scratchpad add-ons \citep{rakhilin2025scratchpad}). With MCP, the harness ships as 15 server tools the agent can discover at runtime; no model fine-tune, no client patch.

- **The benchmark frontier is finally honest about long-context failure.** RULER, HELMET, NoCha, FACTS-Grounding, and SWE-bench *Verified* together cover the failure modes that prompt-only ablations cannot fix. We can therefore *measure* the harness's contribution against problems the field has agreed are hard.

Cognition's *Don't Build Multi-Agents* essay \citep{yan2025dontbuild} also clarified what *not* to build: an army of communicating sub-agents whose context overlap explodes super-linearly. Our harness is single-agent-by-default and deliberately *minimises* multi-agent surface area: Buddhi and Manas are *internal* gates on a single agent's reasoning trace, not separate networked agents. Sākṣī is a witness object, not a "supervisor agent". The plugin therefore stays on the right side of Cognition's principle: *share the full conversation history, don't fragment it across loose sub-agents.*

## 1.4 Contributions

1. **A Vedic-epistemology-to-context-engineering translation table** (Section 4) that gives precise, runtime-checkable definitions of *pratyakṣa, avacchedaka, bādha, buddhi/manas, sākṣī, khyātivāda*, and *vāsanā/saṃskāra*-driven adaptive forgetting in terms of LLM context-window operations. To our knowledge no prior work has attempted this synthesis at this granularity.

2. **An open-source plugin** — `pratyaksha-context-eng-harness` — packaged for Cursor / Claude Code / Claude Desktop, with 15 MCP tools, 3 skills, 3 agents, 4 commands, and 3 lifecycle hooks. The plugin ships **zero** dependencies on the development-time orchestration tools (`attractor-flow`, `ralph-loop`, `triz-engine`) we used to *build* it; the shipped artefact is auditable to be free of those imports (Section 6.5, Appendix B).

3. **A 10-study validation portfolio** with paired permutation tests, bootstrap CIs, Cohen's d, and a Stouffer-Z omnibus statistic — including the live case study and SWE-bench Verified A/B that prior context-engineering position papers conspicuously lacked.

4. **A 6-class Khyātivāda annotation codebook** with two-rater agreement (heuristic + LLM-as-judge) at Cohen's κ = **0.736** on n = 3,000 examples, plus the released annotation pipeline (Appendix E).

5. **A reproducible cost-aware build pipeline** (Section 7.1, Appendix C) that drove the entire 10-study validation under hard CLI rate limits via a custom budget scheduler with disk caching, prompt caching, and 5-hour-window-aware exponential backoff. The total measured CLI budget consumed was logged to `cost_ledger.db` and is included in the reproducibility manifest.

6. **A documented origin story** (Section 3) showing how this work emerged from a TRIZ-style contradiction analysis of context engineering itself, with the `triz-engine` plugin's `log_session_entry` tool used as the audit trail. For this paper we treat that session as a *method note* (contradiction → inventive principle → vocabulary gap) rather than as a claim that TRIZ itself generalises the contribution.

## 1.5 Roadmap

Section 2 surveys the four relevant literatures (long-context LLMs, RAG/memory, agentic coding, hallucination/calibration). Section 3 narrates how a TRIZ contradiction analysis pointed us to Vedic epistemology. Section 4 builds the translation table from Sanskrit philosophical terms to LLM operations. Section 5 describes the harness architecture. Section 6 specifies the shipped plugin. Section 7 gives the validation methodology and our CLI-budget infrastructure. Sections 8, 9, and 10 present results from L1 (public benchmarks), L2 (live case study), and L3 (SWE-bench Verified A/B) respectively. Section 11 discusses limitations and threats to validity. Section 12 concludes and points to future work. Six appendices contain the glossary, MCP tool reference, statistical details, adapter details, the Khyātivāda codebook, and the reproducibility manifest.
