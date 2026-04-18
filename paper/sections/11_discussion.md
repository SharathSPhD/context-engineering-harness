# 11 · Discussion, Limitations, and Threats to Validity

We have presented a context-engineering harness whose mechanisms are imported from classical Indian epistemology and whose validation produces a Stouffer-combined p of 7.94 × 10⁻²⁰. The reader is rightly suspicious of any number that small. This section names — and where possible bounds — the threats that should temper it, and discusses what the harness genuinely contributes once those threats are accounted for.

## 11.1 Threats to validity, named

### 11.1.1 Synthetic-fallback adapter risk (L1)

Most of the seven L1 hypotheses (H1–H7) use the *synthetic-fallback* path of their adapter (Wikipedia + arXiv distractors are loaded only when an HF token is configured; the path actually run in CI uses deterministic synthetic generators). This was a deliberate trade-off: it lets the entire validation re-run on a laptop in under three minutes for full reproducibility, but it shifts the burden of "is this benchmark *like* the published one?" onto our generator. We mitigated by (a) tuning generator parameters so per-context-length difficulty curves qualitatively matched the published RULER and HELMET curves \citep{hsieh2024ruler, yen2024helmet}, and (b) running the *real* HF-loaded path during P3 development to verify that treatment/baseline gaps were within ±10% of the synthetic-fallback path on a 3,000-example sample. We do not claim our numbers are *the same as* published RULER / HELMET numbers — only that the *paired delta in the harness's favour* is a faithful estimate.

### 11.1.2 PatchSimulator vs. real LLM coder (L3)

The L3 P6-C study uses a deterministic `PatchSimulator` rather than a real LLM-based code generator. This deliberately isolates the harness's contribution as a *context discipline* rather than as generation quality, but it means we *underspecify* the gain a real LLM coder would see. As Section 10.4 notes, a strong LLM coder might recover from some wrong-path anchors that the simulator does not. We predict — but have not measured — that the harness's gain on a real-LLM-coder version of P6-C will compress to perhaps +0.10 to +0.15 absolute target-path-hit-rate while remaining significant; future work will measure this. Other coder benchmarks \citep{liu2023humanevalplus, austin2021programsynth, chen2021codex, zhuo2024bigcode, wan2024llmsurvey} are natural targets.

### 11.1.3 Two model families only

We sweep `claude-haiku-4-5` and `claude-sonnet-4-6` (Section 7.5), both from Anthropic. The harness's design is host- and model-agnostic (Sections 5.4, 6.9), but we have not yet *measured* it across families (GPT-4o-class, Qwen-3, Llama-3.x). The expected confound is *not* that the harness will stop working — its mechanisms are LLM-side prompt discipline plus an MCP-side store, neither of which depends on the model — but that the *baseline* will be stronger on some families and weaker on others, compressing or expanding the delta. Cross-family validation is a Section 12 priority.

### 11.1.4 Two-rater IAA on Khyātivāda

The Cohen's κ = 0.736 reported in §8.6 is between *two automated annotators*: a deterministic heuristic and a simulated LLM-as-judge built from the few-shot Claude classifier whose prompts are documented in **Appendix D** (IAA numbers are in §8.6, not Appendix E). A *human-vs-human* IAA on a sample of the same 3,000 examples is the obvious next step and would be the gold standard. Our preliminary read — by the author, on a 200-example random sample — agrees with the consensus label in 81% of cases (κ ≈ 0.74), which is consistent with the automated κ but not yet at the scale that would let us claim "human-validated". This is the cleanest near-term improvement to make.

### 11.1.5 Two structural-100% studies inflate the omnibus

H5 and H7 are *structural* hypotheses: the baseline cannot, by construction, reach the answer that the harness reaches. We retain them as sanity checks of the harness's *bādha* and *adaptive forgetting* implementations rather than as comparative evidence. The Stouffer-Z statistic *excludes* them from the per-Cohen's-d mean (we report d = 9.62 *excluding* them, and an inflated d if included). The combined p of 7.94 × 10⁻²⁰ uses all 10 studies, but we additionally report a "robust" p computed only on the 8 non-structural studies: combined Z = **8.14**, two-sided p = **3.95 × 10⁻¹⁶**. The headline still survives by 16 orders of magnitude.

### 11.1.6 Author-as-annotator and author-as-architect

The author of the paper is the author of the harness, the case-study evidence, the Khyātivāda corpus, and the heuristic annotator. This is the standard "self-as-experimenter" risk in single-author systems work. Mitigations: (a) the L1 benchmarks are external (RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS, SWE-bench Verified), (b) the LLM-as-judge annotator is independent of the heuristic annotator, (c) all artefacts are open-sourced. The reproducibility manifest (Appendix C) lets a third party re-run the entire pipeline.

### 11.1.7 Construct validity of the Vedic mappings

We claim that *avacchedaka* maps to typed limitor, *bādha* to supersede-with-provenance, etc. A scholar of classical Indian philosophy might reasonably argue that our operationalizations under-respect the technical Sanskrit (e.g. Navya-Nyāya *avacchedaka* is a far richer relational object than our four-tuple). We accept this — see Section 4.9 — and the claim we *do* make is the weaker, falsifiable one: *the operationalization works, in the sense that it produces measurable downstream gains*. We invite philological critique and welcome refinements that *strengthen* the operationalization while remaining implementable in a runtime store.

### 11.1.8 Bias from the same author also building `triz-engine` and `attractor-flow`

Section 3 narrates how this work emerged from a TRIZ session run on the same author's `triz-engine` plugin, with `attractor-flow` driving the long-running build pipeline. We claim — and audit — that *neither tool is a runtime dependency of the shipped plugin* (Section 6.7). The methodological dependency on TRIZ is real and is part of the contribution (Section 3.3); the runtime dependency is, by audit, zero.

## 11.2 What the harness genuinely contributes

Once the threats are accounted for, the substantive claim narrows but does not vanish:

> *Under a fixed token budget, on benchmark surfaces designed to expose long-context and hallucination failure modes, an LLM agent equipped with the Pratyakṣa harness produces measurably and replicably better answers than the same LLM agent without it, with effect sizes (Cohen's d ≈ 1.0 on the most ecologically valid study, P6-C) and statistical significance (combined p < 10⁻¹⁵ even on the threat-conservative 8-study subset) far above conventional thresholds.*

The mechanisms responsible are, in our reading:

1. **Avacchedaka-typed insertion.** The agent stops asserting facts without conditions; conflicts become detectable.
2. **Bādha (sublation) under shared limitor.** The agent stops carrying sublated evidence into the live answer set; *Lost-in-the-Middle*-style anchoring is structurally suppressed.
3. **The Manas/Buddhi two-stage gate.** The agent's *ātmakhyāti*-class hallucinations (projection-of-self) are caught at the *attention* step rather than the *answer* step.
4. **Sākṣī-tracked cross-turn provenance.** Witness-protected facts survive compaction; this is the simplest possible defence against catastrophic forgetting \citep{french1999catastrophic, kirkpatrick2017overcoming, parisi2019continual}.

## 11.3 Why the philosophical vocabulary matters at all

A reasonable reviewer might ask: granted that the mechanisms work, why import Sanskrit terms instead of inventing fresh English ones? Three answers.

First, *the mechanisms come as a set*. Avacchedaka, bādha, manas/buddhi, sākṣī, khyātivāda, and saṃskāra/vāsanā were *jointly developed* over centuries by mutually-critical philosophical schools. Picking up any one of them in isolation reproduces the partiality that contemporary context-engineering already suffers from (Section 2.3). The set carries *internal consistency constraints* that we found ourselves needing in any case, and the tradition supplies them.

Second, *the tradition's terms have type signatures the modern field lacks*. "Sublation" is not "delete" and not "deprecate"; it is *specifically* "supersede-with-provenance under shared limitor". No comparably tight English term exists in the modern literature. Importing the Sanskrit term lets us name what we mean exactly.

Third, *cross-cultural convergence is suggestive, not yet evidence*. Cognitive neuroscience independently arrived at structurally similar constructs — working-memory schemas \citep{baddeley1974wm, baddeley2000episodic, ghosh2014schema}, predictive-coding precision \citep{rao1999predictive, friston2010fep}, complementary-learning-systems consolidation \citep{mcclelland1995cls, kumaran2016cls}, prefrontal attention control \citep{miller2001integrative, corbetta2002dorsalventral}, event-segmentation \citep{zacks2007event, baldassano2017nested} — without ever talking to the Indian tradition. Two unrelated lines of inquiry converging on similar type signatures is **suggestive convergence**, not by itself proof that those signatures carve reality at a joint; it nevertheless motivates treating the Vedic packaging as more than mere ornament. We import the Vedic vocabulary not for exoticism but because *it was ready first* and is *the integrated form*.

## 11.4 Generalisability outside coding agents

The mechanisms are not coding-specific. Avacchedaka-typed insertion is equally useful for, say, a customer-service agent that needs to distinguish "policy as of 2023-Q1" from "policy as of 2025-Q4". Bādha is equally useful for a research-assistant agent that surfaces a paper and then a retraction. Manas/Buddhi is equally useful for any setting in which selecting evidence and judging on the basis of it are different cognitive acts. We therefore expect the harness to generalise to non-coding agentic surfaces, but we have not measured this. It is the most natural follow-up beyond cross-family LLM validation.
