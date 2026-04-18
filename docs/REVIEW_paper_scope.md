# Paper Scope-Guardian Review

Scope is judged against the paper’s own stated goals (Section 1 / Abstract): (1) operationalize **seven** Vedic-epistemology constructs as **runtime** mechanisms, (2) validate on L1/L2/L3, (3) ship a **Cursor- and Claude-Code-compatible** plugin — without claiming new architectures, training, SOTA LLM performance, re-deriving Indian philosophy, or replacing cognitive science.

---

## Scope creep — must cut

| Section | Scope-creep instance | Why it’s outside stated goals | Recommended cut |
|--------|----------------------|------------------------------|-----------------|
| **Abstract** (00_frontmatter) | “classical Indian epistemology supplies **exactly** the vocabulary modern agent frameworks have been groping toward” | Strong uniqueness / sufficiency claim; not established by L1–L3 (which test a **specific** harness on **specific** metrics). Reads like philosophy-of-the-field, not falsifiable engineering. | Soften to “offers a **workable** typed vocabulary” or tie explicitly to the **seven** operationalized constructs only; drop “exactly.” |
| **§1.4 contribution (6)** (01_introduction) | TRIZ origin story framed as a method that “**generalizes beyond the present paper**” | Stated goals include TRIZ only as **provenance** for this work, not as a **general scientific method** contribution. | Delete or move to a single sentence in §3: “We do **not** claim TRIZ is a general recipe for research; we document **one** session that led here.” |
| **§3.3** (03_origin) | “The deeper claim is **methodological**” + mapping scratchpads/rerankers/sleep-time compute to Sanskrit constructs + “**two thousand years** of internal critique … as our **adversarial review**” | Expands the paper into a **meta-methodology** and **interpretive** claim about the whole field “re-discovering” Indian epistemology. Not validated by benchmarks or plugin tests. | Cut or compress to 2–3 sentences: TRIZ motivated vocabulary search; Sanskrit terms were chosen for **typed runtime** fit; drop “adversarial review” and broad “re-discovering” unless you add evidence (e.g., blind expert comparison). |
| **§11.3** (11_discussion), third bullet | “**Cross-cultural agreement is evidence**” that type signatures “are tracking **something real about cognition**” | Bridges into **philosophy of mind / cognitive science**; user’s non-goals explicitly disavow replacing cognitive science. No experiment in §§8–10 measures “reality of cognition.” | Remove this bullet or replace with: “Mechanisms **parallel** cognitive-science accounts; we **do not** test neural or behavioral hypotheses.” |
| **Appendix A** (`A_glossary.md`) | **Svataḥ-prāmāṇya / parataḥ-prāmāṇya** row claims it “**influences the prior** on Bayesian aggregation” (`source=user` vs `source=tool:rag`) | If §5.3 / code do **not** implement distinct priors by source (paper text shows a generic Beta update from precisions and votes only), this is an **ungrounded** operationalization — glossary invents mechanism not demonstrated in the validation story. | Either implement + cite in §5.3, or **delete** this row / reword to “**could** inform future priors” and remove from “operationalization” column. |

---

## Premature abstraction — should narrow

| Section | Instance | Why it’s heavy for the headline | Recommended narrow |
|--------|-----------|----------------------------------|-------------------|
| **§2.5–2.6** (02_related_work) | Long cognitive-architecture and neuroscience survey (WM, PFC, CLS, predictive coding, SOAR/ACT-R, dual-process) | Stated goals are **not** to ground the harness in cognitive science; this competes with the epistemology story and lengthens “related work” without new empirical obligations. | Keep one short paragraph: “Parallels exist; we **do not** model the brain” (already in §2.5); **move** the rest to supplementary material or citations only. |
| **§5.3 + §2.2** | Full **Beta–Bernoulli** formalism + Brier/ECE narrative | Headline is **typed context + sublation + witness**; Bayesian fusion is a **separate** design choice. It’s tested (H2) but the **philosophical** tie to pramāṇa-samplava (§3.2, Appendix A) doubles the conceptual stack. | In main text, present Bayesian block as “**one** conflict-resolution implementation”; keep Mīmāṃsā debate in a footnote or Appendix A only. |
| **§5.6** | EventBoundaryCompactor tied to predictive coding, Friston, Zipf fallback stack | H4 tests compaction; the **multi-theory** motivation is more than needed to justify “surprise threshold + episode summary.” | One sentence + citation to event segmentation; push predictive-coding details to appendix. |
| **§7.1 + §1.4 (5)** | **CLIBudgetScheduler**, `cost_ledger.db`, attractor-flow HALT/RESUME | Valuable for **reproducibility** but is **build infrastructure**, not part of the seven constructs or plugin runtime. Risks sounding like a second contribution (“budget OS”). | Reframe everywhere as “**appendix / engineering note**”; demote from numbered “contributions” or merge with reproducibility bullet. |
| **Stouffer omnibus** (§7.4, §8.8, §10.7, Abstract) | Single **combined p ≈ 10⁻¹⁹** headline | Statistically fine as a summary, but **sounds** like a universal “harness fixes agents” claim; includes studies with very different generality (synthetic structural vs P6-C). | Lead with **P6-C + L2** for ecological claims; report omnibus in one sentence with explicit “**includes** structural sanity checks; **robust** subset p = …” (§11.1.5 already does part of this — mirror in Abstract). |

---

## Defensible scope but should be flagged

| Section | Note |
|--------|------|
| **Appendix A (24 glossary rows vs 7 constructs)** | The “extra” rows are mostly **decompositions** of the seven (e.g. **prakāra / viśeṣya** ↔ tuple fields; **six khyāti** classes ↔ H6); **anumāna / pramāṇa** appear in §4.1; **antaḥkaraṇa** in §4.4; **pramāṇa-samplava** in §3.2 and §5.3 narrative. **Defensible** as a single reference table **if** each operational column matches shipped behavior (see Svataḥ/parataḥ row above). |
| **§6.9 / §5.9 “Claude Desktop”** | Text asserts **manual** install exercise on four hosts including Desktop — **not** the same evidence tier as L1–L3 (no protocol, n, or logs cited in the sections read). **Flag:** either add a one-line pointer to `docs/plugin_smoke_transcript.md` / commit evidence for Desktop, or change to “**intended** to work via MCP; **validated** on Cursor + Claude Code **only**.” |
| **§11.4 Generalisability** | Explicitly says non-coding agents are **not measured** — **honest**. Keep, but ensure Abstract/Intro don’t smuggle the same claim without “not measured.” |
| **§11.2** | Narrows claims after threats — **good** alignment; use as the **canonical** strength statement and trim stronger sentences in Abstract/§1.2. |
| **§3 (TRIZ)** | Origin narrative is **on-mission** for “how we built it”; problem is only **generalization** (see must-cut). |
| **Appendix F** | Negative results support **honesty**; references to `attractor-flow-state/` are dev breadcrumbs — fine in appendix if framed as “build diary,” not runtime contribution. |
| **§12.4 future work (8 items)** | Future work is **not** the same as writing bad checks **if** each item is labeled **unfunded / not done**. Items **7–8** (neuroscience consolidation curves; **industry-wide** MCP schema) are **far** from the three-layer validation — flag as **aspirational** or move to a “vision” paragraph, not peer results. |

---

## Scope-aligned content (kept honest)

- **§4** translation table and **§5** architecture mapping **seven** constructs to MCP modules and hypotheses **H1–H7** — core to goal (1).
- **§§8–10** L1 benchmarks, **P6-B** deterministic case study, **P6-C** paired A/B with explicit **PatchSimulator** limitation — core to goal (2); **§11.1.2** correctly bounds L3.
- **§6** plugin inventory (15 tools, skills, agents, hooks) and **Appendix B** tool contracts — core to goal (3).
- **§4.9 / §12.3** explicit **non-claims** (no new architecture, no philological full fidelity) — keeps scope disciplined.
- **§11.1** threat list (synthetic fallback, two models, author-as-annotator, structural studies in omnibus) — strengthens rather than inflates scope.

---

## Note on the “§11 phenomenology” check

The sections read for this review **do not** contain the phrase “phenomenology” or “phenomenological framework” in **§11** (or elsewhere in `paper/sections/*.md` per targeted search). The closest **argument creep** is **§11.3** (why Sanskrit) and the **cross-cultural / cognition** bullet above — recommend trimming that, not a non-existent “phenomenology” subsection.
