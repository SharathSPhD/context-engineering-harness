# Paper Coherence Review

Pratyakṣa v2 preprint — internal coherence pass over `paper/sections/*.md` and `paper/appendices/*.md` (plus cross-checks against stated artefacts). This review flags **document-internal** inconsistencies and ambiguity; it does not audit the code or recomputed statistics.

---

## Contradictions found

1. **“Five public benchmarks” vs six named families (Abstract / §1)**  
   - **What each says:** `00_frontmatter.md` (Abstract) and `01_introduction.md` §1.2 describe **“five public benchmarks”** but immediately enumerate **six** named surfaces: RULER, HELMET, NoCha, HaluEval, TruthfulQA, FACTS-Grounding (SWE-bench Verified is named separately as part of L1).  
   - **Which is correct:** Either the count should be **six** (if each name is a distinct benchmark family) or one name should be folded into another in prose; as written, the numeral and the list disagree.

2. **Appendix E is cited for Khyātivāda / IAA, but Appendix E is the P6-C protocol**  
   - **What each says:** `04_theory.md` §4.6, `08_results_l1.md` §8.6, `01_introduction.md` §1.4 contribution 4, and `11_discussion.md` §11.1.4 point the **3,000-example IAA / classifier / “codebook”** material to **Appendix E** and sometimes to **“Section 11.2 / Appendix E.”**  
   - **Actual file:** `appendices/E_p6c_protocol.md` is **only** the SWE-bench Verified A/B (P6-C) protocol — no Khyātivāda codebook or IAA protocol. Prompts live in `appendices/D_prompts.md`.  
   - **Which is correct:** The **lettered appendix mapping is wrong** in the body: Khyātivāda/IAA citations should not point to Appendix E as currently defined; “Section 11.2” is also wrong for IAA (IAA appears under **§11.1.4**, while §11.2 is “What the harness genuinely contributes”).

3. **Reproducibility manifest cited as Appendix F in §1.4, but Appendix F is negative results**  
   - **What each says:** `01_introduction.md` §1.4 item 5: reproducibility / `cost_ledger` / manifest → **“(Section 7.1, Appendix F).”**  
   - **Actual file:** `appendices/F_negative_results.md` is **Negative and Null Results**; the reproducibility manifest is **`appendices/C_reproducibility.md`**.  
   - **Which is correct:** **Appendix C** matches the described content; **Appendix F** does not.

4. **P6-C synthetic snippet precision ranges: §10 vs Appendix E**  
   - **What each says:** `10_results_l3.md` §10.1: stale precisions **∈ [0.20, 0.45]**, fresh **∈ [0.80, 0.97]**. `appendices/E_p6c_protocol.md` §E.2: stale **∈ [0.05, 0.30]**, fresh **∈ [0.70, 0.95]**, and a different supersession pattern (“one fresh … points at one stale”).  
   - **Which is correct:** **Unknown from the paper alone** — the two protocol descriptions cannot both hold for the same generator without a stated version skew. This is a high-severity **spec contradiction** between the main-results section and the appendix billed as the audit-grade protocol.

5. **H2 post-replacement ECE: §8.2 vs Appendix F**  
   - **What each says:** `08_results_l1.md` §8.2 reports Bayesian ECE **0.041** on a 1,800-example slice. `appendices/F_negative_results.md` §F.1 claims the replacement aggregator took H2 ECE to **~0.07**.  
   - **Which is correct:** **Unclear** — may reflect different slices, metrics, or rounding, but the document does not bridge the two numbers, so readers will infer conflicting “after” calibration.

6. **§11.1.1: “Eight of the nine L1 hypotheses”**  
   - **What each says:** L1 is defined everywhere else as **H1–H7** (seven hypotheses). “Nine” does not match any other table in the methodology/results.  
   - **Which is correct:** **Seven** L1 hypotheses unless the text introduces a different counting convention (e.g., splitting H1/H2 by length *as* separate “hypotheses”), which it does not in §11.1.1.

7. **“10 quantitative studies” vs portfolio wording (P6-B omitted from Stouffer list)**  
   - **What each says:** Abstract / §1.2 describe the validation portfolio as **H1–H7 + P6-B + P6-C** (reads as **nine** headline components if each hypothesis counts once). `10_results_l3.md` §10.7 defines the Stouffer input as **H1×2 lengths + H2×2 lengths + H3–H7 + P6-C per-instance** (= **10** rows) and **does not list P6-B**.  
   - **Which is correct:** **Both can be defended** if “10 studies” means “10 Stouffer rows” and P6-B is intentionally excluded — but that exclusion is **not stated in the Abstract/§1**, so a reader can reasonably believe P6-B is inside the omnibus. **Needs explicit disambiguation** (what the 10 rows are; why P6-B is in/out).

8. **§1.5 roadmap vs actual appendix titles**  
   - **What each says:** Six appendices described as “glossary, MCP tool reference, statistical details, adapter details, the Khyātivāda codebook, and the reproducibility manifest.”  
   - **Actual files:** A glossary; B MCP tools; **C reproducibility**; **D verbatim prompts**; **E P6-C protocol**; **F negative results**. There is **no appendix titled** “statistical details,” “adapter details,” or “Khyātivāda codebook,” and **negative results (F)** are omitted from the roadmap sentence.

9. **Translation table (§4.8): Manas → “`ManasAgent` skill”**  
   - **What each says:** `04_theory.md` §4.8 maps **Manas** to **`ManasAgent` skill** and cites H3. `06_plugin.md` §6.3 places **Manas** in **`agents/manas.md`** as a **sub-agent**; skills are the three `skills/*/SKILL.md` entries (none is `ManasAgent`).  
   - **Which is correct:** **§6** is internally consistent with the plugin layout; **§4.8’s “skill” label for Manas is inconsistent** with §4.4 and §6.

10. **Core MCP tool names: §4.2 / §5.1 vs §6 / Appendix B**  
    - **What each says:** Theory/architecture use names like **`insert`**, **`retrieve_by_qualifier`**, **`retrieve_under_condition`** (`04_theory.md`, `05_architecture.md`). The shipped server uses **`context_insert`**, **`context_retrieve`**, **`context_get`**, **`compact`**, etc. (`06_plugin.md`, `appendices/B_mcp_tools.md`).  
    - **Which is correct:** **Appendix B / §6** match each other; earlier sections read like a **parallel API** unless qualified as pseudonyms.

11. **Compaction hook vs tool name**  
    - **What each says:** `05_architecture.md` §5.7 and `06_plugin.md` §6.5 describe a hook calling **`compact_now(...)`**; Appendix B defines the tool as **`compact`** (no `compact_now` tool).  
    - **Which is correct:** **Appendix B**; the **`compact_now` name appears inconsistent** with the specified MCP surface unless it is a host-side wrapper not shown in B.

12. **L2 results path: §9 vs Appendix C**  
    - **What each says:** `09_results_l2.md` cites `experiments/results/p6b/_summary.json`. `appendices/C_reproducibility.md` §C.4 gives `experiments/results/p6b/summary.json` (no underscore).  
    - **Which is correct:** **Unknown from text alone** — filenames should be unified to the canonical artefact.

13. **Appendix E internal cross-reference**  
    - **What each says:** `appendices/E_p6c_protocol.md` §E.4: “**Section 7.5** describes the rationale” for `PatchSimulator`. In `07_methodology.md`, **§7.5 is “Models tested”;** the PatchSimulator design is in **§7.6.2** / **§10**.  
    - **Which is correct:** **§7.6 / §10**, not §7.5.

---

## Terminology drift

| Term / pattern | Places used | Drift description |
|----------------|-------------|-------------------|
| **Manas output JSON** | §4.4 `{attended, conditions, filter_reasons}`; §6.3 `{attended_ids, conditions, filter_reasons}`; Appendix D `{"selected_ids", "rationale"}` | Three different schemas for the same agent step — implementers cannot know which is canonical. |
| **Avacchedaka vs qualifier / limitor** | §4.2 formal triple (a, Q, R) with R = limitor; glossary A: **Avacchedaka** = `condition` field; prose sometimes calls Q “qualifier” (prakāra) | Conceptually disciplined, but **“qualifier” (Q) vs “limitor” (R)** are easy to conflate; the glossary helps — still, §1.2 calls *avacchedaka* “qualifier-conditioned cognition” while Navya-Nyāya separates **prakāra** vs **avacchedaka**. |
| **“Six-class” vs seven labels** | Abstract “six-class taxonomy”; §4.6 and classifier: **six classes + `none`**; Appendix A: “seven `khyati_class` labels” | Common in ML papers, but **“six-class” headline vs seven-way classifier** can confuse strict readers. |
| **Buddhi/Manas vs two-stage agent** | §1 plain-language summary; §2.3; §4.4; §5.4 | Mostly aligned as **the same two-stage gate**; occasional wording (“sub-agents” vs “internal gates”) is consistent with the single-agent framing. |
| **Agents vs skills** | §1.2 “3 sub-agents”; elsewhere “3 agents”; §4.8 mis-labels Manas as skill | Minor “sub-agents” vs “agents”; **§4.8 Manas row is the real inconsistency**. |
| **MCP tool naming** | §4–5 vs §6 / Appendix B | See contradiction (10) — **pseudocode names vs shipped names**. |

---

## Structural gaps

1. **Khyātivāda codebook / IAA protocol has no appendix that matches the citations** — multiple forward refs to “Appendix E” / “Section 11.2” for material that is not in `E_p6c_protocol.md` and not at §11.2.  
2. **§7.4 defines Stouffer weighting but not dependence between studies** — all rows share models/seeds/adapters; **no discussion of correlation** or robustness of meta-p to dependence (only §11.1.5 removes structural H5/H7 from a *different* summary).  
3. **Appendix E §E.5** (heuristic vs Docker κ ≈ 0.97 on 30 instances) is **not echoed in §10** despite being a validity claim in the protocol appendix.  
4. **Figures/tables** are referenced by ID (F01, T4, etc.) with paths under `experiments/results/p7/`; the markdown sections do not embed them — acceptable for a split build, but **readers of `.md` alone** see **unresolved figure/table pointers** unless they follow file paths.  
5. **§11 does not summarize Appendix F** — negative/null results live only in Appendix F; the discussion’s “threats” list is **not** the same structure as F.1–F.6 (see below).

---

## Ambiguity / reader-divergence

| Location | Candidate readings | Recommended disambiguation |
|----------|--------------------|----------------------------|
| **Abstract “100% … vs 50.3%” on SWE-bench Verified** | (A) End-to-end SWE pass rate on real patches; (B) **Heuristic target-path hit** under **PatchSimulator**, 720 paired runs, fixed 8K budget | §10 already narrows this; **Abstract should flag “heuristic target-path / PatchSimulator”** in one clause to match §10.4–10.6. |
| **“10 quantitative studies” + Stouffer** | (A) Includes P6-B; (B) **Excludes P6-B**, uses **split H1/H2 lengths** as separate rows | State explicitly: **omnibus = 10 rows: …; P6-B reported separately and excluded because …** (or include P6-B with a defined effect-size input). |
| **Stouffer validity (§7.4)** | (A) Formal meta-analysis with independent p-values; (B) **Heuristic portfolio summary** over correlated tests | One sentence on **non-independence** (shared models, seeds, adapters) and why weighted Stouffer is still used as a **descriptive** omnibus vs claim of **independent** evidence. |
| **“Preregistered” (Abstract / §1)** | Industry sense (pre-registered hypotheses) vs **preregistration in a registry** | If not registry-preregistered, prefer **“prespecified”** or cite a frozen preregistration artefact. |
| **IAA “two automated annotators” (§11.1.4)** | Sounds like **unsupervised / circular** agreement vs deliberate **pipeline** with guardrails | Keep, but tie to **Appendix D** (prompts + guardrails) once appendix letters are fixed. |

---

## Redundancy

1. **Single-agent + Buddhi/Manas internal gate + Cognition “Don’t Build Multi-Agents”** — repeated with similar wording in `02_related_work.md` §2.3, `01_introduction.md` §1.3, and `05_architecture.md` / `04_theory.md`. **Recommend:** keep full argument once (e.g. §1.3), shorten later mentions to one sentence + cross-ref.  
2. **PatchSimulator isolates context vs generation** — near-duplicate rationale in `07_methodology.md` §7.6.2, `10_results_l3.md` §10.4, and `11_discussion.md` §11.1.2. **Recommend:** one canonical paragraph in §7 or §10, others reference it.  
3. **“Weaker falsifiable claim” about Sanskrit fidelity** — `04_theory.md` §4.9 and `11_discussion.md` §11.1.7 overlap heavily. **Recommend:** retain §4.9; reduce §11.1.7 to threats unique to reviewer response.

---

## Cross-section consistency table

| Claim | Section(s) | Value stated | OK / issue |
|-------|------------|--------------|------------|
| Stouffer combined Z | Abstract; §10.7; §11 | Z = 9.114 | **OK** (same number) |
| Stouffer two-sided p | Abstract; §10.7; §11 | 7.94 × 10⁻²⁰ | **OK** |
| Mean per-study delta | Abstract; §10.7 | +0.476 | **OK** |
| SWE target-path hit (treatment) | Abstract; §1.2; §10.2 | 100% (720/720) | **OK** with §10 nuance |
| SWE baseline target-path rate | Abstract; §10.2; §10.6 | ~50.3% vs 0.5028 (362/720) | **OK** (rounding) |
| Cohen’s κ (Khyātivāda) | Abstract; §2.4; §4.6; §8.6; §12 | 0.736, n = 3,000 | **OK** |
| “10 quantitative studies” composition | Abstract; §1.2; §10.7 | Stouffer rows omit P6-B; intro lists P6-B | **Issue** — clarify |
| Plugin: 15 MCP / 3 skills / 3 agents / 4 commands / 3 hooks | Abstract; §1; §6; Appendix B | 15 tools align §6 ↔ B | **OK** |
| 720 paired runs | Abstract; §7.6.2; §10; Appendix E | 120×3×2 | **OK** |
| “Five public benchmarks” + list | Abstract | Six names | **Issue** |
| L1 hypothesis count in synthetic-fallback sentence | §11.1.1 | “nine” | **Issue** |
| Appendix for reproducibility manifest | §1.4 | Says F | **Issue** — should be C |
| Appendix for Khyātivāda IAA / codebook | §1.4; §4.6; §8.6; §11.1.4 | Says E / §11.2 | **Issue** — E is P6-C |
| P6-C precision ranges | §10.1 vs Appendix E | Different intervals | **Issue** |
| H2 ECE after Bayesian fusion | §8.2 vs Appendix F | 0.041 vs ~0.07 | **Issue** — reconcile |
| Manas in translation table | §4.8 vs §6 | “skill” vs agent file | **Issue** |
| MCP `insert` vs `context_insert` | §4–5 vs §6 / B | Name mismatch | **Issue** — alias or unify |
| `compact_now` vs `compact` | §5.7; §6.5 vs Appendix B | Name mismatch | **Issue** |
| P6-B summary path | §9 vs Appendix C | `_summary` vs `summary` | **Issue** |
| PatchSimulator rationale § ref | Appendix E | Points to §7.5 | **Issue** |
| §11 “negative results” vs Appendix F | §11; Appendix F | No enumerated parity | **Gap** (not a direct contradiction) |

---

## Targeted high-stakes checks (user-requested)

| Check | Verdict |
|-------|---------|
| **Does §7 methodology license the Stouffer combination in §10?** | **Partially.** §7.4 specifies the **weighted Stouffer** recipe but does **not** justify **independence** (or dependence-robustness) of the combined p-values. §11.1.5 addresses a different concern (structural H5/H7). A reader strict about meta-analysis will **not** see a full license—only a procedural definition. |
| **Does §6 list the same 15 MCP tools as Appendix B?** | **Yes** — tool names and ordering match between `06_plugin.md` §6.2 and `B_mcp_tools.md` §B.1–B.15. |
| **Does §11 “negative results” match Appendix F?** | **No structured match.** §11 is organised as **threats to validity**; Appendix F lists **six** negative/null engineering outcomes (F.1–F.6). Content **overlaps thematically** (e.g. PatchSimulator, synthetic fallback, single-stage Buddhi appears in F.2 vs §11.1.2 Manas/Buddhi threat — related but not the same heading). **Recommend** a short §11 subsection “Relation to Appendix F” or move F summaries into §11. |
| **Does §4’s translation table match §5’s module names?** | **Mostly, modulo MCP naming and the Manas row.** §5 uses module names (`ContextStore`, `EventBoundaryCompactor`, …) consistent with the **ideas** in §4.8, but **MCP tool identifiers** differ from §4.2’s `insert` / `retrieve_*` wording, and **Manas is mis-typed as a skill** in §4.8. |

---

## Summary (1 paragraph)

The headline statistics (Stouffer Z/p, 720 runs, κ, plugin counts) are **internally numerically aligned** across Abstract, results, and discussion, but the manuscript has several **high-severity cross-reference and protocol inconsistencies**: **Appendix E is repeatedly cited for Khyātivāda/IAA while the file is the P6-C protocol**, **§1.4 points reproducibility at Appendix F (which is negative results) instead of Appendix C**, and **§10.1 vs Appendix E disagree on P6-C synthetic precision ranges and supersession details**. Layered on that are **API naming splits** (`insert` vs `context_insert`, `compact_now` vs `compact`), **three different Manas JSON shapes**, a clear **miscount** (“five benchmarks” with six names; “nine L1 hypotheses”), and **under-specified independence assumptions** for Stouffer despite correlated experimental designs. Cleaning appendix-lettering, unifying the P6-C protocol text, and tightening the Abstract’s SWE-bench / omnibus claims would remove the worst reader traps without changing the underlying narrative.
