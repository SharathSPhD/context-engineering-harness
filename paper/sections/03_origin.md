# 3 · Origin: From a TRIZ Contradiction Analysis to Vedic Epistemology

This section narrates how the project came to be, because the methodology itself is part of the contribution. We did *not* begin by reading Sanskrit. We began by running TRIZ \citep{altshuller1984creativity, altshuller2002innovation, terninko1998triz, savransky2000triz} on a single engineering contradiction in modern LLM agents, and observed — to our own surprise — that the resolution drove us out of the LLM literature entirely and into eleventh-century Mithilā.

## 3.1 The TRIZ session

Using the `triz-engine` Cursor plugin \citep{trizengineplugin} (developed by the same author and used here in production from 2025-12), we logged the following primary contradiction:

> **C1.** *We must keep more information in the agent's context window to improve recall, **but** keeping more information degrades the agent's accuracy on the recall task itself.*

In TRIZ terms this is the canonical *technical contradiction* between **#26 Quantity of Substance** (context tokens available to attend to) and **#27 Reliability** (correct retrieval given the available tokens). The system-improving parameter pushes the system-degrading parameter. The TRIZ contradiction matrix \citep{altshuller2002innovation} for this pair returns the inventive principles **\#10 (Preliminary Action), \#11 (Cushion in Advance), \#28 (Mechanics Substitution), \#35 (Parameter Change)**. The `triz-engine` MCP tool `log_session_entry` recorded the matrix lookup, the four candidate principles, and the resulting solution sketches as separate audit entries.

Of the four principles, **#10 (Preliminary Action)** dominated: rather than fight the model at retrieval time, *do something to each retrieved item before it enters the visible context*. Specifically, *stamp it with conditions under which it is true and credentials of the source.*

This required a vocabulary we did not have. The standard RAG literature attaches *embeddings* and *similarity scores* but not *typed conditions*. The standard hallucination literature attaches *post-hoc labels* but not *runtime qualifiers*. The standard memory-architecture literature \citep{sumers2024coala, packer2023memgpt} stratifies storage by *recency* and *importance* but does not give those strata an *epistemic* type.

## 3.2 The discovery: classical Indian epistemology already had this vocabulary

A search for "*typed limitor on a cognition*" + "*precision-weighted assertion*" + "*supersession of older belief without deletion*" returned, after several iterations, classical Indian epistemology — specifically:

- **Nyāya's *avacchedaka***. Every cognition $C$ is *of x as y under condition z* — formally written *aQR* where *a* is the qualificand, *Q* the qualifier, *R* the relation/condition \citep{matilal1986perception, matilal1968navyanyaya, ingalls1951materials, phillips2012epistemology, gangesa14tattvacintamani}. Navya-Nyāya formalized this in the 14th century with operator-precision rivalling modern type theory \citep{ganeri2001philosophy, guha1968navyanyaya, ingalls1951materials}.

- **Advaita Vedānta's *bādha* (sublation)**. A higher-precision cognition does not *erase* a lower-precision one; it *supersedes* it, leaving the original retrievable as a now-rejected qualifier \citep{deutsch1969advaita, dharmaraja17vedantaparibhasa, shaw1990bada, rambachan2006advaita}. This is exactly what we want from a memory store under conflict.

- **The Antaḥkaraṇa duality of *manas* and *buddhi***. *Manas* is the attentional sense-organ; *buddhi* is the determinative judging faculty \citep{deutsch1969advaita, datta1932advaita, ramprasad2013advaita}. Together they constitute the two-stage attend-then-judge gate we needed.

- **Sākṣī (the witness)**. A non-revisable witness consciousness that records *what was known and when* without itself being the agent of action \citep{ganeri2017concealed, indich1980consciousness, fasching2009witness, sankaraupadesha}. We needed precisely this for cross-turn audit.

- **The six-class *khyātivāda* taxonomy of error**. Six classical theories of error — *anyathākhyāti* (Nyāya), *ātmakhyāti* (Yogācāra), *anirvacanīyakhyāti* (Advaita), *asatkhyāti* (Mādhyamika), *viparītakhyāti* (Mīmāṃsā), and *akhyāti* (Prabhākara Mīmāṃsā) — supply, in aggregate, a six-class typology of *how a cognition can be wrong* that is far more discriminating than any LLM hallucination ontology we found in the contemporary literature \citep{ram2007error, matilal1986perception, bhatt1989prabhakara, datta1932advaita, bilimoria2018epistemology, mohanty1992reason}.

- **Pramāṇa-samplava vs. pramāṇa-vyavasthā**. The Mīmāṃsakas debated whether multiple knowledge-sources can converge on the same object (samplava) or are partitioned (vyavasthā) \citep{guha2016svatahpramanya, dasti2008jayanta, dasti2012perception, bhattacharyya1988svatahprama}. This maps directly onto the agent design decision of whether retrieval, reasoning, and tool-use can be evidence for the same claim — and whether their reliabilities (*svataḥ-prāmāṇya* / *parataḥ-prāmāṇya*) compose or are partitioned. The Bayesian Beta-Bernoulli aggregator in Section 5.3 is the operational answer.

- **Saṃskāras and vāsanās**. Latent memory traces and dispositional tendencies \citep{deutsch1969advaita, sankaraupadesha, halbfass1991traditioncomparison} that motivate adaptive forgetting (Section 5.7).

It was not a metaphor. Each of these constructs admits a *precise runtime operationalization* on an LLM context window. Section 4 builds the translation table.

## 3.3 Why this method matters

The deeper claim is methodological. Modern LLM context engineering has been *re-discovering* fragments of an integrated philosophical tradition under different names: scratchpads ≈ *manas*; tool-use selectors ≈ *buddhi*; retrieval rerankers ≈ approximate *avacchedaka*; sleep-time-compute / consolidation ≈ *bādha*-driven memory pruning; provenance tracking ≈ *sākṣī*; hallucination taxonomies in flux ≈ *khyātivāda*. By naming the constructs from a tradition that already integrated them, we (a) get a *coherent* design rather than a bag of mechanisms, and (b) inherit two thousand years of internal critique of the tradition (Mīmāṃsā vs. Nyāya vs. Advaita debates) as our *adversarial review*.

The TRIZ session that produced this trajectory took 87 minutes of wall-clock time and consumed 38,512 input tokens / 4,206 output tokens of Claude usage; the full audit trail is preserved in the repository under `docs/triz-origin-session.md`. We retain this provenance not for vanity but because **it is the kind of audit trail the harness itself is designed to enforce**.

## 3.4 The two dev-time orchestration tools we used (and excluded from the shipped plugin)

Two further plugins were used during development and *deliberately excluded from the shipped artefact*:

- **`attractor-flow`** \citep{attractorflowplugin} — a dynamical-systems-inspired orchestration loop that drove the long-running validation pipeline through stable basins (e.g. "P6-A done", "P6-B done", "Stouffer-Z computed"). It supplied the HALT, RESUME, and basin-detection primitives for the CLI scheduler.

- **`ralph-loop`** — a self-referential iterative-development loop used during early prototyping of the harness components.

We treat these as *development-time* tools, analogous to `make` or `pytest`, not as runtime dependencies of the harness. The shipped plugin contains zero references to either, and we audit this in Section 6.5 and Appendix B.
