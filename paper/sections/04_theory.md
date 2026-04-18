# 4 · Theoretical Foundations: Vedic Epistemology Mapped to LLM Context Engineering

This section constructs a precise translation table from seven classical Indian epistemic constructs to seven runtime operations on an LLM agent's context. Each row is a falsifiable design commitment, and each is operationalized in Section 5 and tested in Sections 8–10.

## 4.1 Pratyakṣa — direct perception as the harness's grounding pramāṇa

In Nyāya \citep{aksapada200nyayasutra, matilal1986perception, phillips2012epistemology} and Advaita \citep{dharmaraja17vedantaparibhasa, datta1932advaita}, *pratyakṣa* (direct perception) is the *first* of the *pramāṇas* — the means of valid cognition — and the *foundational* one against which inference (*anumāna*), comparison (*upamāna*), testimony (*śabda*), implication (*arthāpatti*), and non-cognition (*anupalabdhi*) are measured.

**LLM operationalization.** We treat *pratyakṣa* as the agent's **direct read of its current context window** — what it can in-fact attend to right now — and demote all upstream operations (RAG retrieval, tool outputs, prior-turn summaries) to the status of *secondary pramāṇas* whose claims must be entered into the visible context with their own qualifiers. The harness's name, *Pratyakṣa*, is therefore a methodological commitment: *if it is not in the visible, witness-tracked context, it is not direct evidence for the agent's next action*. Hidden hot caches, opaque rerankers, and "system memories" the agent cannot inspect are, in this discipline, **not allowed to ground a claim** without first entering the *pratyakṣa* surface.

## 4.2 Avacchedaka — qualifier-conditioned cognition as typed limitor

In Navya-Nyāya \citep{matilal1968navyanyaya, ingalls1951materials, gangesa14tattvacintamani, phillips2012epistemology}, every cognition is structurally a triple:

$$
C = \langle a,\; Q,\; R \rangle \quad\text{read as ``}a\text{ as qualified by }Q\text{ under relation/condition }R\text{''.}
$$

Here $a$ is the *qualificand* (viśeṣya), $Q$ the *qualifier* (prakāra), and $R$ the *limitor* (avacchedaka) under which the qualification holds. Crucially, *R* is part of the cognition; "the pot is blue *under daylight*" is a different cognition from "the pot is blue *under candlelight*", and the two can co-exist without contradiction.

**LLM operationalization.** Every item entered into the harness's `ContextStore` carries the four-tuple

$$
\texttt{Item} = \big\langle \texttt{qualificand},\; \texttt{qualifier},\; \texttt{condition},\; \texttt{precision}\in[0,1] \big\rangle.
$$

`condition` is a free-text string like `"as of 2024-09-Django-5.0"` or `"on POSIX, Python ≥ 3.11"`. The MCP tool `insert(qualificand, qualifier, condition, precision, source)` is the only way evidence enters the store. Two items that share `qualificand` but differ in `condition` *do not conflict*; two items that share `qualificand` *and* `condition` but disagree on `qualifier` *do*. This is the operational definition of a contradiction the harness can act on (Section 4.3).

## 4.3 Bādha — sublation as supersede-with-provenance

In Advaita Vedānta \citep{deutsch1969advaita, shaw1990bada, dharmaraja17vedantaparibhasa, sankaraupadesha}, a higher-precision cognition *sublates* (bādhita) a lower-precision one. The classical example is mistaking a rope for a snake: the *snake-cognition* is not erased on closer inspection — the agent must remember it, because *that error is itself an object of subsequent reflection* — but its qualifier is now stamped *bādhita* and it no longer grounds action.

**LLM operationalization.** The MCP tool `sublate_with_evidence(target_id, by_id, reason)` rewrites the target item's `status` to `bādhita`, annotates it with a pointer to the superseding item, and *retains* the original under that status. Subsequent retrievals filter by `status == "live"` for normal use but can be opted-in for audit. Sublation is triggered by two rules:

1. **Explicit pointer**: the newly-arriving item names a prior item via `superseded_by_id`.
2. **Dominance rule**: the newly-arriving item shares `qualificand` and `condition` with an existing item, has *strictly higher* precision, and the older item is `stale=True` while the newer is not.

The dominance rule is the operational form of the Advaita commitment that *higher pramāṇa supersedes lower pramāṇa* under shared limitor; the explicit pointer is the form used when an LLM-side classifier or external evidence has detected a stale-fresh pair.

## 4.4 Buddhi and Manas — the two-stage attend-then-judge gate

The Antaḥkaraṇa quartet of *manas, buddhi, citta, ahaṃkāra* \citep{deutsch1969advaita, ramprasad2013advaita, datta1932advaita} distinguishes the *attentional sense-organ* (*manas*) from the *determinative judging faculty* (*buddhi*). Manas selects and presents; Buddhi decides. The two are sequential and cannot be collapsed without losing the gate.

**LLM operationalization.** The harness defines two sub-agent prompts (Section 6.3):

- **`ManasAgent`** receives the user query and the `ContextStore` snapshot, and returns *which items it has elected to attend to and under which conditions*. Its output is structured JSON: `{attended: [item_id, ...], conditions: [str, ...], filter_reasons: [str, ...]}`. Manas is *forbidden* from emitting a final answer.
- **`BuddhiAgent`** receives the user query, the Manas output, and re-reads the attended items from the `ContextStore`, and returns the final answer plus its own `khyāti_class` (Section 4.6) for the answer. Buddhi is the only agent that may emit a user-visible answer.

This two-stage gate is identical in shape to dual-process accounts of human cognition \citep{evans2003duality, kahneman2011thinking, sloman1996two} and to classic AI cognitive architectures \citep{laird1987soar, anderson1996actr, newell1990unified}, but the names — and the surrounding philosophical apparatus — are Vedic. Critically, *Manas can be wrong without Buddhi being wrong*: if Manas mis-selects, Buddhi's job is to detect and call for re-selection rather than to answer.

## 4.5 Sākṣī — the witness as model-invariant audit frame

In Advaita Vedānta \citep{indich1980consciousness, fasching2009witness, ganeri2017concealed, sankaraupadesha}, *sākṣī* — the witness consciousness — is the *non-revisable, non-acting* observer that *records* what is cognised without itself being the agent of cognition or action. It supplies the stable reference frame against which changes of cognition are intelligible at all.

**LLM operationalization.** The harness instantiates a `SakshiKeeperAgent` (Section 6.3) and a `SakshiPrefix` skill (Section 6.2). On every turn, the keeper appends to a persistent JSON-lines `witness_log.jsonl` an immutable record of:

- The user query.
- The Manas selection (which items, under which conditions).
- The Buddhi judgment (final answer, khyāti class, calibrated posterior).
- All sublations fired this turn (target_id, by_id, reason).
- The token budget consumed, by category (system, retrieval, tool, completion).

Crucially, the witness is *write-once-per-turn* and *read-only* to all agents in the next turn. It is the harness's source of truth for cross-turn provenance, and it is *invariant under model swaps*: the witness log written under Claude 3.5 Sonnet is fully readable, fully diffable, and fully auditable when the agent is later switched to Claude 4.5 Sonnet, GPT-4o, or Qwen-3-72B. The model is replaceable; the witness is not.

## 4.6 Khyātivāda — six-class typology of error

Six classical schools of Indian philosophy proposed six theories of erroneous cognition \citep{ram2007error, matilal1986perception, datta1932advaita, bhatt1989prabhakara, bilimoria2018epistemology, mohanty1992reason}. Each names a specific *kind* of error:

| Class | School | Kernel definition | Modern LLM analogue |
|---|---|---|---|
| **Anyathākhyāti** | Nyāya | "Apprehension of one thing as another *of the same general type*." | Wrong API name (right module, wrong function); citing the right paper but wrong year. |
| **Ātmakhyāti** | Yogācāra | "Projection of an internal cognition onto the external object." | Inventing helper functions / config keys the framework never exposed; hallucinated TODOs the user did not write. |
| **Anirvacanīyakhyāti** | Advaita | "Apprehension of an object that is *neither real nor unreal*; sui generis indeterminate." | Hedged hallucinations: "you might consider X-or-similar" where X is plausible but ungrounded. |
| **Asatkhyāti** | Mādhyamika | "Apprehension of a non-existent object." | Citing a CVE, a person, a paper, or a commit hash that simply does not exist. |
| **Viparītakhyāti** | Mīmāṃsā | "Apprehension of the contrary." | Inverting a boolean flag's semantics; recommending the *opposite* of the documented best practice. |
| **Akhyāti** | Prabhākara Mīmāṃsā | "Failure of correct apprehension by mere conflation of two adjacent cognitions." | Mixing two adjacent functions / two near-duplicate config schemas / two adjacent versions. |

This is not a post-hoc relabelling of an existing modern taxonomy. It is the *original* taxonomy from the Indian philosophical tradition, jointly developed by debating schools over roughly a thousand years. We adopt it directly because (a) it carves the space cleanly along *epistemic* (not merely surface-textual) lines, (b) the six classes are mutually exclusive in the tradition's own usage, and (c) it gives us a *typed signal* the harness can act on (different remediation strategies for different classes).

**LLM operationalization.** A few-shot Claude-side `KhyativadaClassifier` (Section 5.5) tags every Buddhi output with one of seven labels (six classes plus `none`). The classifier is grounded by a 3,000-example, two-rater annotated corpus (Section 11.2 / Appendix E) at Cohen's κ = 0.736 ("substantial").

## 4.7 Saṃskāras and vāsanās — adaptive forgetting

In Advaita and Yoga, *saṃskāras* are residual mental impressions and *vāsanās* are dispositional tendencies that survive the immediate occasion of a cognition \citep{deutsch1969advaita, sankaraupadesha, halbfass1991traditioncomparison}. They explain why we are drawn to the same patterns of thought and why some impressions decay quickly while others persist indefinitely. The Advaitin practice prescribes *removal of obstructive vāsanās* — adaptive forgetting of patterns that no longer ground correct cognition.

**LLM operationalization.** The `AdaptiveForgetting` module (Section 5.7) applies an exponential-decay schedule to non-witnessed items in the `ContextStore`, with three explicit rules:

1. **Witnessed items never decay.** If `sākṣī.witnessed == True` for an item-id, its precision floor is fixed.
2. **Sublated items decay faster.** Items with `status == bādhita` halve their precision every time they survive a budget-pressure compaction event.
3. **High-recency, low-precision items decay slowest among the rest** — the item is too new to know whether it will become a vāsanā or a saṃskāra.

This avoids both the catastrophic-forgetting failure mode of naive context pruning \citep{french1999catastrophic, kirkpatrick2017overcoming, parisi2019continual} and the "everything is forever" failure mode of unconstrained external memory \citep{packer2023memgpt}.

## 4.8 The translation table, summarised

| Vedic construct | LLM operationalization | MCP tool / module | Tested by |
|---|---|---|---|
| Pratyakṣa | Visible-context-only grounding discipline | All `retrieve_*` tools | All hypotheses (architectural commitment) |
| Avacchedaka | Typed limitor `(qualificand, qualifier, condition, precision)` | `insert`, `retrieve_by_qualifier` | H2, H5 |
| Bādha | Supersede-with-provenance under shared limitor | `sublate_with_evidence` | H4, H5, P6-B, P6-C |
| Manas | Pre-judgment attention-selection sub-agent | `ManasAgent` skill | H3 |
| Buddhi | Determinative judgment + khyāti tagging sub-agent | `BuddhiAgent` skill | H3, H6 |
| Sākṣī | Write-once cross-turn witness log | `SakshiKeeperAgent`, `SakshiPrefix` | All (audit invariant) |
| Khyātivāda | 6-class hallucination classifier | `classify_khyati` | H6 |
| Saṃskāra/vāsanā | Adaptive forgetting with witness-protected items | `AdaptiveForgetting` | H7 |

## 4.9 What is *not* being claimed

We are *not* claiming that the historical authors had LLMs in mind, nor that our operationalizations are textually faithful in the way a Sanskrit philologist would require. We are claiming the weaker, falsifiable thing: that the *type signatures* the tradition developed for cognition turn out to *fit* the type signatures we need for an LLM context-engineering harness, and that fitting them produces measurable downstream gains. The next six sections defend that claim empirically.
