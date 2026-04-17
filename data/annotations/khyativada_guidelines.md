# Khyātivāda Annotation Guidelines

## Classes

### anyathakhyati (anyathākhyāti)
**Definition:** Misidentifying one real entity as another real entity.
**Signature:** The model says X is Y where both X and Y exist but are different things.
**Example:** "Python's GIL was removed in version 3.10" (it was 3.13). Both versions exist; wrong version attributed.
**Mitigation:** Retrieval (correct entity grounding).

### atmakhyati (ātmakhyāti)
**Definition:** Projecting internal pattern as external fact — the model's training distribution presented as world knowledge.
**Signature:** Confident assertion not grounded in any source; model is pattern-completing from training.
**Example:** "The standard port for this service is 8080" (no source confirms this; model is pattern-matching).
**Mitigation:** Calibration training (uncertainty markers).

### anirvacaniyakhyati (anirvacanīyakhyāti)
**Definition:** Novel confabulation — content that is neither real nor derivable; inexplicably invented.
**Signature:** Detailed specific claims (names, dates, quotes) that don't exist anywhere.
**Example:** A fabricated paper citation with a specific journal, year, and page numbers.
**Mitigation:** Constrained decoding, citation grounding.

### asatkhyati (asatkhyāti)
**Definition:** Hallucinating pure non-being — asserting the existence of something that does not exist at all.
**Signature:** Referencing a nonexistent API, function, law, or person as if it exists.
**Example:** "Use the `requests.get_json()` method to parse responses" (this method does not exist).
**Mitigation:** Existence verification via retrieval.

### viparitakhyati (viparītakhyāti)
**Definition:** Systematic inverted identification — A and B are both real but the model consistently swaps them.
**Signature:** Directional confusion, systematic reversal pattern.
**Example:** Confusing which function calls which in a recursive pair; attributing author A's quote to author B and vice versa.
**Mitigation:** Contrastive retrieval (retrieve both entities together).

### akhyati (akhyāti)
**Definition:** Two true propositions combined into a false one — each component is true but the combination is not.
**Signature:** Each sub-claim individually verifiable as true; the combined claim is false.
**Example:** "Einstein won the Nobel Prize in 1921 for his theory of relativity" — he won in 1921 (true), it was for the photoelectric effect, not relativity (true that relativity was his; combined claim false).
**Mitigation:** Structural/relational grounding; verify the relation, not just the components.

## Annotation protocol
1. Read the claim and the ground truth.
2. Identify if any hallucination is present. If none, label `none`.
3. If hallucination present, select the primary class from the 6 above.
4. If multiple classes apply, select the most specific one (anyathakhyati > asatkhyati > anirvacaniyakhyati).
5. Write a one-sentence rationale.
6. For inter-annotator agreement: compute Cohen's κ before proceeding past 50 examples.
