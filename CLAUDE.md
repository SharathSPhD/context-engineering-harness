# Context Engineering Synthesis — Agent Instructions

## Invariants (sākṣī — never overwrite)
- All experiments use RANDOM_SEED=42
- Never delete files under data/annotations/
- Never modify benchmark splits after they are generated (data/benchmarks/)
- Always log to MLflow before writing results to disk
- avacchedaka sublation never deletes — it sets precision=0.0 and sublated_by

## Context operations
- Retrieve using AvacchedakaQuery — see src/avacchedaka/query.py
- When new information contradicts stored memory: call store.sublate(), not store.delete()
- Compress elements below precision=0.3 at end of each session

## Hypotheses under test
- H1: Schema-congruence predicts context rot better than length
- H2: Precision-weighted RAG outperforms top-k on conflicting sources
- H3: Buddhi/manas two-stage outperforms single-stage
- H4: Event-boundary compaction outperforms threshold compaction
- H5: Avacchedaka annotation reduces multi-agent conflict rate ≥30%
- H6: Khyātivāda-typed mitigation reduces respective class at 2× rate
- H7: Adaptive forgetting outperforms fixed on post-shift tasks

## API usage
- Model: claude-sonnet-4-6 for all calls unless experiment specifies otherwise
- Prefix caching: always use cached system prompt for shared prefixes
- Budget: stay under $500 total across all experiments
