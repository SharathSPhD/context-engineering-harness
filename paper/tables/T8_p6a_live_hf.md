| bundle (dataset revision)                                                                                     | treatment | baseline | delta   | paired p | cohen's d | n  | status                           |
| ------------------------------------------------------------------------------------------------------------- | --------- | -------- | ------- | -------- | --------- | -- | -------------------------------- |
| H1\_ruler\_8192\_live (`simonjegou/ruler`@24adcea)                                                            | 1.0000    | 0.7667   | +0.2333 | 0.0005   | 0.547     | 60 | complete, target met             |
| H1\_ruler\_16384\_live (`simonjegou/ruler`@24adcea)                                                           | 1.0000    | 0.9333   | +0.0667 | 0.1174   | 0.265     | 60 | complete, underpowered (N=60)    |
| H\_TQA\_live\_v2 (`truthfulqa/truthful_qa`@741b827)                                                           | 0.0500    | 0.0833   | −0.0333 | 0.7386   | −0.091    | 60 | complete, null                   |
| H\_SWEB\_live\_n15, haiku-only (`princeton-nlp/SWE-bench_Verified`@c104f84)                                   | 0.1259    | 0.0167   | +0.1093 | 0.0322   | 0.488     | 30 | partial (CLI-blocked, see App.G) |

Source: `experiments/results/p6a/_summary_live.json` (`H1_*_live`, `H_TQA_live_v2`) and `experiments/results/p6a/swe_bench_outcomes.json` (haiku-only paired slice of `H_SWEB_live_n15`). Full per-seed/per-model records live in the attached JSONL checkpoints under `.cache/live_hf_checkpoints/`.
