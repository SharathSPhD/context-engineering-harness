# Khyātivāda Annotation Report

- **Corpus size**: 3000
- **Master seed**: 0
- **Simulated judge accuracy**: 0.85
- **Percent agreement (raw)**: 0.774
- **Cohen's κ (overall)**: 0.736
- **Landis & Koch band**: substantial

## Per-class κ

| Class | One-vs-rest κ |
|---|---|
| akhyati | 0.788 |
| anirvacaniyakhyati | 0.737 |
| anyathakhyati | 0.771 |
| asatkhyati | 0.771 |
| atmakhyati | 0.620 |
| none | 0.611 |
| viparitakhyati | 0.860 |

## Confusion matrix

Rows = annotator A (heuristic), columns = annotator B (judge).

| | akhyati | anirvacaniyakhyati | anyathakhyati | asatkhyati | atmakhyati | none | viparitakhyati |
|---|---|---|---|---|---|---|---|
| akhyati | 338 | 0 | 48 | 0 | 0 | 23 | 19 |
| anirvacaniyakhyati | 0 | 305 | 0 | 24 | 17 | 8 | 0 |
| anyathakhyati | 30 | 0 | 371 | 0 | 0 | 7 | 21 |
| asatkhyati | 0 | 112 | 0 | 355 | 25 | 12 | 0 |
| atmakhyati | 3 | 16 | 19 | 0 | 290 | 105 | 0 |
| none | 9 | 4 | 21 | 0 | 95 | 295 | 0 |
| viparitakhyati | 19 | 0 | 33 | 0 | 0 | 9 | 367 |

## Marginal class distribution

| Class | Annotator A | Annotator B |
|---|---|---|
| akhyati | 428 | 399 |
| anirvacaniyakhyati | 354 | 437 |
| anyathakhyati | 429 | 492 |
| asatkhyati | 504 | 379 |
| atmakhyati | 433 | 427 |
| none | 424 | 459 |
| viparitakhyati | 428 | 407 |
