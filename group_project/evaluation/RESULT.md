# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | TODO  |
| Framework and version              | TODO  |
| Evaluator model                    | TODO  |
| Generator model                    | TODO  |
| Embedding model                    | TODO  |
| Corpus version/commit              | TODO  |
| Golden dataset size                | TODO  |
| `top_k`                            | TODO  |
| Fallback threshold and calibration | TODO  |

## Configurations

- **Config A — dense-only:** TODO
- **Config B — hybrid + RRF:** TODO

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và `top_k`; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     TODO |     TODO |      TODO |
| Answer relevance  |     TODO |     TODO |      TODO |
| Context recall    |     TODO |     TODO |      TODO |
| Context precision |     TODO |     TODO |      TODO |
| **Average**       |     TODO |     TODO |      TODO |

## A/B comparison

- Cấu hình tốt hơn: TODO
- Evidence: TODO
- Trade-off về latency/cost: TODO

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | TODO     | TODO   |         TODO |      TODO |   TODO |      TODO | retrieval/generation/data | TODO       |
|   2 | TODO     | TODO   |         TODO |      TODO |   TODO |      TODO | retrieval/generation/data | TODO       |
|   3 | TODO     | TODO   |         TODO |      TODO |   TODO |      TODO | retrieval/generation/data | TODO       |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | TODO   | TODO                           | TODO            | TODO          |
|        2 | TODO   | TODO                           | TODO            | TODO          |
|        3 | TODO   | TODO                           | TODO            | TODO          |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| TODO       | TODO     |         TODO |               TODO | TODO       |
