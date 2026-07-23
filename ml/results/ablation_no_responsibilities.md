# Ablation: excluding `responsibilities` / `responsibilities.1`

Split: `test` | matched_score threshold: `0.7` | queries: `434` | candidate JDs: `952`

See docs/PHASE3_FINDINGS.md ("Ground truth contamination via literal text duplication") for why this field is excluded here.

## MRR with vs. without `responsibilities`

| matcher | MRR (original, field included) | MRR (ablated, field excluded) | delta |
|---|---|---|---|
| tfidf | 0.1496 | 0.0072 | -0.1424 |
| sbert_pretrained | 0.0559 | 0.0101 | -0.0458 |
| sbert_finetuned | 0.0547 | 0.0138 | -0.0409 |

## Full ablated metrics

| matcher | n_queries | n_candidates | P@1 | P@3 | P@5 | MRR |
|---|---|---|---|---|---|---|
| tfidf | 434 | 952 | 0.0000 | 0.0000 | 0.0009 | 0.0072 |
| sbert_pretrained | 434 | 952 | 0.0023 | 0.0015 | 0.0009 | 0.0101 |
| sbert_finetuned | 434 | 952 | 0.0046 | 0.0023 | 0.0023 | 0.0138 |
