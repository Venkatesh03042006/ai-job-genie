# Phase 2 Baseline Results

Split: `test` | relevance: same (normalized) category | queries: `24` | candidate JDs: `2174`

> **Caveat**: category labels are a consensus of these same three matchers (`scripts/build_resume_labels_cache.py`) - every matcher below helped define the ground truth it's scored against, so this is not a clean baseline comparison. See `docs/PHASE3_FINDINGS.md`'s "Category label mismatch" section.

## Overall

| matcher | n_queries | n_candidates | P@1 | P@3 | P@5 | MRR |
|---|---|---|---|---|---|---|
| tfidf | 24 | 2174 | 0.5833 | 0.6667 | 0.6500 | 0.7070 |
| sbert_pretrained | 24 | 2174 | 0.6667 | 0.6944 | 0.7083 | 0.7595 |
| sbert_multi_qa_minilm_l6_cos_v1 | 24 | 2174 | 0.6250 | 0.6250 | 0.6417 | 0.7122 |

## By category

| matcher | category | n_queries | P@1 | P@3 | P@5 | MRR |
|---|---|---|---|---|---|---|
| tfidf | Data Science Engineer | 13 | 0.6923 | 0.7436 | 0.7231 | 0.8077 |
| tfidf | Business Development Executive | 3 | 0.6667 | 0.8889 | 0.8667 | 0.8333 |
| tfidf | Mechanical Engineer | 3 | 0.6667 | 0.8889 | 0.8667 | 0.8333 |
| tfidf | Senior Software Engineer | 3 | 0.3333 | 0.3333 | 0.3333 | 0.4087 |
| tfidf | Data Engineer | 2 | 0.0000 | 0.0000 | 0.0000 | 0.1214 |
| sbert_pretrained | Data Science Engineer | 13 | 1.0000 | 0.9487 | 0.9077 | 1.0000 |
| sbert_pretrained | Business Development Executive | 3 | 0.3333 | 0.6667 | 0.8000 | 0.6667 |
| sbert_pretrained | Mechanical Engineer | 3 | 0.6667 | 0.6667 | 0.6667 | 0.7222 |
| sbert_pretrained | Senior Software Engineer | 3 | 0.0000 | 0.1111 | 0.1333 | 0.2333 |
| sbert_pretrained | Data Engineer | 2 | 0.0000 | 0.0000 | 0.2000 | 0.1806 |
| sbert_multi_qa_minilm_l6_cos_v1 | Data Science Engineer | 13 | 0.6154 | 0.6154 | 0.6000 | 0.7059 |
| sbert_multi_qa_minilm_l6_cos_v1 | Business Development Executive | 3 | 1.0000 | 0.8889 | 0.9333 | 1.0000 |
| sbert_multi_qa_minilm_l6_cos_v1 | Mechanical Engineer | 3 | 0.6667 | 0.7778 | 0.8667 | 0.7778 |
| sbert_multi_qa_minilm_l6_cos_v1 | Senior Software Engineer | 3 | 0.6667 | 0.6667 | 0.5333 | 0.6944 |
| sbert_multi_qa_minilm_l6_cos_v1 | Data Engineer | 2 | 0.0000 | 0.0000 | 0.3000 | 0.2500 |
