# AI Job Genie — Semantic Job Matching with Fine-Tuned Embeddings & Explainability

## Project Summary

An AI-powered career assistance platform that recommends jobs based on a user's resume, skills, and experience. Unlike standard job portals (LinkedIn, Naukri, Indeed) that rely on keyword matching or generic pretrained embeddings, this system **fine-tunes SBERT specifically on resume–job description pairs** so it understands domain-specific semantic meaning, and adds a **span-level explainability layer** so users can see exactly _why_ a job matched their resume, instead of an opaque match percentage. It also includes ATS resume scoring, skill gap analysis, voice-based search, and career insights as supporting features.

**One-line novelty statement:**

> "Existing platforms apply general-purpose sentence embeddings to job matching as-is and return an opaque match score. We fine-tune SBERT on domain-specific resume–job description pairs using contrastive learning and add a chunk-level explainability layer, improving both matching accuracy and user trust — validated against TF-IDF and generic-SBERT baselines."

---

## Objective

To reduce the time and effort of job searching by providing personalized, semantically accurate job recommendations that outperform generic embedding-based matching, while making every recommendation explainable and helping users understand resume quality and skill gaps.

---

## Tech Stack

| Layer                | Tools                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------- |
| Frontend             | Next.js                                                                                 |
| Backend              | Django REST Framework                                                                   |
| Database             | MySQL                                                                                   |
| Core ML              | `sentence-transformers` (SBERT), PyTorch                                                |
| Fine-tuning          | Contrastive learning — `MultipleNegativesRankingLoss` / Triplet Loss                    |
| Baselines            | TF-IDF + cosine similarity, generic pretrained SBERT                                    |
| Explainability       | Chunk-level cosine similarity heatmaps (extendable to attention/SHAP-based attribution) |
| Resume parsing / NLP | spaCy / NLTK, PDF/doc parsing libraries                                                 |
| Voice search         | Speech-to-text API (e.g., Web Speech API / Whisper)                                     |
| Job data             | External job APIs (e.g., Adzuna, RapidAPI job listings)                                 |
| Evaluation           | Precision@k, MRR, cosine similarity separation                                          |
| Version control      | Git/GitHub                                                                              |

---

## Step-by-Step Process

### Phase 1 — Data Preparation (for fine-tuning)

1. Source resume and job description datasets (Kaggle resume datasets, scraped/public job postings).
2. Since real "shortlisted" labels are rarely available, use **weak/distant supervision**: resumes and JDs from the same job category/title = positive pairs; cross-category = negative pairs.
3. Build training triples: `(resume_chunk, matching_JD_chunk, non-matching_JD_chunk)`.
4. Clean and chunk resumes/JDs into sentence/bullet-level segments for later explainability use.

### Phase 2 — Baseline Models (the "everyone does this" versions)

5. Implement TF-IDF + cosine similarity matching — **Baseline A**.
6. Implement generic pretrained SBERT (`all-MiniLM-L6-v2` or similar) matching — **Baseline B**.
7. Evaluate both on a held-out test set to establish what "off-the-shelf" performance looks like.

### Phase 3 — Fine-Tuning SBERT (core novelty #1)

8. Start from a pretrained SBERT checkpoint (transfer learning, not training from scratch).
9. Fine-tune using contrastive/triplet loss on your resume–JD training triples.
10. Tune hyperparameters (batch size, learning rate, epochs) and validate on a held-out split.
11. Save the fine-tuned model as your production matching engine.

### Phase 4 — Explainability Layer (core novelty #2)

12. Break resumes and JDs into chunks (sentences/bullets) at inference time.
13. Compute a similarity matrix between resume chunks and JD chunks using the fine-tuned model.
14. Surface the top-matching chunk pairs in the UI (e.g., "Your experience with 'Python, pandas' matched the requirement 'data analysis in Python'").
15. (Stretch goal) Add attention-based or Integrated Gradients attribution for token-level explanation.

### Phase 5 — System Integration

16. Wire the fine-tuned model into the existing pipeline: resume upload → parsing → embedding → similarity scoring → ranked recommendations.
17. Compute ATS compatibility score and skill gap analysis using the same embeddings/matching engine.
18. Integrate voice-based search and resume builder as supporting features around the core matching engine.

### Phase 6 — Evaluation (this is what makes it a thesis)

19. Compare **TF-IDF vs generic SBERT vs fine-tuned SBERT** on:
    - Precision@k / Recall@k
    - Mean Reciprocal Rank (MRR)
    - Cosine similarity separation between true-positive and true-negative pairs
20. Run a small qualitative user study (10–15 people) rating whether the chunk-level match explanations "make sense" and feel trustworthy.
21. Report improvement of fine-tuned model over both baselines with numbers/graphs.

### Phase 7 — Documentation

22. Write a Related Work section explicitly naming LinkedIn/Naukri/Indeed-style keyword matching and generic-SBERT matching, and state precisely what gap you're filling — pre-empt the "this already exists" critique directly in the thesis text.
23. Structure results as: Baseline Comparison → Fine-Tuning Methodology → Explainability Evaluation.

---

## Key Features (final list, novelty-first ordering)

- **Fine-tuned SBERT semantic matching** — domain-adapted embeddings, not generic off-the-shelf ones.
- **Chunk-level explainable matching** — shows _why_ a job matched, not just a score.
- Resume analysis with ATS compatibility scoring.
- Skill gap analysis for target jobs.
- Personalized job recommendations ranked by relevance.
- Voice-based job search.
- Resume builder and profile management.
- Career insights for employability improvement.

---

## One-Minute Viva Explanation

"AI Job Genie is a job recommendation platform that addresses two gaps in existing systems. First, platforms like LinkedIn or Naukri use keyword matching or, at best, generic pretrained sentence embeddings that weren't trained for resumes and job descriptions specifically — we fine-tuned SBERT using contrastive learning on domain-specific resume-JD pairs, which measurably improves match accuracy over both TF-IDF and generic SBERT baselines. Second, existing platforms give an opaque match score with no reasoning — we added a chunk-level explainability layer that shows exactly which parts of your resume matched which job requirements, so users can trust and understand the recommendation. The system also includes ATS scoring, skill gap analysis, voice-based search, and career insights, built using Next.js, Django REST Framework, and MySQL, with the matching engine validated against baseline methods using precision@k, MRR, and similarity separation metrics."

---

Want me to sketch the actual fine-tuning code (dataset construction + `sentence-transformers` training loop) so you can start building this part first, since it's the highest-risk piece of the timeline?
