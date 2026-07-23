# Phase 3 Findings — Hybrid TF-IDF/SBERT Matching

## TL;DR — the whole arc in 30 seconds

- **Contamination (section 5)**: `resume_data.csv` turned out to be a
  synthetic 344-profile × 28-category cross-product, not organic
  resume-JD pairs. The JD side's `responsibilities` field was a
  byte-identical, category-keyed template (28 unique values across 9,544
  rows) — every candidate JD within a category was textually identical, so
  the original "TF-IDF beats SBERT" headline (section 4) mostly reflected
  detecting a repeated template, not real matching. `matched_score` itself
  was very likely a formulaic label from that generation process.
- **Fix**: switched to real, independently-sourced job postings
  (`postings.csv`, 14 categories with genuine title coverage) and dropped
  `matched_score` entirely in favor of weak (category co-membership)
  supervision.
- **Category-label mismatch (section 7.1)**: even after that fix, Phase 2
  per-category results were wildly inconsistent (9 of 14 categories near
  P@1=0). Root cause: the *resume* side is still that same cross-product —
  identical text stamped with every category — so for 6 of 14 categories,
  **not one single test resume's actual content matched its stamped
  label**.
- **Fix**: deduplicated resumes down to their 344 real unique profiles and
  relabeled each by content (consensus of TF-IDF + 2 SBERT checkpoints),
  replacing the stamped `job_position_name`. Explicitly documented as
  partially circular for Phase 2's own baseline comparison (section 7.2).
- **Scope narrowing (section 7.4)**: content-derived labels concentrate
  heavily (Data Science Engineer = 36% of all profiles); categories with
  <25 profiles were dropped from evaluation as statistically unviable,
  leaving 5 categories / 251 resumes.
- **Noise-floor analysis (section 7.5–7.6)**: on the narrowed dataset
  (n=24 test queries), the three Phase 2 matchers' scores are not
  statistically distinguishable, and the ranking of "best matcher"
  literally flips depending on whether the dominant category is included
  or excluded — proof the comparison can't support a "matcher X wins"
  claim at this sample size.
- **Phase 3 decision (section 7.7)**: fine-tuning was **not re-attempted**
  — the Data Science Engineer-only training set (179 triples, ~44 gradient
  steps) is ~60x smaller than the already-modest original fine-tune, and
  any result would face the same n≈13 evaluation ceiling. The thesis
  pivots to presenting the investigative journey itself (contamination →
  circularity → mismatch → noise-floor) as the core methodological
  contribution.
- **Phase 4 matcher (section 7.8)**: TF-IDF, not the nominally
  higher-scoring `sbert_pretrained` — its perfect 1.0000 score on 13 test
  queries is a small-sample artifact, and TF-IDF keeps the explainability
  layer interpretable rather than embedding-based.

Summary of the Phase 3 fine-tuning + hybrid-matching investigation, for the
thesis writeup. Full numbers: `ml/results/phase3_results.json` and
`ml/results/phase3_comparison_table.md` (test split, `matched_score > 0.7`,
434 queries, 952 candidate JDs).

> **Read section 5 before citing section 4's numbers.** A dataset artifact
> discovered during Phase 4 means the "TF-IDF beats SBERT" headline below is
> substantially explained by a data leak, not by TF-IDF's matching ability.
> Section 5 quantifies it and section 6 states what the thesis should
> actually claim.

## 1. Fine-tuning worked, in isolation

SBERT (`all-MiniLM-L6-v2`) was fine-tuned with `MultipleNegativesRankingLoss`
on resume/JD triples (Phase 1). Triplet accuracy on the held-out val split
improved from `0.5496` (pretrained) to `0.6409` (fine-tuned) — the fine-tuning
procedure itself is sound and measurably changes the embedding space in the
intended direction.

## 2. The scale-mismatch bug

The first hybrid attempt combined TF-IDF and SBERT with a plain weighted sum:

```
final_score = alpha * tfidf_score + (1 - alpha) * sbert_score
```

This failed because the two matchers' raw cosine similarities live on
incomparable scales:

- **TF-IDF**: near-zero for most pairs with occasional sharp peaks
  (mean ≈ 0.03, std ≈ 0.07). The peakiness *is* its ranking signal — a
  resume's true match stands out because almost everything else scores ~0.
- **SBERT**: clusters tightly near 1.0 (mean ≈ 0.89, std ≈ 0.04), a typical
  symptom of embedding anisotropy in sentence-transformer output.

Summing raw scores let SBERT's small-but-nonzero variance perturb TF-IDF's
sharp ranking regardless of `alpha`, since SBERT's absolute magnitude
dominated the sum. **Fix**: per-query (per-row) min-max normalization —
scale each matcher's similarity row independently to `[0, 1]` before
blending — so `alpha` controls each matcher's actual contribution to the
*ranking*, not to raw magnitude. Implemented as
`hybrid_tfidf_sbert_normalized` in `ml/src/models/hybrid_baseline.py`
(`_row_min_max_normalize`).

Normalization measurably helped the hybrid (test MRR `0.1118` raw →
`0.1275` normalized), confirming the diagnosis — but did not close the gap
to TF-IDF alone (test MRR `0.1496`).

## 3. Extended alpha sweep: still never beats TF-IDF alone

The alpha sweep was extended toward TF-IDF-heavy weightings (approaching
`1.0`) to check whether the hybrid could recover TF-IDF's standalone
performance as `alpha → 1`. One point looked promising:

- `alpha = 0.85` (normalized): **val MRR = 0.1357**, the best hybrid result
  seen on val.

That result did not hold up. Re-evaluated on the held-out test split, it
collapsed to **MRR = 0.1103** — worse than the other hybrid alphas and far
below TF-IDF alone. Diagnosis: overfitting to the val split, which is small
(464 queries vs. 434 on test) — a single alpha value that happens to rank
well on val's specific query set is not a generalizable choice. The
production alpha sweep was pulled back to the original, less overfit-prone
grid (`[0.3, 0.5, 0.7]`, selected by val MRR); the `0.85` exploratory point
was not kept as a reported result.

## 4. Conclusion

Across every approach tried — TF-IDF, generic pretrained SBERT, a
retrieval-tuned SBERT checkpoint (`multi-qa-MiniLM-L6-cos-v1`), the
fine-tuned SBERT, the raw hybrid, and the normalized hybrid — **TF-IDF alone
remains the strongest matcher** on this dataset's held-out test split:

| matcher | test MRR |
|---|---|
| **tfidf** | **0.1496** |
| hybrid_tfidf_sbert_normalized | 0.1275 |
| hybrid_tfidf_sbert (raw) | 0.1118 |
| sbert_pretrained | 0.0559 |
| sbert_finetuned | 0.0547 |
| sbert_multi_qa_minilm_l6_cos_v1 | 0.0404 |

Fine-tuning improved SBERT's *triplet* accuracy but never translated into a
better *ranking* than TF-IDF on this domain/dataset — plausible drivers are
the dataset's short, keyword-dense resume/JD fields (where lexical overlap
is a strong signal) and the modest fine-tuning set size (~10.7k triples).
This motivates the pivot away from `docs/PROJECT_SPEC.md`'s original plan
(fine-tuned SBERT as the production matcher): **Phase 4 builds the
explainability layer around TF-IDF instead**, since it is the
best-performing matcher on every split tested.

**This conclusion needed revision — see section 5.**

## 5. ⚠️ Ground truth contamination via literal text duplication

Discovered while building Phase 4's chunk-level explainability demo: the
demo's top chunk matches were all exact-duplicate text scoring a suspicious
`1.000` cosine similarity. Tracing it back to the raw dataset
(`ml/data/raw/resume_data.csv`) found the cause.

### What's contaminated

The paired dataset's `columns.paired` config (`ml/config/config.yaml`)
concatenates `responsibilities` into every resume's text and
`responsibilities.1` into every JD's text. Checked across the **entire raw
dataset** (not just the 4,468 rows that clear the `matched_score > 0.7`
positive-pair threshold):

| resume field | job field | exact-match rate |
|---|---|---|
| `responsibilities` | `responsibilities.1` | **9,544 / 9,544 rows (100%)** |
| `skills` | `skills_required` | 10 / 9,544 (0.1% — all 10 are both-`NaN` coincidences, not real overlap) |
| `positions` | `job_position_name` | 0 / 9,544 (0%) |
| `career_objective` | `experiencere_requirement` | 686 / 9,544 (7.2% — all 686 are both-`NaN` coincidences) |
| `career_objective` | `educationaL_requirements` | 0 / 9,544 (0%) |

So the duplication is isolated to one field pair, and it is **not**
specific to the positive-labeled rows — every single row in the dataset has
it, independent of `matched_score`. It's a structural property of how the
dataset was built, not an artifact of the positive-pair selection logic in
`src/data/pairing.py`.

### It's a category template, not per-row content

`responsibilities` has only **28 unique values across all 9,544 rows** —
and there are exactly **28 unique job categories** (`job_position_name`).
Checked directly: it's a strict 1:1 mapping (every category has exactly one
`responsibilities` template; every template belongs to exactly one
category). Concretely, *every* resume and *every* JD in, say, "DevOps
Engineer" carries the identical ~9-line boilerplate block — it doesn't
distinguish the true matching pair from any other same-category candidate,
it only distinguishes category membership.

That block is not a minor addition: median length 181 characters, which is
**~50% of the median JD's total text length** and **~29% of the median
resume's total text length** (`resumes["text"]` / `jobs["text"]` after
`load_paired_dataset`'s field concatenation). Roughly half of what a JD
"says," text-wise, is this shared template.

### It's worse than one field — the whole JD side is a 28-way template, and the dataset is a synthetic cross-product

Checking the *entire* concatenated `jd_text` (all 5 `job_fields`, not just
`responsibilities.1`) directly: **`jobs["text"].nunique() == 28`** — there
are only 28 distinct job descriptions in the whole 9,544-row dataset, one
per category, every JD row within a category is byte-identical to every
other. This holds before *and* after dropping `responsibilities.1`
(`educationaL_requirements`, `experiencere_requirement`, `skills_required`
are each, individually, also a strict 1-value-per-category template — same
check as `responsibilities`: N unique values dataset-wide == N unique
categories that carry a non-null value for that field). **There is no
per-JD-instance text anywhere in this dataset's job side.** The
`ablation_no_responsibilities.py` ablation above didn't reveal a harder but
still-solvable task — it revealed that the task was never solvable beyond
category classification, with or without `responsibilities.1`.

Checking the resume side explains why: grouping rows by the
`(career_objective, skills, positions)` triple gives **344 unique resume
profiles across 9,544 rows, each appearing in the data exactly 28 times**
— once per category. This is a full cross-product: 344 base resume
profiles × 28 job categories ≈ 9,632 rows (9,544 after whatever filtering
produced the released file). `responsibilities`/`matched_score` are then
filled in per (profile, category) pair — i.e. **the dataset was
synthetically generated by pairing every resume against every category**,
not sourced as organic resume-JD pairs with a real hiring outcome. This is
strong circumstantial evidence `matched_score` itself is a formulaic
label from that generation process rather than a human/ATS-verified
relevance judgment — worth stating as an assumption if the thesis relies on
`matched_score` as ground truth.

### Quantifying the impact: an ablation

`ml/scripts/ablation_no_responsibilities.py` re-runs the core Phase 2/3
matchers with `responsibilities`/`responsibilities.1` excluded from
`resume_fields`/`job_fields`, same test split, same
`min_matched_score=0.7` queries (434), same 952-candidate pool. Result
(`ml/results/ablation_no_responsibilities.md`):

| matcher | MRR (original, field included) | MRR (ablated, field excluded) |
|---|---|---|
| tfidf | 0.1496 | **0.0072** |
| sbert_pretrained | 0.0559 | 0.0101 |
| sbert_finetuned | 0.0547 | 0.0138 |

For reference, the **theoretical MRR of a uniformly random ranking** over
this pool (952 candidates) is `H_952 / 952 ≈ 0.0078` (harmonic-number
formula for expected reciprocal rank of a uniformly-placed true item).

**TF-IDF's ablated score (0.0072) is statistically indistinguishable from
random guessing.** Given the JD side has zero per-instance text (previous
subsection), this is expected, not surprising: within any category, every
candidate JD is textually identical, so no matcher can outperform random
tie-breaking among same-category candidates no matter what text survives
on the resume side. SBERT (pretrained and fine-tuned) fares slightly better
than random, and fine-tuned SBERT is now the *best* of the three ablated
matchers — the opposite ranking from section 4's headline table — though
"slightly better than random on an unsolvable task" is a thin signal to
lean on.

This lines up with what the ablated Phase 4 explainability demo shows
directly (`ml/results/phase4_explainability_demo.md`, generated by
`ml/scripts/demo_explainability.py` after excluding exact-duplicate
chunks): the surviving "genuine" top matches score `0.00`–`0.06` and are
frequently semantically nonsensical (e.g. a resume's "Fluent in Spanish"
bullet "matching" a JD's job-title chunk) — consistent with a signal that's
barely above noise.

## 6. Revised interpretation for the thesis

**Section 4's comparison table should not be presented at face value.** It
answers "which matcher best detects a repeated 28-way category template,"
not "which matcher best determines that *this specific* resume matches
*this specific* job description." Those are different tasks, and the
second one — the one the thesis's novelty claim is actually about — is not
just "close to unsolved" but **mathematically unsolvable on this dataset as
released**, because the JD side carries zero per-instance information:
ranking against "952 candidate JDs" is really ranking against duplicates of
28 fixed documents, and the true match is textually indistinguishable from
every other same-category candidate regardless of which fields are kept.

Recommended framing for the writeup:

1. **Disclose this prominently as a limitation**, not a footnote — it
   materially changes what the Phase 2/3 numbers mean, and it's a
   structural property of the dataset (a 344-profile × 28-category
   cross-product), not a bug fixable by dropping one column. Reviewers/
   examiners who dig into the dataset construction would find this
   quickly, and it's far better to have found and disclosed it first.
2. **Don't claim "TF-IDF beats SBERT" as a domain finding.** The evidence
   now points the other way once the leak is closed: TF-IDF collapses to
   exactly random chance, fine-tuned SBERT retains a small (if weak) real
   edge. That contrast is worth reporting, but neither number should be
   read as "solving" resume-JD matching — both are close to the
   information-theoretic ceiling this dataset allows.
3. **Report both numbers side by side** (contaminated vs. ablated) rather
   than picking one silently — the contrast itself is informative and
   demonstrates methodological rigor (a thesis committee will likely credit
   catching this more than they'd penalize the dataset's flaw).
4. **Dropping fields will not fix this** — every job-side field is a
   category template, so no combination of `job_fields` yields per-JD
   variation. The real fix is a different data source for the JD side:
   `docs/PROJECT_SPEC.md`'s Phase 1 already lists "scraped/public job
   postings" as an alternative to the Kaggle resume dataset — that's now a
   requirement, not an option, if Phase 6's headline comparison is meant to
   demonstrate genuine content-level matching. Positioning: keep this
   dataset for the fine-tuning *procedure* (triplet construction,
   contrastive loss mechanics all still work correctly - section 1's
   triplet-accuracy improvement is real), but source real per-posting JD
   text for the *evaluation* Phase 6 reports as the headline result.
5. `matched_score` itself is very likely a formulaic label from the
   344×28 cross-product generation process, not a human/recruiter-verified
   relevance judgment — state this assumption explicitly if the thesis
   leans on `matched_score` as ground-truth relevance.
6. Phase 4's explainability layer is still built on TF-IDF per section 4's
   original (now-revised) reasoning — that choice doesn't need to change
   immediately, but its justification does: it's the cheapest matcher to
   make explainable, not a demonstrated winner. `demo_explainability.py`
   now excludes exact-duplicate chunk matches for this reason, and reports
   how many qualifying resumes have *no* genuine (non-template) match at
   all — that count is itself worth citing, and will likely need to be
   re-run once/if real JD text replaces the templated job side.

## 7. Category label mismatch (after switching to real postings.csv) — and the fix

Section 6 recommended sourcing real per-posting JD text instead of the
templated job side, which was done: `ml/src/data/loader.py` now loads
`ml/data/raw/postings.csv` (real LinkedIn postings, filtered to the 14
categories with genuine title coverage — `src/data/category_keywords.py`)
independently from the resume side, with `matched_score` dropped entirely.
Weak (category co-membership) supervision replaced it, matching what
section 6 already flagged as the honest ground-truth signal available.

Phase 2 baselines were re-run on this real, independently-sourced dataset
(`ml/results/baseline_results.json`, 476 test queries, 4,012 candidate JDs).
Overall numbers looked reasonable (tfidf MRR `0.1259`, sbert_pretrained MRR
`0.1286`), but the **per-category breakdown was not**:

| category | tfidf P@1 | tfidf MRR | sbert P@1 | sbert MRR |
|---|---|---|---|---|
| Manager- HRM | 0.000 | 0.009 | 0.000 | 0.018 |
| Network Support Engineer | 0.000 | 0.015 | 0.000 | 0.046 |
| Marketing Officer | 0.000 | 0.054 | 0.000 | 0.014 |
| Full Stack Developer | 0.000 | 0.028 | 0.000 | 0.063 |
| Civil Engineer | 0.000 | 0.058 | 0.029 | 0.045 |
| DevOps Engineer | 0.000 | 0.052 | 0.059 | 0.121 |
| Data Science Engineer | 0.412 | 0.478 | 0.441 | 0.522 |

9 of 14 categories score near-zero P@1 for every matcher; Data Science
Engineer is a dramatic outlier in the other direction.

### 7.1 Root cause: the resume side is still a cross-product, and content doesn't follow the label

Section 5 already established that `resume_data.csv` is a synthetic
cross-product — 344 unique `(career_objective, skills, positions)` profiles,
each stamped with all 28 (14, post-filtering) category labels. At the time
that fact was used to explain `matched_score`'s formulaic nature. It turns
out it independently corrupts the category label itself, with or without
`matched_score`: because `career_objective`/`skills`/`positions` are
**byte-identical across all 14 category-copies of a profile**, a profile's
actual text content never changes based on which category it's stamped
with — the label is arbitrary with respect to content for most rows.

Quantified directly: for each unique test-split resume text, the target
category whose job pool it's most textually similar to (mean TF-IDF cosine
similarity) was compared against its stamped label.

| stamped category | mismatch rate (stamped label ≠ best content-match category) |
|---|---|
| Database Administrator (DBA) | 100.00% |
| Full Stack Developer | 100.00% |
| DevOps Engineer | 100.00% |
| Network Support Engineer | 100.00% |
| Manager- HRM | 100.00% |
| Marketing Officer | 100.00% |
| Data Engineer | 97.06% |
| Civil Engineer | 97.06% |
| Project Coordinator (Civil) | 97.06% |
| Business Development Executive | 97.06% |
| Senior Software Engineer | 88.24% |
| System Administrator | 88.24% |
| Mechanical Engineer | 85.29% |
| Data Science Engineer | 64.71% |

For 6 of 14 categories, **not one single test resume** stamped with that
label has content that actually best-matches it. Even the best case (Data
Science Engineer) is 65% mismatched — it only "wins" on the aggregate
metrics because a disproportionate share of the underlying 344 profiles
happen to be ML/data-science-flavored career objectives, so it acts as an
attractor for resumes stamped with *other* labels too. Spot-checked
examples: a resume stamped "Manager- HRM" whose actual text is *"Machine
Learning Engineer seeking roles in... NLP, TensorFlow"* ranks a Data
Scientist JD at 0.317 cosine similarity vs. 0.023 for the best HR Manager
JD — the resume is genuinely about ML, not HR; the label is simply wrong
for that instance. Intra-category JD-JD similarity was checked too (mean
0.12–0.22, well above the resume-JD similarity floor of ~0.02–0.04) to rule
out the job side being the problem — postings within a category are
internally coherent; the resumes are what's mislabeled.

This is not a pipeline bug (vocabulary, normalization, or matcher choice)
— no matcher can rank a resume's stamped category highly when the resume's
actual content has nothing to do with it. It is the same structural
cross-product artifact as section 5, manifesting a second, independent way.

### 7.2 Fix: deduplicate, then relabel by content

Two changes, applied together rather than relabeling the cross-product in
place:

1. **Deduplicate** `resume_data.csv` down to its 344 unique profiles
   (`src/data/loader.py`'s `load_resume_profiles`) — the 14x-28x
   category-stamped copies are dropped entirely rather than kept and
   relabeled individually, since they carry no independent information
   (identical text, per 7.1).
2. **Relabel by content**: each of the 344 profiles is assigned the target
   category its text is most similar to, using a **consensus of three
   matchers** (TF-IDF + `all-MiniLM-L6-v2` + `multi-qa-MiniLM-L6-cos-v1` —
   each matcher's per-category mean similarity row-min-max normalized, then
   averaged; argmax over categories) rather than any single matcher
   (`src/data/relabel.py`, `scripts/build_resume_labels_cache.py`).

**Why consensus, not TF-IDF alone (the method used for the 7.1 diagnosis):**
using TF-IDF similarity to both assign ground truth *and* evaluate the
TF-IDF baseline would be circular — TF-IDF's Phase 2 P@k/MRR would be
inflated near-tautologically (it would tend to rediscover the very label
it assigned), while SBERT wouldn't share that inflation, making the two
baselines' scores incomparable. An independent, non-circular signal was
considered (keyword-matching each profile's `positions` field against
`src/data/category_keywords.py`'s title patterns — the same patterns
already used to label the job side) but only covers 52/344 profiles (15%),
too sparse to relabel everyone on its own. Consensus across all three
Phase 2 matchers was the practical middle ground.

> **⚠️ Limitation, stated plainly**: this does not eliminate the
> circularity, only spreads it evenly across the three Phase 2 baselines.
> Every matcher evaluated in Phase 2 on this relabeled data had a hand in
> defining the ground truth it is scored against. **Phase 2's numbers on
> the relabeled data are not a clean baseline comparison** and must not be
> presented as one — in particular, they should not be compared
> apples-to-apples against Phase 3's fine-tuned SBERT model, which
> contributes nothing to labeling and therefore gets no such advantage.
> `scripts/run_baselines.py`'s output carries this caveat in its config
> block and results table for the same reason.

### 7.3 Before/after category distribution — concentration got worse, not better

| category | stamped (cross-product, n≈341/category) | content-derived (n=344 total) |
|---|---|---|
| Data Science Engineer | ~341 (uniform) | **124 (36.0%)** |
| Mechanical Engineer | ~341 | 37 (10.8%) |
| Senior Software Engineer | ~341 | 35 (10.2%) |
| Business Development Executive | ~341 | 29 (8.4%) |
| Data Engineer | ~341 | 26 (7.6%) |
| System Administrator | ~341 | 15 (4.4%) |
| Manager- HRM | ~341 | 14 (4.1%) |
| Full Stack Developer | ~341 | 13 (3.8%) |
| Database Administrator (DBA) | ~341 | 12 (3.5%) |
| Civil Engineer | ~341 | 12 (3.5%) |
| Marketing Officer | ~341 | 9 (2.6%) |
| Network Support Engineer | ~341 | 8 (2.3%) |
| Project Coordinator (Civil) | ~341 | 7 (2.0%) |
| DevOps Engineer | ~341 | 3 (0.9%) |

The stamped labels were artificially uniform (~341/category, by
cross-product construction); the honest, content-derived distribution is
heavily concentrated — Data Science Engineer alone accounts for over a
third of all 344 profiles, while DevOps Engineer drops to 3. Only 116/344
profiles (34%) had all three matchers individually agree on the same
category before consensus-averaging resolved the rest, underscoring how
much genuine ambiguity content-based labeling has to arbitrate.

### 7.4 Scope narrowing: dropping categories without enough resumes to evaluate

A stratified 80/10/10 split on the content-derived labels leaves several
categories with too few resumes for a statistically meaningful val/test
split (e.g. DevOps Engineer, n=3, would get 2 train / 0 val / 1 test).
Reporting per-category P@k/MRR on n<10 would not survive scrutiny, so
`config.yaml`'s `categories.target` was narrowed to the 5 categories with
≥25 content-relabeled profiles, leaving `categories.all` (the original 14)
as the labeling candidate pool so this scope change doesn't feed back into
how profiles are labeled (`src/data/relabel.py` / `scripts/build_resume_labels_cache.py`
always label against all 14 regardless of `categories.target`):

| category | n | share of narrowed pool (n=251) |
|---|---|---|
| Data Science Engineer | 124 | **49.4%** |
| Mechanical Engineer | 37 | 14.7% |
| Senior Software Engineer | 35 | 13.9% |
| Business Development Executive | 29 | 11.6% |
| Data Engineer | 26 | 10.4% |

Concentration is *worse* as a share of the narrowed pool than of the full
344 (36.0% → 49.4%) — narrowing removes categories, not Data Science
Engineer's dominance within what remains. This is disclosed here rather
than smoothed over.

The resulting stratified split gives every surviving category non-trivial
train/val/test coverage:

| category | total | train | val | test |
|---|---|---|---|---|
| Business Development Executive | 29 | 23 | 3 | 3 |
| Data Engineer | 26 | 21 | 3 | 2 |
| Data Science Engineer | 124 | 99 | 12 | 13 |
| Mechanical Engineer | 37 | 30 | 4 | 3 |
| Senior Software Engineer | 35 | 28 | 4 | 3 |

Val/test sizes (2–13 per category) are still small enough that per-category
metrics should be read as indicative, not precise — but this is an honest
reflection of how many genuinely-relevant resume profiles this dataset
actually contains per category, not an artifact of a fixable pipeline
choice. The 9 dropped categories are out of scope for Phase 2/3 evaluation
until a supplementary resume source can provide real per-category volume.

### 7.5 Phase 2 re-run on the narrowed dataset — and Data Science Engineer is carrying the result

`scripts/build_dataset.py` and `scripts/run_baselines.py` were re-run on the
narrowed 251-resume, 5-category, 2,174-JD pool (`ml/results/baseline_results.json`).
Test split: **24 queries total.**

| matcher | P@1 | P@3 | P@5 | MRR |
|---|---|---|---|---|
| tfidf | 0.5833 | 0.6667 | 0.6500 | 0.7070 |
| sbert_pretrained | 0.6667 | 0.6944 | 0.7083 | 0.7595 |
| sbert_multi_qa_minilm_l6_cos_v1 | 0.6250 | 0.6250 | 0.6417 | 0.7122 |

**Per-category, with test-set size flagged for reliability:**

| matcher | category | test n | reliability | P@1 | MRR |
|---|---|---|---|---|---|
| tfidf | Data Science Engineer | 13 | ✅ ok | 0.6923 | 0.8077 |
| tfidf | Business Development Executive | 3 | ⚠️ illustrative only | 0.6667 | 0.8333 |
| tfidf | Mechanical Engineer | 3 | ⚠️ illustrative only | 0.6667 | 0.8333 |
| tfidf | Senior Software Engineer | 3 | ⚠️ illustrative only | 0.3333 | 0.4087 |
| tfidf | Data Engineer | 2 | ⚠️ illustrative only | 0.0000 | 0.1214 |
| sbert_pretrained | Data Science Engineer | 13 | ✅ ok | 1.0000 | 1.0000 |
| sbert_pretrained | Business Development Executive | 3 | ⚠️ illustrative only | 0.3333 | 0.6667 |
| sbert_pretrained | Mechanical Engineer | 3 | ⚠️ illustrative only | 0.6667 | 0.7222 |
| sbert_pretrained | Senior Software Engineer | 3 | ⚠️ illustrative only | 0.0000 | 0.2333 |
| sbert_pretrained | Data Engineer | 2 | ⚠️ illustrative only | 0.0000 | 0.1806 |
| sbert_multi_qa | Data Science Engineer | 13 | ✅ ok | 0.6154 | 0.7059 |
| sbert_multi_qa | Business Development Executive | 3 | ⚠️ illustrative only | 1.0000 | 1.0000 |
| sbert_multi_qa | Mechanical Engineer | 3 | ⚠️ illustrative only | 0.6667 | 0.7778 |
| sbert_multi_qa | Senior Software Engineer | 3 | ⚠️ illustrative only | 0.6667 | 0.6944 |
| sbert_multi_qa | Data Engineer | 2 | ⚠️ illustrative only | 0.0000 | 0.2500 |

Only Data Science Engineer (n=13) clears single digits. The other 4
categories (n=2–3) are single-query-swing territory — e.g. Data Engineer's
P@1=0.0000 for two matchers means "0 out of 2," not a stable rate.

**Sensitivity check: does Data Science Engineer (49% of the pool) carry the
aggregate, and does it do so evenly across matchers?**

| matcher | overall MRR (n=24) | DS Engineer only (n=13) | excl. DS Engineer (n=11) |
|---|---|---|---|
| tfidf | 0.7070 | 0.8077 | 0.5881 |
| sbert_pretrained | 0.7595 | **1.0000** | **0.4753** |
| sbert_multi_qa | 0.7122 | 0.7059 | 0.7197 |

No — and not evenly. **sbert_pretrained's apparent lead is almost entirely
a Data Science Engineer artifact**: a perfect 1.0000 MRR on that one
category, but the *worst* of the three (0.4753, below tfidf) on everything
else. tfidf sits in between (0.8077 → 0.5881, a real but smaller gap).
**sbert_multi_qa is the most robust** — its MRR is essentially flat with or
without DS Engineer (0.7059 vs 0.7197) — and on the excl.-DS-Engineer slice
it is the clear best of the three.

**The ranking flips depending on which slice is read**: overall says
sbert_pretrained > sbert_multi_qa > tfidf; excluding the dominant category
says sbert_multi_qa > tfidf > sbert_pretrained. A single aggregate number
from this run is not just imprecise, it is actively misleading about which
matcher "wins" — any claim drawn from it must specify which slice it's
citing, and given the n=2–3 categories that produce the swing, even the
excl.-DS-Engineer numbers should be read as directional, not conclusive.

### 7.6 Recommended framing for the thesis

**The Phase 2 numbers above do not support a "matcher X wins" claim, and
that headline should be dropped, not hedged.**

1. **The noise floor swamps the signal.** At n=24, the standard error on a
   proportion near 0.5 is ~10 points; a 95% CI is roughly ±20 points. The
   actual spread between matchers (P@1: 0.58–0.67, a 9-point range) sits
   *inside* that noise band — these three matchers are not statistically
   distinguishable at this sample size.
2. **The ranking-flip in 7.5 is proof of non-robustness, not just a
   caveat.** If the winner reverses depending on whether one category
   (49% of the pool) is included, the comparison is not stable enough to
   support a directional claim in either direction.
3. **The excl.-DS-Engineer comparison — arguably the fairer one — runs on
   n=11**, built from categories of 2–3 each: anecdote regime, where one
   query flipping moves MRR by ~9 points.
4. **The labeling-circularity caveat (7.2) compounds this independently of
   sample size.** Even with 10x the data, these would not be clean
   baseline numbers, since the labels were partly defined by the matchers
   being scored.

**Recommended reframe**: present this phase's contribution as
methodological, not competitive. The pipeline is real and working
end-to-end (weak supervision, stratified splitting, chunk-level
explainability); the actual findings worth building the thesis around are
the two independent data-integrity problems discovered and diagnosed with
reproducible evidence — `matched_score`/template contamination (section 5)
and category-label mismatch (section 7) — plus the noise-floor analysis
above showing why a naive baseline comparison on this dataset would have
been methodologically unsound. That is a stronger and more defensible
thesis contribution than a numeric leaderboard, which a committee member
could challenge by re-deriving the same confidence intervals in five
minutes.

**Consequence for Phase 3**: whatever fine-tuned-vs-baseline comparison
runs next hits the same n≈24 test ceiling, since it draws on the same
underlying resume pool — no amount of fine-tuning effort fixes an
evaluation-set-size problem. Before re-attempting Phase 3, the thesis's
quantitative claim should be narrowed to **Data Science Engineer alone**
(124 resumes, 99 train / 12 val / 13 test — the one category with real
statistical volume), with the other 4 categories reported qualitatively /
as case studies rather than as numeric comparisons.

### 7.7 Phase 3 go/no-go decision: not re-attempted

Before spending compute on a fine-tune, the Data Science Engineer-only
slice's actual training volume was checked against
`data/processed/triples.jsonl`: **179 train / 20 val / 25 test triples**
(from the 99/12/13 resume split, `triples.chunks_per_resume=3`).

For context, the original (pre-investigation) Phase 3 fine-tune used ~10.7k
triples across all 14 categories, and that was already characterized as a
"modest, transfer-learning-sized dataset" (`config.yaml` comment). 179
train triples is **~60x smaller** than that already-modest baseline.

**Two independent problems, not one:**

1. **Training signal is too thin to trust.** With `training.batch_size=16`
   and `NO_DUPLICATES` batch sampling: 179 triples ÷ 16 ≈ 11 batches/epoch
   × `training.epochs=4` ≈ **~44 total gradient steps** — too few to
   meaningfully reshape the embedding space, but enough to risk overfitting
   to 99 specific resumes' vocabulary quirks. The 20-triple val set used
   for `load_best_model_at_end` checkpoint selection is equally thin —
   closer to selecting on noise than on genuine generalization.
   Concretely, `training.save_eval_steps=200` exceeds the ~44 total
   training steps, so the current config would never reach a mid-training
   checkpoint+eval — this is a config/scale mismatch, not just a
   statistical concern, and would need fixing before the script could even
   run correctly at this scale.
2. **Evaluation signal is too thin to confirm a result either way**, even
   if training somehow worked. A fine-tuned model would still be scored
   against the same 13 Data Science Engineer test resumes — the identical
   noise-floor problem quantified in 7.5/7.6 (±20-point 95% CIs at n=24;
   worse at n=13). Any few-point MRR improvement from fine-tuning would be
   indistinguishable from noise.

**Decision: Phase 3 fine-tuning is not re-attempted.** Both the input
(training triples) and the output (test queries) sides of the fine-tuning
loop are undersized for this dataset's real volume, so the compute cost of
attempting it would not be repaid with a trustworthy result — the same
trap this investigation avoided for `matched_score` (section 5) and the
category label (section 7.1–7.2) would simply reappear as an unexaminable
fine-tuning number instead. The thesis pivots directly to **Phase 4
(explainability)**, built on the strongest baseline matcher from the Data
Science Engineer slice, with the full Phase 1–3 investigative journey —
contamination, circularity, category mismatch, noise-floor analysis —
presented as the thesis's core methodological contribution rather than a
"fine-tuned model beats baselines" claim.

One open item for the Phase 4 handoff: the nominal "best" DS-Engineer-slice
matcher is `sbert_pretrained` (P@1/MRR = 1.0000, 13/13) but a perfect score
on n=13 is fragile (one miss drops it to 0.92) and more likely reflects
that separating 5 fairly distinct categories is an easy coarse task than
that fine-grained resume-JD matching is solved. `tfidf`'s 0.6923 P@1 on the
same slice is more textured, and `src/explainability/chunk_similarity.py`
is already built around a fitted `TfidfVectorizer` rather than a black-box
embedding — TF-IDF remains the more defensible choice for Phase 4 pending
a final decision.

### 7.8 Phase 4 matcher decision: TF-IDF

**Decided: Phase 4's explainability layer is built on TF-IDF**, not
`sbert_pretrained`, despite the latter's nominally higher DS-Engineer-slice
score. Rationale: a perfect 1.0000 P@1/MRR on 13 test queries is a
small-sample artifact (one miss drops it to 0.92), not evidence of
genuinely better matching — consistent with 7.6's broader point that
scores at this n aren't reliable enough to pick a winner from. TF-IDF's
imperfect-but-textured 0.6923 P@1 is the more trustworthy signal, and it
keeps Phase 4 aligned with `src/explainability/chunk_similarity.py`'s
existing interpretable, `TfidfVectorizer`-based design rather than
introducing a black-box embedding into the explanation path.

## Addendum (Phase 5): mid-word spacing artifacts in postings.csv

Phase 5's ATS keyword-overlap/skill-gap feature occasionally surfaces
garbled tokens (e.g. `"cellent"`, `"equired"`, `"xperience"`, `"rong"`,
`"kills"`). Traced to source: some job postings in `postings.csv` have
literal spurious space characters splitting single words in half, e.g.
`"st rong pay, ex cellent growth"` and `"R equired E xperience"` (doc
index 284 in the target-category pool is a clear example). This is a
PDF-scraping artifact in the original data, not a tokenizer bug — the
words are already split by real whitespace before any tokenization runs,
so `TfidfVectorizer`'s tokenizer is behaving correctly on messy input.
Confirmed by inspecting raw source text directly; left as-is, since a
corpus-wide rejoin heuristic risks false-positive merges elsewhere and is
out of scope relative to the size of the issue.
