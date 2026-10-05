# LLM Usage Log & Verification Record

This document records the interaction, adaptation, and empirical verification of Large Language Model (LLM) suggestions utilized during the engineering of the Document Similarity Engine.

---

## 1. LLM Prompt & Usage Log

| Sr. No. | Purpose | Actual Prompt | How Response Was Used | Modification / Verification |
|---|---|---|---|---|
| 1 | Algorithm Selection | *"Compare Jaccard, Bag-of-Words, and TF-IDF with Cosine similarity for parallel document similarity computation in terms of time/space complexity and load balancing."* | Used to structure literature evaluation and select TF-IDF over sparse vectors. | Verified theoretical complexity $O(N^2 \cdot K)$ and confirmed L2 normalization eliminates vector magnitude division inside pairwise loops. |
| 2 | Parallelization Design | *"How should I parallelize an upper triangular loop $i \in [0, N-2]$, $j \in [i+1, N-1]$ in OpenMP without load imbalance or race conditions?"* | Used OpenMP `#pragma omp parallel for schedule(dynamic, chunk)` and thread-local result vectors. | Rejected static schedule suggestion; empirically verified dynamic schedule eliminated load imbalance across threads. |
| 3 | Sparse Representation | *"What is the most cache-efficient C++ sparse vector representation for fast dot product computation?"* | Selected `std::vector<std::pair<int, double>>` sorted by `term_id`. | Verified sorted two-pointer merge traversal achieved $O(|A| + |B|)$ linear time per document pair comparison. |
| 4 | OpenMP Vocabulary Accumulation | *"How can I count document frequencies across parallel threads without global std::mutex locking on every token?"* | Constructed thread-local `unordered_map` instances per OpenMP thread, followed by thread-sequential dictionary merge. | Verified zero lock contention and complete thread safety during parallel pre-processing. |
| 5 | Correctness Testing | *"Suggest comprehensive edge cases for verifying document similarity correctness."* | Created unit tests for identical documents ($\text{sim} = 1.0$), orthogonal documents ($\text{sim} = 0.0$), stopword-only texts, and numerical equivalence. | Ran tests against sequential engine with tolerance $< 10^{-6}$. |
| 6 | Synthetic Data Generator | *"Write a Python script to generate synthetic document corpora with controlled document lengths, topic clusters, and near-duplicate pairs."* | Adopted script structure for `generate_dataset.py`. | Added deterministic seed configuration and verified near-duplicate generation triggers expected $\ge 0.85$ similarity scores. |
| 7 | Benchmarking & Visualization | *"How to script automated benchmarking in Python and visualize execution time vs. document count and speedup vs. OpenMP threads using matplotlib?"* | Created `run_benchmarks.py` and `plot_results.py`. | Formatted CSV outputs and validated speedup formulas against empirical runs. |
| 8 | Documentation & viva | *"Formulate key viva questions and technical explanations for OpenMP schedule directives, thread-local accumulation, and Amdahl's Law bottlenecks."* | Structured `VIVA_PREPARATION.md` Q&A sections. | Cross-referenced answers with C++17 standard and OpenMP 5.0 specifications. |

---

## 2. Detailed Case Study: Modified / Rejected LLM Suggestion

### LLM Suggestion
During initial algorithm design, the LLM suggested using `#pragma omp parallel for collapse(2)` across the nested pairwise loops:

```cpp
// LLM Initial Suggestion (REJECTED)
#pragma omp parallel for collapse(2) schedule(static)
for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
        double sim = compute_sparse_dot_product(vec[i], vec[j]);
        if (sim >= threshold) {
            #pragma omp critical
            matches.push_back({i, j, sim});
        }
    }
}
```

### Why It Appeared Reasonable
- `collapse(2)` combines nested loops into a single iteration space, theoretically maximizing parallel granularity.
- `#pragma omp critical` guarantees thread-safe insertion into the shared vector `matches`.

### Problem Discovered During Technical Analysis
1. **Invalid Loop Bounds for Collapse**: In OpenMP, `collapse(2)` requires rectangular loop bounds where the inner loop limits are independent of the outer loop variable (e.g. `j = 0; j < M`). In an upper-triangular loop (`j = i + 1`), standard OpenMP 2.0/3.0 specifications reject `collapse(2)` or trigger non-compliant compiler behavior.
2. **Severe Mutex Contention**: Inserting matches inside a `#pragma omp critical` block serializes thread execution whenever similarity threshold conditions are met, creating severe thread stall locks and ruining parallel speedup.

### Modification & Final Architecture
The suggestion was modified to:
1. Use single outer-loop parallelization (`#pragma omp parallel for schedule(dynamic, 8)`).
2. Utilize **thread-local match vectors** (`std::vector<std::vector<SimilarityPair>> thread_matches(threads)`) to eliminate `#pragma omp critical` completely during parallel compute, merging local results after loop completion.

```cpp
// Modified & Verified Final Implementation
#pragma omp parallel num_threads(threads)
{
    int tid = omp_get_thread_num();
    #pragma omp for schedule(dynamic, 8)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double sim = compute_sparse_dot_product(vec[i], vec[j]);
            if (sim >= threshold) {
                thread_matches[tid].push_back({i, j, docs[i].id, docs[j].id, sim});
            }
        }
    }
}
```

### Empirical Verification
- **With `#pragma omp critical`**: Execution time on 8 threads for 1,000 documents was **1,420 ms** (worse than sequential runtime of 1,267 ms due to lock contention).
- **With Thread-Local Accumulation**: Execution time dropped to **132 ms** (**9.59x compute speedup**).
