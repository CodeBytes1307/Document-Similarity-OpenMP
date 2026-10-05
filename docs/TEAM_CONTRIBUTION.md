# Team Contribution Breakdown (2–4 Members)

This document provides a template for allocating responsibilities across team members for project evaluation, grading, and presentation.

---

## Team Overview

- **Project Title**: High-Performance Document Similarity Engine (Sequential vs. OpenMP Parallel)
- **Course**: Design and Analysis of Algorithms / Parallel & Distributed Computing
- **Team Size**: 4 Members (Adaptable to 2 or 3 members)

---

## Member Responsibilities & Deliverables

### Member 1: Core Algorithm Lead & Preprocessing Architecture
- **Primary Modules**: `include/document.hpp`, `src/document.cpp`, `include/preprocessor.hpp`, `src/preprocessor.cpp`.
- **Key Contributions**:
  - Implemented thread-safe document loading and text normalization (lowercasing, stopword stripping, punctuation removal).
  - Designed text tokenization pipeline for C++ std::string processing.
  - Implemented `TextPreprocessor::preprocess_documents_parallel()` with static OpenMP loop partitioning.
- **Viva Focus Areas**: Text preprocessing complexity, stopword filtering algorithms, thread-safe memory management.

### Member 2: TF-IDF Vectorizer & Sparse Computation Lead
- **Primary Modules**: `include/tfidf.hpp`, `src/tfidf.cpp`.
- **Key Contributions**:
  - Formulated smooth IDF logarithmic weighting equation and L2 unit vector normalization.
  - Designed `SparseVector` data structure (`std::vector<std::pair<int, double>>`) sorted by `term_id`.
  - Implemented fast linear two-pointer merge traversal `compute_sparse_dot_product()` operating in $O(|A| + |B|)$ time.
  - Implemented parallel document frequency aggregation using thread-local hash map reductions.
- **Viva Focus Areas**: TF-IDF mathematical formulation, sparse vector dot-product algorithm, memory layout & L1/L2 cache optimization.

### Member 3: Parallel Similarity Engine & Synchronization Lead
- **Primary Modules**: `include/similarity_engine.hpp`, `src/similarity_engine.cpp`, `src/main.cpp`.
- **Key Contributions**:
  - Implemented upper-triangular matrix pair comparison loop ($i < j$).
  - Evaluated OpenMP scheduling strategies and applied `#pragma omp parallel for schedule(dynamic, 8)` to eliminate load imbalance across threads.
  - Designed thread-local match buffer accumulator (`thread_matches`) to completely eliminate mutex lock contention during threshold filtering.
  - Developed unified C++ CLI interface supporting configurable dataset paths, thread counts, thresholds, and benchmark flags.
- **Viva Focus Areas**: OpenMP loop scheduling constructs, load imbalance mitigation, lock-free thread-local result accumulation, Amdahl's law bottlenecks.

### Member 4: Dataset Generator, Benchmarking & Correctness Suite Lead
- **Primary Modules**: `scripts/generate_dataset.py`, `scripts/run_benchmarks.py`, `scripts/plot_results.py`, `tests/test_correctness.cpp`.
- **Key Contributions**:
  - Created reproducible synthetic corpus generator supporting controlled document counts, lengths, topic clusters, and near-duplicate targets.
  - Built automated benchmarking Python harness executing strong/weak scaling sweeps across document counts $N \in [100, 2000]$ and threads $T \in [1, 10]$.
  - Developed visualization script plotting Execution Time, Speedup, Parallel Efficiency, and Stage Breakdowns using `matplotlib`.
  - Authored comprehensive test suite verifying 100% numerical and structural equivalence between sequential and parallel engines.
- **Viva Focus Areas**: Speedup and efficiency metric calculations, benchmark methodology, synthetic corpus design, unit test validation.

---

## Contribution Matrix Summary

| Task / Module | Lead Member | Support Member(s) | Verification Method |
|---|---|---|---|
| Text Tokenization & Preprocessing | Member 1 | Member 2 | Unit Test 4 (Edge cases) |
| TF-IDF & Sorted Sparse Vector Dot Product | Member 2 | Member 1 | Unit Test 1 & 2 (Identical & Orthogonal) |
| OpenMP Parallel Pairwise Loop & Scheduling | Member 3 | Member 2 | Strong Scaling Benchmarks & Unit Test 3 |
| Synthetic Dataset Generator | Member 4 | Member 1 | Reproducible random seed validation |
| Benchmarking Suite & Plot Generation | Member 4 | Member 3 | Empirical CSV log verification |
| Correctness Verification Suite | Member 4 | Member 3 | `make test` test harness |
| Final Project Report & Documentation | Member 1 & 3 | Member 2 & 4 | Section-by-section review |
