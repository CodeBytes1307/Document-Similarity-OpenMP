# High-Performance Document Similarity Engine (Sequential vs. OpenMP Parallel)

[![Language: C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Parallel Model: OpenMP](https://img.shields.io/badge/Parallel%20Model-OpenMP-orange.svg)](https://www.openmp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academically rigorous, high-performance C++ system designed to determine pairwise document similarity across large-scale document collections. The system features a **Sequential Baseline** and a multi-threaded **OpenMP Parallel Implementation**, utilizing a **TF-IDF (Term Frequency-Inverse Document Frequency)** representation and **Cosine Similarity** over sparse vectors.

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Problem Formulation](#2-problem-formulation)
- [3. Key Features](#3-key-features)
- [4. Algorithm & Theoretical Complexity](#4-algorithm--theoretical-complexity)
- [5. Parallelization Strategy](#5-parallelization-strategy)
- [6. Project Architecture](#6-project-architecture)
- [7. System Requirements & Installation](#7-system-requirements--installation)
- [8. Compilation & Execution](#8-compilation--execution)
- [9. Synthetic Dataset Generation](#9-synthetic-dataset-generation)
- [10. Automated Benchmarking & Visualization](#10-automated-benchmarking--visualization)
- [11. Correctness Verification](#11-correctness-verification)
- [12. Benchmark Results & Speedup Analysis](#12-benchmark-results--speedup-analysis)
- [13. Limitations & Future Work](#13-limitations--future-work)
- [14. Academic & Viva Resources](#14-academic--viva-resources)

---

## 1. Executive Summary

As document collections grow into thousands or millions of text items, finding similar document pairs becomes computationally expensive due to the quadratic pairwise comparison complexity $O(N^2 \cdot K)$, where $N$ is the number of documents and $K$ is the average number of unique terms per document.

This project investigates the performance characteristics of calculating document similarity, comparing a single-threaded CPU sequential baseline with an optimized shared-memory multi-core parallel architecture built on **OpenMP**. The implementation incorporates **thread-local result buffers**, **dynamic scheduling for load balancing upper-triangular loops**, and **sparse vector dot-product computations**.

---

## 2. Problem Formulation

Given a collection of $N$ documents $D = \{d_1, d_2, \dots, d_N\}$:
1. Preprocess raw text: lowercasing, non-alphanumeric character removal, stopword filtering.
2. Vectorize documents into normalized sparse TF-IDF vectors $\vec{v}_i \in \mathbb{R}^{V}$.
3. Compute pairwise Cosine Similarity:
   $$\text{CosineSim}(d_i, d_j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\|_2 \|\vec{v}_j\|_2} = \vec{v}_i \cdot \vec{v}_j \quad (\text{since } \|\vec{v}\|_2 = 1.0)$$
4. Extract all document pairs $(d_i, d_j)$ where $i < j$ such that $\text{CosineSim}(d_i, d_j) \ge \tau$, where $\tau \in [0, 1]$ is a configurable similarity threshold.

---

## 3. Key Features
- **Academically Sound Architecture**: Strict separation of Preprocessing, TF-IDF Vectorization, Sparse Dot-Product, and Threshold Extraction.
- **Sorted Sparse Vector Operations**: Sparse dot products computed in $O(|A| + |B|)$ linear time using two-pointer merge traversal.
- **OpenMP Multithreading**: Parallel document tokenization, parallel vocabulary accumulation, parallel TF-IDF vector transformation, and dynamic upper-triangular loop distribution.
- **Reproducible Synthetic Generator**: Python generator for multi-document corpora with configurable document count, word length, vocabulary size, topic clusters, and near-duplicate documents.
- **Automated Benchmarking & Plotting**: Automated harness producing CSV logs and publication-ready charts (Speedup, Efficiency, Execution Time vs. $N$).
- **Strict Verification Suite**: Ensures exact numerical equivalence between sequential and parallel implementations down to $10^{-6}$ tolerance across edge cases.

---

## 4. Algorithm & Theoretical Complexity

### Preprocessing & Tokenization
- Removes non-alphanumeric characters, converts tokens to lowercase, and filters against a 100+ stopword dictionary.
- **Time Complexity**: $O(N \cdot L)$ where $L$ is document text length.
- **Space Complexity**: $O(N \cdot K)$ tokens stored, where $K \le L$ is unique valid terms per document.

### TF-IDF Vector Construction
- **IDF Formula**: Smooth Inverse Document Frequency:
  $$\text{IDF}(t) = \ln\left(\frac{N + 1}{\text{DF}(t) + 1}\right) + 1.0$$
- **L2 Normalization**:
  $$\hat{v}(t, d) = \frac{\text{TF}(t, d) \cdot \text{IDF}(t)}{\sqrt{\sum_{t'} (\text{TF}(t', d) \cdot \text{IDF}(t'))^2}}$$
- **Time Complexity**: $O(N \cdot K \log K)$ to sort sparse term pairs by term ID.

### Pairwise Cosine Similarity Matrix
- Computes $P = \frac{N(N-1)}{2}$ unique pair comparisons.
- **Sequential Time Complexity**: $O(N^2 \cdot \bar{K})$ where $\bar{K}$ is average non-zero terms per document.
- **Space Complexity**: $O(N \cdot \bar{K})$ vector storage + $O(M)$ output pair matches ($M \ll N^2$).

---

## 5. Parallelization Strategy

### 1. Document Preprocessing (`#pragma omp parallel for schedule(static)`)
- Independent document tokenization distributed evenly among threads with static chunk allocation.

### 2. Vocabulary & DF Count (`#pragma omp parallel`)
- Each thread constructs thread-local Document Frequency (DF) hash maps to prevent atomic locks during vocabulary aggregation, followed by a thread-sequential reduction into the global dictionary.

### 3. TF-IDF Vector Building (`#pragma omp parallel for schedule(dynamic, 16)`)
- Vector creation and sorting per document executed in parallel using dynamic allocation.

### 4. Upper-Triangular Pair Comparison (`#pragma omp parallel for schedule(dynamic, 8)`)
- Outer loop index $i \in [0, N-2]$ and inner loop $j \in [i+1, N-1]$.
- **Load Imbalance Mitigation**: Iteration $i$ performs $(N - 1 - i)$ dot products. Static scheduling would assign heavy early iterations to thread 0 and light late iterations to higher threads. `schedule(dynamic, 8)` ensures active threads request work dynamically, eliminating load imbalance.
- **Race Condition Prevention**: Thread-local vectors `thread_matches[tid]` store matches locally during loop execution, avoiding mutex contention on the global match vector.

---

## 6. Project Architecture

```
DAA_T1/
├── Makefile                     # Build setup with macOS/Linux OpenMP autodetect
├── CMakeLists.txt               # Cross-platform CMake configuration
├── README.md                    # Project documentation
├── include/                     # C++ Header Files
│   ├── document.hpp             # Document data structures & loaders
│   ├── preprocessor.hpp         # Text cleaning & tokenization engine
│   ├── tfidf.hpp                # TF-IDF sparse vectorizer & dot product
│   ├── similarity_engine.hpp    # Sequential & Parallel Similarity Engine
│   └── timer.hpp                # High-precision timer harness
├── src/                         # C++ Source Files
│   ├── document.cpp
│   ├── preprocessor.cpp
│   ├── tfidf.cpp
│   ├── similarity_engine.cpp
│   └── main.cpp                 # CLI driver
├── tests/
│   └── test_correctness.cpp     # Unit & numerical equivalence test suite
├── scripts/
│   ├── generate_dataset.py      # Reproducible synthetic document generator
│   ├── run_benchmarks.py        # Automated benchmark orchestrator
│   └── plot_results.py          # Benchmark chart generator
├── data/                        # Generated datasets & corpora
├── results/                     # Empirical benchmark CSVs & PNG graphs
└── docs/                        # Project documentation
    ├── PROJECT_REPORT.md        # Complete 23-section project report
    ├── LLM_USAGE_LOG.md         # Prompt log & suggestion evaluation
    ├── TEAM_CONTRIBUTION.md     # Team responsibility breakdown
    └── VIVA_PREPARATION.md      # Comprehensive Viva Q&A guide
```

---

## 7. System Requirements & Installation

### Operating System & Dependencies
- **OS**: macOS (Apple Silicon / Intel) or Linux (Ubuntu 20.04+) or Windows (MSVC/MinGW).
- **C++ Compiler**: `clang++` (with `libomp`) or `g++` (C++17 standard).
- **Python**: Python 3.8+ with `matplotlib` and `pandas`.

### Installing OpenMP Library (macOS)
```bash
brew install libomp
```

---

## 8. Compilation & Execution

### Build Targets
```bash
# Clean and compile executables
make clean
make all
```

### Run Correctness Verification
```bash
make test
# OR
./run_tests
```

### Basic CLI Usage
```bash
# Generate synthetic dataset first
python3 scripts/generate_dataset.py --num_docs 200 --doc_length 300 --output data/sample_corpus.txt

# Run both sequential and parallel engines
./doc_similarity --input data/sample_corpus.txt --mode both --threshold 0.10 --threads 4 --top 10
```

---

## 9. Synthetic Dataset Generation

To ensure 100% reproducibility without external dataset dependencies, a synthetic dataset generator is provided:

```bash
python3 scripts/generate_dataset.py \
  --num_docs 1000 \
  --doc_length 500 \
  --vocab_size 2000 \
  --num_clusters 8 \
  --near_dup_ratio 0.05 \
  --seed 42 \
  --output data/synthetic_1k.txt
```

---

## 10. Automated Benchmarking & Visualization

Run the complete benchmark suite:
```bash
# 1. Execute scaling experiments (generates results/benchmark_results.csv)
python3 scripts/run_benchmarks.py

# 2. Generate visualization plots (generates results/*.png)
python3 scripts/plot_results.py
```

Generated Graphs:
- `results/execution_time_vs_docs.png`: Sequential vs. Parallel execution time across $N$.
- `results/speedup_vs_threads.png`: Measured speedup vs. OpenMP thread count.
- `results/parallel_efficiency_vs_threads.png`: Parallel efficiency (%) vs. thread count.
- `results/seq_vs_par_comparison.png`: Pipeline stage breakdown comparison.

---

## 11. Correctness Verification

Correctness is validated through four independent tests in `tests/test_correctness.cpp`:
1. **Identical Documents**: Ensures Cosine Similarity equals exactly 1.0.
2. **Orthogonal Documents**: Ensures Cosine Similarity equals 0.0 for disjoint vocabularies.
3. **Equivalence Suite**: Verifies exact matching of pairwise similarity values and match ordering between Sequential and Parallel implementations down to $10^{-6}$ precision.
4. **Edge Cases**: Validates graceful handling of empty documents and stopword-only texts.

---

## 12. Benchmark Results & Speedup Analysis

*(Detailed empirical values generated by `scripts/run_benchmarks.py`)*

### Measured Speedup Formula
$$\text{Speedup} (S_T) = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(T)}$$

$$\text{Efficiency} (E_T) = \frac{S_T}{T} \times 100\%$$

---

## 13. Limitations & Future Work
- **Shared Memory Limit**: OpenMP is bounded by single-node RAM and core count. Distributed models (MPI) can extend scalability to multi-node clusters.
- **GPU Acceleration**: CUDA sparse matrix operations (cuSPARSE) can accelerate dense pair dot products for millions of documents.
- **Approximate Nearest Neighbor (ANN)**: Locality Sensitive Hashing (LSH) or HNSW indices could reduce pair candidate space from $O(N^2)$ to $O(N \log N)$.

---

## 14. Academic & Viva Resources
For viva preparation, algorithm trade-offs, and full documentation, refer to:
- [Project Report](docs/PROJECT_REPORT.md)
- [LLM Usage Log](docs/LLM_USAGE_LOG.md)
- [Team Contribution Template](docs/TEAM_CONTRIBUTION.md)
- [Viva Q&A Guide](docs/VIVA_PREPARATION.md)
