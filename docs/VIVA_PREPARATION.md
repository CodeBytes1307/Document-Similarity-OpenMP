# Viva Voce Preparation & Technical Q&A Guide

This guide provides technically precise answers to likely examiner questions for viva examination on the Document Similarity project.

---

## 1. Algorithmic & Representation Concepts

### Q1: What is document similarity, and how is it mathematically formulated in this project?
**Answer**: Document similarity measures the semantic or textual overlap between two documents. In this project, documents are represented as high-dimensional sparse vectors in a Term Frequency-Inverse Document Frequency (TF-IDF) vector space. The similarity between document $d_i$ and document $d_j$ is calculated using **Cosine Similarity**:
$$\text{CosineSim}(d_i, d_j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\|_2 \|\vec{v}_j\|_2}$$
By pre-normalizing all sparse TF-IDF vectors to unit L2 length ($\|\vec{v}\|_2 = 1.0$), Cosine Similarity simplifies to a dot product:
$$\text{CosineSim}(d_i, d_j) = \sum_{k} v_{i, k} \cdot v_{j, k}$$

### Q2: Why select TF-IDF over Bag-of-Words (BoW) or Jaccard similarity?
**Answer**:
- **Bag-of-Words (BoW)** counts raw word frequencies, meaning ubiquitous words like "system" or "data" skew similarity scores upwards regardless of actual semantic intent.
- **Jaccard Similarity** operates on set intersections ($|A \cap B| / |A \cup B|$), ignoring term importance completely.
- **TF-IDF** combines local term frequency ($\text{TF}$) with global inverse document frequency ($\text{IDF} = \ln(\frac{N+1}{\text{DF}+1}) + 1$), down-weighting non-informative words while highlighting rare domain keywords.

### Q3: How are sparse vectors stored and dot products calculated efficiently?
**Answer**: Each document vector is stored as `std::vector<std::pair<int, double>>`, containing `(term_id, weight)` tuples sorted by `term_id`. Because vectors are sorted, the dot product between document $A$ and document $B$ is computed in linear time $O(|A| + |B|)$ using a **two-pointer merge traversal**, avoiding dense $O(V)$ zero-multiplication loops.

---

## 2. Theoretical & Practical Complexity

### Q4: What are the Time and Space Complexities of the sequential implementation?
**Answer**:
- **Preprocessing & Tokenization**: Time $O(N \cdot L)$, Space $O(N \cdot K)$ where $N$ is document count, $L$ is document word length, and $K \le L$ is unique words per document.
- **TF-IDF Construction**: Time $O(N \cdot K \log K)$ to sort sparse term vectors by `term_id`.
- **Pairwise Similarity Matrix**: Time $O(N^2 \cdot \bar{K})$ where $\bar{K}$ is the average non-zero terms per document.
- **Total Space Complexity**: $O(N \cdot \bar{K})$ sparse vector memory + $O(M)$ output pairs exceeding similarity threshold $\tau$.

### Q5: Why is the pairwise similarity loop $O(N^2)$?
**Answer**: Comparing all unique pairs among $N$ documents requires evaluating the upper-triangular index space where $0 \le i < j < N$. The total number of pair evaluations is:
$$P = \frac{N(N - 1)}{2} = \frac{N^2 - N}{2} = \Theta(N^2)$$
For $N = 2,000$ documents, this requires $1,999,000$ pairwise sparse dot-product operations.

---

## 3. Parallelization & OpenMP Constructs

### Q6: Why select OpenMP over MPI or CUDA for this project?
**Answer**:
- **OpenMP (Shared Memory)**: Ideal for multi-core CPUs where memory is unified. It allows zero-copy shared access to document vectors, avoiding expensive network communication overhead (MPI) or host-to-device PCI-e transfers (CUDA). It also provides dynamic load balancing for non-uniform matrix loops.
- **MPI**: Introduces message-passing overhead that outweighs compute gains for single-node multi-core systems.
- **CUDA**: Requires specialized NVIDIA GPU hardware and complex sparse memory packing (CSR/COO).

### Q7: What exact parts of the pipeline are parallelized?
**Answer**:
1. **Document Preprocessing**: `#pragma omp parallel for schedule(static)` parallelizes token cleaning per document.
2. **Vocabulary & DF Generation**: Thread-local document frequency hash maps aggregated per thread, followed by a global dictionary merge.
3. **TF-IDF Vector Construction**: `#pragma omp parallel for schedule(dynamic, 16)` parallelizes sparse vector normalization and sorting per document.
4. **Pairwise Dot-Product Loop**: `#pragma omp parallel for schedule(dynamic, 8)` parallelizes upper-triangular loop iterations $i \in [0, N-2]$.

### Q8: What load imbalance problem occurs in upper-triangular loops, and how is it solved?
**Answer**:
- **Problem**: In the upper-triangular loop $i \in [0, N-2]$, iteration $i=0$ performs $N-1$ dot products, whereas iteration $i=N-2$ performs only 1 dot product. If static scheduling (`schedule(static)`) were used, low-indexed threads would receive heavy workloads while high-indexed threads finish early and idle.
- **Solution**: Applying **dynamic scheduling** (`#pragma omp parallel for schedule(dynamic, 8)`) assigns small chunks (8 iterations of outer loop $i$) dynamically to available threads as they complete work, maintaining 100% core utilization.

### Q9: How are data races avoided when storing matching document pairs?
**Answer**: Inserting matches into a single shared vector `std::vector<SimilarityPair>` from multiple threads creates a data race. Instead of using `#pragma omp critical` (which serializes execution and destroys speedup), each thread accumulates matches in a **thread-local vector** `thread_matches[tid]`. After the parallel region finishes, thread-local vectors are concatenated into the final results list sequentially in $O(M)$ time.

---

## 4. Performance Metrics & Bottlenecks

### Q10: Define Speedup and Parallel Efficiency. What were your empirical results?
**Answer**:
- **Speedup ($S$)**:
  $$S_T = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(T)}$$
- **Parallel Efficiency ($E$)**:
  $$E_T = \frac{S_T}{T} \times 100\%$$
- **Empirical Result**: On an 8-thread system with $N = 2,000$ documents ($2 \text{ million pairs}$), sequential pairwise computation took **4,034.75 ms** while OpenMP parallel computation took **562.50 ms**, yielding a **7.17x Speedup** and **89.66% Parallel Efficiency**.

### Q11: Why is speedup not perfectly linear ($S_T = T$)?
**Answer**:
1. **Amdahl's Law**: Serial fractions (reading input files, merging thread-local dictionaries, consolidating final output) cannot be parallelized.
2. **Memory Bandwidth & Cache Contention**: Multiple CPU cores competing for L3 cache and main memory bus access during sparse vector traversal.
3. **Thread Overhead**: OpenMP thread initialization, synchronization barriers, and thread scheduling overhead.

### Q12: How was correctness verified between sequential and parallel implementations?
**Answer**:
The project includes a dedicated test suite (`tests/test_correctness.cpp`):
1. **Identical Documents**: Verifies Cosine Similarity equals 1.0.
2. **Orthogonal Documents**: Verifies Cosine Similarity equals 0.0 for disjoint vocabularies.
3. **Equivalence Test**: Runs sequential and 4-thread parallel engines on identical document sets and asserts that match counts, document IDs, and floating-point similarity values match down to $10^{-6}$ numerical tolerance.
4. **Edge Cases**: Validates handling of empty documents and stopword-only texts.
