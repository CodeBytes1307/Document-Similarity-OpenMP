# LLM Usage Log
## Parallel Document Similarity Analysis — DAA Term 1

---

> **Purpose of this document**: This log documents the actual prompts used when interacting with any AI language model (e.g., ChatGPT, Gemini, Claude, Copilot) during the development of this project. It also records how each response was used and verified, and documents at least one instance where an LLM-generated suggestion was modified or rejected.
>
> **Instructions to the team**: Replace every row marked `[EXAMPLE — REPLACE WITH ACTUAL]` with the real prompts and responses you used during development. Do **not** fabricate entries. If you did not use an LLM for a particular category, write "Not used" in that row. The log must reflect authentic interactions.

---

## Section A: Usage Log Table

| Sr. No. | Category | Purpose | Actual Prompt Used | How Response Was Used | Modification / Verification |
|---|---|---|---|---|---|
| 1 | Algorithm Understanding | Understanding TF-IDF weighting and its variants | `[EXAMPLE — REPLACE WITH ACTUAL] "Explain the difference between standard IDF (log(N/df)) and smooth IDF (log((N+1)/(df+1))+1). When does the smooth variant prevent numerical issues?"` | `[EXAMPLE] Used the explanation to understand that standard IDF can be negative (when df > N/e) and undefined (when df=0). Decided to use smooth IDF in the implementation. Verified by checking tfidf.cpp line 41 and cross-referencing with Scikit-learn source.` | `[EXAMPLE] The LLM suggested using log(1 + N/df) as an alternative. We verified it was not equivalent to the sklearn smooth IDF formula and rejected it. Implemented exactly ln((N+1)/(df+1))+1.0 matching sklearn.` |
| 2 | Algorithm Understanding | Understanding cosine similarity with pre-normalized vectors | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL: describe how you used the response]` | `[REPLACE WITH ACTUAL: what was verified or modified]` |
| 3 | Algorithm Selection | Comparing TF-IDF vs LSH vs MinHash for exact pairwise similarity | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 4 | Algorithm Selection | Choosing between dense and sparse vector representations | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 5 | Parallelization | Understanding OpenMP schedule types (static vs dynamic vs guided) | `[EXAMPLE — REPLACE WITH ACTUAL] "In OpenMP, when should I use schedule(dynamic) vs schedule(static) for a for loop? I have a loop where iteration i does i+1 units of work."` | `[EXAMPLE] LLM explained static gives equal chunk sizes ignoring workload, dynamic assigns chunks on demand. Used this to justify dynamic scheduling for the upper-triangular similarity loop where row i does N-1-i comparisons.` | `[EXAMPLE] LLM suggested schedule(guided) for decreasing workloads. We tested both — guided had higher overhead for small chunk completions near end of loop. Kept schedule(dynamic, 8) after empirical comparison.` |
| 6 | Parallelization | Avoiding race conditions in parallel vocabulary building | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 7 | Parallelization | Thread-local storage pattern for result collection | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 8 | Coding | OpenMP parallel for with dynamic scheduling syntax | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 9 | Coding | Implementing two-pointer merge for sparse vector dot product | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 10 | Coding | C++ unordered_map thread safety considerations | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 11 | Debugging | Diagnosing non-deterministic parallel output ordering | `[EXAMPLE — REPLACE WITH ACTUAL] "My OpenMP parallel program produces different orderings of results each run even though the values are correct. How do I make the output deterministic?"` | `[EXAMPLE] LLM suggested sorting results after the parallel region with a deterministic comparator. Implemented in similarity_engine.cpp lines 44-51 (sequential) and 113-120 (parallel): sort by similarity DESC, then doc_i ASC, then doc_j ASC. This makes --verify pass reliably.` | `[EXAMPLE] Verified by running --verify flag 10 consecutive times on the same input — all [PASS]. Also confirmed sort stability is not required because the tie-breaking is by integer indices which are unique.` |
| 12 | Debugging | Identifying superlinear speedup as cache effect | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 13 | Optimization | Choosing optimal OpenMP chunk size for the inner loop | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 14 | Optimization | Reducing false sharing in thread-local result buffers | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 15 | Testing | Designing test cases for numerical equivalence (seq vs par) | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 16 | Testing | Choosing appropriate floating-point tolerance for assertions | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 17 | Documentation | Explaining Amdahl's Law in the context of this project | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |
| 18 | Documentation | Writing complexity analysis for pairwise sparse dot products | `[REPLACE WITH ACTUAL PROMPT]` | `[REPLACE WITH ACTUAL]` | `[REPLACE WITH ACTUAL]` |

---

## Section B: Categories Reference

Use these categories when filling in the table above:

| Category | When to use |
|---|---|
| **Algorithm Understanding** | When an LLM was used to explain a concept, formula, or data structure |
| **Algorithm Selection** | When an LLM was used to compare algorithm options or justify a choice |
| **Parallelization** | When an LLM was used for OpenMP pragmas, thread safety, or scheduling advice |
| **Coding** | When an LLM helped write, debug, or refactor actual C++ code |
| **Debugging** | When an LLM was used to diagnose a bug, error message, or unexpected behavior |
| **Optimization** | When an LLM suggested a performance improvement (accepted or rejected) |
| **Testing** | When an LLM helped design test cases or correctness checks |
| **Documentation** | When an LLM helped write comments, README sections, or report content |

---

## Section C: Rejected / Modified LLM Suggestion

> **Requirement**: Document **at least one** instance where an LLM-generated suggestion was considered but modified or rejected. Be specific about why the suggestion seemed reasonable, what problem was discovered, and how the final decision was verified.

---

### Instance 1 — [REPLACE WITH YOUR ACTUAL INSTANCE TITLE]

> **Instructions**: Replace all fields below with the actual instance from your development process. The example below is clearly labeled as an example and must be replaced.

---

**[EXAMPLE — REPLACE WITH ACTUAL]**

#### LLM Suggestion

When asked how to parallelize the vocabulary construction phase (building the word-to-id map across documents), the LLM suggested:

```
Use #pragma omp critical around the hash map insertion:

#pragma omp parallel for schedule(static)
for (int i = 0; i < n; ++i) {
    for (const auto& term : unique_terms_of_doc_i) {
        #pragma omp critical
        {
            global_df_map[term]++;
        }
    }
}
```

#### Why It Appeared Reasonable

The suggestion correctly identified that concurrent writes to an `unordered_map` are a data race. Using `#pragma omp critical` is a standard OpenMP construct for protecting shared data access and would have produced correct results.

#### Problem Discovered

When implemented and profiled with N = 1000, the `#pragma omp critical` approach was **slower than the sequential baseline** for all thread counts tested (T = 2, 4, 8):

- Sequential vocab build: ~76ms
- Parallel with `omp critical` at T=4: ~143ms (1.88× **slower**)
- Parallel with `omp critical` at T=8: ~229ms (3.0× **slower**)

The reason: every term insertion requires acquiring a global mutex. Each thread competing for the critical section creates heavy lock contention. Since vocabulary construction processes O(N·K) terms and K is large (50–200 unique terms per document), the fraction of time spent waiting for the lock exceeded the fraction of time actually doing work. The implementation essentially serialized vocabulary construction with additional thread management overhead.

#### Modification Made

Replaced `#pragma omp critical` with **thread-local DF maps**:

```cpp
// Thread-local maps — no synchronization needed during parallel phase
std::vector<std::unordered_map<std::string, int>> local_dfs(threads);

#pragma omp parallel num_threads(threads)
{
    int tid = omp_get_thread_num();
    #pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (const auto& term : unique_terms_of_doc_i) {
            local_dfs[tid][term]++;   // no contention — thread-private
        }
    }
}

// Sequential reduction (one time, O(T·V))
for (int t = 0; t < threads; ++t)
    for (auto& kv : local_dfs[t])
        global_df_map[kv.first] += kv.second;
```

#### Experimental Verification

After implementing thread-local maps:

- Sequential vocab build: ~76ms
- Parallel thread-local T=4: ~33ms (2.3× speedup)
- Parallel thread-local T=8: ~12ms (6.3× speedup)

The TF-IDF build phase now scales near-linearly with thread count (confirmed by the benchmark data in `results/benchmark_results.csv`, columns `seq_tfidf_ms` vs `par_tfidf_ms`). The correctness of the reduction was verified by confirming identical vocabulary sizes and `--verify` passing for all configurations.

**Key lesson**: `#pragma omp critical` is suitable only for rare, infrequent updates to shared data. For high-frequency updates (O(N·K) insertions), thread-local reduction patterns are required to avoid serialization through lock contention.

---

### [Add additional rejected/modified instances here if applicable]

---

## Section D: General Notes on LLM Usage

> Replace the notes below with accurate descriptions of your team's LLM usage practices.

1. **Models Used**: *(List the specific AI tools your team used: e.g., ChatGPT-4o, Gemini 1.5 Pro, GitHub Copilot, Claude 3.5 Sonnet)*

2. **Verification Practice**: All LLM-generated code suggestions were tested by:
   - Compiling and running the relevant test case
   - Checking output against the sequential baseline using `--verify`
   - Reviewing the generated code manually before committing

3. **Limitations Encountered**: *(Describe any cases where LLM suggestions were confidently wrong, e.g., incorrect OpenMP syntax, wrong complexity claims, outdated API calls)*

4. **What LLMs Were NOT Used For**: *(List any sections where the team deliberately avoided LLMs, e.g., "Benchmark data was collected entirely by running the code — no LLM was asked to generate or estimate timing numbers")*

---

*This log was prepared as part of the project submission for the Design and Analysis of Algorithms course, Term 1. All entries marked `[EXAMPLE]` or `[REPLACE WITH ACTUAL]` must be updated by the team before final submission.*
