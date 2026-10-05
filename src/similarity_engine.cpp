#include "../include/similarity_engine.hpp"
#include "../include/timer.hpp"
#include <algorithm>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

DocumentSimilarityEngine::DocumentSimilarityEngine(double similarity_threshold)
    : threshold(similarity_threshold) {}

std::vector<SimilarityPair> DocumentSimilarityEngine::compute_pairwise_sequential(
    const std::vector<Document>& docs,
    const std::vector<SparseVector>& tfidf_vectors,
    ExecutionStats& stats
) {
    Timer timer;
    timer.start();

    size_t n = docs.size();
    std::vector<SimilarityPair> matches;
    size_t total_pairs = (n * (n - 1)) / 2;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sim = compute_sparse_dot_product(tfidf_vectors[i], tfidf_vectors[j]);
            if (sim >= threshold) {
                matches.push_back({
                    static_cast<int>(i),
                    static_cast<int>(j),
                    docs[i].id,
                    docs[j].id,
                    sim
                });
            }
        }
    }

    timer.stop();
    stats.similarity_compute_time_ms = timer.elapsed_ms();
    stats.total_pairs_computed = total_pairs;
    stats.threshold_matches = matches.size();

    // Sort matches deterministically: by similarity desc, then doc_i, then doc_j
    std::sort(matches.begin(), matches.end(), [](const SimilarityPair& a, const SimilarityPair& b) {
        if (std::abs(a.similarity - b.similarity) > 1e-9) {
            return a.similarity > b.similarity;
        }
        if (a.doc_i != b.doc_i) return a.doc_i < b.doc_i;
        return a.doc_j < b.doc_j;
    });

    return matches;
}

std::vector<SimilarityPair> DocumentSimilarityEngine::compute_pairwise_parallel(
    const std::vector<Document>& docs,
    const std::vector<SparseVector>& tfidf_vectors,
    ExecutionStats& stats,
    int num_threads
) {
    Timer timer;
    timer.start();

    int n = static_cast<int>(docs.size());
    size_t total_pairs = (static_cast<size_t>(n) * (n - 1)) / 2;

    int threads = num_threads > 0 ? num_threads : 1;
#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    threads = omp_get_max_threads();
#endif

    std::vector<std::vector<SimilarityPair>> thread_matches(threads);

    // Dynamic load balancing over upper triangular matrix iterations
    #pragma omp parallel num_threads(threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        #pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double sim = compute_sparse_dot_product(tfidf_vectors[i], tfidf_vectors[j]);
                if (sim >= threshold) {
                    thread_matches[tid].push_back({
                        i, j, docs[i].id, docs[j].id, sim
                    });
                }
            }
        }
    }

    // Merge thread local results
    std::vector<SimilarityPair> matches;
    size_t total_matches = 0;
    for (int t = 0; t < threads; ++t) {
        total_matches += thread_matches[t].size();
    }
    matches.reserve(total_matches);
    for (int t = 0; t < threads; ++t) {
        matches.insert(matches.end(), thread_matches[t].begin(), thread_matches[t].end());
    }

    timer.stop();
    stats.similarity_compute_time_ms = timer.elapsed_ms();
    stats.total_pairs_computed = total_pairs;
    stats.threshold_matches = matches.size();

    // Sort matches deterministically for identical comparison against sequential
    std::sort(matches.begin(), matches.end(), [](const SimilarityPair& a, const SimilarityPair& b) {
        if (std::abs(a.similarity - b.similarity) > 1e-9) {
            return a.similarity > b.similarity;
        }
        if (a.doc_i != b.doc_i) return a.doc_i < b.doc_i;
        return a.doc_j < b.doc_j;
    });

    return matches;
}
