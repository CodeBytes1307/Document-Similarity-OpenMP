#ifndef SIMILARITY_ENGINE_HPP
#define SIMILARITY_ENGINE_HPP

#include <vector>
#include <string>
#include "tfidf.hpp"
#include "document.hpp"

struct SimilarityPair {
    int doc_i;
    int doc_j;
    std::string doc_i_id;
    std::string doc_j_id;
    double similarity;
};

struct ExecutionStats {
    double preprocess_time_ms;
    double tfidf_build_time_ms;
    double similarity_compute_time_ms;
    double total_time_ms;
    size_t total_pairs_computed;
    size_t threshold_matches;
};

class DocumentSimilarityEngine {
private:
    double threshold;

public:
    explicit DocumentSimilarityEngine(double similarity_threshold = 0.0);

    // Sequential matrix computation
    std::vector<SimilarityPair> compute_pairwise_sequential(
        const std::vector<Document>& docs,
        const std::vector<SparseVector>& tfidf_vectors,
        ExecutionStats& stats
    );

    // Parallel matrix computation (OpenMP)
    std::vector<SimilarityPair> compute_pairwise_parallel(
        const std::vector<Document>& docs,
        const std::vector<SparseVector>& tfidf_vectors,
        ExecutionStats& stats,
        int num_threads = 0
    );
};

#endif // SIMILARITY_ENGINE_HPP
