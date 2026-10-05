#include "../include/tfidf.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

TFIDFVectorparser::TFIDFVectorparser() : num_documents(0) {}

void TFIDFVectorparser::build_vocabulary_sequential(const std::vector<Document>& docs) {
    word_to_id.clear();
    id_to_word.clear();
    doc_frequencies.clear();
    idf_weights.clear();
    num_documents = docs.size();

    if (num_documents == 0) return;

    for (const auto& doc : docs) {
        std::unordered_set<std::string> unique_terms(doc.tokens.begin(), doc.tokens.end());
        for (const auto& term : unique_terms) {
            auto it = word_to_id.find(term);
            if (it == word_to_id.end()) {
                int new_id = static_cast<int>(id_to_word.size());
                word_to_id[term] = new_id;
                id_to_word.push_back(term);
                doc_frequencies.push_back(1);
            } else {
                doc_frequencies[it->second]++;
            }
        }
    }

    size_t vocab_size = id_to_word.size();
    idf_weights.resize(vocab_size);
    double N_double = static_cast<double>(num_documents);
    for (size_t i = 0; i < vocab_size; ++i) {
        // Smooth IDF: ln((N + 1) / (df + 1)) + 1.0
        idf_weights[i] = std::log((N_double + 1.0) / (static_cast<double>(doc_frequencies[i]) + 1.0)) + 1.0;
    }
}

void TFIDFVectorparser::build_vocabulary_parallel(const std::vector<Document>& docs, int num_threads) {
    word_to_id.clear();
    id_to_word.clear();
    doc_frequencies.clear();
    idf_weights.clear();
    num_documents = docs.size();

    if (num_documents == 0) return;

    int threads = num_threads > 0 ? num_threads : 1;
#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
    threads = omp_get_max_threads();
#endif

    // Thread-local maps for vocabulary discovery
    std::vector<std::unordered_map<std::string, int>> local_dfs(threads);

    int n = static_cast<int>(docs.size());
    #pragma omp parallel num_threads(threads)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            std::unordered_set<std::string> unique_terms(docs[i].tokens.begin(), docs[i].tokens.end());
            for (const auto& term : unique_terms) {
                local_dfs[tid][term]++;
            }
        }
    }

    // Reduce thread local DFs into global DF map
    std::unordered_map<std::string, int> global_df_map;
    for (int t = 0; t < threads; ++t) {
        for (const auto& kv : local_dfs[t]) {
            global_df_map[kv.first] += kv.second;
        }
    }

    for (const auto& kv : global_df_map) {
        int new_id = static_cast<int>(id_to_word.size());
        word_to_id[kv.first] = new_id;
        id_to_word.push_back(kv.first);
        doc_frequencies.push_back(kv.second);
    }

    size_t vocab_size = id_to_word.size();
    idf_weights.resize(vocab_size);
    double N_double = static_cast<double>(num_documents);
    for (size_t i = 0; i < vocab_size; ++i) {
        idf_weights[i] = std::log((N_double + 1.0) / (static_cast<double>(doc_frequencies[i]) + 1.0)) + 1.0;
    }
}

void TFIDFVectorparser::compute_tfidf_sequential(const std::vector<Document>& docs, std::vector<SparseVector>& tfidf_vectors) {
    tfidf_vectors.resize(docs.size());

    for (size_t i = 0; i < docs.size(); ++i) {
        const auto& tokens = docs[i].tokens;
        if (tokens.empty()) continue;

        std::unordered_map<int, double> term_counts;
        for (const auto& token : tokens) {
            auto it = word_to_id.find(token);
            if (it != word_to_id.end()) {
                term_counts[it->second] += 1.0;
            }
        }

        double total_terms = static_cast<double>(tokens.size());
        SparseVector vec;
        vec.reserve(term_counts.size());

        double norm_sq = 0.0;
        for (const auto& kv : term_counts) {
            int term_id = kv.first;
            double tf = kv.second / total_terms;
            double tfidf = tf * idf_weights[term_id];
            vec.push_back({term_id, tfidf});
            norm_sq += tfidf * tfidf;
        }

        double norm = std::sqrt(norm_sq);
        if (norm > 0.0) {
            for (auto& elem : vec) {
                elem.second /= norm;
            }
        }

        // Sort by term_id for O(|A| + |B|) dot products
        std::sort(vec.begin(), vec.end(), [](const SparseTerm& a, const SparseTerm& b) {
            return a.first < b.first;
        });

        tfidf_vectors[i] = std::move(vec);
    }
}

void TFIDFVectorparser::compute_tfidf_parallel(const std::vector<Document>& docs, std::vector<SparseVector>& tfidf_vectors, int num_threads) {
    tfidf_vectors.resize(docs.size());
    int n = static_cast<int>(docs.size());

    #pragma omp parallel for schedule(dynamic, 16) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
    for (int i = 0; i < n; ++i) {
        const auto& tokens = docs[i].tokens;
        if (tokens.empty()) continue;

        std::unordered_map<int, double> term_counts;
        for (const auto& token : tokens) {
            auto it = word_to_id.find(token);
            if (it != word_to_id.end()) {
                term_counts[it->second] += 1.0;
            }
        }

        double total_terms = static_cast<double>(tokens.size());
        SparseVector vec;
        vec.reserve(term_counts.size());

        double norm_sq = 0.0;
        for (const auto& kv : term_counts) {
            int term_id = kv.first;
            double tf = kv.second / total_terms;
            double tfidf = tf * idf_weights[term_id];
            vec.push_back({term_id, tfidf});
            norm_sq += tfidf * tfidf;
        }

        double norm = std::sqrt(norm_sq);
        if (norm > 0.0) {
            for (auto& elem : vec) {
                elem.second /= norm;
            }
        }

        std::sort(vec.begin(), vec.end(), [](const SparseTerm& a, const SparseTerm& b) {
            return a.first < b.first;
        });

        tfidf_vectors[i] = std::move(vec);
    }
}
