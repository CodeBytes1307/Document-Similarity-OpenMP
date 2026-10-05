#ifndef TFIDF_HPP
#define TFIDF_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include "document.hpp"

// Sparse element: (term_id, normalized_weight)
using SparseTerm = std::pair<int, double>;
using SparseVector = std::vector<SparseTerm>;

class TFIDFVectorparser {
private:
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    std::vector<int> doc_frequencies; // DF per term_id
    std::vector<double> idf_weights;   // IDF per term_id
    size_t num_documents;

public:
    TFIDFVectorparser();

    void build_vocabulary_sequential(const std::vector<Document>& docs);
    void build_vocabulary_parallel(const std::vector<Document>& docs, int num_threads = 0);

    void compute_tfidf_sequential(const std::vector<Document>& docs, std::vector<SparseVector>& tfidf_vectors);
    void compute_tfidf_parallel(const std::vector<Document>& docs, std::vector<SparseVector>& tfidf_vectors, int num_threads = 0);

    size_t get_vocab_size() const { return id_to_word.size(); }
    const std::vector<std::string>& get_vocab() const { return id_to_word; }
};

// Fast sparse dot product of two sorted sparse vectors
inline double compute_sparse_dot_product(const SparseVector& vecA, const SparseVector& vecB) {
    double dot = 0.0;
    size_t i = 0, j = 0;
    size_t szA = vecA.size();
    size_t szB = vecB.size();

    while (i < szA && j < szB) {
        if (vecA[i].first == vecB[j].first) {
            dot += vecA[i].second * vecB[j].second;
            ++i;
            ++j;
        } else if (vecA[i].first < vecB[j].first) {
            ++i;
        } else {
            ++j;
        }
    }
    return dot;
}

#endif // TFIDF_HPP
