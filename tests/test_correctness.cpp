#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include "../include/document.hpp"
#include "../include/preprocessor.hpp"
#include "../include/tfidf.hpp"
#include "../include/similarity_engine.hpp"

void test_identical_documents() {
    std::cout << "[Test 1] Identical Documents Similarity... ";
    std::vector<Document> docs = {
        {"doc_1", "parallel computing high performance openmp multi core cluster", {}},
        {"doc_2", "parallel computing high performance openmp multi core cluster", {}}
    };

    TextPreprocessor pre;
    pre.preprocess_documents_sequential(docs);

    TFIDFVectorparser tfidf;
    tfidf.build_vocabulary_sequential(docs);
    std::vector<SparseVector> vec;
    tfidf.compute_tfidf_sequential(docs, vec);

    double sim = compute_sparse_dot_product(vec[0], vec[1]);
    assert(std::abs(sim - 1.0) < 1e-5);
    std::cout << "PASSED (Sim = " << sim << ")\n";
}

void test_orthogonal_documents() {
    std::cout << "[Test 2] Disjoint / Orthogonal Documents... ";
    std::vector<Document> docs = {
        {"doc_1", "quantum physics string theory particle accelerator", {}},
        {"doc_2", "gourmet pizza mozzarella basil olive oil database", {}}
    };

    TextPreprocessor pre;
    pre.preprocess_documents_sequential(docs);

    TFIDFVectorparser tfidf;
    tfidf.build_vocabulary_sequential(docs);
    std::vector<SparseVector> vec;
    tfidf.compute_tfidf_sequential(docs, vec);

    double sim = compute_sparse_dot_product(vec[0], vec[1]);
    assert(std::abs(sim - 0.0) < 1e-5);
    std::cout << "PASSED (Sim = " << sim << ")\n";
}

void test_seq_vs_par_equivalence() {
    std::cout << "[Test 3] Sequential vs Parallel Numerical Equivalence... ";
    std::vector<Document> docs;
    for (int i = 0; i < 50; ++i) {
        std::string content = "algorithm data structure parallel performance openmp execution optimization document similarity analysis result iteration " + std::to_string(i % 5);
        if (i % 2 == 0) content += " high throughput matrix computation lock free schedule";
        docs.push_back({"doc_" + std::to_string(i), content, {}});
    }

    TextPreprocessor pre;
    DocumentSimilarityEngine engine(0.01);

    // Sequential
    auto docs_seq = docs;
    pre.preprocess_documents_sequential(docs_seq);
    TFIDFVectorparser tfidf_seq;
    tfidf_seq.build_vocabulary_sequential(docs_seq);
    std::vector<SparseVector> vec_seq;
    tfidf_seq.compute_tfidf_sequential(docs_seq, vec_seq);
    ExecutionStats stats_seq{};
    auto matches_seq = engine.compute_pairwise_sequential(docs_seq, vec_seq, stats_seq);

    // Parallel
    auto docs_par = docs;
    pre.preprocess_documents_parallel(docs_par, 4);
    TFIDFVectorparser tfidf_par;
    tfidf_par.build_vocabulary_parallel(docs_par, 4);
    std::vector<SparseVector> vec_par;
    tfidf_par.compute_tfidf_parallel(docs_par, vec_par, 4);
    ExecutionStats stats_par{};
    auto matches_par = engine.compute_pairwise_parallel(docs_par, vec_par, stats_par, 4);

    assert(matches_seq.size() == matches_par.size());
    for (size_t i = 0; i < matches_seq.size(); ++i) {
        assert(matches_seq[i].doc_i == matches_par[i].doc_i);
        assert(matches_seq[i].doc_j == matches_par[i].doc_j);
        assert(std::abs(matches_seq[i].similarity - matches_par[i].similarity) < 1e-6);
    }
    std::cout << "PASSED (" << matches_seq.size() << " pairs verified across 4 threads)\n";
}

void test_empty_and_short_docs() {
    std::cout << "[Test 4] Empty and Short Edge Case Documents... ";
    std::vector<Document> docs = {
        {"doc_empty", "", {}},
        {"doc_single", "algorithm", {}},
        {"doc_stopwords_only", "the a an in of to and is", {}}
    };

    TextPreprocessor pre;
    pre.preprocess_documents_sequential(docs);

    TFIDFVectorparser tfidf;
    tfidf.build_vocabulary_sequential(docs);
    std::vector<SparseVector> vec;
    tfidf.compute_tfidf_sequential(docs, vec);

    // Should not crash and vectors should be empty or handle gracefully
    assert(vec[0].empty());
    assert(vec[2].empty()); // all stopwords stripped
    assert(!vec[1].empty());
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << "         RUNNING CORRECTNESS SUITE                     \n";
    std::cout << "=======================================================\n";
    test_identical_documents();
    test_orthogonal_documents();
    test_seq_vs_par_equivalence();
    test_empty_and_short_docs();
    std::cout << "=======================================================\n";
    std::cout << "ALL CORRECTNESS TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "=======================================================\n";
    return 0;
}
