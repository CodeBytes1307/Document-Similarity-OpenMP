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

void test_requested_document_combinations() {
    const auto verify_combination = [](const std::string& test_name,
                                       const std::string& doc_1_text,
                                       const std::string& doc_2_text,
                                       double expected_similarity) {
        std::vector<Document> docs = {
            {"doc_1", doc_1_text, {}},
            {"doc_2", doc_2_text, {}}
        };

        TextPreprocessor pre;
        auto docs_seq = docs;
        pre.preprocess_documents_sequential(docs_seq);
        TFIDFVectorparser tfidf_seq;
        tfidf_seq.build_vocabulary_sequential(docs_seq);
        std::vector<SparseVector> vectors_seq;
        tfidf_seq.compute_tfidf_sequential(docs_seq, vectors_seq);

        auto docs_par = docs;
        pre.preprocess_documents_parallel(docs_par, 4);
        TFIDFVectorparser tfidf_par;
        tfidf_par.build_vocabulary_parallel(docs_par, 4);
        std::vector<SparseVector> vectors_par;
        tfidf_par.compute_tfidf_parallel(docs_par, vectors_par, 4);

        const double sequential_similarity =
            compute_sparse_dot_product(vectors_seq[0], vectors_seq[1]);
        const double parallel_similarity =
            compute_sparse_dot_product(vectors_par[0], vectors_par[1]);
        assert(std::abs(sequential_similarity - expected_similarity) < 1e-6);
        assert(std::abs(parallel_similarity - expected_similarity) < 1e-6);

        DocumentSimilarityEngine engine(0.0);
        ExecutionStats seq_stats{};
        ExecutionStats par_stats{};
        const auto seq_matches =
            engine.compute_pairwise_sequential(docs_seq, vectors_seq, seq_stats);
        const auto par_matches =
            engine.compute_pairwise_parallel(docs_par, vectors_par, par_stats, 4);
        assert(seq_matches.size() == 1);
        assert(par_matches.size() == 1);
        assert(seq_matches[0].doc_i_id == "doc_1");
        assert(seq_matches[0].doc_j_id == "doc_2");
        assert(par_matches[0].doc_i_id == "doc_1");
        assert(par_matches[0].doc_j_id == "doc_2");
        std::cout << test_name << " PASSED (Cosine Sim = "
                  << sequential_similarity << ")\n";
    };

    std::cout << "[Test 5] Requested Document Combinations...\n";
    verify_combination("  Case 1 (shivani khushi / nehal mugdha):",
                       "shivani khushi", "nehal mugdha", 0.0);
    verify_combination("  Case 2 (shivani khushi khushi / shivani khushi):",
                       "shivani khushi khushi", "shivani khushi",
                       3.0 / std::sqrt(10.0));
}

int main() {
    std::cout << "=======================================================\n";
    std::cout << "         RUNNING CORRECTNESS SUITE                     \n";
    std::cout << "=======================================================\n";
    test_identical_documents();
    test_orthogonal_documents();
    test_seq_vs_par_equivalence();
    test_empty_and_short_docs();
    test_requested_document_combinations();
    std::cout << "=======================================================\n";
    std::cout << "ALL CORRECTNESS TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "=======================================================\n";
    return 0;
}
