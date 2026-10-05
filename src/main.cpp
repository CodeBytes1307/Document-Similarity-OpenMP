#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../include/document.hpp"
#include "../include/preprocessor.hpp"
#include "../include/tfidf.hpp"
#include "../include/similarity_engine.hpp"
#include "../include/timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --input <file>        Path to input document manifest file (required)\n"
              << "  --mode <seq|par|both> Execution mode (default: both)\n"
              << "  --threads <N>         Number of OpenMP threads for parallel mode (default: auto)\n"
              << "  --threshold <float>   Similarity score threshold [0.0 - 1.0] (default: 0.10)\n"
              << "  --top <K>             Number of top similar pairs to display (default: 10)\n"
              << "  --benchmark           Format output as CSV for benchmarking pipelines\n"
              << "  --verify              Strictly verify numerical/order equivalence between seq and par\n"
              << "  --help                Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string input_path = "";
    std::string mode = "both";
    int num_threads = 0;
    double threshold = 0.10;
    int top_k = 10;
    bool benchmark_mode = false;
    bool verify_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stod(argv[++i]);
        } else if (arg == "--top" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmark_mode = true;
        } else if (arg == "--verify") {
            verify_mode = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (input_path.empty()) {
        std::cerr << "[Error] Input manifest path is required via --input <file>\n";
        print_usage(argv[0]);
        return 1;
    }

    // 1. Load documents
    Timer total_timer;
    total_timer.start();

    auto docs = DocumentLoader::load_from_manifest(input_path);
    if (docs.empty()) {
        std::cerr << "[Error] No valid documents loaded from: " << input_path << "\n";
        return 1;
    }

    int active_threads = num_threads;
#ifdef _OPENMP
    if (active_threads <= 0) active_threads = omp_get_max_threads();
#else
    active_threads = 1;
#endif

    std::vector<SimilarityPair> seq_matches, par_matches;
    ExecutionStats seq_stats{}, par_stats{};

    TextPreprocessor preprocessor;
    DocumentSimilarityEngine engine(threshold);

    // Run Sequential Pipeline
    if (mode == "seq" || mode == "both" || verify_mode) {
        auto docs_seq = docs;
        Timer t_prep;
        t_prep.start();
        preprocessor.preprocess_documents_sequential(docs_seq);
        t_prep.stop();
        seq_stats.preprocess_time_ms = t_prep.elapsed_ms();

        TFIDFVectorparser tfidf_seq;
        Timer t_tfidf;
        t_tfidf.start();
        tfidf_seq.build_vocabulary_sequential(docs_seq);
        std::vector<SparseVector> vectors_seq;
        tfidf_seq.compute_tfidf_sequential(docs_seq, vectors_seq);
        t_tfidf.stop();
        seq_stats.tfidf_build_time_ms = t_tfidf.elapsed_ms();

        seq_matches = engine.compute_pairwise_sequential(docs_seq, vectors_seq, seq_stats);
        seq_stats.total_time_ms = seq_stats.preprocess_time_ms + seq_stats.tfidf_build_time_ms + seq_stats.similarity_compute_time_ms;
    }

    // Run Parallel Pipeline
    if (mode == "par" || mode == "both" || verify_mode) {
        auto docs_par = docs;
        Timer t_prep;
        t_prep.start();
        preprocessor.preprocess_documents_parallel(docs_par, active_threads);
        t_prep.stop();
        par_stats.preprocess_time_ms = t_prep.elapsed_ms();

        TFIDFVectorparser tfidf_par;
        Timer t_tfidf;
        t_tfidf.start();
        tfidf_par.build_vocabulary_parallel(docs_par, active_threads);
        std::vector<SparseVector> vectors_par;
        tfidf_par.compute_tfidf_parallel(docs_par, vectors_par, active_threads);
        t_tfidf.stop();
        par_stats.tfidf_build_time_ms = t_tfidf.elapsed_ms();

        par_matches = engine.compute_pairwise_parallel(docs_par, vectors_par, par_stats, active_threads);
        par_stats.total_time_ms = par_stats.preprocess_time_ms + par_stats.tfidf_build_time_ms + par_stats.similarity_compute_time_ms;
    }

    total_timer.stop();

    // Verify equivalence if requested or running both
    if (verify_mode || (mode == "both" && !benchmark_mode)) {
        bool match = (seq_matches.size() == par_matches.size());
        if (match) {
            for (size_t i = 0; i < seq_matches.size(); ++i) {
                if (seq_matches[i].doc_i != par_matches[i].doc_i ||
                    seq_matches[i].doc_j != par_matches[i].doc_j ||
                    std::abs(seq_matches[i].similarity - par_matches[i].similarity) > 1e-6) {
                    match = false;
                    break;
                }
            }
        }
        if (!benchmark_mode) {
            std::cout << "\n=======================================================\n";
            std::cout << "               CORRECTNESS VERIFICATION\n";
            std::cout << "=======================================================\n";
            if (match) {
                std::cout << "[PASS] Sequential and Parallel results match EXACTLY!\n";
                std::cout << "       Matches count: " << seq_matches.size() << "\n";
            } else {
                std::cout << "[FAIL] Result mismatch detected!\n";
                std::cout << "       Seq count: " << seq_matches.size() << ", Par count: " << par_matches.size() << "\n";
            }
        }
    }

    if (benchmark_mode) {
        // Output CSV row: docs_count,threads,seq_compute_ms,par_compute_ms,seq_total_ms,par_total_ms,speedup,efficiency
        double speedup = (par_stats.similarity_compute_time_ms > 0) ? (seq_stats.similarity_compute_time_ms / par_stats.similarity_compute_time_ms) : 0.0;
        double efficiency = speedup / active_threads;
        std::cout << docs.size() << ","
                  << active_threads << ","
                  << std::fixed << std::setprecision(4)
                  << seq_stats.preprocess_time_ms << ","
                  << par_stats.preprocess_time_ms << ","
                  << seq_stats.tfidf_build_time_ms << ","
                  << par_stats.tfidf_build_time_ms << ","
                  << seq_stats.similarity_compute_time_ms << ","
                  << par_stats.similarity_compute_time_ms << ","
                  << seq_stats.total_time_ms << ","
                  << par_stats.total_time_ms << ","
                  << speedup << ","
                  << efficiency << "\n";
        return 0;
    }

    // Pretty console report
    std::cout << "\n=======================================================\n";
    std::cout << "       DOCUMENT SIMILARITY ANALYSIS REPORT             \n";
    std::cout << "=======================================================\n";
    std::cout << "Total Documents    : " << docs.size() << "\n";
    std::cout << "Total Pairs        : " << (docs.size() * (docs.size() - 1)) / 2 << "\n";
    std::cout << "Active Threads     : " << active_threads << "\n";
    std::cout << "Similarity Thresh  : " << threshold << "\n";
    std::cout << "-------------------------------------------------------\n";

    if (mode == "seq" || mode == "both") {
        std::cout << "[Sequential Implementation]\n";
        std::cout << "  - Preprocessing Time : " << std::fixed << std::setprecision(2) << seq_stats.preprocess_time_ms << " ms\n";
        std::cout << "  - TF-IDF Build Time  : " << seq_stats.tfidf_build_time_ms << " ms\n";
        std::cout << "  - Pairwise Sim Time  : " << seq_stats.similarity_compute_time_ms << " ms\n";
        std::cout << "  - Total Compute Time : " << seq_stats.total_time_ms << " ms\n";
    }

    if (mode == "par" || mode == "both") {
        std::cout << "[Parallel OpenMP Implementation]\n";
        std::cout << "  - Preprocessing Time : " << std::fixed << std::setprecision(2) << par_stats.preprocess_time_ms << " ms\n";
        std::cout << "  - TF-IDF Build Time  : " << par_stats.tfidf_build_time_ms << " ms\n";
        std::cout << "  - Pairwise Sim Time  : " << par_stats.similarity_compute_time_ms << " ms\n";
        std::cout << "  - Total Compute Time : " << par_stats.total_time_ms << " ms\n";
    }

    if (mode == "both") {
        double compute_speedup = seq_stats.similarity_compute_time_ms / par_stats.similarity_compute_time_ms;
        double total_speedup = seq_stats.total_time_ms / par_stats.total_time_ms;
        double efficiency = compute_speedup / active_threads;

        std::cout << "-------------------------------------------------------\n";
        std::cout << "PERFORMANCE METRICS:\n";
        std::cout << "  - Sim Compute Speedup : " << std::setprecision(2) << compute_speedup << "x\n";
        std::cout << "  - Total Speedup       : " << total_speedup << "x\n";
        std::cout << "  - Parallel Efficiency : " << (efficiency * 100.0) << "%\n";
    }

    const auto& display_matches = (mode == "par") ? par_matches : seq_matches;
    std::cout << "-------------------------------------------------------\n";
    std::cout << "TOP " << std::min<size_t>(top_k, display_matches.size()) << " SIMILAR DOCUMENT PAIRS:\n";
    for (size_t i = 0; i < std::min<size_t>(top_k, display_matches.size()); ++i) {
        const auto& pair = display_matches[i];
        std::cout << "  [" << (i + 1) << "] " << pair.doc_i_id << " <---> " << pair.doc_j_id
                  << " | Cosine Sim = " << std::setprecision(4) << pair.similarity << "\n";
    }
    std::cout << "=======================================================\n\n";

    return 0;
}
