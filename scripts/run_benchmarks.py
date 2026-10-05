#!/usr/bin/env python3
"""
Automated Benchmarking Harness for Document Similarity Project
Executes sequential vs parallel scaling experiments and records empirical metrics.
"""

import subprocess
import sys
import os
import csv

EXECUTABLE = "./doc_similarity"
GENERATE_SCRIPT = "scripts/generate_dataset.py"
OUTPUT_CSV = "results/benchmark_results.csv"
TEMP_DATA_DIR = "data/benchmarks"

def ensure_build():
    if not os.path.exists(EXECUTABLE):
        print("[Build] Executable not found. Running make...")
        subprocess.run(["make", "all"], check=True)

def generate_corpus(num_docs, doc_length, file_path):
    cmd = [
        sys.executable, GENERATE_SCRIPT,
        "--num_docs", str(num_docs),
        "--doc_length", str(doc_length),
        "--output", file_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

def run_experiment(corpus_path, threads, threshold=0.10):
    cmd = [
        EXECUTABLE,
        "--input", corpus_path,
        "--mode", "both",
        "--threads", str(threads),
        "--threshold", str(threshold),
        "--benchmark"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    row = res.stdout.strip().split(",")
    return row

def main():
    ensure_build()
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    header = [
        "docs_count", "threads", "seq_prep_ms", "par_prep_ms",
        "seq_tfidf_ms", "par_tfidf_ms", "seq_sim_ms", "par_sim_ms",
        "seq_total_ms", "par_total_ms", "speedup", "efficiency"
    ]

    results = []

    # Experiment 1: Strong & Weak Scaling (varying documents N & thread count T)
    doc_counts = [100, 250, 500, 1000, 2000]
    thread_counts = [1, 2, 4, 8, 10]
    fixed_length = 400

    print("=======================================================")
    print("      STARTING AUTOMATED BENCHMARK SUITE              ")
    print("=======================================================")
    print("[Suite 1/2] Scaling Document Count (N) vs Threads (T)...")
    
    for n in doc_counts:
        corpus_file = os.path.join(TEMP_DATA_DIR, f"corpus_N{n}_L{fixed_length}.txt")
        generate_corpus(n, fixed_length, corpus_file)
        for t in thread_counts:
            print(f" -> Running N={n}, L={fixed_length}, Threads={t}...")
            row = run_experiment(corpus_file, t)
            results.append(row)

    # Experiment 2: Scaling Document Length (L) with fixed N=500, Threads=8
    doc_lengths = [100, 300, 600, 1200]
    fixed_n = 500
    fixed_t = 8

    print("[Suite 2/2] Scaling Document Length (L)...")
    for l in doc_lengths:
        corpus_file = os.path.join(TEMP_DATA_DIR, f"corpus_N{fixed_n}_L{l}.txt")
        generate_corpus(fixed_n, l, corpus_file)
        print(f" -> Running N={fixed_n}, L={l}, Threads={fixed_t}...")
        row = run_experiment(corpus_file, fixed_t)
        results.append(row)

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            writer.writerow(r)

    print("=======================================================")
    print(f"[Success] Saved empirical benchmark results to: {OUTPUT_CSV}")
    print("=======================================================")

if __name__ == "__main__":
    main()
