#!/usr/bin/env python3
"""
Visualization Script for Document Similarity Performance Benchmark Results
Generates publication-quality charts for Execution Time, Speedup, Efficiency, and Stage Breakdown.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_PATH = "results/benchmark_results.csv"
OUTPUT_DIR = "results"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"[Error] Benchmark results CSV not found at: {CSV_PATH}")
        print("Please run `python3 scripts/run_benchmarks.py` first.")
        return

    df = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Execution Time vs Number of Documents (N) for fixed T=max thread count
    max_threads = df['threads'].max()
    df_n = df[df['threads'] == max_threads].sort_values('docs_count')

    plt.figure(figsize=(8, 5))
    plt.plot(df_n['docs_count'], df_n['seq_total_ms'], 'o-', label='Sequential (Total)', color='#e74c3c', linewidth=2)
    plt.plot(df_n['docs_count'], df_n['par_total_ms'], 's-', label=f'Parallel ({max_threads} Threads Total)', color='#2ecc71', linewidth=2)
    plt.plot(df_n['docs_count'], df_n['seq_sim_ms'], 'o--', label='Sequential (Similarity Only)', color='#c0392b', alpha=0.7)
    plt.plot(df_n['docs_count'], df_n['par_sim_ms'], 's--', label=f'Parallel ({max_threads} Threads Sim Only)', color='#27ae60', alpha=0.7)

    plt.title("Execution Time vs Number of Documents (N)", fontsize=13, fontweight='bold')
    plt.xlabel("Number of Documents (N)", fontsize=11)
    plt.ylabel("Execution Time (ms)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "execution_time_vs_docs.png"), dpi=300)
    plt.close()
    print("[Plot] Saved results/execution_time_vs_docs.png")

    # 2. Speedup vs Number of Threads (for fixed N = max_N)
    max_n = df['docs_count'].max()
    df_threads = df[df['docs_count'] == max_n].sort_values('threads')

    plt.figure(figsize=(8, 5))
    plt.plot(df_threads['threads'], df_threads['speedup'], 'o-', label='Measured Compute Speedup', color='#2980b9', linewidth=2.5)
    plt.plot(df_threads['threads'], df_threads['threads'], 'k--', label='Ideal Linear Speedup', alpha=0.7)

    plt.title(f"Compute Speedup vs Thread Count (N = {max_n})", fontsize=13, fontweight='bold')
    plt.xlabel("Number of OpenMP Threads", fontsize=11)
    plt.ylabel("Speedup (Sequential / Parallel)", fontsize=11)
    plt.xticks(df_threads['threads'])
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "speedup_vs_threads.png"), dpi=300)
    plt.close()
    print("[Plot] Saved results/speedup_vs_threads.png")

    # 3. Parallel Efficiency vs Number of Threads
    plt.figure(figsize=(8, 5))
    plt.plot(df_threads['threads'], df_threads['efficiency'] * 100, 's-', label='Parallel Efficiency (%)', color='#8e44ad', linewidth=2.5)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.6, label='Ideal 100% Efficiency')

    plt.title(f"Parallel Efficiency vs Thread Count (N = {max_n})", fontsize=13, fontweight='bold')
    plt.xlabel("Number of OpenMP Threads", fontsize=11)
    plt.ylabel("Efficiency (%)", fontsize=11)
    plt.xticks(df_threads['threads'])
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "parallel_efficiency_vs_threads.png"), dpi=300)
    plt.close()
    print("[Plot] Saved results/parallel_efficiency_vs_threads.png")

    # 4. Stage Breakdown Comparison
    row = df_threads.iloc[-1]
    stages = ['Preprocessing', 'TF-IDF Build', 'Pairwise Similarity']
    seq_times = [row['seq_prep_ms'], row['seq_tfidf_ms'], row['seq_sim_ms']]
    par_times = [row['par_prep_ms'], row['par_tfidf_ms'], row['par_sim_ms']]

    x = range(len(stages))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], seq_times, width, label='Sequential', color='#e74c3c')
    plt.bar([i + width/2 for i in x], par_times, width, label=f'Parallel ({int(row["threads"])} Threads)', color='#2ecc71')

    plt.title("Execution Time Breakdown by Pipeline Stage", fontsize=13, fontweight='bold')
    plt.xticks(x, stages, fontsize=11)
    plt.ylabel("Time (ms)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "seq_vs_par_comparison.png"), dpi=300)
    plt.close()
    print("[Plot] Saved results/seq_vs_par_comparison.png")

    print("\n[Success] All benchmark charts generated successfully in results/")

if __name__ == "__main__":
    main()
