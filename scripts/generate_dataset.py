#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Document Similarity Benchmarking
Generates reproducible synthetic text documents with controlled vocabulary,
document lengths, topic clusters, and near-duplicate pairs.
"""

import argparse
import random
import os

def generate_vocab(vocab_size):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    vocab = set()
    random.seed(42)
    while len(vocab) < vocab_size:
        length = random.randint(4, 10)
        word = "".join(random.choice(alphabet) for _ in range(length))
        vocab.add(word)
    return list(vocab)

def generate_documents(num_docs, doc_length, vocab, num_clusters=5, near_dup_ratio=0.05, seed=42):
    random.seed(seed)
    vocab_size = len(vocab)
    
    # Generate cluster seed topics
    cluster_vocab = []
    cluster_size = max(10, vocab_size // (num_clusters * 2))
    for i in range(num_clusters):
        start_idx = (i * cluster_size) % vocab_size
        cluster_vocab.append(vocab[start_idx : start_idx + cluster_size])

    docs = []
    num_duplicates = int(num_docs * near_dup_ratio)
    
    for i in range(num_docs - num_duplicates):
        doc_id = f"doc_{i+1:05d}"
        cluster_idx = i % num_clusters
        topic_words = cluster_vocab[cluster_idx]
        
        words = []
        for _ in range(doc_length):
            if random.random() < 0.6:  # 60% topic specific
                words.append(random.choice(topic_words))
            else:  # 40% general vocabulary
                words.append(random.choice(vocab))
        
        doc_text = " ".join(words)
        docs.append((doc_id, doc_text))
        
    # Generate near duplicates by modifying existing documents slightly
    for i in range(num_duplicates):
        source_doc_idx = random.randint(0, len(docs) - 1)
        source_id, source_text = docs[source_doc_idx]
        dup_id = f"dup_{i+1:05d}_of_{source_id}"
        
        words = source_text.split()
        # Mutate ~5% of words to create a near-duplicate
        mutations = max(1, int(len(words) * 0.05))
        for _ in range(mutations):
            pos = random.randint(0, len(words) - 1)
            words[pos] = random.choice(vocab)
            
        dup_text = " ".join(words)
        docs.append((dup_id, dup_text))

    return docs

def main():
    parser = argparse.ArgumentParser(description="Synthetic Document Corpus Generator")
    parser.add_argument("--num_docs", type=int, default=500, help="Number of documents to generate")
    parser.add_argument("--doc_length", type=int, default=300, help="Average words per document")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Total vocabulary size")
    parser.add_argument("--num_clusters", type=int, default=5, help="Number of topic clusters")
    parser.add_argument("--near_dup_ratio", type=float, default=0.05, help="Ratio of near-duplicate documents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/synthetic_corpus.txt", help="Output file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    vocab = generate_vocab(args.vocab_size)
    docs = generate_documents(
        num_docs=args.num_docs,
        doc_length=args.doc_length,
        vocab=vocab,
        num_clusters=args.num_clusters,
        near_dup_ratio=args.near_dup_ratio,
        seed=args.seed
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(f"{doc_id}\t{text}\n")

    print(f"[Success] Generated {len(docs)} synthetic documents in: {args.output}")

if __name__ == "__main__":
    main()
