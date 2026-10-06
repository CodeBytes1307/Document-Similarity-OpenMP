#!/usr/bin/env python3
"""
Realistic Corpus Generator for Document Similarity Benchmarking
Generates meaningful, domain-specific English documents covering topics such as
Parallel Computing, Quantum Physics, Finance, Medical Genomics, Astrophysics, and Climate Science.
Includes stop words, natural sentence structures, topic clusters, and near-duplicate articles.
"""

import argparse
import random
import os

# Domain-specific word pools and sentence templates
DOMAINS = {
    "parallel_computing": {
        "keywords": [
            "parallel", "computing", "openmp", "multithreading", "concurrency", "performance",
            "optimization", "thread", "processor", "cache", "memory", "speedup", "efficiency",
            "scalability", "matrix", "vector", "sparse", "algorithm", "execution", "synchronization",
            "lock", "mutex", "workload", "scheduler", "core", "supercomputer", "cluster", "node",
            "bandwidth", "throughput", "latency", "bottleneck", "load", "balancing", "barrier"
        ],
        "templates": [
            "High performance parallel computing enables massive acceleration across multi core architectures.",
            "Efficient OpenMP thread scheduling reduces synchronization overhead and memory latency in sparse matrix operations.",
            "Parallel algorithms must balance computational workloads to achieve linear speedup and high parallel efficiency.",
            "Multi threading applications utilize shared memory caches to optimize data throughput and minimize bandwidth bottlenecks.",
            "Hardware performance counters track thread execution and cache miss ratios across distributed supercomputer nodes."
        ]
    },
    "quantum_physics": {
        "keywords": [
            "quantum", "physics", "superposition", "entanglement", "qubit", "coherence", "wavefunction",
            "mechanics", "spin", "photon", "state", "schrodinger", "particle", "interference",
            "decoherence", "teleportation", "algorithm", "circuit", "gate", "tunneling", "subatomic",
            "electron", "field", "quantum-computing", "hamiltonian", "eigenstate", "probability", "density"
        ],
        "templates": [
            "Quantum entanglement allows subatomic particles to maintain correlated quantum states over vast physical distances.",
            "Superposition principles enable qubits to process complex wavefunctions simultaneously inside quantum circuits.",
            "Decoherence remains a primary challenge in maintaining quantum state stability for quantum gate execution.",
            "Schrodinger equation describes the time dependent probability density of quantum mechanical systems.",
            "Photonic qubits exhibit high quantum coherence making them suitable for long distance quantum teleportation experiments."
        ]
    },
    "finance_economics": {
        "keywords": [
            "finance", "market", "economy", "investment", "portfolio", "asset", "equity", "stock",
            "dividend", "inflation", "banking", "capital", "liquidity", "volatility", "arbitrage",
            "yield", "bond", "monetary", "fiscal", "interest", "derivative", "hedging", "risk",
            "transaction", "valuation", "revenue", "financial", "trader", "speculation", "growth"
        ],
        "templates": [
            "Financial market volatility influences portfolio asset allocation and risk management strategies across equity markets.",
            "Central bank monetary policies adjust interest rates to manage inflation expectations and economic capital growth.",
            "Investment portfolios utilize hedging techniques and derivatives to mitigate downside liquidity risks.",
            "Stock valuations reflect corporate revenue prospects, dividend yields, and broader macroeconomic indicators.",
            "High frequency trading algorithms exploit arbitrage opportunities across global financial exchanges in real time."
        ]
    },
    "biomedical_genomics": {
        "keywords": [
            "genomics", "dna", "rna", "sequence", "gene", "protein", "cell", "mutation", "chromosome",
            "biomedical", "crispr", "expression", "molecular", "enzyme", "pathway", "disease",
            "therapy", "clinical", "genome", "nucleotide", "transcription", "biotechnology", "cancer",
            "amino", "acid", "biological", "organism", "diagnosis", "antibody", "receptor"
        ],
        "templates": [
            "Genomic sequencing technologies identify DNA mutations and gene expression patterns in targeted biomedical therapies.",
            "CRISPR gene editing mechanisms enable precise nucleotide modifications within complex cellular molecular pathways.",
            "Protein synthesis relies on RNA transcription to translate genetic code into functional biological enzymes.",
            "Clinical research investigates biotechnology solutions for diagnosing cancer and engineering neutralizing antibodies.",
            "Chromosome structural variations provide key insights into genetic disease mechanisms and cellular receptors."
        ]
    },
    "astrophysics_space": {
        "keywords": [
            "astronomy", "astrophysics", "galaxy", "stellar", "nebula", "blackhole", "gravitational",
            "orbit", "telescope", "radiation", "spectrum", "cosmology", "supernova", "planet",
            "exoplanet", "space", "lightyear", "universe", "relativity", "dark-matter", "energy",
            "atmosphere", "cosmic", "satellite", "observation", "singularity", "event-horizon", "mass"
        ],
        "templates": [
            "Astrophysical observations using space telescopes detect gravitational waves emitted from merging stellar mass black holes.",
            "Exoplanet atmospheric spectra reveal potential biosignatures and chemical compositions across distant galaxy systems.",
            "Supernova explosions seed interstellar nebulae with heavy chemical elements essential for planetary formation.",
            "General relativity equations describe space time curvature surrounding dense cosmic singularities and event horizons.",
            "Cosmological models analyze dark matter distributions to understand the accelerated expansion of the physical universe."
        ]
    },
    "climate_environment": {
        "keywords": [
            "climate", "environment", "renewable", "sustainability", "carbon", "emissions", "solar",
            "wind", "temperature", "ecosystem", "biodiversity", "atmosphere", "glacier", "sea-level",
            "conservation", "energy", "clean-tech", "meteorology", "footprint", "forest", "ocean",
            "greenhouse", "warming", "resource", "recycling", "pollution", "ecology", "habitat"
        ],
        "templates": [
            "Renewable energy technologies like solar and wind power significantly reduce atmospheric greenhouse gas carbon emissions.",
            "Climate models predict rising sea levels and melting polar glaciers caused by global surface temperature increases.",
            "Ecosystem conservation efforts protect biodiversity hotspots and restore fragile marine and terrestrial habitats.",
            "Clean technology adoption promotes sustainable resource recycling and minimizes environmental industrial pollution.",
            "Oceanographic studies monitor carbon absorption rates and acid levels in global marine ecological systems."
        ]
    }
}

GENERAL_TRANSITIONS = [
    "Furthermore, recent empirical investigations demonstrate that",
    "It is important to note that researchers observed",
    "In addition, comprehensive quantitative analyses show",
    "Consequently, the baseline system performance depends on",
    "As a result, experimental evaluations confirm that",
    "However, alternative theoretical frameworks suggest",
    "Similarly, advanced computational methodology highlights",
    "Overall, systematic comparisons indicate that"
]

COMMON_STOPWORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what"
]

def generate_document_text(domain_key, target_words):
    domain = DOMAINS[domain_key]
    templates = domain["templates"]
    keywords = domain["keywords"]

    text_parts = []
    current_words = 0

    while current_words < target_words:
        # Pick sentence template
        base_sentence = random.choice(templates)
        transition = random.choice(GENERAL_TRANSITIONS) if random.random() < 0.4 else ""
        
        # Add a filler sentence or keyword phrase
        extra_keywords = " ".join(random.sample(keywords, min(4, len(keywords))))
        filler_words = " ".join(random.sample(COMMON_STOPWORDS, 3))

        sentence = f"{transition} {base_sentence} Key concepts include {extra_keywords} along with {filler_words}."
        text_parts.append(sentence)
        current_words += len(sentence.split())

    return " ".join(text_parts)

def generate_dataset(num_docs, doc_length, near_dup_ratio=0.05, seed=42):
    random.seed(seed)
    domain_keys = list(DOMAINS.keys())
    docs = []

    num_duplicates = int(num_docs * near_dup_ratio)
    base_docs_count = num_docs - num_duplicates

    # Generate base documents across domains
    for i in range(base_docs_count):
        domain = domain_keys[i % len(domain_keys)]
        doc_id = f"doc_{i+1:05d}_{domain}"
        text = generate_document_text(domain, doc_length)
        docs.append((doc_id, text))

    # Generate realistic near-duplicates by mutating existing documents slightly
    for i in range(num_duplicates):
        source_id, source_text = docs[random.randint(0, len(docs) - 1)]
        dup_id = f"dup_{i+1:05d}_of_{source_id}"

        words = source_text.split()
        # Mutate ~5% of words to create a realistic near-duplicate article
        mutations = max(1, int(len(words) * 0.05))
        for _ in range(mutations):
            pos = random.randint(0, len(words) - 1)
            # Replace with a word from general stopwords or random keywords
            domain = source_id.split("_")[-1] if source_id.split("_")[-1] in DOMAINS else "parallel_computing"
            words[pos] = random.choice(DOMAINS[domain]["keywords"])

        dup_text = " ".join(words)
        docs.append((dup_id, dup_text))

    return docs

def main():
    parser = argparse.ArgumentParser(description="Meaningful Corpus Generator for Document Similarity Analysis")
    parser.add_argument("--num_docs", type=int, default=500, help="Number of documents to generate")
    parser.add_argument("--doc_length", type=int, default=300, help="Average words per document")
    parser.add_argument("--near_dup_ratio", type=float, default=0.05, help="Ratio of near-duplicate articles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/synthetic_corpus.txt", help="Output file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    docs = generate_dataset(
        num_docs=args.num_docs,
        doc_length=args.doc_length,
        near_dup_ratio=args.near_dup_ratio,
        seed=args.seed
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            f.write(f"{doc_id}\t{text}\n")

    print(f"[Success] Generated {len(docs)} meaningful English documents in: {args.output}")

if __name__ == "__main__":
    main()
