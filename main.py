from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from knn_search_python import python_find_top_k
import numpy as np
import time
import torch
import knn_search


def main():
    dataset = load_dataset("rajpurkar/squad", split="train")
    questions = dataset["question"]
    contexts = dataset["context"]

    unique_contexts = list(set(contexts))
    print(f"Unique contexts: {len(unique_contexts)}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

    docs = model.encode(unique_contexts, show_progress_bar=True)

    query_idx = np.random.randint(0, len(questions))
    query = model.encode([questions[query_idx]])[0]

    top_k = 5

    print(f"Query: {questions[query_idx]}\n")

    # Rust variant
    start_rust = time.time()
    results_rust = knn_search.find_top_k(query.tolist(), docs.tolist(), top_k)
    time_rust = time.time() - start_rust

    # Python variant
    start_py = time.time()
    results_py = python_find_top_k(query, docs, top_k)
    time_py = time.time() - start_py

    print(f"Top {top_k} similar contexts (Rust):")
    for idx, sim in results_rust:
        print(f"- similarity: {sim:.4f} | {unique_contexts[idx]}")

    print(f"\nTop {top_k} similar contexts (Python):")
    for idx, sim in results_py:
        print(f"- similarity: {sim:.4f} | {unique_contexts[idx]}")

    print(f"\nRust time: {time_rust:.4f}s")
    print(f"Python time: {time_py:.4f}s")
    if time_rust < time_py:
        speedup = time_py / time_rust
        print(f"Rust is {speedup:.1f}x faster")
    else:
        speedup = time_rust / time_py
        print(f"Python is {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
