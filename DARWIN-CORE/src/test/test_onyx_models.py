#!/usr/bin/env python3
"""
test_all_onnx.py
────────────────
Single-file test suite for Darwin ONNX models.

Runs:
  1. Embedding sanity test (shape, dtype, normalization)
  2. Reranker sanity test (scores, ordering)
  3. End-to-end embed → search → rerank pipeline
  4. Optional: interactive search demo
  5. Optional: interactive rerank demo

Usage:
    uv run python test_all_onnx.py              # Run tests only
    uv run python test_all_onnx.py --demo         # Run tests + both demos
    uv run python test_all_onnx.py --search       # Run tests + search demo
    uv run python test_all_onnx.py --rerank       # Run tests + rerank demo
"""

import os
os.environ["DARWIN_DISABLE_TORCH"] = "1"

import argparse
import json
import numpy as np
from pathlib import Path

from src.core.Models.ModelManager import ModelManager
from src.core.Models.ModelRegistry import ModelCategory


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

def test_embedding():
    print("\n" + "="*60)
    print("🧪 TEST 1: ONNX Embedding (e5-small-v2-onnx)")
    print("="*60)

    manager = ModelManager()
    loader = manager.get_loader()
    embedder = loader.get(ModelCategory.EMBEDDING)

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "ONNX Runtime provides fast inference on CPU.",
    ]

    embeddings = embedder.encode(texts, normalize_embeddings=True)

    assert embeddings.shape[0] == len(texts), f"Batch size mismatch: {embeddings.shape[0]} != {len(texts)}"
    assert embeddings.shape[1] == 384, f"Expected 384-dim, got {embeddings.shape[1]}"
    assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"

    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings not L2-normalized"

    sim = float(embeddings[0] @ embeddings[1])

    print(f"   ✅ Shape:        {embeddings.shape}")
    print(f"   ✅ Dtype:        {embeddings.dtype}")
    print(f"   ✅ L2 norm:      {norms[0]:.6f}")
    print(f"   ✅ Cosine sim:   {sim:.4f}")
    print("   ✅ PASS")


def test_reranker():
    print("\n" + "="*60)
    print("🧪 TEST 2: ONNX Reranker (ms-marco-MiniLM-L-6-v2-onnx)")
    print("="*60)

    manager = ModelManager()
    loader = manager.get_loader()
    reranker = loader.get(ModelCategory.RERANKER)

    query = "What is machine learning?"
    docs = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "The capital of France is Paris.",
        "Deep learning is a subset of machine learning based on artificial neural networks.",
    ]

    pairs = [(query, d) for d in docs]
    scores = reranker.predict(pairs)

    assert len(scores) == len(docs), f"Score count mismatch: {len(scores)} != {len(docs)}"
    assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    print(f"   ✅ Scores:       {scores.tolist()}")
    print(f"   ✅ Best match:   \"{ranked[0][0][:50]}...\" (score={ranked[0][1]:.4f})")

    # ML-related docs should outrank Paris
    assert scores[0] > scores[1], "ML doc should score higher than Paris doc"
    assert scores[2] > scores[1], "Deep learning doc should score higher than Paris doc"
    print("   ✅ PASS")


def test_end_to_end():
    print("\n" + "="*60)
    print("🧪 TEST 3: End-to-End Embed → Search → Rerank")
    print("="*60)

    manager = ModelManager()
    loader = manager.get_loader()
    embedder = loader.get(ModelCategory.EMBEDDING)
    reranker = loader.get(ModelCategory.RERANKER)

    corpus = [
        "Python is a high-level programming language.",
        "JavaScript runs in web browsers and on servers via Node.js.",
        "Rust provides memory safety without garbage collection.",
        "Go was designed at Google for systems programming.",
        "The Eiffel Tower is located in Paris, France.",
    ]

    query = "Which language is memory-safe and fast?"

    # Embed
    doc_embeddings = embedder.encode(corpus, normalize_embeddings=True)
    query_embedding = embedder.encode([query], normalize_embeddings=True)

    # Search (top-3)
    similarities = (doc_embeddings @ query_embedding.T).squeeze()
    top_k = 3
    top_indices = similarities.argsort()[-top_k:][::-1]
    candidates = [corpus[i] for i in top_indices]

    print(f"   🔍 Top-{top_k} by embedding similarity:")
    for i, idx in enumerate(top_indices, 1):
        print(f"      {i}. [{similarities[idx]:.4f}] {corpus[idx][:55]}...")

    # Rerank
    pairs = [(query, c) for c in candidates]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)

    print(f"   🏆 After reranking:")
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"      {i}. [{score:.4f}] {doc[:55]}...")

    # Sanity: reranker should produce valid scores (not NaN/Inf)
    assert np.isfinite(rerank_scores).all(), "Reranker produced invalid scores"
    # Sanity: reranking should not crash and should return same count
    assert len(reranked) == len(candidates), "Reranker changed candidate count"
    print("   ✅ PASS")


# ═══════════════════════════════════════════════════════════════════════
# DEMOS
# ═══════════════════════════════════════════════════════════════════════

def demo_search():
    print("\n" + "="*60)
    print("🔍 DEMO: Interactive Semantic Search")
    print("="*60)
    print("   Paste documents (blank line to finish), then query.")

    manager = ModelManager()
    loader = manager.get_loader()
    embedder = loader.get(ModelCategory.EMBEDDING)

    docs = []
    print("\n📄 Documents (paste text, blank line to finish):")
    while True:
        line = input("   > ")
        if line.strip() == "":
            break
        docs.append(line.strip())

    if not docs:
        print("   ⚠️  No documents provided.")
        return

    print(f"\n🔢 Embedding {len(docs)} documents...")
    doc_embeddings = embedder.encode(docs, normalize_embeddings=True)
    print(f"   Corpus shape: {doc_embeddings.shape}")

    print("\n💬 Search queries (type 'quit' to exit):")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        q_emb = embedder.encode([query], normalize_embeddings=True)
        similarities = (doc_embeddings @ q_emb.T).squeeze()
        top_indices = similarities.argsort()[-5:][::-1]

        print(f"   Results for: \"{query}\"")
        for rank, idx in enumerate(top_indices, 1):
            snippet = docs[idx][:80]
            print(f"      {rank}. [{similarities[idx]:.4f}] {snippet}...")


def demo_rerank():
    print("\n" + "="*60)
    print("🏆 DEMO: Interactive Reranking")
    print("="*60)

    manager = ModelManager()
    loader = manager.get_loader()
    reranker = loader.get(ModelCategory.RERANKER)

    print("\n💬 Reranking sessions (type 'quit' to exit):")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        candidates = []
        print("   Candidates (blank line to finish):")
        while True:
            line = input("   > ")
            if line.strip() == "":
                break
            candidates.append(line.strip())

        if not candidates:
            print("   ⚠️  No candidates provided.")
            continue

        pairs = [(query, c) for c in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        print(f"\n   🏆 Reranked ({len(ranked)} candidates):")
        for i, (doc, score) in enumerate(ranked, 1):
            snippet = doc[:100]
            print(f"      {i}. [{score:.4f}] {snippet}...")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests():
    test_embedding()
    test_reranker()
    test_end_to_end()
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Darwin ONNX Test Suite")
    parser.add_argument("--demo", action="store_true", help="Run tests + all demos")
    parser.add_argument("--search", action="store_true", help="Run tests + search demo")
    parser.add_argument("--rerank", action="store_true", help="Run tests + rerank demo")
    args = parser.parse_args()

    run_all_tests()

    if args.demo or args.search:
        demo_search()
    if args.demo or args.rerank:
        demo_rerank()


if __name__ == "__main__":
    main()