#!/usr/bin/env python3
"""
Script to retrieve and save query results for manual inspection.
Saves both baseline (FAISS-only) and reranked results side-by-side.
"""

import json
from pathlib import Path
from query_faiss import _discover_collections, search_with_rerank

def main():
    query = "Which companies did Palantir partner with?"
    results_dir = Path("results")

    print(f"Loading collections from {results_dir}...")
    collections = list(_discover_collections(results_dir))
    print(f"Loaded {len(collections)} collections\n")

    print("=" * 80)
    print("BASELINE: FAISS-only (top-5)")
    print("=" * 80)

    # Get baseline results
    baseline_results = search_with_rerank(
        query=query,
        collections=collections,
        initial_k=20,
        final_k=5,
        use_reranking=False,  # FAISS only
    )

    baseline_data = []
    for rank, result in enumerate(baseline_results, start=1):
        print(f"\n{rank}. FAISS Score: {result['faiss_score']:.4f}")
        print(f"   Doc ID: {result['doc_id']}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Collection: {result['collection']}")
        print(f"   URL: {result['source_url']}")
        print(f"\n   Text:\n   {result['text'][:500]}...\n")
        print("-" * 80)

        baseline_data.append({
            "rank": rank,
            "faiss_score": float(result['faiss_score']),
            "doc_id": str(result['doc_id']),
            "chunk_id": int(result['chunk_id']),
            "collection": str(result['collection']),
            "url": str(result['source_url']),
            "text": str(result['text'])
        })

    print("\n" + "=" * 80)
    print("WITH RERANKING: FAISS → Bedrock Rerank (initial_k=20, final_k=5)")
    print("=" * 80)

    # Get reranked results
    reranked_results = search_with_rerank(
        query=query,
        collections=collections,
        initial_k=20,
        final_k=5,
        use_reranking=True,  # Enable reranking
    )

    reranked_data = []
    for rank, result in enumerate(reranked_results, start=1):
        print(f"\n{rank}. Rerank Score: {result['rerank_score']:.4f} | FAISS Score: {result['faiss_score']:.4f}")
        print(f"   Doc ID: {result['doc_id']}")
        print(f"   Chunk ID: {result['chunk_id']}")
        print(f"   Collection: {result['collection']}")
        print(f"   URL: {result['source_url']}")
        print(f"\n   Text:\n   {result['text'][:500]}...\n")
        print("-" * 80)

        reranked_data.append({
            "rank": rank,
            "rerank_score": float(result['rerank_score']),
            "faiss_score": float(result['faiss_score']),
            "doc_id": str(result['doc_id']),
            "chunk_id": int(result['chunk_id']),
            "collection": str(result['collection']),
            "url": str(result['source_url']),
            "text": str(result['text'])
        })

    # Save to JSON files
    output_dir = Path("inspection_results")
    output_dir.mkdir(exist_ok=True)

    baseline_file = output_dir / "baseline_faiss_only.json"
    reranked_file = output_dir / "reranked_results.json"

    with open(baseline_file, 'w') as f:
        json.dump({
            "query": query,
            "method": "FAISS only (top-5)",
            "results": baseline_data
        }, f, indent=2)

    with open(reranked_file, 'w') as f:
        json.dump({
            "query": query,
            "method": "FAISS (top-20) → Bedrock Rerank (top-5)",
            "results": reranked_data
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"Baseline results: {baseline_file}")
    print(f"Reranked results: {reranked_file}")
    print("\nYou can now manually inspect these JSON files to compare the results.")

if __name__ == "__main__":
    main()
