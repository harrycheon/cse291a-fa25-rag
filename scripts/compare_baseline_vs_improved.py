"""Comprehensive evaluation script comparing baseline vs improved methods with timing."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from get_metrics import retrieve_chunks_from_faiss, rate_chunks_with_llm
    from query_faiss_improved import (
        _discover_collections,
        _prepare_query_vector,
        Collection,
    )
    from query_faiss_with_rerank import retrieve_with_rerank
except ImportError as exc:
    raise SystemExit(f"Failed to import required modules: {exc}") from exc

import faiss

# Load environment variables
load_dotenv()

# Define all queries organized by company
QUERIES = {
    "alphabet": {
        "dir": "alphabet",
        "queries": [
            "How is Alphabet doing financially?",
            "How has Alphabet's partnership helped it to grow financially?",
        ],
    },
    "doordash": {
        "dir": "doordash",
        "queries": [
            "What are the revenue, total orders and adjusted EPS for Q2 2025, and the forward guidance provided for Q3 2025?",
            "Summarize the annual shareholders voting results.",
        ],
    },
    "palantir": {
        "dir": "palantir",
        "queries": [
            "What are the carbon emissions trends of the company?",
            "Which companies did Palantir partner with?",
        ],
    },
    "qualcomm": {
        "dir": "qualcomm",
        "queries": [
            "Why is Qualcomm shifting automotive module production to India?",
            "Why might Qualcomm's valuation be considered compelling?",
        ],
    },
    "nvidia": {
        "dir": "nvidia",
        "queries": [
            "Is NVIDIA growth on par with market expectations? Support this with concrete financial evidence.",
            "Is the company overexposed to risk in its current position of cash and financial portfolio? What poses the greatest risk, if any?",
        ],
    },
}


def calculate_f1_score(scores: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score based on relevance scores."""
    if not scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_score": 0.0,
            "num_retrieved": 0,
            "num_relevant": 0,
        }

    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_score": 0.0,
            "num_retrieved": len(scores),
            "num_relevant": 0,
        }

    predicted_relevant = sum(1 for s in valid_scores if s >= threshold)
    num_retrieved = len(valid_scores)
    precision = predicted_relevant / num_retrieved if num_retrieved > 0 else 0.0
    mean_score = np.mean(valid_scores)
    recall = mean_score  # Proxy for recall

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_score": mean_score,
        "num_retrieved": num_retrieved,
        "num_relevant": predicted_relevant,
    }


def retrieve_baseline(
    query: str,
    dir_path: Path,
    top_k: int = 5,
) -> Tuple[List[str], float]:
    """Baseline retrieval using original method."""
    start_time = time.time()
    chunks = retrieve_chunks_from_faiss(query, dir_path, top_k=top_k)
    latency = time.time() - start_time
    return chunks, latency


def retrieve_improved_cached(
    query: str,
    collections: List[Collection],
    top_k: int = 5,
    ef_search: int | None = None,
) -> Tuple[List[str], float]:
    """Improved retrieval with caching and optimized HNSW."""
    start_time = time.time()
    
    # Set ef_search if provided
    if ef_search:
        for collection in collections:
            if isinstance(collection.index, faiss.IndexHNSW):
                collection.index.hnsw.efSearch = ef_search
    
    # Prepare query vector (with caching)
    expected_dim = collections[0].index.d
    query_vec = _prepare_query_vector(
        query,
        index_dim=expected_dim,
        model_id="amazon.titan-embed-text-v2:0",
        region="us-west-2",
        normalize=True,
        use_cache=True,
    )
    
    # Search
    aggregated: List[Tuple[float, str]] = []
    for collection in collections:
        distances, indices = collection.index.search(query_vec, top_k)
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(collection.dataframe):
                continue
            text = collection.dataframe.iloc[idx].get("text", "")
            aggregated.append((float(score), text))
    
    aggregated.sort(key=lambda x: x[0], reverse=True)
    chunks = [text for _, text in aggregated[:top_k]]
    
    latency = time.time() - start_time
    return chunks, latency


def retrieve_improved_rerank(
    query: str,
    collections: List[Collection],
    top_k: int = 5,
    rerank_top_k: int = 20,
    ef_search: int | None = None,
) -> Tuple[List[str], float]:
    """Improved retrieval with reranking."""
    start_time = time.time()
    results = retrieve_with_rerank(
        query=query,
        collections=collections,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        use_cross_encoder=True,
        ef_search=ef_search,
    )
    chunks = [text for text, _, _ in results]
    latency = time.time() - start_time
    return chunks, latency


def evaluate_method(
    method_name: str,
    retrieve_fn: callable,
    query: str,
    company_dir: Path,
    collections: List[Collection] | None,
    top_k: int = 5,
    **kwargs,
) -> Dict:
    """Evaluate a single retrieval method on a query."""
    try:
        if collections is not None:
            # Improved methods that use collections
            chunks, latency = retrieve_fn(query, collections, top_k=top_k, **kwargs)
        else:
            # Baseline method that uses directory path
            chunks, latency = retrieve_fn(query, company_dir, top_k=top_k)
        
        if not chunks:
            return {
                "method": method_name,
                "query": query,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "mean_score": 0.0,
                "num_retrieved": 0,
                "num_relevant": 0,
                "latency_ms": latency * 1000,
                "error": None,
            }
        
        # Rate chunks with LLM
        ratings = rate_chunks_with_llm(query, chunks)
        scores = [item.get("score") for item in ratings]
        
        # Calculate metrics
        metrics = calculate_f1_score(scores, threshold=0.5)
        
        return {
            "method": method_name,
            "query": query,
            "f1_score": metrics["f1_score"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "mean_score": metrics["mean_score"],
            "num_retrieved": metrics["num_retrieved"],
            "num_relevant": metrics["num_relevant"],
            "latency_ms": latency * 1000,
            "error": None,
        }
    except Exception as e:
        return {
            "method": method_name,
            "query": query,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mean_score": 0.0,
            "num_retrieved": 0,
            "num_relevant": 0,
            "latency_ms": 0.0,
            "error": str(e),
        }


def run_comparison(
    base_dir: str = "./data/embeddings",
    top_k: int = 5,
    ef_search: int = 200,
    rerank_top_k: int = 20,
    include_rerank: bool = True,
) -> pd.DataFrame:
    """Run comprehensive comparison of baseline vs improved methods."""
    all_results = []
    
    print("=" * 80)
    print("RAG SYSTEM COMPARISON: Baseline vs Improved Methods")
    print("=" * 80)
    
    for company, config in QUERIES.items():
        company_dir_str = config["dir"]
        if company_dir_str.startswith("../") or company_dir_str.startswith("/"):
            company_dir = (Path(base_dir) / company_dir_str).resolve()
        else:
            company_dir = Path(base_dir) / company_dir_str
        
        if not company_dir.exists():
            print(f"[WARNING] Directory not found: {company_dir}")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Evaluating {company.upper()}")
        print(f"{'=' * 80}")
        
        # Load collections once for improved methods
        collections = list(_discover_collections(company_dir))
        
        queries = config["queries"]
        for query_idx, query in enumerate(queries, 1):
            print(f"\nQuery {query_idx}/{len(queries)}: {query[:70]}...")
            
            # 1. Baseline
            print("  [1/3] Running baseline...", end=" ", flush=True)
            result_baseline = evaluate_method(
                "baseline",
                retrieve_baseline,
                query,
                company_dir,
                collections=None,
                top_k=top_k,
            )
            print(f"✓ (F1: {result_baseline['f1_score']:.3f}, Latency: {result_baseline['latency_ms']:.1f}ms)")
            all_results.append({**result_baseline, "company": company})
            
            # 2. Improved with caching + ef_search
            print("  [2/3] Running improved (cached + ef_search)...", end=" ", flush=True)
            result_improved = evaluate_method(
                "improved_cached",
                retrieve_improved_cached,
                query,
                company_dir,
                collections=collections,
                top_k=top_k,
                ef_search=ef_search,
            )
            print(f"✓ (F1: {result_improved['f1_score']:.3f}, Latency: {result_improved['latency_ms']:.1f}ms)")
            all_results.append({**result_improved, "company": company})
            
            # 3. Improved with reranking
            if include_rerank:
                print("  [3/3] Running improved (with reranking)...", end=" ", flush=True)
                try:
                    result_rerank = evaluate_method(
                        "improved_rerank",
                        retrieve_improved_rerank,
                        query,
                        company_dir,
                        collections=collections,
                        top_k=top_k,
                        rerank_top_k=rerank_top_k,
                        ef_search=ef_search,
                    )
                    print(f"✓ (F1: {result_rerank['f1_score']:.3f}, Latency: {result_rerank['latency_ms']:.1f}ms)")
                    all_results.append({**result_rerank, "company": company})
                except ImportError:
                    print("✗ (sentence-transformers not installed, skipping)")
    
    return pd.DataFrame(all_results)


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics comparing methods."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Overall statistics by method
    print("\nOverall Performance by Method:")
    print("-" * 80)
    summary = df.groupby("method").agg({
        "f1_score": ["mean", "std"],
        "precision": "mean",
        "recall": "mean",
        "latency_ms": ["mean", "std"],
    }).round(3)
    print(summary)
    
    # Per-company comparison
    print("\n\nPer-Company Comparison:")
    print("-" * 80)
    company_method = df.groupby(["company", "method"]).agg({
        "f1_score": "mean",
        "latency_ms": "mean",
    }).round(3)
    print(company_method)
    
    # Improvement metrics
    print("\n\nImprovement Metrics:")
    print("-" * 80)
    baseline_avg = df[df["method"] == "baseline"]["f1_score"].mean()
    improved_avg = df[df["method"] == "improved_cached"]["f1_score"].mean()
    
    if "improved_rerank" in df["method"].values:
        rerank_avg = df[df["method"] == "improved_rerank"]["f1_score"].mean()
        print(f"Baseline F1:           {baseline_avg:.3f}")
        print(f"Improved (cached) F1:  {improved_avg:.3f} (+{(improved_avg - baseline_avg) * 100:.1f}%)")
        print(f"Improved (rerank) F1:  {rerank_avg:.3f} (+{(rerank_avg - baseline_avg) * 100:.1f}%)")
    else:
        print(f"Baseline F1:           {baseline_avg:.3f}")
        print(f"Improved (cached) F1:  {improved_avg:.3f} (+{(improved_avg - baseline_avg) * 100:.1f}%)")
    
    baseline_latency = df[df["method"] == "baseline"]["latency_ms"].mean()
    improved_latency = df[df["method"] == "improved_cached"]["latency_ms"].mean()
    speedup = baseline_latency / improved_latency if improved_latency > 0 else 0
    print(f"\nBaseline Latency:      {baseline_latency:.1f}ms")
    print(f"Improved Latency:      {improved_latency:.1f}ms ({speedup:.1f}x speedup)")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare baseline vs improved RAG methods with timing."
    )
    parser.add_argument(
        "--base-dir",
        default="./data/embeddings",
        help="Base directory containing company embedding folders.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query.",
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=200,
        help="HNSW ef_search parameter for improved methods.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=20,
        help="Number of candidates to retrieve before reranking.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking evaluation (faster).",
    )
    parser.add_argument(
        "--output",
        default="comparison_results.csv",
        help="Output CSV file for results.",
    )
    args = parser.parse_args()
    
    # Check for required API keys
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY not found in environment")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    # Run comparison
    results_df = run_comparison(
        base_dir=args.base_dir,
        top_k=args.top_k,
        ef_search=args.ef_search,
        rerank_top_k=args.rerank_top_k,
        include_rerank=not args.no_rerank,
    )
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {args.output}")
    
    # Print summary
    print_summary(results_df)


if __name__ == "__main__":
    main()

