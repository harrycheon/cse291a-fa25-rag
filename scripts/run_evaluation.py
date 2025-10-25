"""
Comprehensive evaluation script for RAG system.
Runs all queries and computes F1 scores based on LLM relevance ratings.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
import json
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from get_metrics import retrieve_chunks_from_faiss, rate_chunks_with_llm
except ImportError:
    raise SystemExit("Failed to import from get_metrics; run from project root.")

# Load environment variables
load_dotenv()

# Define all queries organized by company
# Note: directory path is relative to base_dir (default: ./data/embeddings)
QUERIES = {
    "alphabet": {
        "dir": "alphabet",
        "queries": [
            "How is Alphabet doing financially?",
            "How has Alphabet's partnership helped it to grow financially?"
        ]
    },
    "doordash": {
        "dir": "doordash",
        "queries": [
            "What are the revenue, total orders and adjusted EPS for Q2 2025, and the forward guidance provided for Q3 2025?",
            "Summarize the annual shareholders voting results."
        ]
    },
    "palantir": {
        "dir": "palantir",
        "queries": [
            "What are the carbon emissions trends of the company?",
            "Which companies did Palantir partner with?"
        ]
    },
    "qualcomm": {
        "dir": "qualcomm",
        "queries": [
            "Why is Qualcomm shifting automotive module production to India?",
            "Why might Qualcomm's valuation be considered compelling?"
        ]
    },
    "nvidia": {
        "dir": "nvidia",
        "queries": [
            "Is NVIDIA growth on par with market expectations? Support this with concrete financial evidence.",
            "Is the company overexposed to risk in its current position of cash and financial portfolio? What poses the greatest risk, if any?"
        ]
    }
}

def calculate_f1_score(scores: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score based on relevance scores.
    
    Args:
        scores: List of relevance scores (0-1) from LLM
        threshold: Score threshold to consider a chunk as relevant
    
    Returns:
        Dictionary with precision, recall, F1, and other metrics
    """
    if not scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_score": 0.0,
            "num_retrieved": 0,
            "num_relevant": 0
        }
    
    # Filter out None scores
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_score": 0.0,
            "num_retrieved": len(scores),
            "num_relevant": 0
        }
    
    # Count how many chunks are above threshold (predicted relevant)
    predicted_relevant = sum(1 for s in valid_scores if s >= threshold)
    
    # For evaluation purposes, we consider:
    # - TP (True Positives): chunks with score >= threshold
    # - FP (False Positives): chunks with score < threshold
    # - Precision = TP / (TP + FP) = relevant_retrieved / total_retrieved
    # - Since we don't have ground truth, we use the LLM scores as proxy
    
    # Calculate weighted precision based on scores
    tp = sum(s for s in valid_scores if s >= threshold)
    fp = sum((1 - s) for s in valid_scores if s < threshold)
    fn = sum((1 - s) for s in valid_scores if s >= threshold)
    
    num_retrieved = len(valid_scores)
    
    # Precision: Of all retrieved chunks, what fraction is relevant?
    precision = predicted_relevant / num_retrieved if num_retrieved > 0 else 0.0
    
    # Recall: We approximate this using the mean score (assumes ideal recall = 1.0)
    # A better approach would be to have ground truth, but we use mean score as proxy
    mean_score = np.mean(valid_scores)
    recall = mean_score  # Proxy for recall
    
    # F1 Score
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
        "num_relevant": predicted_relevant
    }


def evaluate_all_queries(
    top_k: int = 5,
    threshold: float = 0.5,
    base_dir: str = "./data/embeddings"
) -> pd.DataFrame:
    """
    Evaluate all queries across all companies and compute F1 scores.
    
    Args:
        top_k: Number of chunks to retrieve per query
        threshold: Relevance threshold for F1 calculation
        base_dir: Base directory containing company embedding folders
    
    Returns:
        DataFrame with evaluation results
    """
    all_results = []
    
    for company, config in QUERIES.items():
        # Get directory path (can be relative to base_dir or absolute)
        company_dir_str = config["dir"]
        if company_dir_str.startswith("../") or company_dir_str.startswith("/"):
            # Relative to base_dir or absolute path
            company_dir = (Path(base_dir) / company_dir_str).resolve()
        else:
            # Regular subdirectory of base_dir
            company_dir = Path(base_dir) / company_dir_str
        
        queries = config["queries"]
        
        if not company_dir.exists():
            print(f"[WARNING] Directory not found: {company_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {company.upper()}")
        print(f"{'='*60}")
        
        for query_idx, query in enumerate(queries, 1):
            print(f"\nQuery {query_idx}/{len(queries)}: {query[:80]}...")
            
            try:
                # Retrieve chunks
                chunks = retrieve_chunks_from_faiss(
                    query=query,
                    dir_path=company_dir,
                    top_k=top_k
                )
                
                if not chunks:
                    print(f"  [WARNING] No chunks retrieved")
                    all_results.append({
                        "company": company,
                        "query": query,
                        "f1_score": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "mean_score": 0.0,
                        "num_retrieved": 0,
                        "num_relevant": 0
                    })
                    continue
                
                # Rate chunks with LLM
                ratings = rate_chunks_with_llm(query, chunks)
                scores = [item.get("score") for item in ratings]
                
                # Calculate F1 score
                metrics = calculate_f1_score(scores, threshold=threshold)
                
                result = {
                    "company": company,
                    "query": query,
                    "f1_score": metrics["f1_score"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "mean_score": metrics["mean_score"],
                    "num_retrieved": metrics["num_retrieved"],
                    "num_relevant": metrics["num_relevant"]
                }
                all_results.append(result)
                
                # Print summary
                print(f"  F1 Score: {metrics['f1_score']:.3f}")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
                print(f"  Mean Relevance: {metrics['mean_score']:.3f}")
                print(f"  Retrieved/Relevant: {metrics['num_retrieved']}/{metrics['num_relevant']}")
                
            except Exception as e:
                print(f"  [ERROR] Failed to evaluate query: {e}")
                all_results.append({
                    "company": company,
                    "query": query,
                    "f1_score": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "mean_score": 0.0,
                    "num_retrieved": 0,
                    "num_relevant": 0,
                    "error": str(e)
                })
    
    return pd.DataFrame(all_results)


def main():
    """Main execution function."""
    print("Starting RAG System Evaluation")
    print("="*60)
    
    # Check for required API keys
    if not os.environ.get("GROQ_API_KEY"):
        print("[ERROR] GROQ_API_KEY not found in environment")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    # Run evaluation
    results_df = evaluate_all_queries(top_k=5, threshold=0.5)
    
    # Save results
    output_csv = "evaluation_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("="*60)
    
    # Overall statistics
    print(f"\nOverall Performance:")
    print(f"  Mean F1 Score: {results_df['f1_score'].mean():.3f}")
    print(f"  Mean Precision: {results_df['precision'].mean():.3f}")
    print(f"  Mean Recall: {results_df['recall'].mean():.3f}")
    print(f"  Mean Relevance Score: {results_df['mean_score'].mean():.3f}")
    
    # Per-company statistics
    print(f"\nPer-Company Performance:")
    company_stats = results_df.groupby('company')[['f1_score', 'precision', 'recall', 'mean_score']].mean()
    print(company_stats.to_string())
    
    # Save detailed results with formatting
    print("\nDETAILED RESULTS:")
    print("="*60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

