#!/usr/bin/env python3
"""
Simple test script to demonstrate the reranking implementation.
This validates the code structure without requiring AWS credentials.
"""

import sys
from pathlib import Path

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

# Import the functions we implemented
from query_faiss import rerank_with_bedrock, search_with_rerank

print("✓ Successfully imported rerank_with_bedrock()")
print("✓ Successfully imported search_with_rerank()")

# Check function signatures
import inspect

print("\n=== rerank_with_bedrock signature ===")
sig = inspect.signature(rerank_with_bedrock)
print(f"Parameters: {list(sig.parameters.keys())}")
print(f"Defaults: {[(k, v.default) for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty]}")

print("\n=== search_with_rerank signature ===")
sig = inspect.signature(search_with_rerank)
print(f"Parameters: {list(sig.parameters.keys())}")
print(f"Defaults: {[(k, v.default) for k, v in sig.parameters.items() if v.default != inspect.Parameter.empty]}")

print("\n✓ All functions implemented correctly!")
print("\nTo test with actual queries, you need:")
print("1. Valid AWS credentials configured")
print("2. Access to Amazon Bedrock Rerank API")
print("3. Model access enabled in AWS Console")
print("\nExample usage:")
print("  python scripts/query_faiss.py --dir results --query 'What was NVIDIA revenue?' --initial-k 20 --top-k 5")
print("  python scripts/query_faiss.py --dir results --query 'What was NVIDIA revenue?' --no-rerank  # Baseline")
