# Reranking Implementation Summary

## Overview
Successfully implemented two-stage retrieval with AWS Bedrock Rerank API for the financial RAG system on branch `harry-reranking`.

## Changes Made

### 1. Modified File: [scripts/query_faiss.py](scripts/query_faiss.py)

#### Added Functions

**`rerank_with_bedrock()`** (lines 133-194)
- Reranks text chunks using AWS Bedrock Rerank API
- Supports `amazon.rerank-v1:0` and `cohere.rerank-v3-5:0` models
- Parameters:
  - `query`: Search query text
  - `chunks`: List of text chunks to rerank
  - `model_id`: Bedrock reranker model (default: `amazon.rerank-v1:0`)
  - `region`: AWS region (default: `us-west-2`)
  - `top_n`: Number of top results to return (default: 5)
- Returns: List of `(original_index, relevance_score)` tuples

**`search_with_rerank()`** (lines 197-292)
- Implements two-stage retrieval pipeline
- Stage 1: FAISS retrieves top-K candidates (default: K=20)
- Stage 2: Bedrock reranks candidates, returns top-N (default: N=5)
- Parameters:
  - `query`: Search query
  - `collections`: List of FAISS collections
  - `initial_k`: Candidates from FAISS (default: 20)
  - `final_k`: Final results after reranking (default: 5)
  - `model_id`: Embedding model (default: `amazon.titan-embed-text-v2:0`)
  - `rerank_model_id`: Reranker model (default: `amazon.rerank-v1:0`)
  - `region`: AWS region (default: `us-west-2`)
  - `normalize`: L2 normalization flag (default: True)
  - `use_reranking`: Enable/disable reranking (default: True)

#### Updated CLI Arguments

Added three new arguments:
- `--no-rerank`: Disable reranking for baseline comparison
- `--initial-k`: Number of FAISS candidates (default: 20)
- `--rerank-model`: Bedrock reranker model ID (default: `amazon.rerank-v1:0`)

#### Modified Output Format

Results now display both scores when reranking is enabled:
```
rerank=0.8532 faiss=0.7123 doc_id=... chunk=...
```

When reranking is disabled (`--no-rerank`):
```
faiss=0.7123 doc_id=... chunk=...
```

## Usage Examples

### With Reranking (Default)
```bash
python scripts/query_faiss.py \
  --dir results \
  --query "What was NVIDIA's revenue in Q2 2025?" \
  --initial-k 20 \
  --top-k 5
```

### Without Reranking (Baseline)
```bash
python scripts/query_faiss.py \
  --dir results \
  --query "What was NVIDIA's revenue in Q2 2025?" \
  --no-rerank \
  --top-k 5
```

### Using Cohere Reranker
```bash
python scripts/query_faiss.py \
  --dir results \
  --query "What was NVIDIA's revenue in Q2 2025?" \
  --rerank-model cohere.rerank-v3-5:0 \
  --region us-east-1
```

## AWS Setup Requirements

### 1. IAM Permissions
Your AWS IAM role/user needs:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": [
                "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:Rerank"
            ],
            "Resource": "*"
        }
    ]
}
```

### 2. Enable Model Access
1. Go to [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Click "Model access" in left sidebar
3. Click "Modify model access"
4. Select "Amazon Rerank 1.0" (or "Cohere Rerank 3.5")
5. Click "Save changes"

### 3. Supported Regions
- `amazon.rerank-v1:0`: us-west-2, ca-central-1, eu-central-1, ap-northeast-1
- `cohere.rerank-v3-5:0`: Also available in us-east-1

## Testing

A test script is provided to verify the implementation:

```bash
python test_reranking.py
```

This validates that all functions are correctly implemented without requiring AWS credentials.

## Expected Improvements

Reranking should improve:
- **Precision**: Better at surfacing the most relevant chunk as #1
- **Temporal accuracy**: Better at distinguishing Q2 2025 from Q2 2024
- **Numerical queries**: Better at finding chunks with specific figures

Trade-off:
- **Latency**: Adds ~200-500ms per query (acceptable for financial analysis)

## Evaluation

To evaluate the reranking feature:

1. Run queries with reranking:
```bash
python scripts/query_faiss.py --dir results --query "Your query here"
```

2. Run queries without reranking:
```bash
python scripts/query_faiss.py --dir results --query "Your query here" --no-rerank
```

3. Compare F1 scores using the evaluation framework

## Next Steps

1. Configure AWS credentials
2. Enable Bedrock model access in AWS Console
3. Run evaluation on all 10 queries
4. Compare F1 scores with and without reranking
5. Document latency impact

## Files Changed
- [scripts/query_faiss.py](scripts/query_faiss.py): Added reranking functions and updated main query pipeline

## Files Added
- [test_reranking.py](test_reranking.py): Test script to verify implementation
- [RERANKING_IMPLEMENTATION.md](RERANKING_IMPLEMENTATION.md): This summary document

## Branch
All changes are on branch: `harry-reranking`

To view changes:
```bash
git diff main scripts/query_faiss.py
```
