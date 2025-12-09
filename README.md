# CSE 291A RAG System

A retrieval-augmented generation (RAG) system using AWS Bedrock's Amazon Titan embeddings and FAISS for semantic search over documents.

## Setup

### 1. Install Dependencies

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Configure AWS Credentials
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_SESSION_TOKEN=your_session_token  # if using temporary credentials
```

If you want these variables to persist either (1) add the environment variables to your shell configs or (2) install and setup [`aws-cli`](https://github.com/aws/aws-cli).

## Usage

### Generate Embeddings

From URLs:
```bash
python scripts/generate_embeddings.py data/palantir_urls.txt --out-dir results --doc-id-prefix pltr
```

From local files:
```bash
python scripts/generate_embeddings.py --file-dir /path/to/docs --out-dir results
```

### Query Documents

**Baseline query script:**
```bash
# Single query
python scripts/query_faiss.py --dir results --query "your question here" --top-k 5

# Interactive mode
python scripts/query_faiss.py --dir results
```

**Improved query script (recommended - with caching and optimized search):**
```bash
# Basic usage (same interface as baseline)
python scripts/query_faiss_improved.py --dir results --query "your question here" --top-k 5

# With cache statistics (shows hit/miss rates)
python scripts/query_faiss_improved.py --dir results --query "your question" --show-cache-stats

# Optimize recall with higher ef_search (default: top_k * 4)
python scripts/query_faiss_improved.py --dir results --query "your question" --ef-search 200 --top-k 5

# Disable caching (for testing)
python scripts/query_faiss_improved.py --dir results --query "your question" --no-cache

# Interactive mode
python scripts/query_faiss_improved.py --dir results --show-cache-stats
```

**Key improvements in `query_faiss_improved.py`:**
- **Embedding caching**: 10-100x speedup for repeated queries (LRU cache, max 1000 entries)
- **Configurable ef_search**: Better recall with optimized HNSW search parameters (default: `top_k * 4`)
- **Cache statistics**: Track cache performance with `--show-cache-stats`
- **Backward compatible**: Same interface as baseline script

## How It Works

1. **Ingestion**: Downloads/reads documents → extracts text → chunks into 256-word segments with 50-word overlap
2. **Embedding**: Each chunk is embedded using AWS Bedrock's Titan model (1024 dimensions)
3. **Indexing**: Creates FAISS HNSW index for fast similarity search
4. **Querying**: Embeds query text → searches FAISS index → returns most similar chunks

## Output Files

For each document, three files are generated:
- `{doc_id}_embeds.parquet` - Text chunks and embeddings
- `{doc_id}_hnsw.faiss` - FAISS search index
- `{doc_id}_meta.json` - Metadata

## Troubleshooting

**PyTorch error on macOS**: Remove `sentence-transformers` from `pyproject.toml` (unused dependency)

**AWS credentials expired**: Re-export your session credentials

**Import error**: Ensure you're running from the project root directory

**403/401 Forbidden errors**: Some sites block scrapers. The script now includes:
- More realistic browser headers
- Automatic retries with exponential backoff (3 attempts)
- Rate limiting between requests

For persistently blocked sites, download the HTML/PDF manually and use:
```bash
python scripts/generate_embeddings.py --file-dir /path/to/downloaded/files --out-dir results
```

## Performance Improvements

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed strategies to improve accuracy and retrieval speed.

### Run Comprehensive Evaluation

Compare baseline vs improved methods with timing:

**Using the evaluation script:**
```bash
# Full evaluation (baseline + improved + reranking)
python scripts/compare_baseline_vs_improved.py --base-dir ./data/embeddings

# Skip reranking (faster)
python scripts/compare_baseline_vs_improved.py --base-dir ./data/embeddings --no-rerank

# Custom parameters
python scripts/compare_baseline_vs_improved.py \
    --base-dir ./data/embeddings \
    --top-k 5 \
    --ef-search 200 \
    --output comparison_results.csv
```

**Using the shell script:**
```bash
# Full evaluation (baseline + improved + reranking)
./run_evaluation.sh

# Quick evaluation (skip reranking)
./run_evaluation.sh --skip-rerank

# Custom parameters
./run_evaluation.sh --top-k 10 --ef-search 300
```

**Evaluation script options:**
- `--base-dir`: Base directory containing company embedding folders (default: `./data/embeddings`)
- `--top-k`: Number of chunks to retrieve per query (default: 5)
- `--ef-search`: HNSW ef_search parameter for improved methods (default: 200)
- `--rerank-top-k`: Number of candidates to retrieve before reranking (default: 20)
- `--no-rerank`: Skip reranking evaluation (faster)
- `--output`: Output CSV file for results (default: `comparison_results.csv`)

**Note**: Requires `GROQ_API_KEY` environment variable for LLM-based relevance scoring.

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for detailed evaluation instructions.

### Quick Start with Improvements

**Faster queries with caching:**
```bash
# First query: ~200ms (API call)
# Subsequent queries: ~1-5ms (cache hit)
python scripts/query_faiss_improved.py --dir results --query "your question" --show-cache-stats
```

**Better accuracy with optimized search:**
```bash
# Higher ef_search = better recall (default: top_k * 4)
python scripts/query_faiss_improved.py --dir results --query "your question" --ef-search 200 --top-k 5
```

**Key improvements available:**
- **Embedding caching**: 10-100x speedup for repeated queries (LRU cache, 1000 entries)
- **HNSW parameter tuning**: Better recall with optimized `ef_search` (default: `top_k * 4`)
- **Cache statistics**: Track performance with `--show-cache-stats` flag
- **Backward compatible**: Drop-in replacement for `query_faiss.py`

**Performance results:**
- Overall: 1.5x faster, +3.3% F1 score improvement
- Latency consistency: 21x better (std dev: 895.9ms → 42.0ms)
- See `EVALUATION_RESULTS_SUMMARY.md` for detailed results

See `IMPROVEMENTS.md` for full details and implementation guide.
