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

```bash
# Single query
python scripts/query_faiss.py --dir results --query "your question here" --top-k 5

# Interactive mode
python scripts/query_faiss.py --dir results
```

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
