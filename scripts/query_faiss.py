"""CLI utility for querying stored FAISS indices using Titan embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import boto3
import faiss
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from generate_embeddings import get_titan_embedding  
except ImportError as exc:  
    raise SystemExit("Failed to import get_titan_embedding; run from project root.") from exc


@dataclass
class Collection:
    """Bundle the FAISS index, vectors, and metadata for a single dataset."""

    name: str
    index: faiss.Index
    dataframe: pd.DataFrame
    meta: dict
    index_path: Path
    parquet_path: Path
    meta_path: Path | None


def _canonical_name(path: Path) -> str:
    """Derive a human-friendly label from the index filename."""
    stem = path.stem
    if stem.endswith("_hnsw"):
        stem = stem[:-5]
    return stem


def _load_metadata(parquet_path: Path) -> pd.DataFrame:
    """Read chunk metadata and embeddings stored in Parquet."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "embedding" not in df.columns:
        raise ValueError("Parquet file must contain an 'embedding' column.")
    return df


def _load_meta_json(meta_path: Path | None) -> dict:
    """Fetch optional ingestion metadata or return an empty mapping."""
    if not meta_path:
        return {}
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_collection(
    index_path: Path,
    parquet_path: Path,
    meta_path: Path | None = None,
) -> Collection:
    """Load FAISS index plus associated parquet/meta into a `Collection`."""
    try:
        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise SystemExit(f"Failed to load FAISS index {index_path}: {exc}") from exc
    dataframe = _load_metadata(parquet_path)
    meta = _load_meta_json(meta_path)
    return Collection(
        name=_canonical_name(index_path),
        index=index,
        dataframe=dataframe,
        meta=meta,
        index_path=index_path,
        parquet_path=parquet_path,
        meta_path=meta_path,
    )


def _discover_collections(directory: Path) -> Sequence[Collection]:
    """Scan a directory for FAISS/parquet/meta triples and load each."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    collections: list[Collection] = []
    for faiss_path in sorted(directory.glob("*_hnsw.faiss")):
        prefix = faiss_path.name[: -len("_hnsw.faiss")]
        parquet_path = directory / f"{prefix}_embeds.parquet"
        meta_path = directory / f"{prefix}_meta.json"
        if not parquet_path.exists():
            print(f"[WARN] Skipping {faiss_path}: missing {parquet_path}", file=sys.stderr)
            continue
        meta_arg = meta_path if meta_path.exists() else None
        collection = _load_collection(faiss_path, parquet_path, meta_arg)
        collections.append(collection)
    if not collections:
        raise SystemExit(f"No FAISS artifacts discovered in {directory}")
    return collections


def _prepare_query_vector(
    text: str,
    *,
    index_dim: int,
    model_id: str,
    region: str,
    normalize: bool,
) -> np.ndarray:
    """Embed the query text using Titan and shape it for FAISS search."""
    embedding = get_titan_embedding(
        text,
        model_id=model_id,
        dimensions=index_dim,
        normalize=normalize,
        region=region,
    )
    vector = np.asarray(embedding, dtype="float32").reshape(1, -1)
    if normalize:
        faiss.normalize_L2(vector)
    return vector


def rerank_with_bedrock(
    query: str,
    chunks: list[str],
    model_id: str = "amazon.rerank-v1:0",
    region: str = "us-west-2",
    top_n: int = 5,
) -> list[tuple[int, float]]:
    """Rerank chunks using AWS Bedrock Rerank API.

    Args:
        query: The search query
        chunks: List of text chunks to rerank
        model_id: Bedrock reranker model ID. Options:
                  - "amazon.rerank-v1:0" (available in us-west-2, ca-central-1, eu-central-1, ap-northeast-1)
                  - "cohere.rerank-v3-5:0" (also available in us-east-1)
        region: AWS region (must match where model is available)
        top_n: Number of top results to return

    Returns:
        List of (original_index, relevance_score) tuples, sorted by relevance descending
    """
    if not chunks:
        return []

    client = boto3.client("bedrock-agent-runtime", region_name=region)

    # Format chunks as sources for the API
    sources = [
        {
            "inlineDocumentSource": {
                "textDocument": {"text": chunk},
                "type": "TEXT"
            },
            "type": "INLINE"
        }
        for chunk in chunks
    ]

    response = client.rerank(
        queries=[{
            "textQuery": {"text": query},
            "type": "TEXT"
        }],
        rerankingConfiguration={
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {
                    "modelArn": f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
                },
                "numberOfResults": min(top_n, len(chunks))
            },
            "type": "BEDROCK_RERANKING_MODEL"
        },
        sources=sources
    )

    # Extract and return results as (original_index, score) tuples
    results = [
        (item["index"], item["relevanceScore"])
        for item in response["results"]
    ]

    return results


def search_with_rerank(
    query: str,
    collections: list[Collection],
    initial_k: int = 20,
    final_k: int = 5,
    model_id: str = "amazon.titan-embed-text-v2:0",
    rerank_model_id: str = "amazon.rerank-v1:0",
    region: str = "us-west-2",
    normalize: bool = True,
    use_reranking: bool = True,
) -> list[dict]:
    """Two-stage retrieval: FAISS candidate retrieval + Bedrock reranking.

    Args:
        query: Search query
        collections: List of Collection objects containing FAISS indices
        initial_k: Number of candidates to retrieve from FAISS (Stage 1)
        final_k: Number of final results after reranking (Stage 2)
        model_id: Embedding model ID for query encoding
        rerank_model_id: Bedrock reranker model ID
        region: AWS region
        normalize: Whether to normalize embeddings
        use_reranking: If False, skip reranking (for A/B comparison)

    Returns:
        List of result dicts with keys: chunk_idx, text, collection, score, source_url
    """
    # Get embedding dimension from first collection
    expected_dim = collections[0].index.d

    # Encode query
    query_vec = _prepare_query_vector(
        query,
        index_dim=expected_dim,
        model_id=model_id,
        region=region,
        normalize=normalize,
    )

    # Stage 1: FAISS retrieval
    candidates = []
    for collection in collections:
        k = min(initial_k, collection.index.ntotal)
        if k == 0:
            continue
        distances, indices = collection.index.search(query_vec, k)
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(collection.dataframe):
                continue
            row = collection.dataframe.iloc[idx]
            if collection.meta:
                source_url = row.get("source_url", collection.meta.get("source_url", "unknown"))
            else:
                source_url = row.get("source_url", "unknown")
            candidates.append({
                "chunk_idx": int(idx),
                "text": str(row.get("text", "")),
                "collection": collection.name,
                "faiss_score": float(score),
                "source_url": source_url,
                "doc_id": row.get("doc_id", "unknown"),
                "chunk_id": row.get("chunk_id", "unknown"),
            })

    if not candidates:
        return []

    # Sort by FAISS score and take top initial_k across all collections
    candidates.sort(key=lambda x: x["faiss_score"], reverse=True)
    candidates = candidates[:initial_k]

    # Stage 2: Reranking (if enabled)
    if use_reranking and candidates:
        chunk_texts = [c["text"] for c in candidates]
        reranked = rerank_with_bedrock(
            query=query,
            chunks=chunk_texts,
            model_id=rerank_model_id,
            region=region,
            top_n=final_k,
        )

        # Build final results from reranked indices
        final_results = []
        for orig_idx, rerank_score in reranked:
            result = candidates[orig_idx].copy()
            result["rerank_score"] = rerank_score
            result["score"] = rerank_score  # Use rerank score as primary
            final_results.append(result)

        return final_results
    else:
        # No reranking - just return top final_k by FAISS score
        for c in candidates:
            c["score"] = c["faiss_score"]
        return candidates[:final_k]


def _iter_queries(args: argparse.Namespace) -> Iterator[str]:
    """Yield query strings from CLI flags, stdin, or an interactive prompt."""
    if args.query:
        for q in args.query:
            yield q.strip()
        return
    if not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line:
                yield line
        return
    print("Enter queries (empty line to exit):", file=sys.stderr)
    while True:
        try:
            line = input("query> ").strip()
        except EOFError:
            break
        if not line:
            break
        yield line


def _preview_text(text: str, limit: int) -> str:
    """Return a truncated version of `text` for concise CLI output."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def main() -> None:
    """Parse CLI arguments, load FAISS collections, and stream query results."""
    parser = argparse.ArgumentParser(
        description="Query FAISS indices generated by scripts/bedrock.py using Titan embeddings."
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=220,
        help="Number of characters to display from each chunk.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--index",
        help="Path to a single FAISS index (requires --parquet, optional --meta).",
    )
    target_group.add_argument(
        "--dir",
        help="Directory containing *_hnsw.faiss, *_embeds.parquet, and *_meta.json files.",
    )
    parser.add_argument(
        "--parquet",
        help="Parquet file storing chunk metadata/text when using --index.",
    )
    parser.add_argument(
        "--meta",
        help="Optional JSON metadata file emitted during ingestion when using --index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest chunks to return per query.",
    )
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        help="Query string to search for (can be repeated). If omitted, reads from stdin or interactive prompt.",
    )
    parser.add_argument(
        "--model-id",
        default="amazon.titan-embed-text-v2:0",
        help="Bedrock embed model identifier.",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region for the Bedrock runtime client.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization before querying (not recommended).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking (use FAISS scores only, for baseline comparison)",
    )
    parser.add_argument(
        "--initial-k",
        type=int,
        default=20,
        help="Number of candidates to retrieve from FAISS before reranking.",
    )
    parser.add_argument(
        "--rerank-model",
        default="amazon.rerank-v1:0",
        help="Bedrock reranker model ID. Options: amazon.rerank-v1:0, cohere.rerank-v3-5:0",
    )
    args = parser.parse_args()

    collections: list[Collection] = []
    if args.index:
        if not args.parquet:
            parser.error("--parquet is required when using --index")
        index_path = Path(args.index).expanduser()
        parquet_path = Path(args.parquet).expanduser()
        meta_path = Path(args.meta).expanduser() if args.meta else None
        collections = [_load_collection(index_path, parquet_path, meta_path)]
    else:
        if args.parquet or args.meta:
            parser.error("--parquet/--meta can only be used with --index")
        dir_path = Path(args.dir).expanduser()
        collections = list(_discover_collections(dir_path))

    normalize = not args.no_normalize
    expected_dim = None
    for collection in collections:
        if normalize and not getattr(collection.index, "ntotal", 0):
            print(f"Warning: index {collection.name} contains no vectors.", file=sys.stderr)
        if expected_dim is None:
            expected_dim = collection.index.d
        elif expected_dim != collection.index.d:
            raise SystemExit(
                f"Index dimension mismatch: expected {expected_dim}, got {collection.index.d} ({collection.index_path})"
            )
    if expected_dim is None:
        raise SystemExit("No suitable indices were loaded.")

    for query in _iter_queries(args):
        if not query:
            continue
        try:
            # Use two-stage retrieval with reranking
            results = search_with_rerank(
                query=query,
                collections=collections,
                initial_k=args.initial_k,
                final_k=args.top_k,
                model_id=args.model_id,
                rerank_model_id=args.rerank_model,
                region=args.region,
                normalize=normalize,
                use_reranking=not args.no_rerank,
            )
        except Exception as exc:
            print(f"[ERROR] Query failed: {exc}", file=sys.stderr)
            continue

        if not results:
            print(f"\nQuery: {query}\n  No results.", flush=True)
            continue

        print(f"\nQuery: {query}")
        for rank, result in enumerate(results, start=1):
            preview = _preview_text(result["text"], args.preview_chars)
            rerank_score = result.get("rerank_score")
            faiss_score = result.get("faiss_score")

            # Format scores based on whether reranking was used
            if rerank_score is not None:
                score_str = f"rerank={rerank_score:.4f} faiss={faiss_score:.4f}"
            else:
                score_str = f"faiss={faiss_score:.4f}"

            print(
                f"  {rank}. {score_str} doc_id={result['doc_id']} chunk={result['chunk_id']} "
                f"collection={result['collection']} url={result['source_url']}"
            )
            print(f"      {preview}")


if __name__ == "__main__":
    main()
