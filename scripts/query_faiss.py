"""CLI utility for querying stored FAISS indices using Titan embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import faiss
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from scripts.generate_embeddings import get_titan_embedding  
except ImportError as exc:  
    raise SystemExit("Failed to import bedrock.get_titan_embedding; run from project root.") from exc


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
            query_vec = _prepare_query_vector(
                query,
                index_dim=expected_dim,
                model_id=args.model_id,
                region=args.region,
                normalize=normalize,
            )
        except Exception as exc:
            print(f"[ERROR] Embedding query failed: {exc}", file=sys.stderr)
            continue
        aggregated: list[tuple[float, int, Collection]] = []
        for collection in collections:
            distances, indices = collection.index.search(query_vec, args.top_k)
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(collection.dataframe):
                    continue
                aggregated.append((float(score), int(idx), collection))
        if not aggregated:
            print(f"\nQuery: {query}\n  No results.", flush=True)
            continue
        aggregated.sort(key=lambda item: item[0], reverse=True)
        limited = aggregated[: args.top_k]
        print(f"\nQuery: {query}")
        for rank, (score, idx, collection) in enumerate(limited, start=1):
            row = collection.dataframe.iloc[idx]
            preview = _preview_text(str(row.get("text", "")), args.preview_chars)
            doc_id = row.get("doc_id", "n/a")
            chunk_id = row.get("chunk_id", "n/a")
            if collection.meta:
                source_url = row.get("source_url", collection.meta.get("source_url"))
            else:
                source_url = row.get("source_url")
            print(
                f"  {rank}. score={score:.4f} doc_id={doc_id} chunk={chunk_id} collection={collection.name} url={source_url}"
            )
            print(f"      {preview}")


if __name__ == "__main__":
    main()
