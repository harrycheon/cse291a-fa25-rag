"""Ingest documents from URLs, embed with Amazon Titan, and persist FAISS artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import boto3
import faiss
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from scripts_part_2.LLM_chunker import chunk_article_semantically
from struct_output_test2 import chunk_article


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
SUPPORTED_FILE_EXTS = {".pdf", ".txt", ".text", ".md", ".html", ".htm"}


@dataclass
class ParsedDocument:
    url: str
    content_type: str | None
    text: str
    is_pdf: bool


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML stripper that ignores script/style blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _sha256(text: str) -> str:
    """Return a stable SHA256 hex digest for deduplicating text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_like_pdf(url: str, content_type: str | None, data: bytes | None = None) -> bool:
    """Heuristically decide if a response represents a PDF document."""
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base in PDF_MIME_TYPES:
            return True
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return True
    if data and data.startswith(b"%PDF"):
        return True
    return False


def _decode_bytes(data: bytes, charset: str | None) -> str:
    """Decode bytes to text, trying provided charset plus sensible fallbacks."""
    candidates = [charset] if charset else []
    candidates.extend(["utf-8", "latin-1"])
    for enc in candidates:
        if not enc:
            continue
        try:
            return data.decode(enc, errors="ignore")
        except LookupError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise ValueError("Unable to read PDF bytes") from exc
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


def _extract_text_from_html(data: bytes, charset: str | None) -> str:
    """Strip HTML content to plain text while skipping script/style tags."""
    html = _decode_bytes(data, charset)
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    text = parser.get_text()
    return text or html


def parse_document_from_url(url: str, *, timeout: float = 30.0, retries: int = 3) -> ParsedDocument:
    """Download a URL and return extracted text plus content metadata.
    
    Args:
        url: The URL to download
        timeout: Request timeout in seconds
        retries: Number of retry attempts for failed requests
        
    Returns:
        ParsedDocument with extracted text and metadata
    """
    last_exception = None
    
    for attempt in range(retries):
        try:
            # Add delay between retries (except first attempt)
            if attempt > 0:
                delay = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                time.sleep(delay)
                
            request = Request(url, headers=DEFAULT_HEADERS)
            with urlopen(request, timeout=timeout) as response:
                raw_bytes = response.read()
                headers = response.headers
                base_type = None
                charset = None
                if headers:
                    base_type = getattr(headers, "get_content_type", lambda: None)()
                    charset = getattr(headers, "get_content_charset", lambda: None)()
                    
            is_pdf = _looks_like_pdf(url, base_type, raw_bytes)
            if is_pdf:
                text = _extract_text_from_pdf(raw_bytes)
            else:
                text = _extract_text_from_html(raw_bytes, charset)
            if not text.strip() and not is_pdf and _looks_like_pdf(url, None, raw_bytes):
                is_pdf = True
                text = _extract_text_from_pdf(raw_bytes)
            return ParsedDocument(
                url=url,
                content_type=base_type,
                text=text,
                is_pdf=is_pdf,
            )
        except Exception as e:
            last_exception = e
            if attempt < retries - 1:
                print(f"[RETRY {attempt + 1}/{retries}] {url}: {e}", file=sys.stderr)
                continue
            # If all retries failed, raise the last exception
            raise last_exception


def parse_document_from_file(file_path: Path) -> ParsedDocument:
    """Read a local file (PDF, HTML, or plain text) and extract its text."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    data = file_path.read_bytes()
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_text_from_pdf(data)
        content_type = "application/pdf"
        is_pdf = True
    elif suffix in {".html", ".htm"}:
        text = _extract_text_from_html(data, None)
        content_type = "text/html"
        is_pdf = False
    elif suffix in {".txt", ".text", ".md"}:
        text = _decode_bytes(data, None)
        content_type = "text/plain"
        is_pdf = False
    else:
        raise ValueError(f"Unsupported file type for ingestion: {file_path}")
    return ParsedDocument(
        url=str(file_path),
        content_type=content_type,
        text=text,
        is_pdf=is_pdf,
    )


def chunk_text(text: str, *, chunk_size: int = 256, overlap: int = 50) -> list[str]:

    # json_chunks = chunk_article(text)
    # text_chunks = [ e['content'] for e in json_chunks ]
    text_chunks = chunk_article_semantically(text)
    return text_chunks


def get_titan_embedding(
    text: str,
    model_id: str = "amazon.titan-embed-text-v2:0",
    dimensions: int = 1024,
    normalize: bool = True,
    region: str = "us-west-2",
) -> list[float]:
    """Return embedding vector for given text using Titan Text Embeddings V2."""
    client = boto3.client("bedrock-runtime", region_name=region)
    body = {
        "inputText": text,
        "dimensions": dimensions,
        "normalize": normalize,
    }
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )
    resp_body = json.loads(response["body"].read())
    return resp_body["embedding"]


def embed_chunks(
    chunks: Sequence[str],
    *,
    model_id: str = "amazon.titan-embed-text-v2:0",
    dimensions: int = 1024,
    normalize: bool = True,
    region: str = "us-west-2",
) -> np.ndarray:
    """Embed each chunk with Titan and return an L2-normalized matrix."""
    vectors = []
    for chunk in chunks:
        embedding = get_titan_embedding(
            chunk,
            model_id=model_id,
            dimensions=dimensions,
            normalize=normalize,
            region=region,
        )
        vectors.append(embedding)
    if not vectors:
        return np.empty((0, dimensions), dtype="float32")
    array = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(array)
    return array


def write_parquet_batch(
    rows: Iterable[dict],
    out_path: str,
) -> None:
    """Write embedding rows to Parquet.

    Args:
        rows: Dicts with keys matching the schema below.
        out_path: Destination Parquet file path.
    """
    df = pd.DataFrame(list(rows))
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")


def build_faiss_hnsw(
    embeddings: np.ndarray,
    m: int = 64,
    ef_construction: int = 200,
) -> faiss.IndexHNSWFlat:
    """Build an HNSW index over L2-normalized vectors.

    Args:
        embeddings: Shape (n, d) float32 L2-normalized.
        m: HNSW graph degree.
        ef_construction: Construction search depth.

    Returns:
        A FAISS HNSW index with all vectors added.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    if embeddings.size:
        index.add(embeddings)
    return index


def _model_revision(model_id: str) -> str:
    """Return the model revision suffix (text after the colon) if present."""
    if ":" in model_id:
        return model_id.split(":")[-1]
    return "unknown"


def ingest_url(
    url: str,
    *,
    out_dir: str | Path = "data",
    doc_id: str | None = None,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    model_id: str = "amazon.titan-embed-text-v2:0",
    dimensions: int = 1024,
    normalize: bool = True,
    region: str = "us-west-2",
) -> dict[str, Path]:
    """Ingest a single URL end-to-end and return saved artifact paths."""
    parsed = parse_document_from_url(url)
    return _ingest_parsed_document(
        parsed,
        source_ref=url,
        out_dir=out_dir,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_id=model_id,
        dimensions=dimensions,
        normalize=normalize,
        region=region,
    )


def ingest_file(
    file_path: str | Path,
    *,
    out_dir: str | Path = "data",
    doc_id: str | None = None,
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    model_id: str = "amazon.titan-embed-text-v2:0",
    dimensions: int = 1024,
    normalize: bool = True,
    region: str = "us-west-2",
) -> dict[str, Path]:
    """Ingest a local file (PDF/HTML/text) and return saved artifact paths."""
    path = Path(file_path).expanduser().resolve()
    if path.suffix.lower() not in SUPPORTED_FILE_EXTS:
        raise ValueError(f"Unsupported file type: {path}")
    parsed = parse_document_from_file(path)
    return _ingest_parsed_document(
        parsed,
        source_ref=str(path),
        out_dir=out_dir,
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_id=model_id,
        dimensions=dimensions,
        normalize=normalize,
        region=region,
    )


def _ingest_parsed_document(
    parsed: ParsedDocument,
    *,
    source_ref: str,
    out_dir: str | Path,
    doc_id: str | None,
    chunk_size: int,
    chunk_overlap: int,
    model_id: str,
    dimensions: int,
    normalize: bool,
    region: str,
) -> dict[str, Path]:
    """Common persistence routine shared by URL and file ingestion."""
    if not parsed.text.strip():
        raise ValueError(f"No textual content could be extracted from {source_ref}")
    chunks = chunk_text(parsed.text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        raise ValueError(f"Parsed document from {source_ref} produced no chunks")
    doc_identifier = doc_id or _sha256(source_ref)[:12]
    embeddings = embed_chunks(
        chunks,
        model_id=model_id,
        dimensions=dimensions,
        normalize=normalize,
        region=region,
    )
    index = build_faiss_hnsw(embeddings)
    created_at = datetime.utcnow().isoformat()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / f"{doc_identifier}_embeds.parquet"
    faiss_path = out_path / f"{doc_identifier}_hnsw.faiss"
    meta_path = out_path / f"{doc_identifier}_meta.json"

    rows = []
    for chunk_id, (chunk_text_val, vector) in enumerate(zip(chunks, embeddings)):
        rows.append(
            {
                "doc_id": doc_identifier,
                "chunk_id": chunk_id,
                "source_url": source_ref,
                "content_type": parsed.content_type,
                "is_pdf": parsed.is_pdf,
                "created_at": created_at,
                "text": chunk_text_val,
                "text_sha256": _sha256(chunk_text_val),
                "model_id": model_id,
                "model_revision": _model_revision(model_id),
                "normalized": True,
                "model_dim": embeddings.shape[1],
                "embedding": vector.astype("float32").tolist(),
            }
        )
    write_parquet_batch(rows, str(parquet_path))
    faiss.write_index(index, str(faiss_path))
    meta = {
        "doc_id": doc_identifier,
        "source_url": source_ref,
        "content_type": parsed.content_type,
        "is_pdf": parsed.is_pdf,
        "model_id": model_id,
        "model_revision": _model_revision(model_id),
        "normalize": normalize,
        "dim": embeddings.shape[1],
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "created_at": created_at,
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return {
        "parquet": parquet_path,
        "faiss": faiss_path,
        "meta": meta_path,
    }


def main() -> None:
    """CLI entry point for ingesting URLs or local document files."""
    parser = argparse.ArgumentParser(
        description="Download or read documents, create Titan embeddings, and persist FAISS artifacts."
    )
    parser.add_argument(
        "url_file",
        nargs="?",
        help="Path to a text file containing one HTTP(s) URL per line.",
    )
    parser.add_argument(
        "--file-dir",
        help="Directory containing local documents (.pdf, .html, .txt) to ingest.",
    )
    parser.add_argument("--out-dir", default="results", help="Directory where artifacts are written.")
    parser.add_argument(
        "--doc-id-prefix",
        help="Optional prefix used to generate doc IDs for the ingested URLs (e.g. 'pltr').",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of tokens (words) per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Word overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--region",
        default="us-west-2",
        help="AWS region for the Bedrock runtime client.",
    )
    args = parser.parse_args()
    if not args.url_file and not args.file_dir:
        parser.error("Either url_file or --file-dir must be provided.")

    tasks: list[tuple[str, str | Path]] = []
    if args.file_dir:
        dir_path = Path(args.file_dir).expanduser()
        if not dir_path.exists():
            parser.error(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            parser.error(f"--file-dir must point to a directory: {dir_path}")
        files: list[Path] = []
        for candidate in sorted(dir_path.iterdir()):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in SUPPORTED_FILE_EXTS:
                print(f"[WARN] Skipping unsupported file: {candidate}", file=sys.stderr)
                continue
            files.append(candidate)
        if not files:
            parser.error(f"No supported documents found in {dir_path}")
        tasks = [("file", path) for path in files]
    else:
        url_path = Path(args.url_file).expanduser()
        if not url_path.exists():
            parser.error(f"URL file not found: {url_path}")
        with url_path.open("r", encoding="utf-8") as handle:
            raw_lines = [line.strip() for line in handle]
        urls = [
            line for line in raw_lines if line and not line.lstrip().startswith("#")
        ]
        if not urls:
            parser.error(f"No URLs found in {url_path}")
        tasks = [("url", url) for url in urls]

    success = 0
    for idx, (kind, payload) in enumerate(tasks, start=1):
        # Add a small delay between requests to be polite to servers
        if idx > 1 and kind == "url":
            time.sleep(1)
            
        doc_id = None
        if args.doc_id_prefix:
            doc_id = f"{args.doc_id_prefix}-{idx:04d}"
        try:
            if kind == "url":
                artifacts = ingest_url(
                    str(payload),
                    out_dir=args.out_dir,
                    doc_id=doc_id,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    region=args.region,
                )
            else:
                artifacts = ingest_file(
                    payload,
                    out_dir=args.out_dir,
                    doc_id=doc_id,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    region=args.region,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {payload}\n  {exc}", file=sys.stderr)
            continue
        success += 1
        print(f"[OK] {payload}")
        for name, path in artifacts.items():
            print(f"  {name}: {path}")
    if success == 0:
        raise SystemExit("No documents were ingested successfully.")


if __name__ == "__main__":
    main()
