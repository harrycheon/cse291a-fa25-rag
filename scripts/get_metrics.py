import os
from pathlib import Path
import time
import pandas as pd
from typing import List
import json
from dotenv import load_dotenv
from groq import Groq

try:
    from query_faiss import _discover_collections, _prepare_query_vector
except ImportError:
    raise SystemExit("Failed to import query_faiss; run from project root.")

# Load environment variables from .env
load_dotenv()

def retrieve_chunks_from_faiss(
    query: str,
    dir_path: str | Path,
    top_k: int = 5,
    model_id: str = "amazon.titan-embed-text-v2:0",
    region: str = "us-west-2",
    normalize: bool = True,
) -> list[str]:
    """Retrieve top-k chunks from FAISS for a query."""
    dir_path = Path(dir_path)
    collections = _discover_collections(dir_path)

    # Prepare query vector
    expected_dim = collections[0].index.d
    query_vec = _prepare_query_vector(
        query,
        index_dim=expected_dim,
        model_id=model_id,
        region=region,
        normalize=normalize,
    )

    aggregated: list[tuple[float, str]] = []
    for collection in collections:
        distances, indices = collection.index.search(query_vec, top_k)
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(collection.dataframe):
                continue
            text = collection.dataframe.iloc[idx].get("text", "")
            aggregated.append((float(score), text))

    aggregated.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in aggregated[:top_k]]


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def rate_chunks_with_llm(query: str, chunks: List[str]) -> List[dict]:
    """Rate each chunk independently for relevance (0-1) using Groq LLM."""
    if not chunks:
        return []

    messages = [
        {"role": "system", "content": "You are a document relevance judge."},
        {
            "role": "user",
            "content": (
                f"Rate the following documents for relevance to the query.\n"
                f"Query: {query}\n"
                f"Documents: {chunks}\n"
                f"Return a score between 0 (irrelevant) and 1 (highly relevant) for each document.\n"
                f"Respond strictly in JSON format as a list of objects with fields 'chunk' and 'score'."
            )
        }
    ]

    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=messages,
            temperature=0,
            max_tokens=2048,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "relevance_ratings",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chunk": {"type": "string"},
                                "score": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["chunk", "score"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print(f"[ERROR] LLM rating failed: {e}")
        return [{"chunk": chunk, "score": None} for chunk in chunks]


def evaluate_queries(
    queries: List[str],
    dir_path: str | Path,
    top_k: int = 5
) -> pd.DataFrame:
    """Evaluate queries and return a DataFrame with relevance scores."""
    records = []
    for query in queries:
        start_time = time.time()
        chunks = retrieve_chunks_from_faiss(query, dir_path, top_k=top_k)
        retrieval_time = time.time() - start_time

        ratings = rate_chunks_with_llm(query, chunks)
        for item in ratings:
            records.append({
                "query": query,
                "chunk_text": item.get("chunk"),
                "llm_score": item.get("score"),
                "retrieval_time_sec": retrieval_time
            })

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    queries = [
        "What was Qualcomm’s total revenue in the last fiscal year?",
        "Break down Qualcomm’s revenue by segment.",
        "Titan embeddings in AWS Bedrock"
    ]
    dir_path = "./data/embeddings/qualcomm"

    df = evaluate_queries(queries, dir_path, top_k=5)
    # Keep only first 30 characters of each chunk
    df['chunk_text'] = df['chunk_text'].astype(str).str.slice(0, 30)
    df['llm_score'] = df['llm_score'].apply(lambda x: float(x) if x is not None else None)
    csv_path = "faiss_evaluation.csv"
    df.to_csv(csv_path, index=False)

    # Print instead of saving to Parquet (for now)
    print(df)

