import argparse
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import umap

def load_embeddings(parquet_path: str):
    df = pd.read_parquet(parquet_path)
    df["embedding"] = df["embedding"].apply(np.array)
    X = np.vstack(df["embedding"].values)
    return df, X

def reduce_dim(X, method="umap", n_components=2):
    if method == "pca":
        reducer = PCA(n_components=n_components)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    return reducer.fit_transform(X)

def visualize(parquet_path: str, method="umap", color_by="is_pdf"):
    df, X = load_embeddings(parquet_path)
    coords = reduce_dim(X, method)
    df["x"], df["y"] = coords[:,0], coords[:,1]

    fig = px.scatter(
        df,
        x="x", y="y",
        color=color_by,
        hover_data=["chunk_id", "text", "source_url"],
        title=f"Embedding Visualization ({method.upper()})"
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Titan embeddings from parquet file")
    parser.add_argument("parquet", help="Path to the parquet file, e.g. data/xyz_embeds.parquet")
    parser.add_argument("--method", choices=["pca", "umap"], default="umap")
    parser.add_argument("--color-by", default="is_pdf")
    args = parser.parse_args()
    visualize(args.parquet, args.method, args.color_by)
