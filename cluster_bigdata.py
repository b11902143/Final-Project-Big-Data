#!/usr/bin/env python3
# cluster_bigdata.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_and_preprocess(csv_path: Path):
    df = pd.read_csv(csv_path)
    ids = df["id"].copy()
    X = df.drop(columns=["id"]).values.astype(float)
    X_std = StandardScaler().fit_transform(X)
    return ids, X_std


def kmeans_cluster(X_std: np.ndarray, k: int, random_state: int = 42):
    km = KMeans(
        n_clusters=k,
        n_init=20,
        init="k-means++",
        max_iter=300,
        tol=1e-4,
        random_state=random_state,
    )
    labels = km.fit_predict(X_std)
    return km, labels


def save_submission(ids: pd.Series, labels: np.ndarray, out_path: Path):
    df_out = pd.DataFrame({"id": ids.astype(int), "label": labels.astype(int)})
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] Saved submission CSV to {out_path.resolve()}")


def plot_pca(X_std: np.ndarray, labels: np.ndarray, k: int, fname: Path = None):
    xy = PCA(n_components=2, random_state=0).fit_transform(X_std)
    plt.figure(figsize=(7.5, 6))
    sc = plt.scatter(xy[:, 0], xy[:, 1], c=labels, s=6, cmap="tab20")
    plt.title(f"PCA projection of clusters (K={k})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.colorbar(sc, label="cluster")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300)
        print(f"[INFO] Plot saved to {fname.resolve()}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Big-Data clustering (K-Means)")
    ap.add_argument("-i", "--input", type=Path, required=True,
                    help="Path to public_data.csv")
    ap.add_argument("-o", "--output", type=Path, default="public_submission.csv",
                    help="Output CSV with id,label")
    ap.add_argument("--k", type=int, default=15, help="Number of clusters")
    ap.add_argument("--plot", action="store_true", help="Plot PCA scatter")
    args = ap.parse_args()

    ids, X_std = load_and_preprocess(args.input)
    print(f"[INFO] Loaded {len(ids):,} samples")

    _, labels = kmeans_cluster(X_std, args.k)
    save_submission(ids, labels, args.output)

    if args.plot:
        plot_pca(X_std, labels, args.k, fname=Path("plot.png"))


if __name__ == "__main__":
    main()
