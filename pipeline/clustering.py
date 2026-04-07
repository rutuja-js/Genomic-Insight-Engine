"""
Unsupervised Clustering
Discovers natural groupings in genomic data without using disease labels.
This simulates a real discovery scenario: "what structure exists in the data?"

Methods:
- KMeans           → fast baseline clustering
- Agglomerative    → hierarchical, captures non-spherical clusters
- UMAP             → 2D embedding for visualization (preserves local structure)
- Silhouette score → evaluates cluster quality
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import umap


class GenomicClustering:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.agglomerative_ = None
        self.umap_embedding_ = None
        self.results_ = {}

    # ------------------------------------------------------------------
    # KMeans
    # ------------------------------------------------------------------
    def run_kmeans(self, X: pd.DataFrame) -> np.ndarray:
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            max_iter=300,
            random_state=self.random_state,
        )
        labels = self.kmeans_.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"[Clustering] KMeans → Silhouette: {sil:.4f}")
        return labels

    # ------------------------------------------------------------------
    # Agglomerative (Ward linkage)
    # ------------------------------------------------------------------
    def run_agglomerative(self, X: pd.DataFrame) -> np.ndarray:
        self.agglomerative_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage="ward",
        )
        labels = self.agglomerative_.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"[Clustering] Agglomerative (Ward) → Silhouette: {sil:.4f}")
        return labels

    # ------------------------------------------------------------------
    # UMAP embedding
    # ------------------------------------------------------------------
    def run_umap(self, X: pd.DataFrame, n_neighbors: int = 30) -> pd.DataFrame:
        print("[Clustering] Running UMAP (this may take ~30s)...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=self.random_state,
        )
        embedding = reducer.fit_transform(X)
        self.umap_embedding_ = pd.DataFrame(
            embedding, columns=["UMAP1", "UMAP2"], index=X.index
        )
        print("[Clustering] UMAP embedding complete")
        return self.umap_embedding_

    # ------------------------------------------------------------------
    # Evaluate clustering vs ground truth labels
    # ------------------------------------------------------------------
    @staticmethod
    def evaluate(cluster_labels: np.ndarray, true_labels: np.ndarray, method: str):
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        print(f"[Clustering] {method} → ARI: {ari:.4f} | NMI: {nmi:.4f}")
        return {"method": method, "ARI": ari, "NMI": nmi}

    # ------------------------------------------------------------------
    # Full clustering pipeline
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y_true: pd.Series) -> "GenomicClustering":
        print("\n=== CLUSTERING ANALYSIS ===")

        km_labels = self.run_kmeans(X)
        agg_labels = self.run_agglomerative(X)
        self.run_umap(X)

        km_eval = self.evaluate(km_labels, y_true.values, "KMeans")
        agg_eval = self.evaluate(agg_labels, y_true.values, "Agglomerative")

        self.results_ = {
            "kmeans_labels": km_labels,
            "agglomerative_labels": agg_labels,
            "umap_embedding": self.umap_embedding_,
            "evaluations": [km_eval, agg_eval],
        }
        return self
