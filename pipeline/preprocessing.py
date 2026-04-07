"""
Preprocessing Pipeline
Handles: QC filtering, log-normalization, batch correction, train/test split.

Real genomic preprocessing involves several domain-specific steps:
1. Low-variance gene filtering  → removes uninformative genes
2. Quantile normalization       → corrects for sequencing depth differences
3. Batch effect correction      → ComBat-style mean-centering per batch
4. Outlier sample removal       → based on Mahalanobis distance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.stats import zscore


class GenomicPreprocessor:
    def __init__(
        self,
        variance_threshold: float = 0.1,
        outlier_zscore_threshold: float = 3.5,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.variance_threshold = variance_threshold
        self.outlier_zscore_threshold = outlier_zscore_threshold
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_genes = None

    # ------------------------------------------------------------------
    # Step 1: Low-variance gene filter
    # ------------------------------------------------------------------
    def filter_low_variance_genes(self, X: pd.DataFrame) -> pd.DataFrame:
        gene_variances = X.var(axis=0)
        self.selected_genes = gene_variances[gene_variances > self.variance_threshold].index
        filtered = X[self.selected_genes]
        print(f"[Preprocess] Variance filter: {X.shape[1]} → {filtered.shape[1]} genes retained")
        return filtered

    # ------------------------------------------------------------------
    # Step 2: Quantile normalization (rank-based)
    # ------------------------------------------------------------------
    @staticmethod
    def quantile_normalize(X: pd.DataFrame) -> pd.DataFrame:
        ranks = X.rank(axis=0, method="average")
        sorted_means = np.sort(X.values, axis=0).mean(axis=1)
        quantile_map = np.argsort(np.argsort(X.values, axis=0), axis=0)
        normalized = sorted_means[quantile_map]
        print(f"[Preprocess] Quantile normalization applied")
        return pd.DataFrame(normalized, columns=X.columns, index=X.index)

    # ------------------------------------------------------------------
    # Step 3: Batch correction (mean-centering per batch)
    # ------------------------------------------------------------------
    @staticmethod
    def correct_batch_effects(X: pd.DataFrame, batch_ids: pd.Series) -> pd.DataFrame:
        corrected = X.copy()
        global_mean = X.mean(axis=0)
        for batch in batch_ids.unique():
            mask = batch_ids == batch
            batch_mean = X[mask].mean(axis=0)
            corrected[mask] = X[mask] - batch_mean + global_mean
        print(f"[Preprocess] Batch correction applied across {batch_ids.nunique()} batches")
        return corrected

    # ------------------------------------------------------------------
    # Step 4: Outlier sample removal (per-sample z-score)
    # ------------------------------------------------------------------
    def remove_outlier_samples(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        sample_means = X.mean(axis=1)
        z_scores = np.abs(zscore(sample_means))
        keep = z_scores < self.outlier_zscore_threshold
        print(f"[Preprocess] Outlier removal: {(~keep).sum()} samples removed → {keep.sum()} retained")
        return X[keep], y[keep]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> dict:
        print("\n=== PREPROCESSING PIPELINE ===")
        
        labels_raw = df["disease_label"]
        batch_ids = df["batch_id"]
        X_raw = df.drop(columns=["disease_label", "batch_id"])

        # Encode labels
        y = pd.Series(
            self.label_encoder.fit_transform(labels_raw),
            index=labels_raw.index,
            name="label",
        )

        # Pipeline steps
        X = self.filter_low_variance_genes(X_raw)
        X = self.quantile_normalize(X)
        X = self.correct_batch_effects(X, batch_ids.loc[X.index])
        X, y = self.remove_outlier_samples(X, y)

        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )

        # Train/test split — stratified to preserve class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        print(f"[Preprocess] Final: {X_scaled.shape[0]} samples × {X_scaled.shape[1]} genes")
        print(f"[Preprocess] Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"[Preprocess] Classes: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "X_full": X_scaled, "y_full": y,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "selected_genes": self.selected_genes,
        }
