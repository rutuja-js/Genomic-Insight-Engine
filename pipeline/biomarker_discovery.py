"""
Biomarker Discovery
Identifies genes most predictive of disease status using multiple complementary methods:

1. Variance-based ranking     → finds highly variable genes across cohort
2. ANOVA F-test               → genes with significant between-group differences
3. Random Forest importance   → non-linear feature contribution
4. Consensus scoring          → combines all three for robust biomarker candidates
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from typing import Optional


class BiomarkerDiscovery:
    def __init__(self, top_k: int = 50, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.biomarker_scores_ = None
        self.top_biomarkers_ = None
        self.pca_ = None
        self.pca_components_ = None

    # ------------------------------------------------------------------
    # Method 1: Variance ranking
    # ------------------------------------------------------------------
    def _variance_scores(self, X: pd.DataFrame) -> pd.Series:
        scores = X.var(axis=0)
        return (scores - scores.min()) / (scores.max() - scores.min())  # normalize 0-1

    # ------------------------------------------------------------------
    # Method 2: ANOVA F-statistic (between-group variance)
    # ------------------------------------------------------------------
    def _anova_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        f_stats = []
        groups = [X[y == cls].values for cls in y.unique()]
        for gene_idx in range(X.shape[1]):
            gene_groups = [g[:, gene_idx] for g in groups]
            f_val, _ = f_oneway(*gene_groups)
            f_stats.append(f_val if not np.isnan(f_val) else 0.0)
        scores = pd.Series(f_stats, index=X.columns)
        return (scores - scores.min()) / (scores.max() - scores.min())

    # ------------------------------------------------------------------
    # Method 3: Random Forest feature importance
    # ------------------------------------------------------------------
    def _rf_importance_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            n_jobs=-1,
            random_state=self.random_state,
        )
        rf.fit(X, y)
        scores = pd.Series(rf.feature_importances_, index=X.columns)
        return (scores - scores.min()) / (scores.max() - scores.min())

    # ------------------------------------------------------------------
    # Consensus scoring + PCA on top biomarkers
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BiomarkerDiscovery":
        print("\n=== BIOMARKER DISCOVERY ===")
        print("[Biomarker] Computing variance scores...")
        var_scores = self._variance_scores(X)

        print("[Biomarker] Computing ANOVA F-scores...")
        anova_scores = self._anova_scores(X, y)

        print("[Biomarker] Computing Random Forest importances (200 trees)...")
        rf_scores = self._rf_importance_scores(X, y)

        # Weighted consensus (equal weight — can be tuned)
        self.biomarker_scores_ = (var_scores + anova_scores + rf_scores) / 3.0
        self.biomarker_scores_.name = "consensus_score"
        self.biomarker_scores_.sort_values(ascending=False, inplace=True)

        self.top_biomarkers_ = self.biomarker_scores_.head(self.top_k).index.tolist()

        print(f"[Biomarker] Top {self.top_k} biomarkers identified")
        print(f"[Biomarker] Top 10: {self.top_biomarkers_[:10]}")

        # PCA on full gene set for visualization
        print("[Biomarker] Fitting PCA (50 components)...")
        self.pca_ = PCA(n_components=50, random_state=self.random_state)
        pca_result = self.pca_.fit_transform(X)
        self.pca_components_ = pd.DataFrame(
            pca_result[:, :2],
            columns=["PC1", "PC2"],
            index=X.index,
        )
        explained = self.pca_.explained_variance_ratio_[:2] * 100
        print(f"[Biomarker] PC1 explains {explained[0]:.1f}% | PC2 explains {explained[1]:.1f}% variance")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return dataset reduced to top biomarker genes."""
        return X[self.top_biomarkers_]

    def get_summary(self) -> pd.DataFrame:
        return self.biomarker_scores_.head(self.top_k).reset_index().rename(
            columns={"index": "gene", "consensus_score": "score"}
        )
