"""
Genomic Insight Engine — Main Pipeline Orchestrator

Run: python main.py

Full pipeline:
  1. Generate synthetic genomic dataset
  2. Preprocess (QC → normalize → batch correct → scale → split)
  3. Biomarker discovery (variance + ANOVA + RF consensus)
  4. Unsupervised clustering (KMeans + Agglomerative + UMAP)
  5. Supervised classification (RF + GBM + SVM with 5-fold CV)
  6. Visualization (7 publication-quality plots)
"""

import sys
import time
from pathlib import Path

# Add project root to path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_data import generate_dataset
from pipeline.preprocessing import GenomicPreprocessor
from pipeline.biomarker_discovery import BiomarkerDiscovery
from pipeline.clustering import GenomicClustering
from pipeline.classification import DiseaseClassifier
from visualization.plots import generate_all_plots


def main():
    t0 = time.time()
    print("=" * 60)
    print("  GENOMIC INSIGHT ENGINE — BIOMARKER DISCOVERY PIPELINE")
    print("=" * 60)

    # ─── 1. DATA GENERATION ───────────────────────────────────────
    df = generate_dataset(save_path="data/genomic_dataset.csv")

    # ─── 2. PREPROCESSING ─────────────────────────────────────────
    preprocessor = GenomicPreprocessor(
        variance_threshold=0.1,
        outlier_zscore_threshold=3.5,
        test_size=0.2,
    )
    proc = preprocessor.fit_transform(df)
    X_train, X_test = proc["X_train"], proc["X_test"]
    y_train, y_test = proc["y_train"], proc["y_test"]
    X_full, y_full  = proc["X_full"], proc["y_full"]
    le = proc["label_encoder"]
    class_names = list(le.classes_)

    # ─── 3. BIOMARKER DISCOVERY ───────────────────────────────────
    bd = BiomarkerDiscovery(top_k=50)
    bd.fit(X_train, y_train)
    biomarker_summary = bd.get_summary()

    # Reduce to top biomarkers for downstream tasks
    
    X_train_bm = bd.transform(X_train)
    X_test_bm  = bd.transform(X_test)
    X_full_bm  = bd.transform(X_full)

    # Re-fit PCA on FULL biomarker set so all 998 sample indices are covered
    from sklearn.decomposition import PCA as _PCA
    import pandas as _pd
    _pca_full = _PCA(n_components=2, random_state=42)
    _pca_coords = _pca_full.fit_transform(X_full_bm)
    bd.pca_components_ = _pd.DataFrame(_pca_coords, columns=["PC1","PC2"], index=X_full_bm.index)
    bd.pca_.explained_variance_ratio_ = _pca_full.explained_variance_ratio_

    # ─── 4. CLUSTERING ────────────────────────────────────────────
    clusterer = GenomicClustering(n_clusters=len(class_names))
    clusterer.fit(X_full_bm, y_full)

    # ─── 5. CLASSIFICATION ────────────────────────────────────────
    classifier = DiseaseClassifier()
    cv_results  = classifier.cross_validate_all(X_train_bm, y_train)
    test_results = classifier.evaluate_on_test(
        X_train_bm, y_train,
        X_test_bm,  y_test,
        class_names,
    )
    classifier.print_best_report(class_names)

    # ─── 6. VISUALIZATION ─────────────────────────────────────────
    generate_all_plots(
        biomarker_summary    = biomarker_summary,
        pca_df               = bd.pca_components_,
        pca_variance         = bd.pca_.explained_variance_ratio_,
        umap_results         = clusterer.results_,
        clustering_results   = clusterer.results_,
        X_full               = X_full_bm,
        y_full               = y_full,
        label_encoder        = le,
        top_biomarkers       = bd.top_biomarkers_,
        cv_results           = cv_results,
        test_results         = test_results,
        best_model_name      = classifier.best_model_name_,
        roc_data             = classifier.roc_data_,
        class_names          = class_names,
    )

    # ─── SUMMARY ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    best = classifier.best_model_name_
    best_test = test_results[best]

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Runtime         : {elapsed:.1f}s")
    print(f"  Best model      : {best}")
    print(f"  Test Accuracy   : {best_test['accuracy']:.4f}")
    print(f"  Test F1 (macro) : {best_test['f1_macro']:.4f}")
    print(f"  Cohen's Kappa   : {best_test['kappa']:.4f}")
    print(f"  Top biomarkers  : {', '.join(bd.top_biomarkers_[:5])}, ...")
    print(f"  Outputs saved   : outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
