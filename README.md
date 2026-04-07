# 🧬 Genomic Insight Engine

A modular, end-to-end bioinformatics pipeline for genomic biomarker discovery, unsupervised clustering, and disease classification — producing publication-quality visualizations at each stage.

---

## Overview

The Genomic Insight Engine takes raw (or synthetic) genomic expression data and runs it through a full analytical pipeline: quality control, biomarker selection, clustering, and multi-model classification, with 7 plots generated as final outputs.

```
Raw Genomic Data
      │
      ▼
 Preprocessing          QC → Normalization → Batch Correction → Train/Test Split
      │
      ▼
Biomarker Discovery     Variance Filter + ANOVA + Random Forest (consensus top-K)
      │
      ▼
  Clustering            KMeans + Agglomerative + UMAP
      │
      ▼
 Classification         Random Forest + GBM + SVM (5-fold CV)
      │
      ▼
 Visualization          7 publication-quality plots → outputs/
```

---

## Project Structure

```
genomic-insight-engine/
├── main.py                    # Pipeline orchestrator
├── data/
│   ├── generate_data.py       # Synthetic genomic dataset generator
│   └── genomic_dataset.csv    # Generated dataset (created at runtime)
├── pipeline/
│   ├── preprocessing.py       # QC, normalization, batch correction, splitting
│   ├── biomarker_discovery.py # Variance + ANOVA + RF consensus biomarker selection
│   ├── clustering.py          # KMeans, Agglomerative, UMAP
│   └── classification.py      # RF, GBM, SVM with cross-validation
├── visualization/
│   └── plots.py               # All plot generation logic
└── outputs/                   # Generated figures (created at runtime)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

That's it. The pipeline will generate a synthetic dataset, run all stages, print a summary, and save plots to `outputs/`.

---

## Pipeline Stages

### 1. Data Generation
Produces a synthetic genomic dataset with realistic structure (multiple disease classes, batch effects, noisy features) and saves it to `data/genomic_dataset.csv`.

### 2. Preprocessing (`GenomicPreprocessor`)
- Low-variance feature filtering
- Z-score outlier removal
- Normalization and batch correction
- Standard scaling
- Stratified train/test split (default 80/20)

### 3. Biomarker Discovery (`BiomarkerDiscovery`)
Consensus selection across three methods retains the top-K features (default: 50) that appear important across all three:
- **Variance filtering** — removes near-constant features
- **ANOVA F-test** — selects features with significant class separation
- **Random Forest importance** — captures non-linear relevance

### 4. Clustering (`GenomicClustering`)
Unsupervised structure discovery on the biomarker-reduced feature set:
- **KMeans** and **Agglomerative** clustering
- **UMAP** for 2D embedding and visualization

### 5. Classification (`DiseaseClassifier`)
Three models evaluated with 5-fold cross-validation, then tested on the held-out set:
- Random Forest
- Gradient Boosting Machine (GBM)
- Support Vector Machine (SVM)

Metrics reported: Accuracy, F1 (macro), Cohen's Kappa, ROC-AUC (per class).

### 6. Visualization
Seven plots saved to `outputs/`:
- Biomarker importance ranking
- PCA scatter (colored by class)
- UMAP embedding
- Clustering results
- Cross-validation performance comparison
- Confusion matrix (best model)
- ROC curves (one-vs-rest, per class)

---

## Example Output

```
============================================================
  PIPELINE COMPLETE
============================================================
  Runtime         : 42.3s
  Best model      : GradientBoosting
  Test Accuracy   : 0.9347
  Test F1 (macro) : 0.9301
  Cohen's Kappa   : 0.9129
  Top biomarkers  : GENE_042, GENE_117, GENE_008, GENE_231, GENE_059, ...
  Outputs saved   : outputs/
============================================================
```

---

## Configuration

Key parameters can be adjusted when instantiating pipeline components:

| Component | Parameter | Default | Description |
|---|---|---|---|
| `GenomicPreprocessor` | `variance_threshold` | `0.1` | Min variance to retain a feature |
| `GenomicPreprocessor` | `outlier_zscore_threshold` | `3.5` | Z-score cutoff for outlier removal |
| `GenomicPreprocessor` | `test_size` | `0.2` | Fraction of data held out for testing |
| `BiomarkerDiscovery` | `top_k` | `50` | Number of consensus biomarkers to select |
| `GenomicClustering` | `n_clusters` | auto (= num classes) | Number of clusters for KMeans / Agglomerative |

---

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- umap-learn
- matplotlib
- seaborn

Install all at once:

```bash
pip install scikit-learn pandas numpy umap-learn matplotlib seaborn
```

---

## License

MIT
