"""
Visualization Module - 7 publication-quality plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PALETTE = {
    "Healthy":         "#2ecc71",
    "Type2_Diabetes":  "#e74c3c",
    "Early_Cancer":    "#9b59b6",
    "Alzheimers":      "#f39c12",
}
STYLE = "dark_background"
OUTPUT_DIR = Path("outputs")


def _save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Viz] Saved → {path}")


def plot_biomarker_scores(biomarker_summary, top_n=20):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        top = biomarker_summary.head(top_n)
        ax.barh(top["gene"][::-1].values, top["score"][::-1].values,
                color=plt.cm.plasma(np.linspace(0.3, 0.9, top_n)))
        ax.set_xlabel("Consensus Biomarker Score", fontsize=12)
        ax.set_title(f"Top {top_n} Disease Biomarkers\n(Variance + ANOVA + RF Importance)", fontsize=13)
        ax.grid(axis="x", alpha=0.3)
        _save(fig, "1_biomarker_scores.png")


def plot_pca(pca_df, y, label_encoder, explained_variance):
    # Use positional numpy arrays to avoid index mismatches
    label_names = label_encoder.inverse_transform(y.values)
    pc1 = pca_df["PC1"].values
    pc2 = pca_df["PC2"].values
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 7))
        for cls in np.unique(label_names):
            mask = label_names == cls
            ax.scatter(pc1[mask], pc2[mask],
                       label=cls, color=PALETTE.get(cls, "#aaa"), alpha=0.6, s=15, edgecolors="none")
        ax.set_xlabel(f"PC1 ({explained_variance[0]*100:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({explained_variance[1]*100:.1f}%)", fontsize=11)
        ax.set_title("PCA: Gene Expression Space", fontsize=13)
        ax.legend(framealpha=0.3)
        _save(fig, "2_pca_scatter.png")


def plot_umap(umap_df, y_true, cluster_labels, label_encoder):
    label_names = label_encoder.inverse_transform(y_true.values)
    u1 = umap_df["UMAP1"].values
    u2 = umap_df["UMAP2"].values
    with plt.style.context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for cls in np.unique(label_names):
            mask = label_names == cls
            ax1.scatter(u1[mask], u2[mask], label=cls,
                        color=PALETTE.get(cls, "#aaa"), alpha=0.6, s=10, edgecolors="none")
        ax1.set_title("UMAP — True Disease Labels", fontsize=12)
        ax1.legend(framealpha=0.3, markerscale=2)

        cmap = plt.cm.tab10(np.linspace(0, 1, len(np.unique(cluster_labels))))
        for c in np.unique(cluster_labels):
            mask = cluster_labels == c
            ax2.scatter(u1[mask], u2[mask], label=f"Cluster {c}",
                        color=cmap[c], alpha=0.6, s=10, edgecolors="none")
        ax2.set_title("UMAP — KMeans Clusters (Unsupervised)", fontsize=12)
        ax2.legend(framealpha=0.3, markerscale=2)
        for ax in [ax1, ax2]:
            ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        fig.suptitle("UMAP Embedding of Genomic Space", fontsize=14, y=1.01)
        _save(fig, "3_umap.png")


def plot_heatmap(X, y, label_encoder, top_biomarkers, top_n=30):
    label_names = label_encoder.inverse_transform(y.values)
    sort_order = np.argsort(y.values)
    genes = top_biomarkers[:top_n]
    data = X[genes].iloc[sort_order].T
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.heatmap(data, cmap="RdBu_r", center=0,
                    xticklabels=False, yticklabels=True, ax=ax,
                    cbar_kws={"shrink": 0.5, "label": "Normalized Expression"},
                    linewidths=0)
        ax.set_title(f"Top {top_n} Biomarkers — Expression Heatmap", fontsize=13)
        ax.set_xlabel("Samples"); ax.set_ylabel("Gene")
        _save(fig, "4_expression_heatmap.png")


def plot_model_comparison(cv_results):
    models = list(cv_results.keys())
    metrics = ["accuracy_mean", "f1_mean", "auc_mean"]
    labels  = ["Accuracy", "F1 (Macro)", "AUC (OVR)"]
    x = np.arange(len(models)); width = 0.25
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, (m, lbl) in enumerate(zip(metrics, labels)):
            vals = [cv_results[mod][m] for mod in models]
            errs = [cv_results[mod][m.replace("mean","std")] for mod in models]
            ax.bar(x + i*width, vals, width, label=lbl, yerr=errs, capsize=4, alpha=0.85)
        ax.set_xticks(x + width); ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0.5, 1.05); ax.set_ylabel("Score", fontsize=12)
        ax.set_title("5-Fold CV Model Comparison", fontsize=13)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        _save(fig, "5_model_comparison.png")


def plot_confusion_matrix(cm, class_names, model_name):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, vmin=0, vmax=1)
        ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name} (Normalized)", fontsize=13)
        _save(fig, "6_confusion_matrix.png")


def plot_roc_curves(roc_data):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        for cls, data in roc_data.items():
            ax.plot(data["fpr"], data["tpr"],
                    label=f"{cls} (AUC={data['auc']:.3f})",
                    color=PALETTE.get(cls, "#aaa"), linewidth=2)
        ax.plot([0, 1], [0, 1], "w--", alpha=0.4, linewidth=1)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — One-vs-Rest (Best Model)", fontsize=13)
        ax.legend(loc="lower right"); ax.grid(alpha=0.2)
        _save(fig, "7_roc_curves.png")


def generate_all_plots(biomarker_summary, pca_df, pca_variance, umap_results, clustering_results,
                       X_full, y_full, label_encoder, top_biomarkers, cv_results, test_results,
                       best_model_name, roc_data, class_names):
    print("\n=== GENERATING VISUALIZATIONS ===")
    plot_biomarker_scores(biomarker_summary)
    plot_pca(pca_df, y_full, label_encoder, pca_variance)
    plot_umap(umap_results["umap_embedding"], y_full, umap_results["kmeans_labels"], label_encoder)
    plot_heatmap(X_full, y_full, label_encoder, top_biomarkers)
    plot_model_comparison(cv_results)
    plot_confusion_matrix(test_results[best_model_name]["confusion_matrix"], class_names, best_model_name)
    plot_roc_curves(roc_data)
    print("[Viz] All 7 plots generated in outputs/")
