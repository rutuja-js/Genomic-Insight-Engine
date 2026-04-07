"""
Synthetic Genomic Data Generator
Simulates realistic gene expression data with disease labels.

Design rationale:
- 1000 samples, 500 genes (manageable but realistic scale)
- 4 classes: Healthy, Type-II Diabetes, Cancer (Early), Alzheimer's
- Each disease has a set of "driver genes" with distinct expression signatures
- Gaussian noise + batch effects simulate real sequencing variability
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 1000
N_GENES = 500
DISEASE_CLASSES = {
    0: "Healthy",
    1: "Type2_Diabetes",
    2: "Early_Cancer",
    3: "Alzheimers",
}

# Each disease affects a subset of genes with distinct fold-change
DISEASE_SIGNATURES = {
    1: {"upregulated": list(range(0, 30)),   "downregulated": list(range(30, 50))},
    2: {"upregulated": list(range(50, 90)),   "downregulated": list(range(90, 110))},
    3: {"upregulated": list(range(110, 140)), "downregulated": list(range(140, 155))},
}


def generate_expression_matrix(n_samples: int, n_genes: int, rng: np.random.Generator) -> np.ndarray:
    """Base log-normal expression matrix (mimics RNA-seq counts after log-norm)."""
    return rng.normal(loc=6.0, scale=1.5, size=(n_samples, n_genes))


def apply_disease_signature(
    matrix: np.ndarray,
    labels: np.ndarray,
    signatures: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shift expression values for disease-driver genes."""
    modified = matrix.copy()
    for disease_id, sig in signatures.items():
        mask = labels == disease_id
        up_genes = sig["upregulated"]
        dn_genes = sig["downregulated"]
        # Upregulated: +2 to +4 fold (log-scale shift)
        modified[np.ix_(mask, up_genes)] += rng.uniform(2.0, 4.0, size=(mask.sum(), len(up_genes)))
        # Downregulated: -1.5 to -3 fold
        modified[np.ix_(mask, dn_genes)] -= rng.uniform(1.5, 3.0, size=(mask.sum(), len(dn_genes)))
    return modified


def add_batch_effects(matrix: np.ndarray, n_batches: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate sequencing batch effects — a common real-world challenge."""
    batch_ids = rng.integers(0, n_batches, size=matrix.shape[0])
    batch_offsets = rng.normal(0, 0.5, size=(n_batches, matrix.shape[1]))
    return matrix + batch_offsets[batch_ids], batch_ids


def generate_dataset(save_path: str = "data/genomic_dataset.csv") -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    # Balanced class distribution
    samples_per_class = N_SAMPLES // len(DISEASE_CLASSES)
    labels = np.repeat(list(DISEASE_CLASSES.keys()), samples_per_class)

    expression = generate_expression_matrix(N_SAMPLES, N_GENES, rng)
    expression = apply_disease_signature(expression, labels, DISEASE_SIGNATURES, rng)
    expression, batch_ids = add_batch_effects(expression, n_batches=5, rng=rng)

    # Clip to realistic range [0, 15] and add tiny noise floor
    expression = np.clip(expression, 0, 15)

    gene_names = [f"GENE_{i:04d}" for i in range(N_GENES)]
    sample_ids = [f"SAMPLE_{i:05d}" for i in range(N_SAMPLES)]

    df = pd.DataFrame(expression, columns=gene_names, index=sample_ids)
    df.insert(0, "disease_label", [DISEASE_CLASSES[l] for l in labels])
    df.insert(1, "batch_id", batch_ids)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    print(f"[DataGen] Dataset saved → {save_path}")
    print(f"[DataGen] Shape: {df.shape} | Classes: {df['disease_label'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    generate_dataset()
