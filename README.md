# ctl-coding-clustering

CTL FNN coding score and clustering pipeline, implementing the Garrett 2026 convention.

## Installation

```bash
pip install git+https://github.com/AllenNeuralDynamics/ctl-coding-clustering.git
```

Optional UMAP support:

```bash
pip install "ctl-coding-clustering[umap] @ git+https://github.com/AllenNeuralDynamics/ctl-coding-clustering.git"
```

## Usage

```python
from ctl_coding_clustering import build_matched_cell_index, compute_all_coding_scores
from ctl_coding_clustering import run_clustering_pipeline, plot_coding_heatmap

# 1. Build matched-cell index (requires ROICaT and GLM assets mounted under CTL_DATA_DIR)
cell_index = build_matched_cell_index(
    subject_ids=[782149, 788406, 790322, 800792, 800995, 804363],
    session_table_path="/path/to/session_table.csv",
)

# 2. Compute 12-feature adjVE coding scores
scores = compute_all_coding_scores(cell_index, session_table_path="/path/to/session_table.csv")
X = scores.values  # (n_cells, 12)

# 3. Cluster
labels, C = run_clustering_pipeline(X, k=12)

# 4. Visualize
fig = plot_coding_heatmap(X, labels, out_path="heatmap.png")
```

Submodules can also be imported directly:

```python
from ctl_coding_clustering import cell_matching, coding_score, clustering, visualize
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CTL_DATA_DIR` | `/data` | Root directory where GLM and ROICaT assets are mounted (Code Ocean convention) |
| `CTL_SESSION_TABLE` | — | Path to the ground-truth session table CSV (required if not passed explicitly) |

## Modules

| Module | Purpose |
|---|---|
| `cell_matching` | ROICaT-based cross-session cell identity; builds matched-cell index |
| `coding_score` | adjVE computation and cross-session normalization → 12-feature score table |
| `clustering` | Gap statistic, spectral co-clustering (150×), hierarchical labels, prediction strength, LOMO, shuffle control |
| `visualize` | Heatmap, cluster profiles, gap/eigengap, co-clustering matrix, UMAP, prediction strength plots |

## Reference

Garrett et al. 2026 — Methods: "Across session normalization of coding scores"
