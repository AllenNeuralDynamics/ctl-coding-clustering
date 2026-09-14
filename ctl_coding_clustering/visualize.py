"""
Visualization for the FNN coding clustering analysis.

Functions:
  plot_gap_statistic      — gap vs k with std ribbon
  plot_eigengap           — sorted Laplacian eigenvalues + gaps
  plot_cocluster_heatmap  — co-clustering probability matrix
  plot_coding_heatmap     — 12-feature heatmap sorted by cluster
  plot_cluster_profiles   — mean coding score grid per cluster
  plot_prediction_strength — PS per cluster with threshold line
  plot_ps_vs_k            — mean PS across k values
  plot_umap               — 2D UMAP colored by cluster or feature
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

FEATURE_LABELS = ["Images\nF", "Images\nN", "Images\nN+",
                  "Omissions\nF", "Omissions\nN", "Omissions\nN+",
                  "Task\nF", "Task\nN", "Task\nN+",
                  "Behavior\nF", "Behavior\nN", "Behavior\nN+"]

CLUSTER_CMAP = "tab20"


def _save(fig, path: str | Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── Gap statistic ──────────────────────────────────────────────────────────────

def plot_gap_statistic(
    gap_df: pd.DataFrame,
    k_selected: int | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = gap_df.index.values
    gap = gap_df["gap"].values
    std = gap_df["gap_std"].values
    ax.plot(ks, gap, "o-", color="steelblue", lw=2)
    ax.fill_between(ks, gap - std, gap + std, alpha=0.25, color="steelblue")
    if k_selected is not None:
        ax.axvline(k_selected, color="tomato", ls="--", lw=1.5, label=f"k={k_selected}")
        ax.legend()
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Gap statistic")
    ax.set_title("Gap statistic for k selection")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path)
    return fig


# ── Eigengap ───────────────────────────────────────────────────────────────────

def plot_eigengap(
    eigvals: np.ndarray,
    k_eig: int | None = None,
    k_max: int = 40,
    out_path: str | Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ks = np.arange(len(eigvals[:k_max + 2]))
    axes[0].plot(ks, eigvals[:k_max + 2], "o-", ms=3)
    axes[0].set_xlabel("Eigenvalue index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Laplacian eigenvalues")

    gaps = np.diff(eigvals[:k_max + 2])
    axes[1].bar(np.arange(1, len(gaps) + 1), gaps)
    if k_eig is not None:
        axes[1].axvline(k_eig, color="tomato", ls="--", lw=1.5, label=f"k={k_eig}")
        axes[1].legend()
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Eigengap (Δλ_k)")
    axes[1].set_title("Eigengap")

    fig.tight_layout()
    if out_path:
        _save(fig, out_path)
    return fig


# ── Co-clustering heatmap ─────────────────────────────────────────────────────

def plot_cocluster_heatmap(
    C: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path | None = None,
) -> plt.Figure:
    order = np.argsort(labels)
    C_sorted = C[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C_sorted, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="none")
    plt.colorbar(im, ax=ax, label="Co-clustering probability")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Cell")
    ax.set_title("Co-clustering matrix (sorted by cluster)")

    # Draw cluster boundaries
    clusters = np.unique(labels)
    sizes = [(labels[order] == c).sum() for c in clusters]
    boundaries = np.cumsum(sizes)[:-1]
    for b in boundaries:
        ax.axhline(b - 0.5, color="red", lw=0.5)
        ax.axvline(b - 0.5, color="red", lw=0.5)

    fig.tight_layout()
    if out_path:
        _save(fig, out_path)
    return fig


# ── 12-feature coding score heatmap ───────────────────────────────────────────

def plot_coding_heatmap(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    order = np.argsort(labels, kind="stable")
    X_sorted = X[order]
    labels_sorted = labels[order]

    n_clusters = len(np.unique(labels))
    fig_h = max(6, n_clusters * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    im = ax.imshow(X_sorted, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   interpolation="none")
    plt.colorbar(im, ax=ax, label="Coding score", fraction=0.02)

    # x-axis: feature labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(FEATURE_LABELS, fontsize=7, rotation=45, ha="right")

    # y-axis: cluster boundaries + labels
    clusters = np.unique(labels)
    sizes = [(labels_sorted == c).sum() for c in clusters]
    boundaries = np.cumsum(sizes)

    ytick_pos = []
    ytick_labels = []
    prev = 0
    for c, sz, b in zip(clusters, sizes, boundaries):
        ytick_pos.append(prev + sz / 2)
        ytick_labels.append(f"C{c}\n(n={sz})")
        if b < len(labels_sorted):
            ax.axhline(b - 0.5, color="white", lw=0.8)
        prev = b

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=7)
    ax.set_ylabel("Cluster")
    ax.set_title(f"FNN coding scores ({len(X)} cells, k={n_clusters})")

    fig.tight_layout()
    if out_path:
        _save(fig, out_path)
    return fig


# ── Per-cluster mean profiles ─────────────────────────────────────────────────

def plot_cluster_profiles(
    X: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path | None = None,
) -> plt.Figure:
    clusters = np.unique(labels)
    k = len(clusters)
    ncols = min(4, k)
    nrows = (k + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3),
                             squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    for i, c in enumerate(clusters):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        mu = X[labels == c].mean(axis=0)
        err = X[labels == c].std(axis=0)
        bar = ax.bar(range(12), mu, yerr=err, color="steelblue", alpha=0.8, capsize=2)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Im.F","Im.N","Im.N+","Om.F","Om.N","Om.N+",
                             "Ta.F","Ta.N","Ta.N+","Be.F","Be.N","Be.N+"],
                            fontsize=6, rotation=45, ha="right")
        ax.set_title(f"C{c} (n={(labels==c).sum()})", fontsize=9)
        ax.set_ylabel("Coding score" if i % ncols == 0 else "")

    fig.suptitle("Mean coding score per cluster (±1 SD)", fontsize=12)
    fig.tight_layout()
    if out_path:
        _save(fig, out_path)
    return fig


# ── Prediction strength ────────────────────────────────────────────────────────

def plot_prediction_strength(
    ps_df: pd.DataFrame,
    threshold: float = 0.7,
    out_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(6, len(ps_df) * 0.6), 4))
    clusters = ps_df.index.values
    means = ps_df["ps_mean"].values
    sems = ps_df["ps_sem"].values
    ax.bar(clusters, means, yerr=sems, capsize=4, color="steelblue", alpha=0.8)
    ax.axhline(threshold, color="tomato", ls="--", lw=1.5, label=f"Threshold={threshold}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Prediction strength")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(clusters)
    ax.set_xticklabels([f"C{c}" for c in clusters], fontsize=8)
    ax.legend()
    ax.set_title("Prediction strength per cluster (50/50 split × 10, KNN k=5)")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path)
    return fig


# ── UMAP ──────────────────────────────────────────────────────────────────────

def plot_umap(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray | None = None,
    out_path: str | Path | None = None,
    seed: int = 42,
) -> plt.Figure:
    try:
        import umap
    except ImportError:
        print("umap-learn not installed; skipping UMAP plot")
        return None

    reducer = umap.UMAP(n_components=2, random_state=seed)
    emb = reducer.fit_transform(X)

    ncols = 2 if subject_ids is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    clusters = np.unique(labels)
    cmap = plt.get_cmap(CLUSTER_CMAP, len(clusters))
    for i, c in enumerate(clusters):
        mask = labels == c
        axes[0].scatter(emb[mask, 0], emb[mask, 1], s=4, alpha=0.6,
                        color=cmap(i), label=f"C{c}")
    axes[0].set_title("UMAP — colored by cluster")
    axes[0].legend(markerscale=3, fontsize=6, ncol=2)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    if subject_ids is not None:
        mice = np.unique(subject_ids)
        mcmap = plt.get_cmap("Set2", len(mice))
        for i, m in enumerate(mice):
            mask = subject_ids == m
            axes[1].scatter(emb[mask, 0], emb[mask, 1], s=4, alpha=0.6,
                            color=mcmap(i), label=str(m))
        axes[1].set_title("UMAP — colored by mouse")
        axes[1].legend(markerscale=3, fontsize=8)
        axes[1].set_xlabel("UMAP 1")
        axes[1].set_ylabel("UMAP 2")

    fig.suptitle(f"FNN coding score UMAP (n={len(X)} cells)")
    fig.tight_layout()
    if out_path:
        _save(fig, out_path)
    return fig
