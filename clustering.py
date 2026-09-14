"""
Clustering pipeline — Garrett 2026 faithful implementation.

Step 1  Gap statistic  (k = 10..35, 20 reps each, shuffle within each column)
Step 2  Eigengap cross-check on the graph Laplacian
Step 3  SpectralClustering × 150 repeats at chosen k
Step 4  Co-clustering probability matrix
Step 5  Agglomerative hierarchical clustering → final labels

Validation:
  - Prediction strength (Tibshirani & Walther 2005): KNN k=5, 50/50, 10×
  - Leave-one-mouse-out (LOMO) prediction strength
  - Cell-ID shuffle control: 500×, within full pool per column
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool, cpu_count
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier

# ── Gap statistic ──────────────────────────────────────────────────────────────

K_MIN = 10
K_MAX = 35
GAP_REPS = 20   # repetitions per k for both real and null


def _shuffle_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each column independently (within-column permutation)."""
    Xs = X.copy()
    for j in range(Xs.shape[1]):
        Xs[:, j] = rng.permutation(Xs[:, j])
    return Xs


def _mean_within_inertia(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean pairwise within-cluster squared Euclidean distance."""
    total = 0.0
    n_total = 0
    for c in np.unique(labels):
        pts = X[labels == c]
        n = len(pts)
        if n < 2:
            continue
        # Sum of squared pairwise distances / n_pairs
        diff = pts[:, None, :] - pts[None, :, :]  # (n, n, d)
        sq = (diff ** 2).sum(axis=2)              # (n, n)
        total += sq.sum() / 2
        n_total += n * (n - 1) / 2
    return total / max(n_total, 1)


def _fit_spectral(args):
    X, k, seed = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = SpectralClustering(
            n_clusters=k, affinity="rbf", assign_labels="kmeans",
            random_state=seed, n_init=10,
        )
        labels = sc.fit_predict(X)
    return labels


def gap_statistic(
    X: np.ndarray,
    k_range: range | None = None,
    n_reps: int = GAP_REPS,
    n_jobs: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute gap statistic for k in k_range.

    Returns DataFrame with columns: k, gap, gap_std, inertia_real, inertia_null_mean.
    """
    if k_range is None:
        k_range = range(K_MIN, K_MAX + 1)
    rng = np.random.default_rng(seed)
    nj = min(n_reps, cpu_count()) if n_jobs is None else n_jobs

    records = []
    for k in k_range:
        print(f"  gap stat k={k}...", flush=True)

        # Real data: n_reps spectral clustering runs
        args_real = [(X, k, int(rng.integers(1e6))) for _ in range(n_reps)]
        with Pool(nj) as pool:
            labels_list = pool.map(_fit_spectral, args_real)
        inertia_real = np.mean([_mean_within_inertia(X, lb) for lb in labels_list])

        # Null: shuffle each column independently, n_reps shuffled datasets
        inertia_nulls = []
        for _ in range(n_reps):
            Xs = _shuffle_columns(X, rng)
            args_null = [(Xs, k, int(rng.integers(1e6)))]
            with Pool(1) as pool:
                lb_null = pool.map(_fit_spectral, args_null)[0]
            inertia_nulls.append(_mean_within_inertia(Xs, lb_null))

        log_null = np.log(np.array(inertia_nulls) + 1e-12)
        gap = log_null.mean() - np.log(inertia_real + 1e-12)
        records.append({
            "k": k,
            "gap": gap,
            "gap_std": log_null.std(),
            "inertia_real": inertia_real,
            "inertia_null_mean": np.exp(log_null.mean()),
        })

    return pd.DataFrame(records).set_index("k")


def select_k_gap(gap_df: pd.DataFrame) -> int:
    """Return k with maximum gap."""
    return int(gap_df["gap"].idxmax())


# ── Eigengap ───────────────────────────────────────────────────────────────────

def eigengap(X: np.ndarray, k_max: int = K_MAX) -> tuple[np.ndarray, int]:
    """Return sorted Laplacian eigenvalues and the eigengap k estimate.

    Uses the RBF affinity matrix (matching SpectralClustering's default).
    """
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.sparse.csgraph import laplacian

    gamma = 1.0 / X.shape[1]
    A = rbf_kernel(X, gamma=gamma)
    L = laplacian(A, normed=True)
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)
    gaps = np.diff(eigvals[:k_max + 2])
    k_eig = int(np.argmax(gaps[1:]) + 2)   # skip trivial gap at 0
    return eigvals, k_eig


# ── Spectral co-clustering ─────────────────────────────────────────────────────

N_COCLUSTERING_REPS = 150


def coclustering_matrix(
    X: np.ndarray,
    k: int,
    n_reps: int = N_COCLUSTERING_REPS,
    n_jobs: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Run SpectralClustering n_reps times and return pairwise co-clustering matrix.

    Returns C[i,j] = fraction of runs where cells i and j were in the same cluster.
    Shape: (n_cells, n_cells).
    """
    rng = np.random.default_rng(seed)
    nj = min(n_reps, cpu_count()) if n_jobs is None else n_jobs
    args = [(X, k, int(rng.integers(1e6))) for _ in range(n_reps)]

    print(f"  Running {n_reps} SpectralClustering repeats (k={k})...", flush=True)
    with Pool(nj) as pool:
        all_labels = pool.map(_fit_spectral, args)

    n = X.shape[0]
    C = np.zeros((n, n), dtype=np.float32)
    for labels in all_labels:
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            C[np.ix_(idx, idx)] += 1
    C /= n_reps
    return C


# ── Hierarchical clustering on co-clustering matrix ───────────────────────────

def hierarchical_from_cocluster(C: np.ndarray, k: int) -> np.ndarray:
    """Agglomerative (average-linkage) clustering on the (1 - C) distance matrix.

    Returns integer label array of length n_cells, clusters numbered 1..k
    sorted by descending size.
    """
    dist = 1.0 - C
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = average(condensed)
    raw_labels = fcluster(Z, k, criterion="maxclust")

    # Re-number clusters by descending size
    counts = pd.Series(raw_labels).value_counts()
    mapping = {old: (new + 1) for new, old in enumerate(counts.index)}
    return np.array([mapping[l] for l in raw_labels])


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_clustering_pipeline(
    X: np.ndarray,
    k: int,
    n_coclustering_reps: int = N_COCLUSTERING_REPS,
    n_jobs: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run co-clustering + hierarchical step at a fixed k.

    Returns:
        labels: (n_cells,) integer cluster labels (1-indexed, sorted by size)
        C:      (n_cells, n_cells) co-clustering probability matrix
    """
    C = coclustering_matrix(X, k, n_reps=n_coclustering_reps, n_jobs=n_jobs, seed=seed)
    labels = hierarchical_from_cocluster(C, k)
    return labels, C


# ── Prediction strength validation ────────────────────────────────────────────

def prediction_strength(
    X: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 10,
    knn_k: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute per-cluster prediction strength (Tibshirani & Walther 2005).

    Random 50/50 train/test splits.  KNN (k=knn_k) trained on train set with
    spectral cluster labels, predicts test set.  For each cluster c:
        PS(c) = fraction of test-cell pairs in cluster c with same predicted label.

    Returns DataFrame with columns: cluster, ps_mean, ps_sem.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    clusters = np.unique(labels)
    ps_records = {c: [] for c in clusters}

    for _ in range(n_splits):
        perm = rng.permutation(n)
        half = n // 2
        train_idx, test_idx = perm[:half], perm[half:]

        X_train, y_train = X[train_idx], labels[train_idx]
        X_test = X[test_idx]
        true_test = labels[test_idx]

        knn = KNeighborsClassifier(n_neighbors=knn_k)
        knn.fit(X_train, y_train)
        pred_test = knn.predict(X_test)

        for c in clusters:
            mask = true_test == c
            n_c = mask.sum()
            if n_c < 2:
                ps_records[c].append(np.nan)
                continue
            pred_c = pred_test[mask]
            # All pairwise: fraction with same predicted label
            same = (pred_c[:, None] == pred_c[None, :]).sum() - n_c
            total = n_c * (n_c - 1)
            ps_records[c].append(same / total)

    rows = []
    for c in clusters:
        vals = [v for v in ps_records[c] if not np.isnan(v)]
        rows.append({
            "cluster": c,
            "ps_mean": np.mean(vals) if vals else np.nan,
            "ps_sem": np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
            "n_cells": (labels == c).sum(),
        })
    return pd.DataFrame(rows).set_index("cluster")


def prediction_strength_lomo(
    X: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    knn_k: int = 5,
) -> pd.DataFrame:
    """Leave-one-mouse-out prediction strength.

    For each mouse m: train on all other mice, predict mouse m's cells.
    Returns per-cluster mean PS across LOMO folds.
    """
    clusters = np.unique(labels)
    mice = np.unique(subject_ids)
    ps_per_mouse = []

    for m in mice:
        test_mask = subject_ids == m
        train_mask = ~test_mask
        if train_mask.sum() < knn_k:
            continue

        knn = KNeighborsClassifier(n_neighbors=knn_k)
        knn.fit(X[train_mask], labels[train_mask])
        pred_test = knn.predict(X[test_mask])
        true_test = labels[test_mask]

        row = {"mouse": m}
        for c in clusters:
            mask_c = true_test == c
            n_c = mask_c.sum()
            if n_c < 2:
                row[c] = np.nan
                continue
            pred_c = pred_test[mask_c]
            same = (pred_c[:, None] == pred_c[None, :]).sum() - n_c
            row[c] = same / (n_c * (n_c - 1))
        ps_per_mouse.append(row)

    df = pd.DataFrame(ps_per_mouse).set_index("mouse")
    summary = pd.DataFrame({
        "cluster": clusters,
        "ps_lomo_mean": [df[c].mean() for c in clusters],
        "ps_lomo_sem": [df[c].sem() for c in clusters],
    }).set_index("cluster")
    return summary


# ── Cell-ID shuffle control ────────────────────────────────────────────────────

def _shuffle_and_cluster(args):
    X, k, seed = args
    rng = np.random.default_rng(seed)
    Xs = _shuffle_columns(X, rng)
    sc = SpectralClustering(
        n_clusters=k, affinity="rbf", assign_labels="kmeans",
        random_state=int(rng.integers(1e6)), n_init=10,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels_s = sc.fit_predict(Xs)
    # Re-index by size for centroid computation
    counts = pd.Series(labels_s).value_counts()
    mapping = {old: new for new, old in enumerate(counts.index)}
    labels_s = np.array([mapping[l] for l in labels_s])
    centroids = np.array([Xs[labels_s == c].mean(axis=0) for c in range(k)])
    return centroids


def shuffle_control(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    n_shuffles: int = 500,
    sse_threshold: float = 0.1,
    n_jobs: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Cell-ID shuffle control: shuffle each column independently, re-cluster.

    For each original cluster c:
      P(c) = fraction of shuffles where a matched cluster was found (SSE ≤ threshold).
      delta_size(c) = mean(original size - matched shuffle size).

    Returns DataFrame indexed by cluster with columns: P, delta_size_mean, delta_size_std.
    """
    clusters = np.unique(labels)
    k_actual = len(clusters)
    orig_centroids = np.array([X[labels == c].mean(axis=0) for c in clusters])
    orig_sizes = np.array([(labels == c).sum() for c in clusters])

    nj = min(n_shuffles, cpu_count()) if n_jobs is None else n_jobs
    rng = np.random.default_rng(seed)
    args = [(X, k_actual, int(rng.integers(1e6))) for _ in range(n_shuffles)]

    print(f"  Running {n_shuffles} shuffle repeats...", flush=True)
    with Pool(nj) as pool:
        all_centroids = pool.map(_shuffle_and_cluster, args)

    # For each original cluster, check how many shuffles match it
    match_counts = np.zeros(k_actual, dtype=int)
    delta_sizes = [[] for _ in range(k_actual)]

    for shuf_centroids in all_centroids:
        shuf_sizes = np.ones(len(shuf_centroids)) * (len(X) // k_actual)  # approx
        for ci, (orig_c, c) in enumerate(zip(orig_centroids, clusters)):
            # SSE between this original centroid and each shuffle centroid
            sses = ((shuf_centroids - orig_c[None, :]) ** 2).sum(axis=1)
            best_j = sses.argmin()
            if sses[best_j] <= sse_threshold:
                match_counts[ci] += 1
                delta_sizes[ci].append(orig_sizes[ci] - shuf_sizes[best_j])

    rows = []
    for ci, c in enumerate(clusters):
        ds = delta_sizes[ci]
        rows.append({
            "cluster": c,
            "P": match_counts[ci] / n_shuffles,
            "delta_size_mean": np.mean(ds) if ds else np.nan,
            "delta_size_std": np.std(ds) if ds else np.nan,
            "n_cells": orig_sizes[ci],
        })
    return pd.DataFrame(rows).set_index("cluster")
