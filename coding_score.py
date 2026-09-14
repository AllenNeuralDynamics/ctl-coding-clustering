"""
Coding score computation — Garrett 2026 convention (corrected).

Uses adjusted variance explained (adj_VE) restricted to each feature's support
frames — the timepoints where the feature's design-matrix columns are non-zero.

Cross-session normalization:
    cs_norm(feature i, session k) =
        (adj_ve_full(i,k)  -  adj_ve_model(i,k))
        / max_{k'} adj_ve_full(i,k')

where:
    adj_ve_full(i,k)  = VE of the FULL model at feature i's support frames in session k
    adj_ve_model(i,k) = VE of the DROPOUT model at the same support frames
    max_{k'}          = max across F, N, N+ sessions for that cell

Cell filtering (per session): exclude cells where < 1% of support frames are active.
Score filtering: zero out where max_k(adj_ve_full) < 0.005 OR per-session
adj_ve_model < 0.005.  All scores clipped to [0, 1].

References:
    Garrett 2026 Methods, "Across session normalization of coding scores"
    AIND: /AIND-ophys-mFISH-GLM/code/coding_score.py
          /AIND-ophys-mFISH-GLM/code/glm_fit_tools.py

Environment variables
---------------------
CTL_SESSION_TABLE
    Path to the ground-truth session table CSV.
    Required when session_table_path is not passed explicitly to
    compute_all_coding_scores().
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cell_matching import find_glm_asset_dir

FULL_MODEL = "Full"
DROPOUT_MODELS = {
    "Images":    "all-images",
    "Omissions": "omissions",
    "Task":      "task",
    "Behavior":  "behavioral",
}
FEATURES = list(DROPOUT_MODELS.keys())  # Images, Omissions, Task, Behavior
VE_THRESHOLD = 0.005        # 0.5 %
PROP_ACTIVE_MIN = 0.01      # exclude cells active in < 1% of trimmed frames


# ── Core: support-frame VE ─────────────────────────────────────────────────────

def _adj_ve(y_vals: np.ndarray, y_hat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Variance explained at support frames, referenced to global mean.

    Matches AIND glm_fit_tools.compute_adjusted_variance_explained exactly.

    y_vals : (T, C)  neural activity (events trace)
    y_hat  : (T, C)  model predictions
    mask   : (T,)  bool — True at support frames
    Returns: (C,)  VE ∈ (-∞, 1]; negative = model worse than mean at support
    """
    resid = y_vals - y_hat
    mu_y = y_vals.mean(axis=0)    # (C,) global mean of y  (over ALL timestamps)
    mu_r = resid.mean(axis=0)     # (C,) global mean of residual (over ALL timestamps)

    var_total = ((y_vals[mask] - mu_y) ** 2).mean(axis=0)   # (C,)
    var_resid = ((resid[mask]  - mu_r) ** 2).mean(axis=0)   # (C,)

    with np.errstate(invalid="ignore", divide="ignore"):
        ve = (var_total - var_resid) / np.where(var_total > 0, var_total, np.nan)
    return np.nan_to_num(ve, nan=0.0)


# ── Session-level adj_VE ───────────────────────────────────────────────────────

def compute_session_adj_ve(
    subject_id: str | int,
    session_date: str,
    roi_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Compute per-cell adj_ve_full and adj_ve_model for each feature in one session.

    Returns DataFrame indexed by cell_roi_id with columns:
        {feat}_adj_ve_full, {feat}_adj_ve_model  for each feat in FEATURES.

    If roi_ids is given, the result is reindexed to that list (zeros for any
    missing cell, i.e. cells excluded by the activity filter or absent from GLM).
    """
    glm_dir = find_glm_asset_dir(subject_id, session_date)
    if glm_dir is None:
        raise FileNotFoundError(f"No GLM asset dir for {subject_id} {session_date}")

    # ── Load GLM result (weights, VE, use_indices) ────────────────────────────
    npy_files = sorted(glm_dir.glob("glm_results_v*_events.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No glm_results npy in {glm_dir}")
    glm = np.load(npy_files[-1], allow_pickle=True).item()

    use_idx = np.asarray(glm["use_indices"])   # integer indices into full frame array

    # ── Load design matrix and activity trace ─────────────────────────────────
    X_xr  = xr.open_dataarray(glm_dir / "design_matrix.nc", mmap=False)
    y_xr  = xr.open_dataarray(glm_dir / "events_activity_trace_matrix.nc", mmap=False)
    with open(glm_dir / "run_params.json") as fh:
        run_params = json.load(fh)

    # Trim to active frames
    X_trim = X_xr.values[use_idx]        # (T, n_weights)
    y_trim = y_xr.values[use_idx]        # (T, n_all_cells)
    all_weights   = X_xr.coords["weights"].values       # (n_weights,)
    all_cells_all = y_xr.coords["cell_roi_id"].values   # (n_all_cells,)

    # ── Filter low-activity cells (< 1% active frames in trimmed window) ───────
    prop_active = (y_trim > 0).mean(axis=0)     # (n_all_cells,)
    active_mask = prop_active >= PROP_ACTIVE_MIN
    y_active    = y_trim[:, active_mask]         # (T, n_active_cells)
    active_cells = all_cells_all[active_mask]

    # ── Mean W across CV folds, aligned to design matrix weight order ──────────
    mean_W_xr = glm["W_cv"].mean(dim="test_fold_ind")   # (model, weights, cell_roi_id)
    # Reorder weights to match design matrix (critical for numpy matmul)
    mean_W_xr = mean_W_xr.sel(weights=all_weights.tolist())

    glm_cells = mean_W_xr.coords["cell_roi_id"].values

    # Intersection: must be active AND in GLM result
    common_cells = active_cells[np.isin(active_cells, glm_cells)]
    cell_in_active = np.isin(active_cells, common_cells)
    y_vals = y_active[:, cell_in_active]    # (T, C)

    mean_W = mean_W_xr.sel(cell_roi_id=common_cells)   # (model, weights, C)

    # ── Full-model predictions (shared across features) ────────────────────────
    W_full_vals = mean_W.sel(model=FULL_MODEL).values   # (n_weights, C)
    y_hat_full  = X_trim @ W_full_vals                  # (T, C)

    # ── Per-feature adj_VE ─────────────────────────────────────────────────────
    result_cols: dict[str, np.ndarray] = {}

    for feat, model in DROPOUT_MODELS.items():
        rp        = run_params["dropouts"][model]
        is_single = rp["is_single"]

        # Feature kernel names (excluding intercept)
        feat_kernels = rp["kernels"] if is_single else rp["dropped_kernels"]
        feat_kernels = [k for k in feat_kernels if k != "intercept"]

        # Feature weight columns: any weight name that contains a feature kernel name
        feat_w_flag = np.array(
            [any(k in w for k in feat_kernels) for w in all_weights], dtype=bool
        )
        # Support mask: frames where any feature column is non-zero
        support = (X_trim[:, feat_w_flag] != 0).any(axis=1)   # (T,)

        if support.sum() == len(X_trim):
            # All frames are support — adj_VE equals the stored mean-model VE
            adj_ve_f = mean_W.sel(model=FULL_MODEL).values.mean(axis=0) * 0  # placeholder
            ve_mm    = glm["var_explained_mean_model"]
            adj_ve_f = ve_mm.sel(model=FULL_MODEL, cell_roi_id=common_cells).values.copy()
            adj_ve_m = ve_mm.sel(model=model,      cell_roi_id=common_cells).values.copy()
        else:
            # Build reduced model column set
            if is_single:
                # Single model: intercept + feature weights
                intercept_flag = np.array(["intercept" in w for w in all_weights], dtype=bool)
                run_flag = intercept_flag | feat_w_flag
            else:
                # Dropout model: all weights except the feature's
                run_flag = ~feat_w_flag

            run_cols      = np.where(run_flag)[0]
            run_w_names   = all_weights[run_cols].tolist()
            W_model_vals  = mean_W.sel(model=model, weights=run_w_names).values  # (n_run, C)
            y_hat_model   = X_trim[:, run_cols] @ W_model_vals                   # (T, C)

            adj_ve_m = _adj_ve(y_vals, y_hat_model, support)   # (C,)
            adj_ve_f = _adj_ve(y_vals, y_hat_full,  support)   # (C,)

        result_cols[f"{feat}_adj_ve_full"]  = adj_ve_f
        result_cols[f"{feat}_adj_ve_model"] = adj_ve_m

    df = pd.DataFrame(result_cols, index=common_cells)
    df.index.name = "cell_roi_id"

    if roi_ids is not None:
        df = df.reindex(roi_ids, fill_value=0.0)

    return df


# ── Cross-session normalization ────────────────────────────────────────────────

def normalize_across_sessions(
    session_adj_ve: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Cross-session normalization per Garrett 2026.

    Args:
        session_adj_ve: {'F': df_F, 'N': df_N, 'N+': df_Np}
            Each df indexed by cell_roi_id (same index across sessions).
            Columns: {feat}_adj_ve_full, {feat}_adj_ve_model for each feature.

    Returns:
        DataFrame (n_cells × 12) with columns {feat}_{F|N|Np}.
        Values ∈ [0, 1], positive convention (higher = stronger coding).
    """
    labels     = list(session_adj_ve.keys())            # e.g. ['F', 'N', 'N+']
    label_cols = [l.replace("+", "p") for l in labels]  # ['F', 'N', 'Np']

    result: dict[str, pd.Series] = {}
    for feat in FEATURES:
        full_by_sess  = {l: session_adj_ve[l][f"{feat}_adj_ve_full"]  for l in labels}
        model_by_sess = {l: session_adj_ve[l][f"{feat}_adj_ve_model"] for l in labels}

        # Max adj_ve_full across sessions per cell
        ve_max = pd.concat(list(full_by_sess.values()), axis=1).max(axis=1)

        for l, lc in zip(labels, label_cols):
            raw  = (full_by_sess[l] - model_by_sess[l]).clip(lower=0.0)
            norm = (raw / ve_max.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

            # Filter: zero where max adj_ve_full < threshold (non-encoding cell)
            norm[ve_max < VE_THRESHOLD] = 0.0
            # Filter: zero where this session's adj_ve_model < threshold
            norm[model_by_sess[l] < VE_THRESHOLD] = 0.0

            result[f"{feat}_{lc}"] = norm

    return pd.DataFrame(result)


# ── Per-subject pipeline ───────────────────────────────────────────────────────

def compute_subject_coding_scores(
    subject_id: str | int,
    label_date_map: dict[str, str],  # {'F': 'YYYY-MM-DD', 'N': '...', 'N+': '...'}
    cell_index: pd.DataFrame,        # rows = matched cells; cols include roi_id_F/N/Np
) -> pd.DataFrame:
    """Compute normalized 12-feature coding scores for one subject."""
    sid = str(subject_id)
    session_adj_ve: dict[str, pd.DataFrame] = {}

    for label, date in label_date_map.items():
        col     = f"roi_id_{label.replace('+', 'p')}"
        roi_ids = cell_index[col].values
        print(f"  {label} ({date}): adj_VE for {len(roi_ids)} cells ...", flush=True)
        df = compute_session_adj_ve(sid, date, roi_ids=roi_ids)
        df.index = cell_index.index   # re-index by cell_id
        session_adj_ve[label] = df

    norm = normalize_across_sessions(session_adj_ve)
    norm.index = pd.MultiIndex.from_arrays(
        [np.full(len(norm), int(sid)), norm.index],
        names=["subject_id", "cell_id"],
    )
    return norm


# ── Batch: all subjects ────────────────────────────────────────────────────────

def compute_all_coding_scores(
    cell_index: pd.DataFrame,
    session_table_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute normalized coding scores for all subjects in cell_index.

    Args:
        cell_index: from cell_matching.build_matched_cell_index.
                    Columns: cell_id, subject_id, roi_id_F, roi_id_N, roi_id_Np.
        session_table_path: path to ground-truth session table CSV.
            If None, reads the CTL_SESSION_TABLE environment variable.

    Returns:
        DataFrame (n_cells × 12) with MultiIndex (subject_id, cell_id).
    """
    if session_table_path is None:
        env = os.environ.get("CTL_SESSION_TABLE")
        if not env:
            raise ValueError(
                "session_table_path is required. "
                "Pass it explicitly or set the CTL_SESSION_TABLE environment variable."
            )
        session_table_path = env

    st = pd.read_csv(session_table_path)
    st["date"] = st["raw_asset_name"].str.extract(r"_(\d{4}-\d{2}-\d{2})_")

    pieces = []
    for sid, sub_idx in cell_index.groupby("subject_id"):
        sub_idx = sub_idx.set_index("cell_id")
        sub_st = st[
            (st["subject_id"].astype(str) == str(sid)) &
            (st["session_type"].str.contains("OPHYS_1|OPHYS_4", na=False))
        ]
        o1 = sub_st[sub_st["session_type"].str.startswith("OPHYS_1")].sort_values("date")
        o4 = sub_st[sub_st["session_type"].str.startswith("OPHYS_4")].sort_values("date")
        label_date = {
            "F":  o1.iloc[-1]["date"],
            "N":  o4.iloc[0]["date"],
            "N+": o4.iloc[1]["date"],
        }
        print(
            f"Subject {sid}: F={label_date['F']}  N={label_date['N']}"
            f"  N+={label_date['N+']}  ({len(sub_idx)} cells)",
            flush=True,
        )
        try:
            scores = compute_subject_coding_scores(str(sid), label_date, sub_idx)
            pieces.append(scores)
        except Exception as exc:
            import traceback
            print(f"  ERROR for {sid}: {exc}")
            traceback.print_exc()

    return pd.concat(pieces) if pieces else pd.DataFrame()
