"""
Cell matching across sessions via ROICaT tracking results.

Builds a DataFrame with columns [cell_id, subject_id, roi_id_F, roi_id_N, roi_id_Np]
containing the GLM cell_roi_id for each matched valid cell in each target session.

Cell identity: (fov_name, ucid) within a subject → cell_id = "{fov_name}_{ucid}".
GLM cell_roi_id: "{fov_name}_{roi_session_index:04d}".
Valid cells: matched=True AND ucid >= 0 in ROICaT, AND present in GLM result for
all three sessions (F, N, N+).

Environment variables
---------------------
CTL_DATA_DIR
    Root directory where GLM and ROICaT assets are mounted.
    Defaults to "/data" (Code Ocean convention).

CTL_SESSION_TABLE
    Path to the ground-truth session table CSV.
    Required when session_table_path is not passed explicitly to
    build_matched_cell_index().
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

DATA_DIR = Path(os.environ.get("CTL_DATA_DIR", "/data"))
_DATE_RE = re.compile(r"multiplane-ophys_\d+_(\d{4}-\d{2}-\d{2})_")

# Regex to find the glm .npy result file inside an asset subdirectory
_GLM_NPY_RE = re.compile(r"glm_results_v\d+_(\d+)_(\d{4}-\d{2}-\d{2})_events\.npy$")

# ── GLM asset discovery ────────────────────────────────────────────────────────

def find_glm_asset_dir(subject_id: str | int, session_date: str) -> Path | None:
    """Return the inner subdirectory of the mounted GLM asset for (subject, date).

    Asset mount names follow: multiplane-ophys_{subject}_{date}_{time}_glm_*
    Inner subdir: {subject}_{date}_glm_v*
    """
    sid = str(subject_id)
    pattern = f"multiplane-ophys_{sid}_{session_date}_*_glm_*"
    mounts = sorted(DATA_DIR.glob(pattern))
    if not mounts:
        return None
    mount = mounts[-1]  # latest capture if multiple
    inner = sorted(mount.glob(f"{sid}_{session_date}_glm_v*"))
    return inner[-1] if inner else None


def find_glm_npy(subject_id: str | int, session_date: str) -> Path | None:
    """Return path to glm_results_v*_events.npy for (subject, session_date)."""
    d = find_glm_asset_dir(subject_id, session_date)
    if d is None:
        return None
    npys = sorted(d.glob("glm_results_v*_events.npy"))
    return npys[-1] if npys else None


def get_glm_cell_ids(subject_id: str | int, session_date: str) -> np.ndarray | None:
    """Return the cell_roi_id array from the GLM result, or None if unavailable."""
    npy = find_glm_npy(subject_id, session_date)
    if npy is None:
        return None
    data = np.load(npy, allow_pickle=True).item()
    ve = data["var_explained_test_cv"]
    if isinstance(ve, xr.DataArray):
        return ve.coords["cell_roi_id"].values
    return None


# ── Session label assignment ───────────────────────────────────────────────────

def assign_fnn_labels(
    subject_id: str | int,
    session_table: pd.DataFrame,
) -> dict[str, str]:
    """Return {session_date: label} for F, N, N+ sessions of one subject.

    Args:
        session_table: must have columns session_key, session_type, filtered to
                       this subject (OPHYS_1 and OPHYS_4 rows only).

    Returns:
        {date_str: 'F'|'N'|'N+'}  e.g. {'2025-04-16': 'F', '2025-04-28': 'N', ...}
    """
    sid = str(subject_id)
    df = session_table[session_table["subject_id"].astype(str) == sid].copy()
    df["date"] = df["raw_asset_name"].str.extract(r"_(\d{4}-\d{2}-\d{2})_")
    df = df.sort_values("date")

    o1 = df[df["session_type"].str.startswith("OPHYS_1")]
    o4 = df[df["session_type"].str.startswith("OPHYS_4")]

    if o1.empty or len(o4) < 2:
        raise ValueError(
            f"Subject {sid}: need ≥1 OPHYS_1 and ≥2 OPHYS_4 sessions, "
            f"got {len(o1)} OPHYS_1, {len(o4)} OPHYS_4"
        )

    labels: dict[str, str] = {}
    labels[o1.iloc[-1]["date"]] = "F"   # last OPHYS_1
    labels[o4.iloc[0]["date"]] = "N"    # first OPHYS_4
    labels[o4.iloc[1]["date"]] = "N+"   # second OPHYS_4
    return labels


# ── ROICaT loading ─────────────────────────────────────────────────────────────

def _find_roicat_dir(subject_id: str | int) -> Path | None:
    sid = str(subject_id)
    candidates = sorted(DATA_DIR.glob(f"multiplane-ophys_{sid}_ROICat_*"))
    return candidates[-1] if candidates else None


def load_roicat(subject_id: str | int) -> pd.DataFrame:
    """Load all FOV ROICaT tracking CSVs for a subject and return combined DataFrame.

    Adds columns:
        date      — YYYY-MM-DD extracted from session_name
        cell_id   — "{fov_name}_{ucid}" (unique within subject, cross-session identity)
        glm_roi_id— "{fov_name}_{roi_session_index:04d}" (matches GLM cell_roi_id)
    """
    base = _find_roicat_dir(subject_id)
    if base is None:
        raise FileNotFoundError(f"No ROICaT dir for subject {subject_id}")

    dfs = []
    for fov_dir in sorted(base.glob("[0-9]*")):
        csv = fov_dir / "ROICaT.tracking.results.csv"
        if csv.exists():
            dfs.append(pd.read_csv(csv))

    if not dfs:
        raise FileNotFoundError(f"No ROICaT CSVs found in {base}")

    df = pd.concat(dfs, ignore_index=True)

    # Extract raw session date from session_name
    df["date"] = df["session_name"].str.extract(_DATE_RE.pattern)

    # Cross-session cell identity within this subject
    df["cell_id"] = df["fov_name"].astype(str) + "_" + df["ucid"].astype(str)

    # GLM cell_roi_id: "{fov_name}_{roi_session_index:04d}"
    df["glm_roi_id"] = (
        df["fov_name"].astype(str)
        + "_"
        + df["roi_session_index"].apply(lambda i: f"{int(i):04d}")
    )

    return df


# ── Per-subject matched cell index ────────────────────────────────────────────

def build_subject_cell_index(
    subject_id: str | int,
    label_map: dict[str, str],   # {date: 'F'|'N'|'N+'}
) -> pd.DataFrame:
    """Return matched-cell DataFrame for one subject using ROICaT tracking.

    Columns: cell_id, subject_id, roi_id_F, roi_id_N, roi_id_Np
        cell_id    = "{fov_name}_{ucid}" — cross-session cell identity
        roi_id_*   = GLM cell_roi_id for that session (e.g. 'VISp_0_0003')

    Keeps only cells that are:
        - matched=True and ucid >= 0 in ROICaT
        - present in GLM result for all three sessions
    """
    sid = str(subject_id)
    roicat = load_roicat(sid)

    # Valid matched cells only
    valid = roicat[(roicat["matched"] == True) & (roicat["ucid"] >= 0)].copy()

    frames: dict[str, pd.DataFrame] = {}
    for date, label in label_map.items():
        col = f"roi_id_{label.replace('+', 'p')}"

        sess = (
            valid[valid["date"] == date][["cell_id", "glm_roi_id"]]
            .rename(columns={"glm_roi_id": col})
            .drop_duplicates("cell_id")
            .set_index("cell_id")
        )

        # Intersect with cells that have GLM data for this session
        glm_ids = get_glm_cell_ids(sid, date)
        if glm_ids is None:
            print(f"  WARNING: no GLM data for {sid} {date} ({label}), skipping")
            return pd.DataFrame()
        sess = sess[sess[col].isin(set(glm_ids))]
        frames[label] = sess

    # Inner join across all three sessions — keeps only cells present in F ∩ N ∩ N+
    base = frames["F"].join(frames["N"], how="inner").join(frames["N+"], how="inner")
    base.insert(0, "subject_id", int(sid))
    return base


# ── Multi-subject build ────────────────────────────────────────────────────────

def _resolve_session_table(session_table_path: str | Path | None) -> Path:
    if session_table_path is None:
        env = os.environ.get("CTL_SESSION_TABLE")
        if not env:
            raise ValueError(
                "session_table_path is required. "
                "Pass it explicitly or set the CTL_SESSION_TABLE environment variable."
            )
        session_table_path = env
    return Path(session_table_path)


def build_matched_cell_index(
    subject_ids: Sequence[int | str],
    session_table_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build matched-cell index across all subjects using ROICaT.

    Returns DataFrame with columns:
        cell_id, subject_id, roi_id_F, roi_id_N, roi_id_Np
    cell_id is unique within a subject but may collide across subjects;
    use (subject_id, cell_id) as the compound key.

    Args:
        subject_ids: list of subject IDs to process.
        session_table_path: path to the ground-truth session table CSV.
            If None, reads the CTL_SESSION_TABLE environment variable.
    """
    st = pd.read_csv(_resolve_session_table(session_table_path))
    st = st[st["session_type"].str.contains("OPHYS_1|OPHYS_4", na=False)]

    pieces = []
    for sid in subject_ids:
        sid_str = str(sid)
        try:
            label_map = assign_fnn_labels(sid_str, st)
        except ValueError as e:
            print(f"  SKIP {sid}: {e}")
            continue
        dates = {l: d for d, l in label_map.items()}
        print(f"{sid}: F={dates['F']}  N={dates['N']}  N+={dates['N+']}")
        idx = build_subject_cell_index(sid_str, label_map)
        if idx.empty:
            print(f"  SKIP {sid}: no matched cells")
            continue
        print(f"  {len(idx)} matched cells")
        pieces.append(idx.reset_index())  # cell_id becomes a column

    if not pieces:
        return pd.DataFrame()

    return pd.concat(pieces, ignore_index=True)
