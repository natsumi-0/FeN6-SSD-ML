#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

"""
Leak-free nested cross-validation for RandomForest classification with
group-based splitting by FeN6 core identifier (``rename_id``).

This script performs nested cross-validation while enforcing group separation
between training and test data. All preprocessing and feature-selection steps
are fit only on the outer training split and then applied to the corresponding
test split by column alignment.

Main features
-------------
- Group-based outer and inner cross-validation using ``rename_id``
- Leak-free training-only preprocessing
- Removal of columns containing NaN, inf, or excessively large values
- Variance filtering
- Streaming correlation pruning without constructing a full correlation matrix
- Assertion of train/test group disjointness in each outer fold
- Optional fold-wise and aggregated SHAP summaries

Cross-validation
----------------
Outer CV:
    StratifiedGroupKFold repeated by changing the random seed

Inner CV:
    StratifiedGroupKFold, 5 splits, used within GridSearchCV

Leak-free preprocessing policy
------------------------------
All filtering and selection are fit on the outer training split only:
1. Replace inf with NaN
2. Mask excessively large values (> float32 max) to NaN
3. Drop training columns containing any NaN
4. Apply variance filtering on training data
5. Apply streaming correlation pruning on training data

The resulting selected columns are then aligned onto the outer test split.

Outputs
-------
Default output directory:
    ./results_rf_group_leakfree

Per fold:
    <out_dir>/<state>/<descriptor>/<fold>/
        train_ids_<fold>.csv
        test_ids_<fold>.csv
        train_groups_<fold>.csv
        test_groups_<fold>.csv
        metrics.csv
        importance.csv
        cols_used.txt
        shap_mean_abs.csv              # if --compute-shap
        shap_values.csv.gz             # if --compute-shap --save-shap-matrix

Summary files:
    <out_dir>/summary_<state>.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

THIS_FILE = Path(__file__).resolve()
LOGGER = logging.getLogger(__name__)


def find_repo_root(start: Path) -> Path:
    """
    Locate the repository root by searching upward from ``start``.

    The repository root is defined as a directory containing:
    - ``descriptors/``
    - ``ML/``
    - ``FeN6-SSD/FeN6-SSD_500_spin_labeled.csv``

    Parameters
    ----------
    start
        Starting path for upward search.

    Returns
    -------
    Path
        Detected repository root.

    Raises
    ------
    RuntimeError
        If no matching parent directory is found.
    """
    for p in [start] + list(start.parents):
        if (p / "descriptors").is_dir() and (p / "ML").is_dir():
            if (p / "FeN6-SSD" / "FeN6-SSD_500_spin_labeled.csv").is_file():
                return p
    raise RuntimeError(
        "Repository root not found. Expected 'descriptors/', 'ML/', and "
        "'FeN6-SSD/FeN6-SSD_500_spin_labeled.csv' in a parent directory."
    )


ROOT_DIR = find_repo_root(THIS_FILE.parent)
DEFAULT_LABEL_CSV = ROOT_DIR / "FeN6-SSD" / "FeN6-SSD_500_spin_labeled.csv"
DEFAULT_DESC_DIR = ROOT_DIR / "descriptors"
DEFAULT_OUT_DIR = THIS_FILE.parent / "results_rf_group_leakfree"

COL_ID = "ccdc_id"
COL_STATE = "Spin-state of crystal structure"
COL_BEHAV = "Spin-state behaviour"
COL_RENAME_ID = "rename_id"

POSITIVE_LABEL = "spin-crossover"
DEFAULT_STATES = ("high-spin", "low-spin")

DEFAULT_N_ESTIMATORS = (50, 100, 300)
DEFAULT_MAX_FEATURES = ("None", "sqrt", "log2")

OUTER_SPLITS = 5
OUTER_REPEATS = 3
OUTER_SEED = 42

INNER_SPLITS = 5
INNER_SEED = 0

VAR_THRESH_DEFAULT = 1e-12
CORR_THRESH_DEFAULT = 0.90
SEED = 42

FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class FilterConfig:
    """
    Configuration for training-only feature filtering.

    Parameters
    ----------
    var_thresh
        Variance threshold for low-variance feature removal.
    corr_thresh
        Correlation threshold used in streaming correlation pruning.
    use_abs_corr
        If True, prune by absolute correlation; otherwise use signed correlation.
    corr_block_size
        Number of candidate columns processed per block during correlation pruning.
    corr_keep_first
        Minimum number of top-variance features retained before blockwise pruning.
    corr_max_keep
        Maximum number of retained features. A value of 0 disables the cap.
    """

    var_thresh: float = VAR_THRESH_DEFAULT
    corr_thresh: float = CORR_THRESH_DEFAULT
    use_abs_corr: bool = True
    corr_block_size: int = 512
    corr_keep_first: int = 1
    corr_max_keep: int = 0


def configure_logging(level: str) -> None:
    """
    Configure module-level logging.

    Parameters
    ----------
    level
        Logging level name such as ``"INFO"`` or ``"DEBUG"``.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s")


def ensure_dir(p: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    p
        Directory path to create.
    """
    p.mkdir(parents=True, exist_ok=True)


def parse_max_features(values: Sequence[str]) -> List[object]:
    """
    Parse ``max_features`` values from command-line strings.

    Strings equal to ``"none"`` or ``"null"`` are converted to ``None``.
    Other values are returned as strings.

    Parameters
    ----------
    values
        Raw argument values.

    Returns
    -------
    List[object]
        Parsed values suitable for scikit-learn.
    """
    out: List[object] = []
    for v in values:
        v2 = v.strip()
        if v2.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(v2)
    return out


def fold_name(i: int, outer_splits: int) -> str:
    """
    Generate a fold name in the form ``r<repeat>_f<fold>``.

    Parameters
    ----------
    i
        Zero-based fold index across all repeats.
    outer_splits
        Number of folds in each repeat.

    Returns
    -------
    str
        Fold label.
    """
    rep = i // outer_splits + 1
    f = i % outer_splits + 1
    return f"r{rep}_f{f}"


def write_run_config(out_root: Path, args: argparse.Namespace) -> None:
    """
    Save runtime configuration and package versions as JSON.

    Parameters
    ----------
    out_root
        Output directory.
    args
        Parsed command-line arguments.
    """
    ensure_dir(out_root)
    cfg = {
        "python": sys.version,
        "platform": platform.platform(),
        "args": vars(args),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }
    try:
        import sklearn  # noqa

        cfg["sklearn"] = sklearn.__version__
    except Exception:
        cfg["sklearn"] = None
    try:
        import shap  # noqa

        cfg["shap"] = shap.__version__
    except Exception:
        cfg["shap"] = None

    (out_root / "run_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _pick_id_column(raw: pd.DataFrame, label_ids: pd.Index) -> Optional[str]:
    """
    Infer the ID column in a descriptor table.

    Priority:
    1. Exact match to ``ccdc_id``
    2. Case-insensitive match to ``ccdc_id``
    3. Non-numeric column with the largest overlap with ``label_ids``
       provided the overlap is at least 5
    4. Otherwise return ``None``

    Parameters
    ----------
    raw
        Raw descriptor table.
    label_ids
        Valid label IDs used for overlap-based inference.

    Returns
    -------
    Optional[str]
        Selected ID column name, or ``None`` if no suitable column is found.
    """
    if raw.empty:
        return None

    if COL_ID in raw.columns:
        return COL_ID

    lower_map = {str(c).strip().lower(): c for c in raw.columns}
    if COL_ID.lower() in lower_map:
        return str(lower_map[COL_ID.lower()])

    label_set = set(label_ids.astype(str))
    best_col: Optional[str] = None
    best_overlap = 0

    for c in list(raw.columns):
        s = raw[c]
        if getattr(s.dtype, "kind", "") in "biufc":
            continue
        ss = s.dropna().astype(str).str.strip()
        if ss.empty:
            continue
        overlap = len(set(ss) & label_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = str(c)

    if best_col is not None and best_overlap >= 5:
        return best_col
    return None


def read_labels(labels_path: Path) -> pd.DataFrame:
    """
    Read and normalize the label table.

    Parameters
    ----------
    labels_path
        Path to the label CSV file.

    Returns
    -------
    pd.DataFrame
        Label table indexed by ``ccdc_id``.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    df = pd.read_csv(labels_path)
    missing = [c for c in (COL_ID, COL_STATE, COL_BEHAV, COL_RENAME_ID) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in label CSV: {missing}")

    df[COL_ID] = df[COL_ID].astype(str).str.strip()
    df[COL_STATE] = df[COL_STATE].astype(str).str.strip()
    df[COL_BEHAV] = df[COL_BEHAV].astype(str).str.strip()
    df[COL_RENAME_ID] = df[COL_RENAME_ID].astype(str).str.strip()
    df = df.replace({"nan": np.nan, "NaN": np.nan})
    return df.set_index(COL_ID)


def load_descriptors(desc_root: Path, *, label_ids: pd.Index) -> Dict[str, pd.DataFrame]:
    """
    Load descriptor tables from subdirectories of ``desc_root``.

    Each descriptor is expected at:
        ``<desc_root>/<name>/<name>.csv``

    The returned data frames:
    - are indexed by inferred ID column
    - retain numeric columns only
    - are not globally filtered for missing values

    Parameters
    ----------
    desc_root
        Root descriptor directory.
    label_ids
        Label IDs used for robust ID-column inference.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from descriptor name to numeric descriptor matrix.

    Raises
    ------
    FileNotFoundError
        If ``desc_root`` does not exist.
    RuntimeError
        If no valid descriptor tables are found.
    """
    if not desc_root.is_dir():
        raise FileNotFoundError(f"Descriptor directory not found: {desc_root}")

    out: Dict[str, pd.DataFrame] = {}
    for name in sorted(os.listdir(desc_root)):
        sub = desc_root / name
        if not sub.is_dir():
            continue
        csv_path = sub / f"{name}.csv"
        if not csv_path.is_file():
            continue

        raw = pd.read_csv(csv_path)
        id_col = _pick_id_column(raw, label_ids)

        if id_col is not None:
            df = raw.set_index(id_col)
        else:
            df = pd.read_csv(csv_path, index_col=0)

        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.select_dtypes(include=[np.number])

        if df.shape[1] == 0:
            LOGGER.warning("Descriptor '%s' has no numeric columns. Skipped.", name)
            continue
        out[name] = df

    if not out:
        raise RuntimeError(f"No descriptor CSVs found under: {desc_root}")

    LOGGER.info("Loaded %d descriptors.", len(out))
    for k, v in out.items():
        LOGGER.info("  - %s: %s", k, v.shape)
    return out


def drop_low_variance_cols_train_only(Xtr: pd.DataFrame, var_thresh: float) -> List[str]:
    """
    Keep training columns whose sample variance exceeds ``var_thresh``.

    Parameters
    ----------
    Xtr
        Training feature matrix.
    var_thresh
        Variance threshold.

    Returns
    -------
    List[str]
        Names of retained columns.
    """
    if Xtr.shape[1] == 0:
        return []
    v = Xtr.astype(np.float64, copy=False).var(axis=0, ddof=1)
    return v[v > var_thresh].index.astype(str).tolist()


def select_low_correlation_cols_streaming_train_only(
    Xtr: pd.DataFrame,
    *,
    corr_thresh: float,
    use_abs: bool,
    block_size: int,
    keep_first: int,
    max_keep: int,
) -> List[str]:
    """
    Perform greedy correlation pruning on training data without forming a full
    correlation matrix.

    Features are ordered by decreasing standard deviation. A seed set of
    ``keep_first`` columns is retained first, after which remaining columns are
    processed blockwise. Exact correlations are computed between retained
    columns and candidate blocks after standardization.

    Parameters
    ----------
    Xtr
        Training feature matrix.
    corr_thresh
        Correlation threshold for pruning.
    use_abs
        If True, use absolute correlation values.
    block_size
        Number of candidate columns processed in each block.
    keep_first
        Number of highest-variance columns retained before blockwise pruning.
    max_keep
        Maximum number of columns to retain. A value of 0 disables the cap.

    Returns
    -------
    List[str]
        Names of retained columns.
    """
    if Xtr.shape[1] <= 1:
        return list(Xtr.columns.astype(str))

    A = Xtr.to_numpy(dtype=np.float64, copy=False)
    cols_all = np.asarray(Xtr.columns.astype(str), dtype=object)

    finite_cols = np.isfinite(A).all(axis=0)
    A = A[:, finite_cols]
    cols = cols_all[finite_cols]
    if A.shape[1] <= 1:
        return cols.astype(str).tolist()

    mu = A.mean(axis=0)
    sd = A.std(axis=0, ddof=1)
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    A = A[:, ok]
    cols = cols[ok]
    sd = sd[ok]
    if A.shape[1] <= 1:
        return cols.astype(str).tolist()

    Z = (A - mu[ok]) / sd
    order = np.argsort(sd)[::-1]
    denom = max(Z.shape[0] - 1, 1)

    keep_idx: List[int] = []
    initial_take = min(max(int(keep_first), 1), len(order))
    init = order[:initial_take]
    keep_idx.extend(init.tolist())
    K = Z[:, init].copy()

    if max_keep > 0 and len(keep_idx) >= max_keep:
        keep_idx = keep_idx[:max_keep]
        return cols[keep_idx].astype(str).tolist()

    rest = order[initial_take:]
    bs = max(int(block_size), 1)

    for start in range(0, len(rest), bs):
        blk = rest[start:start + bs]
        if blk.size == 0:
            continue

        B = Z[:, blk]
        C = (K.T @ B) / denom
        if use_abs:
            C = np.abs(C)

        maxcorr = np.nanmax(C, axis=0)
        good = np.isfinite(maxcorr) & (maxcorr < corr_thresh)
        if not np.any(good):
            continue

        kept_blk = blk[good]
        keep_idx.extend(kept_blk.tolist())
        K = np.concatenate([K, Z[:, kept_blk]], axis=1)

        if max_keep > 0 and len(keep_idx) >= max_keep:
            keep_idx = keep_idx[:max_keep]
            break

    return cols[keep_idx].astype(str).tolist()


def build_state_dataset(labels: pd.DataFrame, state: str) -> Tuple[pd.Index, pd.Series, pd.Series]:
    """
    Build the dataset for one spin state.

    Rows with ``mixture`` behaviour are excluded. The target is binarized such
    that ``spin-crossover`` is treated as the positive class and all remaining
    rows are mapped to the specified state label.

    Parameters
    ----------
    labels
        Full label table.
    state
        Spin state to process.

    Returns
    -------
    Tuple[pd.Index, pd.Series, pd.Series]
        Sample IDs, target labels, and group labels (``rename_id``).
    """
    df = labels.copy()

    df = df[df[COL_STATE].isin(["high-spin", "low-spin"])]
    df = df[df[COL_STATE] == state]

    beh = df[COL_BEHAV].astype(str).str.strip()
    beh = beh[beh != "mixture"]
    df = df.loc[beh.index].copy()
    beh = beh.loc[df.index]

    g = df[COL_RENAME_ID].astype(str).str.strip()
    y = beh.apply(lambda v: POSITIVE_LABEL if v == POSITIVE_LABEL else state)

    mask = y.notna() & g.notna() & (g != "")
    y = y[mask]
    g = g[mask]
    return y.index, y, g


def build_outer_splits_group(
    y: pd.Series,
    groups: pd.Series,
    n_splits: int,
    n_repeats: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build repeated outer group-based cross-validation splits.

    Parameters
    ----------
    y
        Target labels.
    groups
        Group labels used to enforce group disjointness.
    n_splits
        Number of folds in each repeat.
    n_repeats
        Number of repeated outer CV runs.
    seed
        Base random seed. Each repeat uses ``seed + repeat_index``.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Outer train/test index pairs.
    """
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(np.zeros(len(y)), y.to_numpy(), groups=groups.to_numpy()):
            splits.append((tr, te))
    return splits


def _extract_positive_class_shap_array(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    positive_label: Optional[str] = None,
) -> np.ndarray:
    """
    Extract SHAP values for one class from a fitted RandomForest classifier.

    Parameters
    ----------
    model
        Fitted RandomForest classifier.
    X
        Evaluation matrix.
    positive_label
        Class label for which SHAP values should be extracted. If omitted,
        class index 1 is used for binary classification when available.

    Returns
    -------
    np.ndarray
        SHAP value matrix with shape ``(n_samples, n_features)``.

    Raises
    ------
    RuntimeError
        If SHAP is not installed or the returned SHAP array has an
        unsupported shape.
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError(
            "SHAP calculation requested but 'shap' is not installed. "
            "Install with: pip install shap"
        ) from e

    explainer = shap.TreeExplainer(model)

    try:
        raw = explainer.shap_values(X, check_additivity=False)
    except TypeError:
        try:
            exp = explainer(X, check_additivity=False)
        except TypeError:
            exp = explainer(X)
        raw = exp.values
    except Exception:
        try:
            exp = explainer(X, check_additivity=False)
        except TypeError:
            exp = explainer(X)
        raw = exp.values

    classes = list(model.classes_)
    if positive_label is not None and positive_label in classes:
        pos_idx = classes.index(positive_label)
    else:
        pos_idx = 1 if len(classes) > 1 else 0

    if isinstance(raw, list):
        arr = np.asarray(raw[pos_idx], dtype=np.float64)
    else:
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2:
            pass
        elif arr.ndim == 3:
            if arr.shape[2] == len(classes):
                arr = arr[:, :, pos_idx]
            elif arr.shape[0] == len(classes):
                arr = arr[pos_idx]
            else:
                raise RuntimeError(f"Unsupported SHAP array shape: {arr.shape}")
        else:
            raise RuntimeError(f"Unsupported SHAP array ndim: {arr.ndim}")

    if arr.shape != (X.shape[0], X.shape[1]):
        raise RuntimeError(
            f"Unexpected SHAP shape {arr.shape}; expected {(X.shape[0], X.shape[1])}"
        )
    return arr


def save_fold_shap_outputs(
    *,
    model: RandomForestClassifier,
    X_eval: pd.DataFrame,
    fold_dir: Path,
    positive_label: Optional[str],
    save_shap_matrix: bool,
) -> pd.DataFrame:
    """
    Compute and save SHAP outputs for one fold.

    Parameters
    ----------
    model
        Fitted RandomForest classifier.
    X_eval
        Evaluation matrix used for SHAP calculation.
    fold_dir
        Output directory for the current fold.
    positive_label
        Optional target class for SHAP extraction.
    save_shap_matrix
        If True, save the raw SHAP matrix as a compressed CSV file.

    Returns
    -------
    pd.DataFrame
        Table of mean absolute SHAP values per feature.
    """
    shap_arr = _extract_positive_class_shap_array(model, X_eval, positive_label=positive_label)

    mean_abs = np.abs(shap_arr).mean(axis=0)
    shap_mean_df = pd.DataFrame(
        {
            "feature": X_eval.columns.astype(str),
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_mean_df.to_csv(fold_dir / "shap_mean_abs.csv", index=False)

    if save_shap_matrix:
        shap_mat = pd.DataFrame(
            shap_arr.astype(np.float32),
            index=X_eval.index.astype(str),
            columns=X_eval.columns.astype(str),
        )
        shap_mat.index.name = COL_ID
        shap_mat.to_csv(fold_dir / "shap_values.csv.gz", compression="gzip")

    return shap_mean_df


def write_fold_lists(out_dir: Path, fold: str, ids_tr: pd.Index, ids_te: pd.Index, g_tr: pd.Series, g_te: pd.Series) -> None:
    """
    Save sample-ID and group-ID lists for one fold.

    Parameters
    ----------
    out_dir
        Fold output directory.
    fold
        Fold label.
    ids_tr
        Training sample IDs.
    ids_te
        Test sample IDs.
    g_tr
        Training group labels.
    g_te
        Test group labels.
    """
    ensure_dir(out_dir)
    pd.Series(ids_tr.astype(str), name=COL_ID).to_csv(out_dir / f"train_ids_{fold}.csv", index=False)
    pd.Series(ids_te.astype(str), name=COL_ID).to_csv(out_dir / f"test_ids_{fold}.csv", index=False)
    pd.Series(pd.Series(g_tr).astype(str).unique(), name=COL_RENAME_ID).to_csv(out_dir / f"train_groups_{fold}.csv", index=False)
    pd.Series(pd.Series(g_te).astype(str).unique(), name=COL_RENAME_ID).to_csv(out_dir / f"test_groups_{fold}.csv", index=False)


def empty_metrics(state: str, descriptor: str, fold: str, reason: str, *, max_abs_train: float = np.nan) -> dict:
    """
    Create an empty metrics record for skipped folds.

    Parameters
    ----------
    state
        Spin state.
    descriptor
        Descriptor name.
    fold
        Fold label.
    reason
        Reason for skipping the fold.
    max_abs_train
        Maximum absolute training value observed before filtering, if available.

    Returns
    -------
    dict
        Metrics dictionary populated with NaN or null placeholders.
    """
    return {
        "state": state,
        "descriptor": descriptor,
        "fold": fold,
        "n_features": 0,
        "n_train": np.nan,
        "n_test": np.nan,
        "n_train_groups": np.nan,
        "n_test_groups": np.nan,
        "max_abs_train": max_abs_train,
        "train_accuracy": np.nan,
        "test_accuracy": np.nan,
        "train_mcc": np.nan,
        "test_mcc": np.nan,
        "train_precision": np.nan,
        "test_precision": np.nan,
        "train_recall": np.nan,
        "test_recall": np.nan,
        "train_f1": np.nan,
        "test_f1": np.nan,
        "best_n_estimators": None,
        "best_max_features": None,
        "skipped_reason": reason,
    }


def run_fold_group(
    *,
    state: str,
    descriptor: str,
    fold: str,
    X_master: pd.DataFrame,
    y_master: pd.Series,
    g_master: pd.Series,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    out_dir: Path,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
    filter_cfg: FilterConfig,
    grid_n_jobs: int,
    rf_n_jobs: int,
    no_corr_prune: bool,
    compute_shap: bool,
    save_shap_matrix: bool,
    shap_on: str,
    shap_positive_label: Optional[str],
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """
    Run a single outer fold for one descriptor and one spin state using
    group-based splitting.

    The function applies training-only filtering, performs inner group-aware
    grid search for RandomForest hyperparameters, evaluates train and test
    performance, and optionally computes SHAP summaries.

    Parameters
    ----------
    state
        Spin state currently being processed.
    descriptor
        Descriptor name.
    fold
        Fold label.
    X_master
        Descriptor matrix indexed by master IDs.
    y_master
        Target vector indexed by master IDs.
    g_master
        Group vector indexed by master IDs.
    tr_idx
        Outer training indices.
    te_idx
        Outer test indices.
    out_dir
        Output directory for this fold.
    n_estimators_grid
        Candidate values for ``n_estimators``.
    max_features_grid
        Candidate values for ``max_features``.
    filter_cfg
        Feature-filter configuration.
    grid_n_jobs
        ``n_jobs`` passed to GridSearchCV.
    rf_n_jobs
        ``n_jobs`` passed to RandomForestClassifier.
    no_corr_prune
        If True, disable correlation pruning.
    compute_shap
        If True, compute SHAP outputs.
    save_shap_matrix
        If True, save raw SHAP values.
    shap_on
        Dataset split used for SHAP computation, either ``"train"`` or ``"test"``.
    shap_positive_label
        Optional target class for SHAP extraction.

    Returns
    -------
    Tuple[dict, Optional[pd.DataFrame]]
        Metrics dictionary and optional SHAP summary table.

    Raises
    ------
    RuntimeError
        If train and test groups overlap within a fold.
    """
    ensure_dir(out_dir)

    X_tr = X_master.iloc[tr_idx].copy()
    X_te = X_master.iloc[te_idx].copy()
    y_tr = y_master.iloc[tr_idx].copy()
    y_te = y_master.iloc[te_idx].copy()
    g_tr = g_master.iloc[tr_idx].copy()
    g_te = g_master.iloc[te_idx].copy()

    tr_groups = set(pd.Series(g_tr).astype(str).unique())
    te_groups = set(pd.Series(g_te).astype(str).unique())
    inter = tr_groups.intersection(te_groups)
    if inter:
        raise RuntimeError(f"[BUG] train/test group overlap in {state}/{descriptor}/{fold}: {list(sorted(inter))[:5]}")

    write_fold_lists(out_dir, fold, X_tr.index, X_te.index, g_tr, g_te)

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te = X_te.replace([np.inf, -np.inf], np.nan)

    X_tr = X_tr.mask(np.abs(X_tr.astype(np.float64, copy=False)) > FLOAT32_MAX, np.nan)
    X_te = X_te.mask(np.abs(X_te.astype(np.float64, copy=False)) > FLOAT32_MAX, np.nan)

    try:
        max_abs_train = float(np.nanmax(np.abs(X_tr.to_numpy(dtype=np.float64, copy=False))))
    except Exception:
        max_abs_train = np.nan

    keep_no_nan = ~X_tr.isna().any(axis=0)
    X_tr = X_tr.loc[:, keep_no_nan]
    X_te = X_te.loc[:, keep_no_nan]
    if X_tr.shape[1] == 0:
        return empty_metrics(state, descriptor, fold, "no_features_after_train_nan_drop", max_abs_train=max_abs_train), None

    keep_var = drop_low_variance_cols_train_only(X_tr, filter_cfg.var_thresh)
    if not keep_var:
        return empty_metrics(state, descriptor, fold, "no_features_after_train_variance_filter", max_abs_train=max_abs_train), None
    X_tr = X_tr.loc[:, keep_var]
    X_te = X_te.loc[:, keep_var]

    if no_corr_prune or X_tr.shape[1] <= 1:
        keep_cols = list(X_tr.columns.astype(str))
    else:
        keep_cols = select_low_correlation_cols_streaming_train_only(
            Xtr=X_tr,
            corr_thresh=float(filter_cfg.corr_thresh),
            use_abs=bool(filter_cfg.use_abs_corr),
            block_size=int(filter_cfg.corr_block_size),
            keep_first=int(filter_cfg.corr_keep_first),
            max_keep=int(filter_cfg.corr_max_keep),
        )
    if not keep_cols:
        return empty_metrics(state, descriptor, fold, "no_features_after_train_corr_prune", max_abs_train=max_abs_train), None

    X_tr = X_tr.loc[:, keep_cols]
    X_te = X_te.loc[:, keep_cols]

    if not np.isfinite(X_tr.to_numpy(dtype=np.float64, copy=False)).all():
        return empty_metrics(state, descriptor, fold, "non_finite_remaining_in_train", max_abs_train=max_abs_train), None
    if not np.isfinite(X_te.to_numpy(dtype=np.float64, copy=False)).all():
        return empty_metrics(state, descriptor, fold, "non_finite_remaining_in_test", max_abs_train=max_abs_train), None

    pd.Series(X_tr.columns.astype(str), name="feature").to_csv(out_dir / "selected_features.csv", index=False)

    inner_cv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)

    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=OUTER_SEED,
        n_jobs=int(rf_n_jobs),
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid={"n_estimators": list(n_estimators_grid), "max_features": list(max_features_grid)},
        cv=inner_cv,
        scoring=make_scorer(matthews_corrcoef),
        n_jobs=int(grid_n_jobs),
        error_score="raise",
    )
    grid.fit(X_tr, y_tr, groups=g_tr)

    model: RandomForestClassifier = grid.best_estimator_
    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    metrics = {
        "state": state,
        "descriptor": descriptor,
        "fold": fold,
        "n_features": int(X_tr.shape[1]),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "n_train_groups": int(pd.Series(g_tr).nunique()),
        "n_test_groups": int(pd.Series(g_te).nunique()),
        "max_abs_train": max_abs_train,
        "train_accuracy": float(accuracy_score(y_tr, y_tr_pred)),
        "test_accuracy": float(accuracy_score(y_te, y_te_pred)),
        "train_precision": float(precision_score(y_tr, y_tr_pred, average="weighted", zero_division=0)),
        "test_precision": float(precision_score(y_te, y_te_pred, average="weighted", zero_division=0)),
        "train_recall": float(recall_score(y_tr, y_tr_pred, average="weighted", zero_division=0)),
        "test_recall": float(recall_score(y_te, y_te_pred, average="weighted", zero_division=0)),
        "train_f1": float(f1_score(y_tr, y_tr_pred, average="weighted", zero_division=0)),
        "test_f1": float(f1_score(y_te, y_te_pred, average="weighted", zero_division=0)),
        "train_mcc": float(matthews_corrcoef(y_tr, y_tr_pred)),
        "test_mcc": float(matthews_corrcoef(y_te, y_te_pred)),
        "best_n_estimators": int(model.n_estimators),
        "best_max_features": None if model.max_features is None else str(model.max_features),
        "skipped_reason": "",
    }

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame({"feature": X_tr.columns.astype(str), "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False) \
        .to_csv(out_dir / "importance.csv", index=False)
    (out_dir / "cols_used.txt").write_text("\n".join(X_tr.columns.astype(str).tolist()) + "\n", encoding="utf-8")

    shap_mean_df: Optional[pd.DataFrame] = None
    if compute_shap:
        X_for_shap = X_te if shap_on == "test" else X_tr
        shap_mean_df = save_fold_shap_outputs(
            model=model,
            X_eval=X_for_shap,
            fold_dir=out_dir,
            positive_label=shap_positive_label,
            save_shap_matrix=save_shap_matrix,
        )

    return metrics, shap_mean_df


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    p = argparse.ArgumentParser(description="Leak-free nested-CV RF with GROUP split by rename_id.")
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--descriptors", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    p.add_argument("--descs", nargs="*", default=None)

    p.add_argument("--n-estimators", nargs="+", type=int, default=list(DEFAULT_N_ESTIMATORS))
    p.add_argument("--max-features", nargs="+", default=list(DEFAULT_MAX_FEATURES))

    p.add_argument("--outer-splits", type=int, default=OUTER_SPLITS)
    p.add_argument("--outer-repeats", type=int, default=OUTER_REPEATS)
    p.add_argument("--outer-seed", type=int, default=OUTER_SEED)

    p.add_argument("--var-thresh", type=float, default=VAR_THRESH_DEFAULT)
    p.add_argument("--corr-thresh", type=float, default=CORR_THRESH_DEFAULT)
    p.add_argument("--signed-corr", action="store_true", help="Use signed correlation (default abs).")
    p.add_argument("--no-corr-prune", action="store_true", help="Disable TRAIN-only correlation pruning.")
    p.add_argument("--corr-block-size", type=int, default=512)
    p.add_argument("--corr-keep-first", type=int, default=1)
    p.add_argument("--corr-max-keep", type=int, default=0)

    p.add_argument("--grid-n-jobs", type=int, default=1)
    p.add_argument("--rf-n-jobs", type=int, default=1)

    p.add_argument("--compute-shap", action="store_true", help="Compute fold-wise SHAP summaries.")
    p.add_argument("--save-shap-matrix", action="store_true", help="Also save raw SHAP matrix as shap_values.csv.gz.")
    p.add_argument("--shap-on", choices=["train", "test"], default="test", help="Compute SHAP on train or test fold.")
    p.add_argument(
        "--shap-positive-label",
        type=str,
        default=POSITIVE_LABEL,
        help="Positive class label for SHAP extraction. Default: spin-crossover",
    )

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """
    Run nested cross-validation for all requested states and descriptors using
    group-based splitting.

    Parameters
    ----------
    argv
        Optional command-line argument list. If omitted, ``sys.argv[1:]`` is used.
    """
    args = build_argparser().parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)

    labels_path = args.labels or DEFAULT_LABEL_CSV
    desc_root = args.descriptors or DEFAULT_DESC_DIR
    out_root = args.out_dir or DEFAULT_OUT_DIR
    ensure_dir(out_root)
    write_run_config(out_root, args)

    n_estimators_grid = list(args.n_estimators)
    max_features_grid = parse_max_features(args.max_features)

    filter_cfg = FilterConfig(
        var_thresh=float(args.var_thresh),
        corr_thresh=float(args.corr_thresh),
        use_abs_corr=not bool(args.signed_corr),
        corr_block_size=int(args.corr_block_size),
        corr_keep_first=int(args.corr_keep_first),
        corr_max_keep=int(args.corr_max_keep),
    )

    LOGGER.info("Repo root: %s", ROOT_DIR)
    LOGGER.info("Labels: %s", labels_path)
    LOGGER.info("Descriptors: %s", desc_root)
    LOGGER.info("Output: %s", out_root)
    LOGGER.info("Grid: n_estimators=%s, max_features=%s", n_estimators_grid, max_features_grid)
    LOGGER.info(
        "Filtering: var_thresh=%g corr_thresh=%g corr=%s block=%d keep_first=%d max_keep=%s",
        float(args.var_thresh),
        float(args.corr_thresh),
        "abs" if filter_cfg.use_abs_corr else "signed",
        int(args.corr_block_size),
        int(args.corr_keep_first),
        ("NO_CAP" if int(args.corr_max_keep) == 0 else str(int(args.corr_max_keep))),
    )
    LOGGER.info(
        "Thread env: OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
    )
    LOGGER.info("Parallel: RF n_jobs=%d | GridSearch n_jobs=%d", int(args.rf_n_jobs), int(args.grid_n_jobs))
    LOGGER.info(
        "SHAP: %s | on=%s | save_matrix=%s | positive_label=%s",
        "ON" if args.compute_shap else "OFF",
        str(args.shap_on),
        "ON" if args.save_shap_matrix else "OFF",
        str(args.shap_positive_label),
    )
    LOGGER.info(
        "CV: outer=%dx%d (seed=%d), inner=%d",
        int(args.outer_splits),
        int(args.outer_repeats),
        int(args.outer_seed),
        INNER_SPLITS,
    )

    labels = read_labels(labels_path)
    descriptors = load_descriptors(desc_root, label_ids=labels.index)

    if args.descs:
        wanted = set(args.descs)
        descriptors = {k: v for k, v in descriptors.items() if k in wanted}
        LOGGER.info("Filtered descriptors: %d -> %s", len(descriptors), sorted(descriptors.keys()))
        if not descriptors:
            raise RuntimeError(f"--descs specified but none matched. wanted={sorted(wanted)}")

    for state in args.states:
        LOGGER.info("=== STATE: %s ===", state)

        master_ids, y_master, g_master = build_state_dataset(labels, state=state)
        if len(master_ids) == 0:
            LOGGER.warning("No usable rows after filtering for state=%s", state)
            continue

        common = master_ids
        for X in descriptors.values():
            common = common.intersection(X.index)

        if len(common) == 0:
            LOGGER.warning("No common IDs across labels+descriptors for state=%s", state)
            continue

        y_master = y_master.loc[common]
        g_master = g_master.loc[common]
        X_master_dict = {d: X.reindex(common) for d, X in descriptors.items()}

        splits = build_outer_splits_group(
            y=y_master,
            groups=g_master,
            n_splits=int(args.outer_splits),
            n_repeats=int(args.outer_repeats),
            seed=int(args.outer_seed),
        )

        rows: List[dict] = []
        for desc_name, Xmat in X_master_dict.items():
            if Xmat.shape[0] == 0 or Xmat.shape[1] == 0:
                continue

            LOGGER.info("Descriptor: %s | X=%s", desc_name, Xmat.shape)
            fold_shap_rows: List[pd.DataFrame] = []

            for i, (tr_idx, te_idx) in enumerate(splits):
                fold = fold_name(i, int(args.outer_splits))
                fold_out = out_root / state / desc_name / fold
                ensure_dir(fold_out)

                metrics_path = fold_out / "metrics.csv"
                if metrics_path.is_file() and (not args.overwrite):
                    try:
                        rows.append(pd.read_csv(metrics_path).iloc[0].to_dict())
                        if args.compute_shap:
                            shp = fold_out / "shap_mean_abs.csv"
                            if shp.is_file():
                                tmp = pd.read_csv(shp)
                                tmp["fold"] = fold
                                fold_shap_rows.append(tmp)
                        continue
                    except Exception:
                        pass

                res, shap_mean_df = run_fold_group(
                    state=state,
                    descriptor=desc_name,
                    fold=fold,
                    X_master=Xmat,
                    y_master=y_master,
                    g_master=g_master,
                    tr_idx=tr_idx,
                    te_idx=te_idx,
                    out_dir=fold_out,
                    n_estimators_grid=n_estimators_grid,
                    max_features_grid=max_features_grid,
                    filter_cfg=filter_cfg,
                    grid_n_jobs=int(args.grid_n_jobs),
                    rf_n_jobs=int(args.rf_n_jobs),
                    no_corr_prune=bool(args.no_corr_prune),
                    compute_shap=bool(args.compute_shap),
                    save_shap_matrix=bool(args.save_shap_matrix),
                    shap_on=str(args.shap_on),
                    shap_positive_label=args.shap_positive_label,
                )
                rows.append(res)

                if shap_mean_df is not None:
                    tmp = shap_mean_df.copy()
                    tmp["fold"] = fold
                    fold_shap_rows.append(tmp)

            if args.compute_shap and fold_shap_rows:
                shap_cat = pd.concat(fold_shap_rows, ignore_index=True)
                shap_mean_over_folds = (
                    shap_cat.groupby("feature", as_index=False)["mean_abs_shap"]
                    .mean()
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True)
                )
                ensure_dir(out_root / state / desc_name)
                shap_mean_over_folds.to_csv(
                    out_root / state / desc_name / "shap_mean_abs_over_folds.csv",
                    index=False,
                )

        if rows:
            out_csv = out_root / f"summary_{state}.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            LOGGER.info("Saved: %s", out_csv)
        else:
            LOGGER.warning("No folds executed for state=%s", state)

    LOGGER.info("DONE.")


if __name__ == "__main__":
    main()