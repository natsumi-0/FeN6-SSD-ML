#!/usr/bin/env python3
from __future__ import annotations

import os

for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

"""
Leak-free nested cross-validation for RandomForest classification with
streaming correlation pruning and optional SHAP output.

This script performs repeated outer cross-validation with inner
hyperparameter tuning by GridSearchCV. All feature filtering steps are fit
only on the outer training split and then applied to the corresponding test
split by column alignment.

Main features
-------------
- Leak-free feature filtering on outer training data only
- Removal of columns containing NaN, inf, or excessively large values
- Variance filtering
- Streaming correlation pruning without constructing a full correlation matrix
- RandomForest hyperparameter tuning by nested cross-validation
- Optional fold-wise and aggregated SHAP summaries

Cross-validation
----------------
Outer CV:
    RepeatedStratifiedKFold, 5 splits × 3 repeats
Inner CV:
    StratifiedKFold, 5 splits

Outputs
-------
<out_dir>/<state>/<descriptor>/<fold>/
    train_ids_<fold>.csv
    test_ids_<fold>.csv
    metrics.csv
    importance.csv
    selected_features.csv
    shap_mean_abs.csv              # if --compute-shap
    shap_values.csv.gz             # if --compute-shap --save-shap-matrix

<out_dir>/<state>/<descriptor>/
    shap_mean_abs_over_folds.csv   # if --compute-shap

<out_dir>/
    summary_<state>.csv

Example
-------
python rf-5fold_mbtrsafe_shap.py \
    --descs mbtr-cif \
    --states high-spin low-spin \
    --grid-n-jobs 1 \
    --rf-n-jobs 1 \
    --compute-shap \
    --shap-on test
"""

import argparse
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

THIS_FILE = Path(__file__).resolve()
LOGGER = logging.getLogger(__name__)

COL_ID = "ccdc_id"
COL_STATE = "Spin-state of crystal structure"
COL_TARGET = "Spin-state behaviour"

DEFAULT_STATES = ("high-spin", "low-spin")

OUTER_SPLITS = 5
OUTER_REPEATS = 3
OUTER_SEED = 42

INNER_SPLITS = 5
INNER_SEED = 0

DEFAULT_N_ESTIMATORS = (50, 100, 300)
DEFAULT_MAX_FEATURES = ("None", "sqrt", "log2")

VAR_THRESH_DEFAULT = 1e-12
CORR_THRESH_DEFAULT = 0.90

FLOAT32_MAX = float(np.finfo(np.float32).max)


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
DEFAULT_OUT_DIR = THIS_FILE.parent / "results_rf"


@dataclass(frozen=True)
class FeatureFilterConfig:
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
    corr_compute_dtype
        Floating-point dtype used during correlation computation.
        Must be ``"float64"`` or ``"float32"``.
    """

    var_thresh: float = VAR_THRESH_DEFAULT
    corr_thresh: float = CORR_THRESH_DEFAULT
    use_abs_corr: bool = True
    corr_block_size: int = 512
    corr_keep_first: int = 1
    corr_max_keep: int = 0
    corr_compute_dtype: str = "float64"


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
        pd.Series(cfg).to_json(indent=2, force_ascii=False), encoding="utf-8"
    )


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
    missing = [c for c in (COL_ID, COL_STATE, COL_TARGET) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in label CSV: {missing}")

    df[COL_ID] = df[COL_ID].astype(str).str.strip()
    df[COL_STATE] = df[COL_STATE].astype(str).str.strip()
    df[COL_TARGET] = df[COL_TARGET].astype(str).str.strip()
    df = df.replace({"nan": np.nan, "NaN": np.nan})
    return df.set_index(COL_ID)


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


def load_descriptors(
    desc_root: Path,
    *,
    label_ids: pd.Index,
    float32_mode: bool,
) -> Dict[str, pd.DataFrame]:
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
    float32_mode
        If True, store loaded descriptor matrices as ``float32``.

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

        if float32_mode:
            df = df.astype(np.float32, copy=False)

        if df.shape[1] == 0:
            LOGGER.warning("Descriptor '%s' has no numeric columns. Skipped.", name)
            continue
        out[name] = df

    if not out:
        raise RuntimeError(f"No descriptor CSVs found under: {desc_root}")

    LOGGER.info("Loaded %d descriptors.", len(out))
    for k, v in out.items():
        head = v.index[:3].tolist()
        LOGGER.info(
            "  - %s: %s (id_head=%s, dtype=%s)",
            k,
            v.shape,
            head,
            str(v.dtypes.iloc[0]),
        )
    return out


def drop_any_nan_cols_train_only(Xtr: pd.DataFrame) -> List[str]:
    """
    Keep columns that contain no missing values in the training split.

    Parameters
    ----------
    Xtr
        Training feature matrix.

    Returns
    -------
    List[str]
        Names of columns retained after missing-value filtering.
    """
    if Xtr.shape[1] == 0:
        return []
    ok = ~Xtr.isna().any(axis=0)
    return Xtr.columns[ok].tolist()


def drop_low_variance_cols_train_only(Xtr: pd.DataFrame, var_thresh: float) -> List[str]:
    """
    Keep columns whose sample variance exceeds ``var_thresh``.

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
    keep = v[v > var_thresh].index.tolist()
    return keep


def _standardize_train_only(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute column-wise training means and sample standard deviations.

    Parameters
    ----------
    X
        Numeric matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Column means and column standard deviations.
    """
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    return mu, sd


def select_low_correlation_cols_streaming_train_only(
    Xtr: pd.DataFrame,
    corr_thresh: float,
    *,
    use_abs: bool,
    block_size: int,
    keep_first: int,
    max_keep: int,
    compute_dtype: str,
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
    compute_dtype
        Storage dtype used for initial array conversion.

    Returns
    -------
    List[str]
        Names of retained columns.
    """
    if Xtr.shape[1] == 0:
        return []
    if Xtr.shape[1] == 1:
        return list(Xtr.columns)

    dtype = np.float64 if compute_dtype == "float64" else np.float32
    X = Xtr.to_numpy(dtype=dtype, copy=False)
    n, _ = X.shape
    cols_all = np.asarray(Xtr.columns, dtype=object)

    finite_mask = np.isfinite(X).all(axis=0)
    X = X[:, finite_mask]
    cols = cols_all[finite_mask]
    if X.shape[1] <= 1:
        return cols.astype(str).tolist()

    mu, sd = _standardize_train_only(X.astype(np.float64, copy=False))
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    X = X[:, ok]
    cols = cols[ok]
    sd = sd[ok]
    if X.shape[1] <= 1:
        return cols.astype(str).tolist()

    Z = (X.astype(np.float64, copy=False) - mu[ok]) / sd
    order = np.argsort(sd)[::-1]
    denom = max(n - 1, 1)

    initial_take = min(max(int(keep_first), 1), len(order))
    keep_idx: List[int] = order[:initial_take].tolist()
    K = Z[:, keep_idx].copy()

    if max_keep and len(keep_idx) >= max_keep:
        return cols[keep_idx].astype(str).tolist()

    rest = order[initial_take:]
    for start in range(0, len(rest), int(block_size)):
        if max_keep and len(keep_idx) >= max_keep:
            break

        blk = rest[start:start + int(block_size)]
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
        if max_keep:
            space = max_keep - len(keep_idx)
            if space <= 0:
                break
            if kept_blk.size > space:
                kept_blk = kept_blk[:space]

        keep_idx.extend(kept_blk.tolist())
        K = np.concatenate([K, Z[:, kept_blk]], axis=1)

    return cols[keep_idx].astype(str).tolist()


def make_master_ids(y: pd.Series, descriptors: Dict[str, pd.DataFrame]) -> pd.Index:
    """
    Compute the intersection of label IDs and all descriptor IDs.

    Parameters
    ----------
    y
        Label vector indexed by sample ID.
    descriptors
        Descriptor matrices indexed by sample ID.

    Returns
    -------
    pd.Index
        Common IDs shared by labels and all descriptor matrices.
    """
    common = y.index
    for X in descriptors.values():
        common = common.intersection(X.index)
    return common


def build_outer_splits(y_master: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build outer repeated stratified cross-validation splits.

    Parameters
    ----------
    y_master
        Target vector for the current state.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Outer train/test index pairs.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=OUTER_SEED,
    )
    dummy_X = np.zeros((len(y_master), 1))
    return list(rskf.split(dummy_X, y_master.to_numpy()))


def fold_names(n_folds: int) -> List[str]:
    """
    Generate fold names in the form ``r<repeat>_f<fold>``.

    Parameters
    ----------
    n_folds
        Total number of outer folds.

    Returns
    -------
    List[str]
        Fold labels.
    """
    names: List[str] = []
    for i in range(n_folds):
        rep = i // OUTER_SPLITS + 1
        fold = i % OUTER_SPLITS + 1
        names.append(f"r{rep}_f{fold}")
    return names


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
        raw = explainer.shap_values(X)
    except Exception:
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
    shap_arr = _extract_positive_class_shap_array(
        model,
        X_eval,
        positive_label=positive_label,
    )

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


def empty_metrics(state: str, descriptor: str, fold: str) -> dict:
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
        "train_acc": np.nan,
        "test_acc": np.nan,
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
        "n_train": 0,
        "n_test": 0,
        "max_abs_train": np.nan,
    }


def run_fold(
    *,
    state: str,
    descriptor: str,
    fold: str,
    X_master: pd.DataFrame,
    y_master: pd.Series,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    out_dir: Path,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
    filter_cfg: FeatureFilterConfig,
    rf_n_jobs: int,
    grid_n_jobs: int,
    no_corr_prune: bool,
    compute_shap: bool,
    save_shap_matrix: bool,
    shap_on: str,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    """
    Run a single outer fold for one descriptor and one spin state.

    The function applies training-only filtering, performs inner-grid search
    for RandomForest hyperparameters, evaluates train and test performance,
    and optionally computes SHAP summaries.

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
    rf_n_jobs
        ``n_jobs`` passed to RandomForestClassifier.
    grid_n_jobs
        ``n_jobs`` passed to GridSearchCV.
    no_corr_prune
        If True, disable correlation pruning.
    compute_shap
        If True, compute SHAP outputs.
    save_shap_matrix
        If True, save raw SHAP values.
    shap_on
        Dataset split used for SHAP computation, either ``"train"`` or ``"test"``.

    Returns
    -------
    Tuple[dict, Optional[pd.DataFrame]]
        Metrics dictionary and optional SHAP summary table.
    """
    ensure_dir(out_dir)

    X_tr_raw = X_master.iloc[tr_idx].copy()
    X_te_raw = X_master.iloc[te_idx].copy()
    y_tr = y_master.iloc[tr_idx].copy()
    y_te = y_master.iloc[te_idx].copy()

    pd.Series(X_tr_raw.index, name=COL_ID).to_csv(
        out_dir / f"train_ids_{fold}.csv",
        index=False,
        header=True,
    )
    pd.Series(X_te_raw.index, name=COL_ID).to_csv(
        out_dir / f"test_ids_{fold}.csv",
        index=False,
        header=True,
    )

    X_tr = X_tr_raw.replace([np.inf, -np.inf], np.nan)
    X_te = X_te_raw.replace([np.inf, -np.inf], np.nan)

    X_tr = X_tr.mask(np.abs(X_tr.astype(np.float64, copy=False)) > FLOAT32_MAX, np.nan)
    X_te = X_te.mask(np.abs(X_te.astype(np.float64, copy=False)) > FLOAT32_MAX, np.nan)

    try:
        max_abs_train = float(np.nanmax(np.abs(X_tr.to_numpy(dtype=np.float64, copy=False))))
    except Exception:
        max_abs_train = np.nan

    keep_no_nan = drop_any_nan_cols_train_only(X_tr)
    if not keep_no_nan:
        LOGGER.warning(
            "%s/%s/%s: no features after TRAIN NaN-column removal.",
            state,
            descriptor,
            fold,
        )
        m = empty_metrics(state, descriptor, fold)
        m["max_abs_train"] = max_abs_train
        return m, None

    X_tr = X_tr.loc[:, keep_no_nan]
    X_te = X_te.loc[:, keep_no_nan]

    keep_var = drop_low_variance_cols_train_only(X_tr, filter_cfg.var_thresh)
    if not keep_var:
        LOGGER.warning(
            "%s/%s/%s: no features after TRAIN variance filter.",
            state,
            descriptor,
            fold,
        )
        m = empty_metrics(state, descriptor, fold)
        m["max_abs_train"] = max_abs_train
        return m, None

    X_tr = X_tr.loc[:, keep_var]
    X_te = X_te.loc[:, keep_var]

    if no_corr_prune or X_tr.shape[1] <= 1:
        keep_corr = list(X_tr.columns)
    else:
        keep_corr = select_low_correlation_cols_streaming_train_only(
            X_tr,
            filter_cfg.corr_thresh,
            use_abs=filter_cfg.use_abs_corr,
            block_size=filter_cfg.corr_block_size,
            keep_first=filter_cfg.corr_keep_first,
            max_keep=filter_cfg.corr_max_keep,
            compute_dtype=filter_cfg.corr_compute_dtype,
        )

    if not keep_corr:
        LOGGER.warning(
            "%s/%s/%s: no features after TRAIN correlation pruning.",
            state,
            descriptor,
            fold,
        )
        m = empty_metrics(state, descriptor, fold)
        m["max_abs_train"] = max_abs_train
        return m, None

    X_tr_fin = X_tr.loc[:, keep_corr]
    X_te_fin = X_te.loc[:, keep_corr]

    if not np.isfinite(X_tr_fin.to_numpy(dtype=np.float64, copy=False)).all():
        LOGGER.warning(
            "%s/%s/%s: non-finite remains in TRAIN after filtering; skipping.",
            state,
            descriptor,
            fold,
        )
        m = empty_metrics(state, descriptor, fold)
        m["max_abs_train"] = max_abs_train
        return m, None

    if not np.isfinite(X_te_fin.to_numpy(dtype=np.float64, copy=False)).all():
        LOGGER.warning(
            "%s/%s/%s: non-finite remains in TEST after filtering; skipping.",
            state,
            descriptor,
            fold,
        )
        m = empty_metrics(state, descriptor, fold)
        m["max_abs_train"] = max_abs_train
        return m, None

    pd.Series(X_tr_fin.columns, name="feature").to_csv(
        out_dir / "selected_features.csv",
        index=False,
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=INNER_SEED,
    )

    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=OUTER_SEED,
        n_jobs=rf_n_jobs,
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid={
            "n_estimators": list(n_estimators_grid),
            "max_features": list(max_features_grid),
        },
        cv=inner_cv,
        scoring=make_scorer(matthews_corrcoef),
        n_jobs=grid_n_jobs,
        error_score="raise",
    )

    grid.fit(X_tr_fin, y_tr)
    model: RandomForestClassifier = grid.best_estimator_

    ytr_pred = model.predict(X_tr_fin)
    yte_pred = model.predict(X_te_fin)

    metrics = {
        "state": state,
        "descriptor": descriptor,
        "fold": fold,
        "n_features": int(X_tr_fin.shape[1]),
        "n_train": int(X_tr_fin.shape[0]),
        "n_test": int(X_te_fin.shape[0]),
        "max_abs_train": max_abs_train,
        "train_acc": float(accuracy_score(y_tr, ytr_pred)),
        "test_acc": float(accuracy_score(y_te, yte_pred)),
        "train_mcc": float(matthews_corrcoef(y_tr, ytr_pred)),
        "test_mcc": float(matthews_corrcoef(y_te, yte_pred)),
        "train_precision": float(
            precision_score(y_tr, ytr_pred, average="weighted", zero_division=0)
        ),
        "test_precision": float(
            precision_score(y_te, yte_pred, average="weighted", zero_division=0)
        ),
        "train_recall": float(
            recall_score(y_tr, ytr_pred, average="weighted", zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_te, yte_pred, average="weighted", zero_division=0)
        ),
        "train_f1": float(
            f1_score(y_tr, ytr_pred, average="weighted", zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_te, yte_pred, average="weighted", zero_division=0)
        ),
        "best_n_estimators": int(model.n_estimators),
        "best_max_features": None if model.max_features is None else str(model.max_features),
    }

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {"feature": X_tr_fin.columns, "importance": model.feature_importances_}
    ).to_csv(out_dir / "importance.csv", index=False)

    shap_mean_df: Optional[pd.DataFrame] = None
    if compute_shap:
        X_for_shap = X_te_fin if shap_on == "test" else X_tr_fin
        shap_mean_df = save_fold_shap_outputs(
            model=model,
            X_eval=X_for_shap,
            fold_dir=out_dir,
            positive_label=None,
            save_shap_matrix=save_shap_matrix,
        )

    return metrics, shap_mean_df


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


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    p = argparse.ArgumentParser(
        description="Nested-CV RF (leak-free, no impute) with MBTR-safe corr pruning."
    )
    p.add_argument("--labels", type=Path, default=None, help="Path to FeN6-SSD_500_spin_labeled.csv")
    p.add_argument("--descriptors", type=Path, default=None, help="Path to descriptors/ directory")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: ./results_rf)")

    p.add_argument("--states", nargs="+", default=list(DEFAULT_STATES), help="Spin states to run.")
    p.add_argument("--descs", nargs="*", default=None, help="Run only these descriptor names (folder names).")

    p.add_argument("--n-estimators", nargs="+", type=int, default=list(DEFAULT_N_ESTIMATORS))
    p.add_argument("--max-features", nargs="+", default=list(DEFAULT_MAX_FEATURES))

    p.add_argument("--var-thresh", type=float, default=VAR_THRESH_DEFAULT)
    p.add_argument("--corr-thresh", type=float, default=CORR_THRESH_DEFAULT)
    p.add_argument("--signed-corr", action="store_true", help="Use signed correlation (default abs).")

    p.add_argument("--no-corr-prune", action="store_true", help="Disable correlation pruning.")
    p.add_argument("--corr-block-size", type=int, default=512, help="Block size for streaming corr pruning.")
    p.add_argument("--corr-keep-first", type=int, default=1, help="Keep at least this many top-variance features.")
    p.add_argument(
        "--corr-max-keep",
        type=int,
        default=0,
        help="Safety cap for number of kept features. 0 means no cap.",
    )
    p.add_argument(
        "--corr-dtype",
        choices=["float64", "float32"],
        default="float64",
        help="dtype for correlation computation",
    )

    p.add_argument("--rf-n-jobs", type=int, default=1, help="n_jobs for RandomForestClassifier")
    p.add_argument("--grid-n-jobs", type=int, default=1, help="n_jobs for GridSearchCV")
    p.add_argument("--float32-mode", action="store_true", help="Store descriptor matrices as float32")

    p.add_argument("--compute-shap", action="store_true", help="Compute fold-wise SHAP summaries.")
    p.add_argument("--save-shap-matrix", action="store_true", help="Also save raw SHAP matrix as shap_values.csv.gz.")
    p.add_argument("--shap-on", choices=["train", "test"], default="test", help="Compute SHAP on train or test fold.")

    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """
    Run nested cross-validation for all requested states and descriptors.

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

    filter_cfg = FeatureFilterConfig(
        var_thresh=float(args.var_thresh),
        corr_thresh=float(args.corr_thresh),
        use_abs_corr=not bool(args.signed_corr),
        corr_block_size=int(args.corr_block_size),
        corr_keep_first=int(args.corr_keep_first),
        corr_max_keep=int(args.corr_max_keep),
        corr_compute_dtype=str(args.corr_dtype),
    )

    LOGGER.info("Repo root     : %s", ROOT_DIR)
    LOGGER.info("Labels        : %s", labels_path)
    LOGGER.info("Descriptors   : %s", desc_root)
    LOGGER.info("Output        : %s", out_root)
    LOGGER.info(
        "Grid          : n_estimators=%s, max_features=%s",
        n_estimators_grid,
        [("None" if v is None else v) for v in max_features_grid],
    )
    LOGGER.info(
        "Filtering     : train-only drop NaN/inf/huge; var_thresh=%g; corr_thresh=%g; corr=%s",
        filter_cfg.var_thresh,
        filter_cfg.corr_thresh,
        "abs" if filter_cfg.use_abs_corr else "signed",
    )
    LOGGER.info(
        "Corr-prune    : %s | block=%d | keep_first=%d | max_keep=%s | dtype=%s",
        "OFF" if args.no_corr_prune else "ON(streaming)",
        filter_cfg.corr_block_size,
        filter_cfg.corr_keep_first,
        ("NO_CAP" if filter_cfg.corr_max_keep == 0 else str(filter_cfg.corr_max_keep)),
        filter_cfg.corr_compute_dtype,
    )
    LOGGER.info(
        "Thread env    : OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
    )
    LOGGER.info(
        "Parallel      : RF n_jobs=%d | GridSearch n_jobs=%d",
        int(args.rf_n_jobs),
        int(args.grid_n_jobs),
    )
    LOGGER.info("Float32 mode  : %s", "ON" if args.float32_mode else "OFF")
    LOGGER.info(
        "SHAP          : %s | on=%s | save_matrix=%s",
        "ON" if args.compute_shap else "OFF",
        args.shap_on,
        "ON" if args.save_shap_matrix else "OFF",
    )

    labels = read_labels(labels_path)
    descriptors = load_descriptors(
        desc_root,
        label_ids=labels.index,
        float32_mode=bool(args.float32_mode),
    )

    if args.descs:
        wanted = set(args.descs)
        descriptors = {k: v for k, v in descriptors.items() if k in wanted}
        LOGGER.info(
            "Filtered descriptors: %d -> %s",
            len(descriptors),
            sorted(descriptors.keys()),
        )
        if not descriptors:
            raise RuntimeError(f"--descs specified but none matched. Wanted={sorted(wanted)}")

    for state in args.states:
        LOGGER.info("State: %s", state)

        y_all = labels.loc[labels[COL_STATE] == state, COL_TARGET].copy()
        y_all = y_all.replace({"nan": np.nan, "NaN": np.nan}).dropna()
        if y_all.empty:
            LOGGER.warning("No labeled rows found for state '%s'. Skipped.", state)
            continue

        master_ids = make_master_ids(y_all, descriptors)
        if len(master_ids) == 0:
            LOGGER.warning(
                "No common IDs across labels and descriptors for state '%s'. Skipped.",
                state,
            )
            continue

        y_master = y_all.loc[master_ids]
        splits = build_outer_splits(y_master)
        fnames = fold_names(len(splits))

        X_master_dict: Dict[str, pd.DataFrame] = {
            d: X.reindex(master_ids) for d, X in descriptors.items()
        }

        rows: List[dict] = []
        for desc_name, Xmat in X_master_dict.items():
            LOGGER.info("Descriptor: %s | X=%s", desc_name, Xmat.shape)

            fold_shap_rows: List[pd.DataFrame] = []

            for (tr, te), fname in zip(splits, fnames):
                fold_out = out_root / state / desc_name / fname
                res, shap_mean_df = run_fold(
                    state=state,
                    descriptor=desc_name,
                    fold=fname,
                    X_master=Xmat,
                    y_master=y_master,
                    tr_idx=tr,
                    te_idx=te,
                    out_dir=fold_out,
                    n_estimators_grid=n_estimators_grid,
                    max_features_grid=max_features_grid,
                    filter_cfg=filter_cfg,
                    rf_n_jobs=int(args.rf_n_jobs),
                    grid_n_jobs=int(args.grid_n_jobs),
                    no_corr_prune=bool(args.no_corr_prune),
                    compute_shap=bool(args.compute_shap),
                    save_shap_matrix=bool(args.save_shap_matrix),
                    shap_on=str(args.shap_on),
                )
                rows.append(res)

                if shap_mean_df is not None:
                    tmp = shap_mean_df.copy()
                    tmp["fold"] = fname
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
            summary_path = out_root / f"summary_{state}.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            LOGGER.info("Saved: %s", summary_path)


if __name__ == "__main__":
    main()