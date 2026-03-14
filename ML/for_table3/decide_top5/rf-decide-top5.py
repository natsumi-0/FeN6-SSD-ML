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
Leaky Top-K feature-selection pipeline for supplementary-information use, with
optional SHAP outputs.

This script intentionally performs feature filtering on the full dataset before
cross-validation. For each spin state and descriptor, the workflow is:

1. Build the dataset
2. Apply global feature filtering once on the full dataset
3. Run repeated 5-fold cross-validation
4. Save per-fold feature importances
5. Optionally save per-fold SHAP summaries
6. Save mean feature importance across folds
7. Save the Top-K feature list

This design is intentionally leaky and should not be interpreted as providing
an unbiased estimate of generalization performance.

Outputs
-------
<out_dir>/<spin>/<descriptor>/
    global_selected_features.csv
    importance_mean.csv
    shap_mean_abs_over_folds.csv         # if --compute-shap
    top5_<spin>_<descriptor>.csv
    r1_f1/
        train_ids_r1_f1.csv
        test_ids_r1_f1.csv
        metrics.csv
        importance.csv
        shap_mean_abs.csv                # if --compute-shap
        shap_values.csv.gz               # if --compute-shap --save-shap-matrix
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
SCRIPT_DIR = THIS_FILE.parent
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

TOPK_DEFAULT = 5

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


ROOT_DIR = find_repo_root(SCRIPT_DIR)
DEFAULT_LABEL_CSV = ROOT_DIR / "FeN6-SSD" / "FeN6-SSD_500_spin_labeled.csv"
DEFAULT_DESC_DIR = ROOT_DIR / "descriptors"
DEFAULT_OUT_DIR = SCRIPT_DIR / "results_globalfilter_topk"


@dataclass(frozen=True)
class FeatureFilterConfig:
    """
    Configuration for global feature filtering.

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
        pd.Series(cfg).to_json(indent=2, force_ascii=False),
        encoding="utf-8",
    )


def read_labels(path: Path) -> pd.DataFrame:
    """
    Read and normalize the label table.

    Parameters
    ----------
    path
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
    df = pd.read_csv(path)
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
        LOGGER.info("  - %s: %s", k, v.shape)
    return out


def drop_any_nan_cols(X: pd.DataFrame) -> List[str]:
    """
    Keep columns that contain no missing values.

    Parameters
    ----------
    X
        Feature matrix.

    Returns
    -------
    List[str]
        Names of columns retained after missing-value filtering.
    """
    if X.shape[1] == 0:
        return []
    ok = ~X.isna().any(axis=0)
    return X.columns[ok].tolist()


def drop_float32_unsafe_cols(X: pd.DataFrame) -> List[str]:
    """
    Keep columns whose values are finite and within float32 range.

    Parameters
    ----------
    X
        Feature matrix.

    Returns
    -------
    List[str]
        Names of retained columns.
    """
    if X.shape[1] == 0:
        return []
    A = X.to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(A).all(axis=0) & (np.abs(A) <= FLOAT32_MAX).all(axis=0)
    return X.columns[ok].tolist()


def drop_low_variance_cols(X: pd.DataFrame, var_thresh: float) -> List[str]:
    """
    Keep columns whose sample variance exceeds ``var_thresh``.

    Parameters
    ----------
    X
        Feature matrix.
    var_thresh
        Variance threshold.

    Returns
    -------
    List[str]
        Names of retained columns.
    """
    if X.shape[1] == 0:
        return []
    v = X.astype(np.float64, copy=False).var(axis=0, ddof=1)
    return v[v > var_thresh].index.tolist()


def _standardize_np(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute column-wise means and sample standard deviations.

    Parameters
    ----------
    A
        Numeric matrix.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Column means and column standard deviations.
    """
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0, ddof=1)
    return mu, sd


def select_low_correlation_cols_streaming(
    X: pd.DataFrame,
    corr_thresh: float,
    *,
    use_abs: bool,
    block_size: int,
    keep_first: int,
    max_keep: int,
    compute_dtype: str,
) -> List[str]:
    """
    Perform greedy correlation pruning without forming a full correlation matrix.

    Features are ordered by decreasing standard deviation. A seed set of
    ``keep_first`` columns is retained first, after which remaining columns are
    processed blockwise. Exact correlations are computed between retained
    columns and candidate blocks after standardization.

    Parameters
    ----------
    X
        Feature matrix.
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
    if X.shape[1] == 0:
        return []
    if X.shape[1] == 1:
        return list(X.columns)

    dtype = np.float64 if compute_dtype == "float64" else np.float32
    A = X.to_numpy(dtype=dtype, copy=False)
    cols_all = np.asarray(X.columns, dtype=object)

    finite_cols = np.isfinite(A).all(axis=0)
    A = A[:, finite_cols]
    cols = cols_all[finite_cols]
    if A.shape[1] <= 1:
        return cols.astype(str).tolist()

    mu, sd = _standardize_np(A.astype(np.float64, copy=False))
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    A = A[:, ok]
    cols = cols[ok]
    sd = sd[ok]
    if A.shape[1] <= 1:
        return cols.astype(str).tolist()

    Z = (A.astype(np.float64, copy=False) - mu[ok]) / sd
    order = np.argsort(sd)[::-1]
    denom = max(Z.shape[0] - 1, 1)

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


def preprocess_global_leaky(
    X: pd.DataFrame,
    cfg: FeatureFilterConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply intentionally global feature filtering to the full dataset.

    Steps:
    1. Replace inf values with NaN
    2. Mask excessively large values to NaN
    3. Remove columns containing NaN
    4. Remove float32-unsafe columns
    5. Remove low-variance columns
    6. Apply streaming correlation pruning

    Parameters
    ----------
    X
        Full feature matrix.
    cfg
        Feature-filter configuration.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Globally filtered feature matrix and retained column names.
    """
    Xg = X.replace([np.inf, -np.inf], np.nan)
    Xg = Xg.mask(np.abs(Xg.astype(np.float64, copy=False)) > FLOAT32_MAX, np.nan)

    keep_no_nan = drop_any_nan_cols(Xg)
    Xg = Xg.loc[:, keep_no_nan]
    if Xg.shape[1] == 0:
        return Xg, []

    keep_safe = drop_float32_unsafe_cols(Xg)
    Xg = Xg.loc[:, keep_safe]
    if Xg.shape[1] == 0:
        return Xg, []

    keep_var = drop_low_variance_cols(Xg, cfg.var_thresh)
    Xg = Xg.loc[:, keep_var]
    if Xg.shape[1] == 0:
        return Xg, []

    keep_corr = select_low_correlation_cols_streaming(
        Xg,
        cfg.corr_thresh,
        use_abs=cfg.use_abs_corr,
        block_size=cfg.corr_block_size,
        keep_first=cfg.corr_keep_first,
        max_keep=cfg.corr_max_keep,
        compute_dtype=cfg.corr_compute_dtype,
    )
    Xg = Xg.loc[:, keep_corr]
    return Xg, list(Xg.columns)


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


def build_outer_splits(y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build repeated stratified outer cross-validation splits.

    Parameters
    ----------
    y
        Target vector.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        Outer train/test index pairs.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=OUTER_SEED,
    )
    dummy = np.zeros((len(y), 1))
    return list(cv.split(dummy, y.to_numpy()))


def fold_name(i: int) -> str:
    """
    Generate a fold name in the form ``r<repeat>_f<fold>``.

    Parameters
    ----------
    i
        Zero-based fold index across all repeats.

    Returns
    -------
    str
        Fold label.
    """
    rep = i // OUTER_SPLITS + 1
    f = i % OUTER_SPLITS + 1
    return f"r{rep}_f{f}"


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
    shap_on: str,
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
    shap_on
        Label describing whether SHAP was computed on train or test data.

    Returns
    -------
    pd.DataFrame
        Table of mean absolute SHAP values per feature.
    """
    X_shap = X_eval

    shap_arr = _extract_positive_class_shap_array(
        model,
        X_shap,
        positive_label=positive_label,
    )

    mean_abs = np.abs(shap_arr).mean(axis=0)
    shap_mean_df = pd.DataFrame(
        {
            "feature": X_shap.columns.astype(str),
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_mean_df.to_csv(fold_dir / "shap_mean_abs.csv", index=False)

    if save_shap_matrix:
        shap_mat = pd.DataFrame(
            shap_arr.astype(np.float32),
            index=X_shap.index.astype(str),
            columns=X_shap.columns.astype(str),
        )
        shap_mat.index.name = COL_ID
        shap_mat.to_csv(fold_dir / "shap_values.csv.gz", compression="gzip")

    return shap_mean_df


def run_fixed_feature_cv(
    *,
    state: str,
    descriptor: str,
    X_fixed: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
    grid_n_jobs: int,
    rf_n_jobs: int,
    compute_shap: bool,
    save_shap_matrix: bool,
    shap_on: str,
) -> Tuple[List[dict], pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run repeated cross-validation on a fixed globally filtered feature set.

    Parameters
    ----------
    state
        Spin state.
    descriptor
        Descriptor name.
    X_fixed
        Globally filtered feature matrix.
    y
        Target vector aligned to ``X_fixed``.
    splits
        Outer train/test index pairs.
    out_dir
        Descriptor output directory.
    n_estimators_grid
        Candidate values for ``n_estimators``.
    max_features_grid
        Candidate values for ``max_features``.
    grid_n_jobs
        ``n_jobs`` passed to GridSearchCV.
    rf_n_jobs
        ``n_jobs`` passed to RandomForestClassifier.
    compute_shap
        If True, compute SHAP outputs.
    save_shap_matrix
        If True, save raw SHAP values.
    shap_on
        Dataset split used for SHAP computation, either ``"train"`` or ``"test"``.

    Returns
    -------
    Tuple[List[dict], pd.DataFrame, Optional[pd.DataFrame]]
        Per-fold metrics, mean importance table, and optional aggregated SHAP table.
    """
    ensure_dir(out_dir)

    rows: List[dict] = []
    imp_all: List[pd.DataFrame] = []
    shap_all: List[pd.DataFrame] = []

    for i, (tr_idx, te_idx) in enumerate(splits):
        fold = fold_name(i)
        fold_dir = out_dir / fold
        ensure_dir(fold_dir)

        X_tr = X_fixed.iloc[tr_idx].copy()
        X_te = X_fixed.iloc[te_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_te = y.iloc[te_idx].copy()

        pd.Series(X_tr.index, name=COL_ID).to_csv(fold_dir / f"train_ids_{fold}.csv", index=False)
        pd.Series(X_te.index, name=COL_ID).to_csv(fold_dir / f"test_ids_{fold}.csv", index=False)

        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)

        grid = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=OUTER_SEED,
                n_jobs=rf_n_jobs,
            ),
            param_grid={
                "n_estimators": list(n_estimators_grid),
                "max_features": list(max_features_grid),
            },
            cv=inner,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=int(grid_n_jobs),
            error_score="raise",
        )
        grid.fit(X_tr, y_tr)

        model: RandomForestClassifier = grid.best_estimator_
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

        mrow = {
            "state": state,
            "descriptor": descriptor,
            "fold": fold,
            "n_features": int(X_tr.shape[1]),
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "train_acc": float(accuracy_score(y_tr, y_tr_pred)),
            "test_acc": float(accuracy_score(y_te, y_te_pred)),
            "train_mcc": float(matthews_corrcoef(y_tr, y_tr_pred)),
            "test_mcc": float(matthews_corrcoef(y_te, y_te_pred)),
            "train_precision": float(precision_score(y_tr, y_tr_pred, average="weighted", zero_division=0)),
            "test_precision": float(precision_score(y_te, y_te_pred, average="weighted", zero_division=0)),
            "train_recall": float(recall_score(y_tr, y_tr_pred, average="weighted", zero_division=0)),
            "test_recall": float(recall_score(y_te, y_te_pred, average="weighted", zero_division=0)),
            "train_f1": float(f1_score(y_tr, y_tr_pred, average="weighted", zero_division=0)),
            "test_f1": float(f1_score(y_te, y_te_pred, average="weighted", zero_division=0)),
            "best_n_estimators": int(model.n_estimators),
            "best_max_features": None if model.max_features is None else str(model.max_features),
        }
        pd.DataFrame([mrow]).to_csv(fold_dir / "metrics.csv", index=False)

        imp_df = pd.DataFrame(
            {
                "feature": X_fixed.columns.astype(str),
                "importance": model.feature_importances_,
                "fold": fold,
            }
        )
        imp_df.to_csv(fold_dir / "importance.csv", index=False)

        rows.append(mrow)
        imp_all.append(imp_df)

        if compute_shap:
            X_for_shap = X_te if shap_on == "test" else X_tr
            shap_mean_df = save_fold_shap_outputs(
                model=model,
                X_eval=X_for_shap,
                fold_dir=fold_dir,
                positive_label=None,
                save_shap_matrix=save_shap_matrix,
                shap_on=shap_on,
            )
            shap_mean_df["fold"] = fold
            shap_all.append(shap_mean_df)

    imp_cat = pd.concat(imp_all, ignore_index=True)
    imp_mean = (
        imp_cat.groupby("feature", as_index=False)["importance"]
        .mean()
        .rename(columns={"importance": "importance_mean"})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    imp_mean.to_csv(out_dir / "importance_mean.csv", index=False)

    shap_mean_over_folds: Optional[pd.DataFrame] = None
    if compute_shap and shap_all:
        shap_cat = pd.concat(shap_all, ignore_index=True)
        shap_mean_over_folds = (
            shap_cat.groupby("feature", as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_mean_over_folds.to_csv(out_dir / "shap_mean_abs_over_folds.csv", index=False)

    return rows, imp_mean, shap_mean_over_folds


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    p = argparse.ArgumentParser(
        description="Leaky global-filter RF pipeline for Top-K feature selection."
    )
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--descriptors", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)

    p.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    p.add_argument("--descs", nargs="*", default=None)

    p.add_argument("--topk", type=int, default=TOPK_DEFAULT)

    p.add_argument("--n-estimators", nargs="+", type=int, default=list(DEFAULT_N_ESTIMATORS))
    p.add_argument("--max-features", nargs="+", default=list(DEFAULT_MAX_FEATURES))

    p.add_argument("--var-thresh", type=float, default=VAR_THRESH_DEFAULT)
    p.add_argument("--corr-thresh", type=float, default=CORR_THRESH_DEFAULT)
    p.add_argument("--signed-corr", action="store_true")
    p.add_argument("--corr-block-size", type=int, default=512)
    p.add_argument("--corr-keep-first", type=int, default=1)
    p.add_argument("--corr-max-keep", type=int, default=0)
    p.add_argument("--corr-dtype", choices=["float64", "float32"], default="float64")

    p.add_argument("--grid-n-jobs", type=int, default=1)
    p.add_argument("--rf-n-jobs", type=int, default=1)
    p.add_argument("--float32-mode", action="store_true")

    p.add_argument("--compute-shap", action="store_true", help="Compute fold-wise SHAP summaries.")
    p.add_argument("--save-shap-matrix", action="store_true", help="Also save sample-wise SHAP matrix (.csv.gz).")
    p.add_argument("--shap-on", choices=["train", "test"], default="test", help="Compute SHAP on train or test fold.")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """
    Run the leaky global-filter feature-selection workflow for all requested
    states and descriptors.

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

    LOGGER.info("Repo root    : %s", ROOT_DIR)
    LOGGER.info("Labels       : %s", labels_path)
    LOGGER.info("Descriptors  : %s", desc_root)
    LOGGER.info("Output       : %s", out_root)
    LOGGER.info("TopK         : %d", int(args.topk))
    LOGGER.info("Grid         : n_estimators=%s, max_features=%s", n_estimators_grid, max_features_grid)
    LOGGER.info(
        "Thread env   : OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
    )
    LOGGER.info("Parallel     : RF n_jobs=%d | GridSearch n_jobs=%d", int(args.rf_n_jobs), int(args.grid_n_jobs))
    LOGGER.info(
        "SHAP         : %s | on=%s | save_matrix=%s",
        "ON" if args.compute_shap else "OFF",
        args.shap_on,
        "ON" if args.save_shap_matrix else "OFF",
    )

    labels = read_labels(labels_path)
    descriptors = load_descriptors(desc_root, label_ids=labels.index, float32_mode=bool(args.float32_mode))

    if args.descs:
        wanted = set(args.descs)
        descriptors = {k: v for k, v in descriptors.items() if k in wanted}
        LOGGER.info("Filtered descriptors: %d -> %s", len(descriptors), sorted(descriptors.keys()))
        if not descriptors:
            raise RuntimeError(f"--descs specified but none matched. wanted={sorted(wanted)}")

    for state in args.states:
        LOGGER.info("State: %s", state)

        y_all = labels.loc[labels[COL_STATE] == state, COL_TARGET].copy()
        y_all = y_all.replace({"nan": np.nan, "NaN": np.nan}).dropna()
        if y_all.empty:
            LOGGER.warning("No labeled rows for state=%s. Skipped.", state)
            continue

        master_ids = make_master_ids(y_all, descriptors)
        if len(master_ids) == 0:
            LOGGER.warning("No common IDs across labels/descriptors for state=%s. Skipped.", state)
            continue

        y_master = y_all.loc[master_ids]
        splits = build_outer_splits(y_master)

        for desc_name, X in descriptors.items():
            LOGGER.info("Descriptor: %s", desc_name)

            X_master = X.reindex(master_ids)
            X_fixed, kept_cols = preprocess_global_leaky(X_master, filter_cfg)

            if X_fixed.shape[1] == 0:
                LOGGER.warning("No features after global filtering: state=%s desc=%s", state, desc_name)
                continue

            desc_out = out_root / state / desc_name
            ensure_dir(desc_out)

            pd.Series(kept_cols, name="feature").to_csv(desc_out / "global_selected_features.csv", index=False)

            rows, imp_mean, _ = run_fixed_feature_cv(
                state=state,
                descriptor=desc_name,
                X_fixed=X_fixed,
                y=y_master.loc[X_fixed.index],
                splits=splits,
                out_dir=desc_out,
                n_estimators_grid=n_estimators_grid,
                max_features_grid=max_features_grid,
                grid_n_jobs=int(args.grid_n_jobs),
                rf_n_jobs=int(args.rf_n_jobs),
                compute_shap=bool(args.compute_shap),
                save_shap_matrix=bool(args.save_shap_matrix),
                shap_on=str(args.shap_on),
            )

            top = imp_mean.head(int(args.topk)).copy()
            top.to_csv(desc_out / f"top{int(args.topk)}_{state}_{desc_name}.csv", index=False)

            pd.DataFrame(rows).to_csv(desc_out / "summary_folds.csv", index=False)
            LOGGER.info(
                "Saved state=%s desc=%s | p_global=%d | top%d ready",
                state, desc_name, int(X_fixed.shape[1]), int(args.topk)
            )


if __name__ == "__main__":
    main()