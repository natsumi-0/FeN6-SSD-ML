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
Leak-free nested cross-validation pipeline for RandomForest classification on
FeN6-SSD data, with fold-local full-model and Top-K-model evaluation.

This script performs repeated outer cross-validation with inner hyperparameter
tuning. For each outer fold, two stages are run:

1. A full-feature model trained on fold-local filtered features
2. A Top-K model trained on the fold-local top-ranked features selected from
   the Stage 1 model fitted on the outer-training split

All preprocessing and feature-selection steps are fit only on the outer
training data and then applied to the corresponding outer test data.

Main features
-------------
- Strict outer train/test separation
- Fold-local training-only feature filtering
- Fold-local Top-K feature selection from Stage 1 training fit
- Optional training-fit/test-transform imputation for Top-K models
- Optional SHAP summaries for both Stage 1 and Stage 2

Leak-free preprocessing policy
------------------------------
Feature filtering is computed on the outer training split only and then applied
to the test split. Depending on settings, fold-local filtering may include:
- Removal of float32-unsafe or non-finite columns
- Low-variance filtering
- Streaming correlation pruning without constructing a full correlation matrix

Typical run
-----------
python3 rf_outercv_full_top5_rewrite.py --filtering on --topk 5
"""

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

LOGGER = logging.getLogger(__name__)

COL_ID = "ccdc_id"
COL_STATE = "Spin-state of crystal structure"
COL_BEHAV = "Spin-state behaviour"

POSITIVE_LABEL = "spin-crossover"
SPIN_STATES = ("high-spin", "low-spin")

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


@dataclass(frozen=True)
class Paths:
    """
    Repository and output paths used by the pipeline.

    Parameters
    ----------
    root
        Repository root directory.
    labels_csv
        Path to the labels CSV file.
    descriptors_dir
        Path to the descriptors directory.
    out_dir
        Path to the output directory.
    """

    root: Path
    labels_csv: Path
    descriptors_dir: Path
    out_dir: Path


@dataclass(frozen=True)
class CorrPruneConfig:
    """
    Configuration for streaming correlation pruning.

    Parameters
    ----------
    corr_thresh
        Correlation threshold used for pruning.
    use_abs_corr
        If True, prune by absolute correlation; otherwise use signed correlation.
    block_size
        Number of candidate columns processed per block.
    keep_first
        Minimum number of top-variance features retained before blockwise pruning.
    max_keep
        Maximum number of retained features. A value of 0 disables the cap.
    """

    corr_thresh: float
    use_abs_corr: bool
    block_size: int
    keep_first: int
    max_keep: int


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
        "Repository root not found. Expected: descriptors/, ML/, "
        "FeN6-SSD/FeN6-SSD_500_spin_labeled.csv"
    )


def resolve_paths(args: argparse.Namespace) -> Paths:
    """
    Resolve repository-relative input and output paths.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
    Paths
        Resolved path bundle.
    """
    this_file = Path(__file__).resolve()
    root = find_repo_root(this_file.parent)
    labels_csv = args.labels or (root / "FeN6-SSD" / "FeN6-SSD_500_spin_labeled.csv")
    descriptors_dir = args.descriptors or (root / "descriptors")
    out_dir = args.out_dir or (this_file.parent / "results_outercv_full_topk")
    return Paths(root=root, labels_csv=labels_csv, descriptors_dir=descriptors_dir, out_dir=out_dir)


def write_run_config(out_dir: Path, args: argparse.Namespace) -> None:
    """
    Save runtime configuration and package versions as JSON.

    Parameters
    ----------
    out_dir
        Output directory.
    args
        Parsed command-line arguments.
    """
    ensure_dir(out_dir)
    out = {
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
    import sklearn  # noqa: WPS433

    out["sklearn"] = sklearn.__version__
    try:
        import shap  # type: ignore  # noqa: WPS433

        out["shap"] = shap.__version__
    except Exception:
        out["shap"] = None

    (out_dir / "run_config.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


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
    for c in (COL_ID, COL_STATE, COL_BEHAV):
        if c not in df.columns:
            raise KeyError(f"Missing required column in labels CSV: {c}")

    df[COL_ID] = df[COL_ID].astype(str).str.strip()
    df[COL_STATE] = df[COL_STATE].astype(str).str.strip()
    df[COL_BEHAV] = df[COL_BEHAV].astype(str).str.strip()
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


def load_descriptors(desc_root: Path, *, label_ids: pd.Index) -> Dict[str, pd.DataFrame]:
    """
    Load descriptor tables from subdirectories of ``desc_root``.

    Each descriptor is expected at:
        ``<desc_root>/<name>/<name>.csv``

    The returned data frames:
    - are indexed by inferred ID column
    - retain numeric columns only
    - replace inf values with NaN during loading

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
    for sub in sorted(desc_root.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = sub / f"{sub.name}.csv"
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
        df = df.replace([np.inf, -np.inf], np.nan)

        if df.shape[1] == 0:
            LOGGER.warning("Descriptor '%s' has no numeric columns. Skipped.", sub.name)
            continue
        out[sub.name] = df

    if not out:
        raise RuntimeError(f"No descriptor CSVs found under: {desc_root}")

    LOGGER.info("Loaded %d descriptors.", len(out))
    for k, v in out.items():
        LOGGER.info("  - %s: %s", k, v.shape)
    return out


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
        description="Leak-free nested-CV RF for FeN6-SSD (Stage① full -> Stage② fold-local TopK)."
    )
    p.add_argument("--labels", type=Path, default=None, help="Override labels CSV path.")
    p.add_argument("--descriptors", type=Path, default=None, help="Override descriptors/ directory.")
    p.add_argument("--out-dir", type=Path, default=None, help="Override output directory.")

    p.add_argument("--descs", nargs="*", default=None, help="Run only these descriptor names (folder names).")
    p.add_argument("--topk", type=int, default=5, help="K for Top-K (default: 5).")

    p.add_argument(
        "--filtering",
        choices=["on", "off", "both"],
        default="on",
        help="Fold-local filtering mode for Stage①. Default=on.",
    )
    p.add_argument(
        "--allow-nonfinite",
        action="store_true",
        help='If set and filtering="off", do NOT drop non-finite columns in Stage① (not recommended).',
    )

    p.add_argument("--var-thresh", type=float, default=VAR_THRESH_DEFAULT)
    p.add_argument("--corr-thresh", type=float, default=CORR_THRESH_DEFAULT)
    p.add_argument("--signed-corr", action="store_true", help="Use signed correlation (default: abs).")
    p.add_argument("--corr-block-size", type=int, default=512, help="Block size for streaming corr pruning.")
    p.add_argument("--corr-keep-first", type=int, default=1, help="Always keep first N features by variance.")
    p.add_argument(
        "--corr-max-keep",
        type=int,
        default=0,
        help="Optional cap on number of kept features after corr pruning. 0 means no cap.",
    )

    p.add_argument("--n-estimators", nargs="+", type=int, default=list(DEFAULT_N_ESTIMATORS))
    p.add_argument("--max-features", nargs="+", default=list(DEFAULT_MAX_FEATURES))

    p.add_argument("--grid-n-jobs", type=int, default=1, help="n_jobs for GridSearchCV.")
    p.add_argument("--rf-n-jobs", type=int, default=1, help="n_jobs for RandomForest.")

    p.add_argument(
        "--impute-strategy",
        choices=["off", "median", "mean", "most_frequent", "constant"],
        default="off",
        help="Top-K model imputation: TRAIN-fit -> TEST-transform. Default=off.",
    )
    p.add_argument("--impute-fill-value", type=float, default=0.0)

    p.add_argument("--no-shap", action="store_true", help="Disable SHAP (default: enabled if shap exists).")
    p.add_argument(
        "--shap-max-test-samples",
        type=int,
        default=0,
        help="If >0, compute SHAP using at most this many outer-test rows.",
    )

    p.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist.")
    p.add_argument("--log-level", default="INFO")
    return p


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


def make_master_ids(y_all: pd.Series, X_dict: Dict[str, pd.DataFrame]) -> pd.Index:
    """
    Compute the intersection of label IDs and all descriptor IDs.

    Parameters
    ----------
    y_all
        Label vector indexed by sample ID.
    X_dict
        Descriptor matrices indexed by sample ID.

    Returns
    -------
    pd.Index
        Common IDs shared by labels and all descriptor matrices.
    """
    common = y_all.index
    for X in X_dict.values():
        common = common.intersection(X.index)
    return common


def float32_safe_columns_mask_chunked(X: pd.DataFrame, *, block: int = 2048) -> np.ndarray:
    """
    Identify columns whose values are finite and within float32 range.

    Parameters
    ----------
    X
        Feature matrix.
    block
        Number of columns processed at once.

    Returns
    -------
    np.ndarray
        Boolean mask over columns.
    """
    p = X.shape[1]
    if p == 0:
        return np.zeros(0, dtype=bool)
    ok = np.ones(p, dtype=bool)
    thr = FLOAT32_MAX
    for start in range(0, p, int(block)):
        end = min(p, start + int(block))
        blk = X.iloc[:, start:end].to_numpy(dtype=np.float64, copy=False)
        ok[start:end] = np.isfinite(blk).all(axis=0) & (np.abs(blk) <= thr).all(axis=0)
    return ok


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
    return v[v > var_thresh].index.astype(str).tolist()


def select_low_correlation_cols_streaming(X: pd.DataFrame, cfg: CorrPruneConfig) -> List[str]:
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
    cfg
        Correlation-pruning configuration.

    Returns
    -------
    List[str]
        Names of retained columns.
    """
    if X.shape[1] <= 1:
        return list(X.columns.astype(str))

    A = X.to_numpy(dtype=np.float64, copy=False)
    cols_all = np.asarray(X.columns.astype(str), dtype=object)

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
    initial_take = min(max(int(cfg.keep_first), 1), len(order))
    init = order[:initial_take]
    keep_idx.extend(init.tolist())
    K = Z[:, init].copy()

    if cfg.max_keep > 0 and len(keep_idx) >= cfg.max_keep:
        keep_idx = keep_idx[: cfg.max_keep]
        return cols[keep_idx].astype(str).tolist()

    rest = order[initial_take:]
    for start in range(0, len(rest), int(cfg.block_size)):
        blk = rest[start:start + int(cfg.block_size)]
        if blk.size == 0:
            continue

        B = Z[:, blk]
        C = (K.T @ B) / denom
        if cfg.use_abs_corr:
            C = np.abs(C)

        maxcorr = np.nanmax(C, axis=0)
        good = np.isfinite(maxcorr) & (maxcorr < cfg.corr_thresh)
        if not np.any(good):
            continue

        kept_blk = blk[good]
        keep_idx.extend(kept_blk.tolist())
        K = np.concatenate([K, Z[:, kept_blk]], axis=1)

        if cfg.max_keep > 0 and len(keep_idx) >= cfg.max_keep:
            keep_idx = keep_idx[: cfg.max_keep]
            break

    return cols[keep_idx].astype(str).tolist()


def apply_foldlocal_filtering(
    *,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    filtering: bool,
    drop_nonfinite_always: bool,
    var_thresh: float,
    corr_cfg: CorrPruneConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply fold-local training-only filtering and align the test split.

    Parameters
    ----------
    X_tr
        Outer-training feature matrix.
    X_te
        Outer-test feature matrix.
    filtering
        Whether to apply variance and correlation filtering.
    drop_nonfinite_always
        Whether to always remove float32-unsafe or non-finite columns.
    var_thresh
        Variance threshold.
    corr_cfg
        Correlation-pruning configuration.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict]
        Filtered training matrix, filtered test matrix, and filtering statistics.
    """
    stats = {
        "n_input_features": int(X_tr.shape[1]),
        "n_after_float32_safe": 0,
        "n_after_variance": 0,
        "n_after_corr": 0,
    }

    if X_tr.shape[1] == 0:
        return X_tr.iloc[:, :0], X_te.iloc[:, :0], stats

    if (not filtering) and (not drop_nonfinite_always):
        stats["n_after_float32_safe"] = int(X_tr.shape[1])
        stats["n_after_variance"] = int(X_tr.shape[1])
        stats["n_after_corr"] = int(X_tr.shape[1])
        return X_tr, X_te, stats

    safe_mask = float32_safe_columns_mask_chunked(X_tr, block=2048)
    X_tr2 = X_tr.loc[:, safe_mask]
    X_te2 = X_te.loc[:, safe_mask]
    stats["n_after_float32_safe"] = int(X_tr2.shape[1])
    if X_tr2.shape[1] == 0:
        return X_tr2, X_te2, stats

    if not filtering:
        stats["n_after_variance"] = int(X_tr2.shape[1])
        stats["n_after_corr"] = int(X_tr2.shape[1])
        return X_tr2, X_te2, stats

    keep_var = drop_low_variance_cols(X_tr2, var_thresh)
    X_tr3 = X_tr2.loc[:, keep_var]
    X_te3 = X_te2.loc[:, keep_var]
    stats["n_after_variance"] = int(X_tr3.shape[1])
    if X_tr3.shape[1] == 0:
        return X_tr3, X_te3, stats

    keep_corr = select_low_correlation_cols_streaming(X_tr3, corr_cfg)
    X_tr4 = X_tr3.loc[:, keep_corr]
    X_te4 = X_te3.loc[:, keep_corr]
    stats["n_after_corr"] = int(X_tr4.shape[1])
    return X_tr4, X_te4, stats


def pick_topk_from_importance(features: Sequence[str], importances: np.ndarray, k: int) -> List[str]:
    """
    Select Top-K feature names by descending importance.

    Parameters
    ----------
    features
        Feature names.
    importances
        Importance values aligned to ``features``.
    k
        Number of features to retain.

    Returns
    -------
    List[str]
        Top-K feature names.
    """
    if k <= 0:
        return []
    if len(features) != len(importances):
        raise ValueError("features and importances length mismatch.")
    order = np.argsort(importances)[::-1]
    top_idx = order[: min(k, len(order))]
    return [str(features[i]) for i in top_idx]


def _assert_float32_safe_df(X: pd.DataFrame, *, context: str) -> None:
    """
    Validate that a matrix is finite and within float32 range.

    Parameters
    ----------
    X
        Feature matrix.
    context
        Context string used in raised error messages.

    Raises
    ------
    ValueError
        If non-finite values remain or values exceed float32 range.
    """
    A = X.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(A).all():
        raise ValueError(f"Non-finite values remain. {context}")
    if np.abs(A).max(initial=0.0) > FLOAT32_MAX:
        raise ValueError(f"Values exceed float32 max (>{FLOAT32_MAX:g}). {context}")


def fit_transform_impute_or_validate(
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    *,
    strategy: str,
    fill_value: float,
    context: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply training-fit/test-transform imputation or validate finiteness.

    Parameters
    ----------
    X_tr
        Training feature matrix.
    X_te
        Test feature matrix.
    strategy
        Imputation strategy, or ``"off"`` to disable imputation.
    fill_value
        Fill value used when ``strategy="constant"``.
    context
        Context string used in raised error messages.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Processed training and test matrices.

    Raises
    ------
    ValueError
        If imputation is disabled and missing values remain, or if output values
        are not float32-safe.
    """
    X_tr2 = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te2 = X_te.replace([np.inf, -np.inf], np.nan)

    if strategy == "off":
        if np.isnan(X_tr2.to_numpy()).any() or np.isnan(X_te2.to_numpy()).any():
            raise ValueError(f"Non-finite values found but imputation is OFF. {context}")
        _assert_float32_safe_df(X_tr2, context=context + " [train]")
        _assert_float32_safe_df(X_te2, context=context + " [test]")
        return X_tr2, X_te2

    if strategy == "constant":
        imp = SimpleImputer(strategy="constant", fill_value=fill_value)
    else:
        imp = SimpleImputer(strategy=strategy)

    Xt = pd.DataFrame(imp.fit_transform(X_tr2), index=X_tr2.index, columns=X_tr2.columns)
    Xv = pd.DataFrame(imp.transform(X_te2), index=X_te2.index, columns=X_te2.columns)

    _assert_float32_safe_df(Xt, context=context + " [train]")
    _assert_float32_safe_df(Xv, context=context + " [test]")
    return Xt, Xv


def _pick_positive_class_index(model: RandomForestClassifier) -> int:
    """
    Infer the positive-class index for SHAP extraction.

    Parameters
    ----------
    model
        Fitted RandomForest classifier.

    Returns
    -------
    int
        Positive-class index.
    """
    try:
        classes = [str(c).lower() for c in model.classes_]
        for i, c in enumerate(classes):
            if ("spin-crossover" in c) or ("crossover" in c) or (c == "sco"):
                return i
    except Exception:
        pass
    return 1


def compute_shap_importance_on_test(
    *,
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    out_path: Path,
    max_test_samples: int,
) -> None:
    """
    Compute and save SHAP feature summaries on the test split.

    Parameters
    ----------
    model
        Fitted RandomForest classifier.
    X_test
        Test feature matrix.
    out_path
        Output CSV path.
    max_test_samples
        Maximum number of test rows used for SHAP. A value of 0 uses all rows.
    """
    import shap  # type: ignore  # noqa: WPS433

    Xt = X_test
    if max_test_samples and max_test_samples > 0 and Xt.shape[0] > max_test_samples:
        Xt = Xt.iloc[:max_test_samples].copy()

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(Xt)

    if isinstance(shap_vals, list):
        pos_idx = _pick_positive_class_index(model)
        sv = shap_vals[pos_idx]
    else:
        sv = shap_vals

    mean_abs = np.abs(sv).mean(axis=0)
    mean_signed = sv.mean(axis=0)

    pd.DataFrame(
        {
            "feature": Xt.columns.astype(str),
            "mean_abs_shap_test": mean_abs,
            "mean_shap_test": mean_signed,
            "n_test_rows_used": int(Xt.shape[0]),
        }
    ).sort_values("mean_abs_shap_test", ascending=False).to_csv(out_path, index=False)


def write_ids(out_fold_dir: Path, fold: str, X_tr: pd.DataFrame, X_te: pd.DataFrame) -> None:
    """
    Save training and test sample IDs for one fold.

    Parameters
    ----------
    out_fold_dir
        Fold output directory.
    fold
        Fold label.
    X_tr
        Training feature matrix.
    X_te
        Test feature matrix.
    """
    ensure_dir(out_fold_dir)
    pd.Series(X_tr.index.astype(str)).to_csv(out_fold_dir / f"train_ids_used_{fold}.csv", index=False, header=False)
    pd.Series(X_te.index.astype(str)).to_csv(out_fold_dir / f"test_ids_used_{fold}.csv", index=False, header=False)


def write_metrics_and_importance(
    *,
    out_fold_dir: Path,
    fold: str,
    prefix: str,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    y_tr_pred: np.ndarray,
    y_te_pred: np.ndarray,
    model: RandomForestClassifier,
    best_params: dict,
    extra: Optional[dict] = None,
) -> dict:
    """
    Save per-fold metrics and Gini importance output.

    Parameters
    ----------
    out_fold_dir
        Fold output directory.
    fold
        Fold label.
    prefix
        Output filename prefix.
    X_tr
        Training feature matrix.
    X_te
        Test feature matrix.
    y_tr
        Training labels.
    y_te
        Test labels.
    y_tr_pred
        Training predictions.
    y_te_pred
        Test predictions.
    model
        Fitted RandomForest classifier.
    best_params
        Best hyperparameters returned by GridSearchCV.
    extra
        Optional additional metrics fields.

    Returns
    -------
    dict
        Metrics row written to disk.
    """
    ensure_dir(out_fold_dir)

    metrics_row = {
        "fold": fold,
        "n_features": int(X_tr.shape[1]),
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
        "best_n_estimators": best_params.get("n_estimators"),
        "best_max_features": best_params.get("max_features"),
    }
    if extra:
        metrics_row.update(extra)

    pd.DataFrame([metrics_row]).to_csv(out_fold_dir / f"metrics_{prefix}_{fold}.csv", index=False)
    pd.DataFrame(
        {"feature": X_tr.columns.astype(str), "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        out_fold_dir / f"importance_gini_{prefix}_{fold}.csv", index=False
    )
    return metrics_row


def run_outercv_full_then_topk(
    *,
    spin: str,
    descriptor_name: str,
    X_master: pd.DataFrame,
    y_master: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    out_full_dir: Path,
    out_topk_dir: Path,
    topk: int,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
    filtering: bool,
    drop_nonfinite_always: bool,
    var_thresh: float,
    corr_cfg: CorrPruneConfig,
    impute_strategy: str,
    impute_fill_value: float,
    overwrite: bool,
    compute_shap: bool,
    shap_max_test_samples: int,
    grid_n_jobs: int,
    rf_n_jobs: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Run the two-stage outer-fold workflow: full model followed by fold-local Top-K model.

    Parameters
    ----------
    spin
        Spin-state label.
    descriptor_name
        Descriptor name.
    X_master
        Descriptor matrix indexed by master IDs.
    y_master
        Target vector indexed by master IDs.
    splits
        Outer train/test index pairs.
    out_full_dir
        Output directory for Stage 1 full-model results.
    out_topk_dir
        Output directory for Stage 2 Top-K results.
    topk
        Number of top features to retain.
    n_estimators_grid
        Candidate values for ``n_estimators``.
    max_features_grid
        Candidate values for ``max_features``.
    filtering
        Whether fold-local filtering is enabled.
    drop_nonfinite_always
        Whether to always remove non-finite or float32-unsafe columns.
    var_thresh
        Variance threshold.
    corr_cfg
        Correlation-pruning configuration.
    impute_strategy
        Imputation strategy for Top-K models.
    impute_fill_value
        Fill value used for constant imputation.
    overwrite
        Whether to overwrite existing fold outputs.
    compute_shap
        Whether to compute SHAP summaries.
    shap_max_test_samples
        Maximum number of test rows used for SHAP.
    grid_n_jobs
        ``n_jobs`` passed to GridSearchCV.
    rf_n_jobs
        ``n_jobs`` passed to RandomForestClassifier.

    Returns
    -------
    Tuple[List[dict], List[dict]]
        Per-fold metrics rows for Stage 1 and Stage 2.
    """
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    base = RandomForestClassifier(class_weight="balanced", random_state=OUTER_SEED, n_jobs=rf_n_jobs)

    full_rows: List[dict] = []
    topk_rows: List[dict] = []

    for i, (tr_idx, te_idx) in enumerate(splits):
        fold = fold_name(i)
        LOGGER.info("Running fold=%s | spin=%s | desc=%s", fold, spin, descriptor_name)

        out_fold_full = out_full_dir / fold
        out_fold_topk = out_topk_dir / fold
        ensure_dir(out_fold_full)
        ensure_dir(out_fold_topk)

        full_metrics_path = out_fold_full / f"metrics_full_{fold}.csv"
        topk_metrics_path = out_fold_topk / f"metrics_topk_samefold_{fold}.csv"
        topk_txt_path = out_fold_topk / f"topk_features_{fold}.txt"

        if (not overwrite) and full_metrics_path.is_file() and topk_metrics_path.is_file() and topk_txt_path.is_file():
            try:
                full_rows.append(pd.read_csv(full_metrics_path).iloc[0].to_dict())
                topk_rows.append(pd.read_csv(topk_metrics_path).iloc[0].to_dict())
                continue
            except Exception:
                pass

        X_tr = X_master.iloc[tr_idx].copy()
        X_te = X_master.iloc[te_idx].copy()
        y_tr = y_master.iloc[tr_idx].copy()
        y_te = y_master.iloc[te_idx].copy()

        X_tr_f, X_te_f, filter_stats = apply_foldlocal_filtering(
            X_tr=X_tr,
            X_te=X_te,
            filtering=filtering,
            drop_nonfinite_always=drop_nonfinite_always,
            var_thresh=var_thresh,
            corr_cfg=corr_cfg,
        )

        if X_tr_f.shape[1] == 0:
            LOGGER.warning("Skipped full/topk (no features after filtering): spin=%s desc=%s fold=%s", spin, descriptor_name, fold)
            topk_txt_path.write_text("", encoding="utf-8")
            continue

        if np.isnan(X_tr_f.to_numpy()).any() or np.isnan(X_te_f.to_numpy()).any():
            LOGGER.warning("Skipped full/topk (NaN remains after filtering): spin=%s desc=%s fold=%s", spin, descriptor_name, fold)
            topk_txt_path.write_text("", encoding="utf-8")
            continue

        try:
            _assert_float32_safe_df(X_tr_f, context=f"[full] spin={spin} desc={descriptor_name} fold={fold} train")
            _assert_float32_safe_df(X_te_f, context=f"[full] spin={spin} desc={descriptor_name} fold={fold} test")
        except Exception as e:
            LOGGER.warning("Skipped full/topk (float32 unsafe): %s", str(e))
            topk_txt_path.write_text("", encoding="utf-8")
            continue

        full_grid = GridSearchCV(
            base,
            param_grid={"n_estimators": list(n_estimators_grid), "max_features": list(max_features_grid)},
            cv=inner,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=int(grid_n_jobs),
        )
        full_grid.fit(X_tr_f, y_tr)
        full_model: RandomForestClassifier = full_grid.best_estimator_

        write_ids(out_fold_full, fold, X_tr_f, X_te_f)
        y_tr_pred_full = full_model.predict(X_tr_f)
        y_te_pred_full = full_model.predict(X_te_f)

        full_extra = {
            "n_input_features": int(filter_stats["n_input_features"]),
            "n_after_float32_safe": int(filter_stats["n_after_float32_safe"]),
            "n_after_variance": int(filter_stats["n_after_variance"]),
            "n_after_corr": int(filter_stats["n_after_corr"]),
        }
        full_row = write_metrics_and_importance(
            out_fold_dir=out_fold_full,
            fold=fold,
            prefix="full",
            X_tr=X_tr_f,
            X_te=X_te_f,
            y_tr=y_tr,
            y_te=y_te,
            y_tr_pred=y_tr_pred_full,
            y_te_pred=y_te_pred_full,
            model=full_model,
            best_params=full_grid.best_params_,
            extra=full_extra,
        )
        full_rows.append(full_row)

        if compute_shap:
            try:
                compute_shap_importance_on_test(
                    model=full_model,
                    X_test=X_te_f,
                    out_path=out_fold_full / f"importance_shap_test_full_{fold}.csv",
                    max_test_samples=shap_max_test_samples,
                )
            except Exception as e:
                LOGGER.warning("SHAP failed (full): spin=%s desc=%s fold=%s (%s)", spin, descriptor_name, fold, str(e))

        imp_train = pd.DataFrame(
            {"feature": X_tr_f.columns.astype(str), "importance": full_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        imp_train.to_csv(out_fold_topk / f"importance_gini_train_for_topk_{fold}.csv", index=False)

        topk_feats = pick_topk_from_importance(X_tr_f.columns.astype(str).tolist(), full_model.feature_importances_, topk)
        topk_txt_path.write_text("\n".join(topk_feats) + ("\n" if topk_feats else ""), encoding="utf-8")

        if not topk_feats:
            LOGGER.warning("Skipped topk model (empty TopK): spin=%s desc=%s fold=%s", spin, descriptor_name, fold)
            continue

        X_tr_k = X_tr_f.loc[:, topk_feats].copy()
        X_te_k = X_te_f.loc[:, topk_feats].copy()

        try:
            X_tr_k2, X_te_k2 = fit_transform_impute_or_validate(
                X_tr_k,
                X_te_k,
                strategy=impute_strategy,
                fill_value=impute_fill_value,
                context=f"[topk samefold] spin={spin} desc={descriptor_name} fold={fold}",
            )
        except Exception as e:
            LOGGER.warning("Skipped topk model: %s", str(e))
            continue

        topk_grid = GridSearchCV(
            base,
            param_grid={"n_estimators": list(n_estimators_grid), "max_features": list(max_features_grid)},
            cv=inner,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=int(grid_n_jobs),
        )
        topk_grid.fit(X_tr_k2, y_tr)
        topk_model: RandomForestClassifier = topk_grid.best_estimator_

        write_ids(out_fold_topk, fold, X_tr_k2, X_te_k2)
        (out_fold_topk / f"cols_used_{fold}.txt").write_text("\n".join(topk_feats) + "\n", encoding="utf-8")

        y_tr_pred_topk = topk_model.predict(X_tr_k2)
        y_te_pred_topk = topk_model.predict(X_te_k2)

        topk_extra = {
            "n_features_full_after_filter": int(X_tr_f.shape[1]),
            "n_features_topk": int(X_tr_k2.shape[1]),
        }
        topk_row = write_metrics_and_importance(
            out_fold_dir=out_fold_topk,
            fold=fold,
            prefix="topk_samefold",
            X_tr=X_tr_k2,
            X_te=X_te_k2,
            y_tr=y_tr,
            y_te=y_te,
            y_tr_pred=y_tr_pred_topk,
            y_te_pred=y_te_pred_topk,
            model=topk_model,
            best_params=topk_grid.best_params_,
            extra=topk_extra,
        )
        topk_rows.append(topk_row)

        if compute_shap:
            try:
                compute_shap_importance_on_test(
                    model=topk_model,
                    X_test=X_te_k2,
                    out_path=out_fold_topk / f"importance_shap_test_topk_samefold_{fold}.csv",
                    max_test_samples=shap_max_test_samples,
                )
            except Exception as e:
                LOGGER.warning("SHAP failed (topk): spin=%s desc=%s fold=%s (%s)", spin, descriptor_name, fold, str(e))

    return full_rows, topk_rows


def write_summary(rows: List[dict], out_csv: Path) -> None:
    """
    Save per-fold rows and aggregate mean/std summaries.

    Parameters
    ----------
    rows
        Per-fold result rows.
    out_csv
        Output CSV path for the fold-level table.
    """
    ensure_dir(out_csv.parent)
    if not rows:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    metric_cols = [
        c
        for c in df.columns
        if c.startswith("train_") or c.startswith("test_") or c.startswith("n_")
    ]
    agg = {}
    for c in metric_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            agg[f"{c}_mean"] = df[c].mean()
            agg[f"{c}_std"] = df[c].std(ddof=1)
    pd.DataFrame([agg]).to_csv(out_csv.with_name(out_csv.stem + "_summary.csv"), index=False)


def main(argv: Optional[List[str]] = None) -> None:
    """
    Run the full nested cross-validation workflow for all requested descriptors
    and spin states.

    Parameters
    ----------
    argv
        Optional command-line argument list. If omitted, ``sys.argv[1:]`` is used.
    """
    args = build_argparser().parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)
    paths = resolve_paths(args)
    ensure_dir(paths.out_dir)
    write_run_config(paths.out_dir, args)

    labels = read_labels(paths.labels_csv)
    descriptors = load_descriptors(paths.descriptors_dir, label_ids=labels.index)

    if args.descs:
        wanted = set(args.descs)
        descriptors = {k: v for k, v in descriptors.items() if k in wanted}
        LOGGER.info("Filtered descriptors: %d -> %s", len(descriptors), sorted(descriptors.keys()))
        if not descriptors:
            raise RuntimeError(f"--descs specified but none matched. wanted={sorted(wanted)}")

    n_estimators_grid = list(args.n_estimators)
    max_features_grid = parse_max_features(args.max_features)
    compute_shap = not bool(args.no_shap)
    shap_max_test_samples = int(args.shap_max_test_samples)

    if args.filtering == "both":
        filtering_modes = [("filterON", True), ("filterOFF", False)]
    elif args.filtering == "on":
        filtering_modes = [("filterON", True)]
    else:
        filtering_modes = [("filterOFF", False)]

    corr_cfg = CorrPruneConfig(
        corr_thresh=float(args.corr_thresh),
        use_abs_corr=not bool(args.signed_corr),
        block_size=int(args.corr_block_size),
        keep_first=int(args.corr_keep_first),
        max_keep=int(args.corr_max_keep),
    )

    LOGGER.info("Repo root: %s", paths.root)
    LOGGER.info("Labels: %s", paths.labels_csv)
    LOGGER.info("Descriptors: %s (%d)", paths.descriptors_dir, len(descriptors))
    LOGGER.info("Output: %s", paths.out_dir)
    LOGGER.info("Grid: n_estimators=%s, max_features=%s", n_estimators_grid, max_features_grid)
    LOGGER.info(
        "Thread env: OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
    )
    LOGGER.info("Parallel: grid_n_jobs=%d rf_n_jobs=%d", int(args.grid_n_jobs), int(args.rf_n_jobs))
    LOGGER.info(
        "TopK=%d | filtering=%s | impute=%s | SHAP=%s",
        int(args.topk),
        args.filtering,
        args.impute_strategy,
        "on" if compute_shap else "off",
    )
    LOGGER.info(
        "Filtering params: var_thresh=%g corr_thresh=%g corr=%s block=%d keep_first=%d max_keep=%d",
        float(args.var_thresh),
        corr_cfg.corr_thresh,
        "abs" if corr_cfg.use_abs_corr else "signed",
        corr_cfg.block_size,
        corr_cfg.keep_first,
        corr_cfg.max_keep,
    )
    LOGGER.info("Float32 max guard enabled: max_abs <= %g (TRAIN-only column keep)", FLOAT32_MAX)

    per_spin_data: Dict[str, dict] = {}
    for spin in SPIN_STATES:
        y_all = labels.loc[labels[COL_STATE] == spin, COL_BEHAV].dropna().astype(str)
        if y_all.empty:
            LOGGER.warning("No labeled rows for spin=%s", spin)
            continue

        y_bin = y_all.apply(lambda v: POSITIVE_LABEL if v == POSITIVE_LABEL else spin)
        master_ids = make_master_ids(y_bin, descriptors)
        if len(master_ids) == 0:
            LOGGER.warning("No common IDs across labels+descriptors for spin=%s", spin)
            continue

        y_master = y_bin.loc[master_ids]
        splits = build_outer_splits(y_master)
        per_spin_data[spin] = {"master_ids": master_ids, "y_master": y_master, "splits": splits}
        LOGGER.info("Spin=%s | n=%d | outer_folds=%d", spin, len(master_ids), len(splits))

    for filter_tag, filtering_on in filtering_modes:
        drop_nonfinite_always = True if filtering_on else (not args.allow_nonfinite)
        mode_out_dir = paths.out_dir / filter_tag
        ensure_dir(mode_out_dir)

        out_stage1_full = mode_out_dir / "stage1_full"
        out_stage2_topk = mode_out_dir / "stage2_topk_samefold"
        out_summary = mode_out_dir / "summaries"
        for p in (out_stage1_full, out_stage2_topk, out_summary):
            ensure_dir(p)

        LOGGER.info("=== MODE %s (filtering=%s) ===", filter_tag, filtering_on)

        for spin in SPIN_STATES:
            if spin not in per_spin_data:
                continue

            master_ids = per_spin_data[spin]["master_ids"]
            y_master = per_spin_data[spin]["y_master"]
            splits = per_spin_data[spin]["splits"]

            for desc_name, Xdesc in descriptors.items():
                X_master = Xdesc.reindex(master_ids)
                if X_master.shape[0] == 0 or X_master.shape[1] == 0:
                    continue

                LOGGER.info("Full -> TopK same-fold: mode=%s spin=%s desc=%s", filter_tag, spin, desc_name)
                full_rows, topk_rows = run_outercv_full_then_topk(
                    spin=spin,
                    descriptor_name=desc_name,
                    X_master=X_master,
                    y_master=y_master,
                    splits=splits,
                    out_full_dir=out_stage1_full / spin / desc_name,
                    out_topk_dir=out_stage2_topk / spin / desc_name,
                    topk=int(args.topk),
                    n_estimators_grid=n_estimators_grid,
                    max_features_grid=max_features_grid,
                    filtering=filtering_on,
                    drop_nonfinite_always=drop_nonfinite_always,
                    var_thresh=float(args.var_thresh),
                    corr_cfg=corr_cfg,
                    impute_strategy=str(args.impute_strategy),
                    impute_fill_value=float(args.impute_fill_value),
                    overwrite=bool(args.overwrite),
                    compute_shap=compute_shap,
                    shap_max_test_samples=shap_max_test_samples,
                    grid_n_jobs=int(args.grid_n_jobs),
                    rf_n_jobs=int(args.rf_n_jobs),
                )

                write_summary(
                    full_rows,
                    out_summary / f"summary_stage1_full__{filter_tag}__{spin}__{desc_name}.csv",
                )
                write_summary(
                    topk_rows,
                    out_summary / f"summary_stage2_topk_samefold__{filter_tag}__{spin}__{desc_name}.csv",
                )

    LOGGER.info("DONE.")


if __name__ == "__main__":
    main()