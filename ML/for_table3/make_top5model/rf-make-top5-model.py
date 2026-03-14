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
Random Forest evaluation with fixed Top-K descriptors.

This script evaluates Random Forest classifiers using preselected Top-K
descriptor sets. Feature selection is performed externally and the selected
feature lists are reused for model training and evaluation.

For each target spin state, outer-fold train/test splits are reused from
previously generated results, while Top-K feature lists can originate from
either the same spin state or the opposite spin state. Hyperparameters are
optimized using inner 5-fold cross-validation, and optional SHAP analysis can
be computed for each fold.

Parameters
----------
Input structure : directory tree
    <topk_root_dir>/<spin>/<descriptor>/
        top5_<spin>_<descriptor>.csv
        r1_f1/
            train_ids_r1_f1.csv
            test_ids_r1_f1.csv
        ...

Output structure : directory tree
    <out_dir>/<target_spin>/<descriptor>/topkmodel_<tag>/
        r1_f1/
            metrics.csv
            importance.csv
            selected_features.csv
            shap_mean_abs.csv
            shap_values.csv.gz
        shap_mean_abs_over_folds.csv

Notes
-----
The script does not perform feature selection itself. It strictly reuses
precomputed Top-K feature sets to preserve reproducibility.
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
LOGGER = logging.getLogger(__name__)

COL_ID = "ccdc_id"
COL_STATE = "Spin-state of crystal structure"
COL_BEHAV = "Spin-state behaviour"

POSITIVE_LABEL = "spin-crossover"
SPIN_STATES = ("high-spin", "low-spin")

INNER_SPLITS = 5
INNER_SEED = 0

DEFAULT_N_ESTIMATORS = (50, 100, 300)
DEFAULT_MAX_FEATURES = ("None", "sqrt", "log2")
TOPK_DEFAULT = 5


def find_repo_root(start: Path) -> Path:
    """
    Find the repository root from a starting path.

    Parameters
    ----------
    start : pathlib.Path
        Starting directory for upward search.

    Returns
    -------
    pathlib.Path
        Repository root directory.

    Raises
    ------
    RuntimeError
        If the expected repository structure cannot be found.
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


@dataclass(frozen=True)
class Paths:
    labels_csv: Path
    descriptors_dir: Path
    topk_root_dir: Path
    out_dir: Path


def configure_logging(level: str) -> None:
    """
    Configure module-level logging.

    Parameters
    ----------
    level : str
        Logging level name.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s")


def ensure_dir(p: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    p : pathlib.Path
        Directory path to create.
    """
    p.mkdir(parents=True, exist_ok=True)


def write_run_config(out_root: Path, args: argparse.Namespace) -> None:
    """
    Write runtime configuration metadata to disk.

    Parameters
    ----------
    out_root : pathlib.Path
        Output root directory.
    args : argparse.Namespace
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
    Read the label table.

    Parameters
    ----------
    path : pathlib.Path
        Path to the label CSV file.

    Returns
    -------
    pandas.DataFrame
        Label table indexed by ``ccdc_id``.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    df = pd.read_csv(path)
    missing = [c for c in (COL_ID, COL_STATE, COL_BEHAV) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in label CSV: {missing}")

    df[COL_ID] = df[COL_ID].astype(str).str.strip()
    df[COL_STATE] = df[COL_STATE].astype(str).str.strip()
    df[COL_BEHAV] = df[COL_BEHAV].astype(str).str.strip()
    df = df.replace({"nan": np.nan, "NaN": np.nan})
    return df.set_index(COL_ID)


def _pick_id_column(raw: pd.DataFrame, label_ids: pd.Index) -> Optional[str]:
    """
    Infer the identifier column in a descriptor table.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw descriptor table.
    label_ids : pandas.Index
        Known label identifiers.

    Returns
    -------
    str or None
        Inferred identifier column name, or ``None`` if no suitable column is
        found.
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
    Load descriptor tables.

    Parameters
    ----------
    desc_root : pathlib.Path
        Root directory containing descriptor subdirectories.
    label_ids : pandas.Index
        Known label identifiers used to infer index columns.

    Returns
    -------
    dict of str to pandas.DataFrame
        Mapping from descriptor name to cleaned numeric descriptor table.

    Raises
    ------
    FileNotFoundError
        If the descriptor root directory does not exist.
    RuntimeError
        If no usable descriptor tables are found.
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
        df = df.dropna(axis=1)

        if df.shape[1] == 0:
            LOGGER.warning(
                "Descriptor '%s' has no usable numeric columns after cleaning. Skipped.",
                sub.name,
            )
            continue

        out[sub.name] = df

    if not out:
        raise RuntimeError(f"No descriptor CSVs found under: {desc_root}")

    LOGGER.info("Loaded %d descriptors.", len(out))
    for k, v in out.items():
        LOGGER.info("  - %s: %s", k, v.shape)
    return out


def make_master_ids(y_all: pd.Series, descriptors: Dict[str, pd.DataFrame]) -> pd.Index:
    """
    Compute IDs shared across labels and all descriptor tables.

    Parameters
    ----------
    y_all : pandas.Series
        Label series indexed by sample ID.
    descriptors : dict of str to pandas.DataFrame
        Descriptor tables indexed by sample ID.

    Returns
    -------
    pandas.Index
        IDs present in all inputs.
    """
    common = y_all.index
    for X in descriptors.values():
        common = common.intersection(X.index)
    return y_all.index[y_all.index.isin(common)]


def fold_sort_key(name: str) -> Tuple[int, int]:
    """
    Generate a sortable key for fold directory names.

    Parameters
    ----------
    name : str
        Fold directory name such as ``r1_f1``.

    Returns
    -------
    tuple of int
        Replicate and fold indices, or a large fallback tuple on parse failure.
    """
    try:
        rep_str, fold_str = name.split("_")
        rep = int(rep_str[1:])
        fold = int(fold_str[1:])
        return rep, fold
    except Exception:
        return (10**9, 10**9)


def discover_saved_folds(base_dir: Path) -> List[str]:
    """
    Discover saved fold directories.

    Parameters
    ----------
    base_dir : pathlib.Path
        Directory containing fold subdirectories.

    Returns
    -------
    list of str
        Sorted fold directory names.
    """
    folds: List[str] = []
    if not base_dir.is_dir():
        return folds
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("r") and "_f" in p.name:
            folds.append(p.name)
    return sorted(folds, key=fold_sort_key)


def read_id_list(path: Path) -> pd.Index:
    """
    Read a one-column ID list from CSV.

    Parameters
    ----------
    path : pathlib.Path
        CSV file containing IDs.

    Returns
    -------
    pandas.Index
        Parsed identifier list.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing ID file: {path}")

    df = pd.read_csv(path, header=None)
    if df.shape[1] == 0:
        return pd.Index([], dtype=object)

    s = df.iloc[:, 0].astype(str).str.strip()
    s = s[s != COL_ID]
    s = s[s != ""]
    return pd.Index(s.tolist(), dtype=object)


def load_saved_fold_ids(
    topk_root_dir: Path,
    fold_spin: str,
    desc: str,
    fold: str,
) -> Tuple[pd.Index, pd.Index]:
    """
    Load saved train/test IDs for a fold.

    Parameters
    ----------
    topk_root_dir : pathlib.Path
        Root directory containing saved fold outputs.
    fold_spin : str
        Spin state used to define the fold split.
    desc : str
        Descriptor name.
    fold : str
        Fold name.

    Returns
    -------
    tuple of pandas.Index
        Train IDs and test IDs.
    """
    fold_dir = topk_root_dir / fold_spin / desc / fold
    train_path = fold_dir / f"train_ids_{fold}.csv"
    test_path = fold_dir / f"test_ids_{fold}.csv"

    train_ids = read_id_list(train_path)
    test_ids = read_id_list(test_path)
    return train_ids, test_ids


def load_topk_features(path: Path) -> List[str]:
    """
    Load a Top-K feature list.

    Parameters
    ----------
    path : pathlib.Path
        Path to the Top-K CSV file.

    Returns
    -------
    list of str
        Feature names in the stored order.

    Raises
    ------
    KeyError
        If the required ``feature`` column is missing.
    """
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise KeyError(f"TopK file missing 'feature' column: {path}")
    return df["feature"].astype(str).tolist()


def parse_max_features(values: Sequence[str]) -> List[object]:
    """
    Parse Random Forest ``max_features`` values from CLI strings.

    Parameters
    ----------
    values : sequence of str
        Input strings.

    Returns
    -------
    list of object
        Parsed values, with ``"None"`` and ``"null"`` mapped to ``None``.
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
    Extract SHAP values for the positive class.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Fitted classifier.
    X : pandas.DataFrame
        Evaluation matrix.
    positive_label : str or None, default=None
        Positive class label. If omitted or absent, the second class is used
        for binary classification.

    Returns
    -------
    numpy.ndarray
        SHAP array with shape ``(n_samples, n_features)``.

    Raises
    ------
    RuntimeError
        If SHAP is unavailable or returns an unsupported output shape.
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
    Save fold-level SHAP outputs.

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        Fitted classifier.
    X_eval : pandas.DataFrame
        Evaluation matrix used for SHAP calculation.
    fold_dir : pathlib.Path
        Output directory for the current fold.
    positive_label : str or None
        Positive class label.
    save_shap_matrix : bool
        Whether to save the full sample-by-feature SHAP matrix.

    Returns
    -------
    pandas.DataFrame
        Mean absolute SHAP values by feature.
    """
    shap_arr = _extract_positive_class_shap_array(
        model, X_eval, positive_label=positive_label
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


def run_topk_fixed_model_from_ids(
    *,
    X_master: pd.DataFrame,
    y_master: pd.Series,
    fold_ids: List[Tuple[str, pd.Index, pd.Index]],
    model_out_dir: Path,
    overwrite: bool,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
    grid_n_jobs: int,
    rf_n_jobs: int,
    compute_shap: bool,
    save_shap_matrix: bool,
    shap_on: str,
    positive_label: Optional[str],
) -> Tuple[List[dict], Optional[pd.DataFrame]]:
    """
    Train and evaluate fixed-TopK models on predefined folds.

    Parameters
    ----------
    X_master : pandas.DataFrame
        Master feature matrix.
    y_master : pandas.Series
        Master label vector.
    fold_ids : list of tuple
        Fold definitions as ``(fold_name, train_ids, test_ids)``.
    model_out_dir : pathlib.Path
        Output directory for model results.
    overwrite : bool
        Whether to overwrite existing fold outputs.
    n_estimators_grid : sequence of int
        Grid values for ``n_estimators``.
    max_features_grid : sequence of object
        Grid values for ``max_features``.
    grid_n_jobs : int
        Number of jobs for grid search.
    rf_n_jobs : int
        Number of jobs for the Random Forest model.
    compute_shap : bool
        Whether to compute SHAP outputs.
    save_shap_matrix : bool
        Whether to save the full SHAP matrix.
    shap_on : str
        Dataset split used for SHAP calculation, either ``"train"`` or ``"test"``.
    positive_label : str or None
        Positive class label for SHAP extraction.

    Returns
    -------
    tuple
        Fold metric rows and aggregated mean absolute SHAP values across folds.
    """
    rows: List[dict] = []
    shap_all: List[pd.DataFrame] = []

    for fold, train_ids_raw, test_ids_raw in fold_ids:
        out_fold_dir = model_out_dir / fold
        ensure_dir(out_fold_dir)

        metrics_path = out_fold_dir / "metrics.csv"
        if (not overwrite) and metrics_path.is_file():
            try:
                old = pd.read_csv(metrics_path).iloc[0].to_dict()
                rows.append(old)
                continue
            except Exception:
                pass

        train_ids = [i for i in train_ids_raw if i in X_master.index and i in y_master.index]
        test_ids = [i for i in test_ids_raw if i in X_master.index and i in y_master.index]

        X_tr = X_master.loc[train_ids].copy()
        X_te = X_master.loc[test_ids].copy()
        y_tr = y_master.loc[train_ids].copy()
        y_te = y_master.loc[test_ids].copy()

        nunique = X_tr.nunique(dropna=False)
        keep_cols = nunique[nunique > 1].index.tolist()
        X_tr = X_tr.loc[:, keep_cols]
        X_te = X_te.loc[:, keep_cols]

        if X_tr.shape[1] > 0:
            bad_tr = ~np.isfinite(X_tr.to_numpy(dtype=np.float64, copy=False)).all(axis=1)
            if bad_tr.any():
                X_tr = X_tr.loc[~bad_tr]
                y_tr = y_tr.loc[X_tr.index]

        if X_te.shape[1] > 0:
            bad_te = ~np.isfinite(X_te.to_numpy(dtype=np.float64, copy=False)).all(axis=1)
            if bad_te.any():
                X_te = X_te.loc[~bad_te]
                y_te = y_te.loc[X_te.index]

        if X_tr.shape[0] == 0 or X_te.shape[0] == 0 or X_tr.shape[1] == 0:
            LOGGER.warning(
                "Empty fold after fixed-TopK preparation: %s (ntr=%d nte=%d p=%d)",
                fold,
                X_tr.shape[0],
                X_te.shape[0],
                X_tr.shape[1],
            )
            continue

        pd.Series(X_tr.index.astype(str), name=COL_ID).to_csv(
            out_fold_dir / f"train_ids_{fold}.csv", index=False
        )
        pd.Series(X_te.index.astype(str), name=COL_ID).to_csv(
            out_fold_dir / f"test_ids_{fold}.csv", index=False
        )
        pd.Series(X_tr.columns.astype(str), name="feature").to_csv(
            out_fold_dir / "selected_features.csv", index=False
        )

        inner = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=INNER_SEED,
        )
        grid = GridSearchCV(
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
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
            "fold": fold,
            "n_features": int(X_tr.shape[1]),
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "train_accuracy": float(accuracy_score(y_tr, y_tr_pred)),
            "test_accuracy": float(accuracy_score(y_te, y_te_pred)),
            "train_precision": float(
                precision_score(y_tr, y_tr_pred, average="weighted", zero_division=0)
            ),
            "test_precision": float(
                precision_score(y_te, y_te_pred, average="weighted", zero_division=0)
            ),
            "train_recall": float(
                recall_score(y_tr, y_tr_pred, average="weighted", zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_te, y_te_pred, average="weighted", zero_division=0)
            ),
            "train_f1": float(
                f1_score(y_tr, y_tr_pred, average="weighted", zero_division=0)
            ),
            "test_f1": float(
                f1_score(y_te, y_te_pred, average="weighted", zero_division=0)
            ),
            "train_mcc": float(matthews_corrcoef(y_tr, y_tr_pred)),
            "test_mcc": float(matthews_corrcoef(y_te, y_te_pred)),
            "best_n_estimators": grid.best_params_.get("n_estimators"),
            "best_max_features": grid.best_params_.get("max_features"),
        }
        pd.DataFrame([mrow]).to_csv(metrics_path, index=False)

        pd.DataFrame(
            {"feature": X_tr.columns.astype(str), "importance": model.feature_importances_}
        ).to_csv(out_fold_dir / "importance.csv", index=False)

        if compute_shap:
            X_for_shap = X_te if shap_on == "test" else X_tr
            shap_mean_df = save_fold_shap_outputs(
                model=model,
                X_eval=X_for_shap,
                fold_dir=out_fold_dir,
                positive_label=positive_label,
                save_shap_matrix=save_shap_matrix,
            )
            shap_mean_df["fold"] = fold
            shap_all.append(shap_mean_df)

        rows.append(mrow)

    shap_mean_over_folds: Optional[pd.DataFrame] = None
    if compute_shap and shap_all:
        shap_cat = pd.concat(shap_all, ignore_index=True)
        shap_mean_over_folds = (
            shap_cat.groupby("feature", as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_mean_over_folds.to_csv(
            model_out_dir / "shap_mean_abs_over_folds.csv",
            index=False,
        )

    return rows, shap_mean_over_folds


def build_argparser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    p = argparse.ArgumentParser(
        description="Fixed TopK RF evaluation with target-fixed fold reuse."
    )
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--descriptors", type=Path, default=None)
    p.add_argument(
        "--topk-root-dir",
        type=Path,
        default=None,
        help="Directory produced by rf_globalfilter_importance_for_topk.py",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    p.add_argument("--descs", nargs="*", default=None)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument(
        "--n-estimators",
        nargs="+",
        type=int,
        default=list(DEFAULT_N_ESTIMATORS),
    )
    p.add_argument(
        "--max-features",
        nargs="+",
        default=list(DEFAULT_MAX_FEATURES),
    )
    p.add_argument("--grid-n-jobs", type=int, default=1)
    p.add_argument("--rf-n-jobs", type=int, default=1)

    p.add_argument(
        "--compute-shap",
        action="store_true",
        help="Compute fold-wise SHAP summaries.",
    )
    p.add_argument(
        "--save-shap-matrix",
        action="store_true",
        help="Also save sample-wise SHAP matrix (.csv.gz).",
    )
    p.add_argument(
        "--shap-on",
        choices=["train", "test"],
        default="test",
        help="Compute SHAP on train or test fold.",
    )

    p.add_argument("--log-level", default="INFO")
    return p


def resolve_paths(args: argparse.Namespace) -> Paths:
    """
    Resolve effective input and output paths.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Paths
        Resolved path bundle.
    """
    labels_csv = args.labels or (ROOT_DIR / "FeN6-SSD" / "FeN6-SSD_500_spin_labeled.csv")
    descriptors_dir = args.descriptors or (ROOT_DIR / "descriptors")
    topk_root_dir = args.topk_root_dir or (
        ROOT_DIR / "ML" / "for_table3" / "decide_top5" / "results_globalfilter_topk"
    )
    out_dir = args.out_dir or (
        SCRIPT_DIR / f"results_top{int(args.topk)}_fixed_exactsplits"
    )

    return Paths(
        labels_csv=labels_csv,
        descriptors_dir=descriptors_dir,
        topk_root_dir=topk_root_dir,
        out_dir=out_dir,
    )


def main(argv: Optional[List[str]] = None) -> None:
    """
    Run the fixed-TopK Random Forest evaluation workflow.

    Parameters
    ----------
    argv : list of str or None, default=None
        Command-line arguments. If ``None``, arguments are read from
        ``sys.argv``.
    """
    args = build_argparser().parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)

    paths = resolve_paths(args)
    ensure_dir(paths.out_dir)
    write_run_config(paths.out_dir, args)

    n_estimators_grid = list(args.n_estimators)
    max_features_grid = parse_max_features(args.max_features)

    LOGGER.info("Repo root    : %s", ROOT_DIR)
    LOGGER.info("Labels       : %s", paths.labels_csv)
    LOGGER.info("Descriptors  : %s", paths.descriptors_dir)
    LOGGER.info("TopK root    : %s", paths.topk_root_dir)
    LOGGER.info("Output       : %s", paths.out_dir)
    LOGGER.info(
        "Thread env   : OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
    )
    LOGGER.info(
        "Parallel     : RF n_jobs=%d | GridSearch n_jobs=%d",
        int(args.rf_n_jobs),
        int(args.grid_n_jobs),
    )
    LOGGER.info(
        "SHAP         : %s | on=%s | save_matrix=%s",
        "ON" if args.compute_shap else "OFF",
        args.shap_on,
        "ON" if args.save_shap_matrix else "OFF",
    )

    labels = read_labels(paths.labels_csv)
    descriptors = load_descriptors(paths.descriptors_dir, label_ids=labels.index)

    if args.descs:
        wanted = set(args.descs)
        descriptors = {k: v for k, v in descriptors.items() if k in wanted}
        LOGGER.info(
            "Filtered descriptors: %d -> %s",
            len(descriptors),
            sorted(descriptors.keys()),
        )
        if not descriptors:
            raise RuntimeError(
                f"--descs specified but none matched. wanted={sorted(wanted)}"
            )

    for target_spin in SPIN_STATES:
        LOGGER.info("Target spin: %s", target_spin)

        y_all = labels.loc[labels[COL_STATE] == target_spin, COL_BEHAV].copy()
        y_all = y_all.replace({"nan": np.nan, "NaN": np.nan}).dropna().astype(str)
        if y_all.empty:
            LOGGER.warning("No labeled rows for target_spin=%s. Skipped.", target_spin)
            continue

        y_bin = y_all.apply(
            lambda v: POSITIVE_LABEL if v == POSITIVE_LABEL else target_spin
        )

        master_ids = make_master_ids(y_bin, descriptors)
        LOGGER.info(
            "master_ids across ALL descriptors for %s: n=%d",
            target_spin,
            len(master_ids),
        )
        if len(master_ids) == 0:
            LOGGER.warning("No common IDs for target_spin=%s. Skipped.", target_spin)
            continue

        y_master = y_bin.loc[master_ids]
        summary_rows: List[dict] = []

        for desc, Xdesc in descriptors.items():
            X_master_full = Xdesc.reindex(master_ids)
            if X_master_full.shape[0] == 0 or X_master_full.shape[1] == 0:
                LOGGER.warning(
                    "Empty X_master_full: target=%s desc=%s",
                    target_spin,
                    desc,
                )
                continue

            fold_base = paths.topk_root_dir / target_spin / desc
            folds = discover_saved_folds(fold_base)
            if not folds:
                LOGGER.warning(
                    "No saved folds found under target-fold base: %s",
                    fold_base,
                )
                continue

            fold_ids: List[Tuple[str, pd.Index, pd.Index]] = []
            fold_load_failed = False
            for fold in folds:
                try:
                    train_ids, test_ids = load_saved_fold_ids(
                        paths.topk_root_dir,
                        fold_spin=target_spin,
                        desc=desc,
                        fold=fold,
                    )
                except Exception as e:
                    LOGGER.warning(
                        "Failed to load target-fold IDs: target=%s desc=%s fold=%s (%s)",
                        target_spin,
                        desc,
                        fold,
                        str(e),
                    )
                    fold_load_failed = True
                    break
                fold_ids.append((fold, train_ids, test_ids))

            if fold_load_failed or not fold_ids:
                continue

            for feature_source_spin in SPIN_STATES:
                topk_path = (
                    paths.topk_root_dir
                    / feature_source_spin
                    / desc
                    / f"top{int(args.topk)}_{feature_source_spin}_{desc}.csv"
                )
                if not topk_path.is_file():
                    LOGGER.warning("Missing TopK file: %s", topk_path)
                    continue

                try:
                    topk_features = load_topk_features(topk_path)
                except Exception as e:
                    LOGGER.warning(
                        "Failed to read TopK file %s (%s)",
                        topk_path,
                        str(e),
                    )
                    continue

                cols = [c for c in topk_features if c in X_master_full.columns]
                if not cols:
                    LOGGER.warning(
                        "No matching TopK columns: target=%s desc=%s feature_source=%s",
                        target_spin,
                        desc,
                        feature_source_spin,
                    )
                    continue

                X_master = X_master_full.loc[:, cols].copy()

                tag = f"{target_spin[:2].upper()}using{feature_source_spin[:2].upper()}"
                model_out_dir = paths.out_dir / target_spin / desc / f"topkmodel_{tag}"

                LOGGER.info(
                    "Run: target=%s desc=%s feature_source=%s fold_source=%s tag=%s n=%d p=%d folds=%d",
                    target_spin,
                    desc,
                    feature_source_spin,
                    target_spin,
                    tag,
                    int(X_master.shape[0]),
                    int(X_master.shape[1]),
                    len(fold_ids),
                )

                try:
                    fold_metrics, _ = run_topk_fixed_model_from_ids(
                        X_master=X_master,
                        y_master=y_master,
                        fold_ids=fold_ids,
                        model_out_dir=model_out_dir,
                        overwrite=bool(args.overwrite),
                        n_estimators_grid=n_estimators_grid,
                        max_features_grid=max_features_grid,
                        grid_n_jobs=int(args.grid_n_jobs),
                        rf_n_jobs=int(args.rf_n_jobs),
                        compute_shap=bool(args.compute_shap),
                        save_shap_matrix=bool(args.save_shap_matrix),
                        shap_on=str(args.shap_on),
                        positive_label=POSITIVE_LABEL,
                    )
                except Exception as e:
                    LOGGER.warning(
                        "Training failed: target=%s desc=%s feature_source=%s (%s)",
                        target_spin,
                        desc,
                        feature_source_spin,
                        str(e),
                    )
                    continue

                for m in fold_metrics:
                    summary_rows.append(
                        {
                            "descriptor": desc,
                            "target_spin": target_spin,
                            "feature_source_spin": feature_source_spin,
                            "fold_source_spin": target_spin,
                            "fold": str(m.get("fold")),
                            "n_ids_master": int(X_master.shape[0]),
                            "n_train": int(m.get("n_train", 0)),
                            "n_test": int(m.get("n_test", 0)),
                            "test_mcc": float(m.get("test_mcc"))
                            if pd.notna(m.get("test_mcc"))
                            else np.nan,
                            "test_accuracy": float(m.get("test_accuracy"))
                            if pd.notna(m.get("test_accuracy"))
                            else np.nan,
                            "test_f1": float(m.get("test_f1"))
                            if pd.notna(m.get("test_f1"))
                            else np.nan,
                            "n_features": int(m.get("n_features"))
                            if pd.notna(m.get("n_features"))
                            else np.nan,
                            "best_n_estimators": m.get("best_n_estimators"),
                            "best_max_features": m.get("best_max_features"),
                        }
                    )

        if summary_rows:
            out_csv = paths.out_dir / f"summary_{target_spin}.csv"
            pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
            LOGGER.info("Saved: %s", out_csv)
        else:
            LOGGER.warning("No models were run for target_spin=%s", target_spin)


if __name__ == "__main__":
    main()