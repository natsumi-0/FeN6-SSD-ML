#!/usr/bin/env python3
from __future__ import annotations

"""
RandomForest with nested cross-validation.

Outer CV: RepeatedStratifiedKFold (5 splits × 3 repeats)
Inner CV: GridSearchCV with StratifiedKFold (5 splits)

Design goals:
- Reproducible and Git-ready (no noisy prints; structured logging only)
- Path-robust (auto-detect repo root and inputs)
- Reusable outer splits across descriptors via a master ID list per spin state
- Minimal, clear outputs per fold + per-state summary CSVs
"""

import argparse
import logging
import os
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

# -----------------------------------------------------------------------------
# Repo / path resolution
# -----------------------------------------------------------------------------


def find_repo_root(start: Path) -> Path:
    """
    Repo-root detection (strict but portable).

    Required:
      - descriptors/ (dir)
      - ML/          (dir)
      - FeN6-SSD/FeN6-SSD_500_spin_labeled.csv (file)
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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COL_ID = "ccdc_id"
COL_STATE = "Spin-state of crystal structure"
COL_BEHAV = "Spin-state behaviour"

DEFAULT_STATES = ("high-spin", "low-spin")

# Model / CV
DEFAULT_N_ESTIMATORS = (50, 100, 300)
DEFAULT_MAX_FEATURES = ("None", "sqrt", "log2")  # parsed to [None, "sqrt", "log2"]

OUTER_SPLITS = 5
OUTER_REPEATS = 3
OUTER_SEED = 42

INNER_SPLITS = 5
INNER_SEED = 0

# Feature filtering
VAR_THRESH = 1e-12
CORR_THRESH = 0.90


@dataclass(frozen=True)
class FeatureFilterConfig:
    pre_split_filter: bool = True
    per_fold_filter: bool = True
    var_thresh: float = VAR_THRESH
    corr_thresh: float = CORR_THRESH


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    missing = [c for c in (COL_ID, COL_STATE, COL_BEHAV) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in label CSV: {missing}")

    df[COL_ID] = df[COL_ID].astype(str).str.strip()
    df[COL_STATE] = df[COL_STATE].astype(str).str.strip()
    df[COL_BEHAV] = df[COL_BEHAV].astype(str).str.strip()
    return df.set_index(COL_ID)


def load_descriptors(desc_root: Path) -> Dict[str, pd.DataFrame]:
    """
    Load descriptor tables from:
      <desc_root>/<name>/<name>.csv

    Only numeric columns are retained; NaN/Inf columns are dropped.
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

        df = pd.read_csv(csv_path, index_col=0)
        df.index = df.index.astype(str).str.strip()

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1)

        if df.shape[1] == 0:
            LOGGER.warning("Descriptor '%s' has no usable numeric columns after cleaning. Skipped.", name)
            continue

        out[name] = df

    if not out:
        raise RuntimeError(f"No descriptor CSVs found under: {desc_root}")

    LOGGER.info("Loaded %d descriptors.", len(out))
    for k, v in out.items():
        LOGGER.info("  - %s: %s", k, v.shape)
    return out


# -----------------------------------------------------------------------------
# Feature selection utilities
# -----------------------------------------------------------------------------


def drop_low_variance_cols(X: pd.DataFrame, var_thresh: float) -> List[str]:
    if X.shape[1] == 0:
        return []
    v = X.var(axis=0, ddof=1)
    keep = v[v > var_thresh].index.tolist()
    return keep


def select_low_correlation_cols_fullcorr(
    X: pd.DataFrame,
    corr_thresh: float,
    use_abs: bool = True,
) -> pd.DataFrame:
    """
    Greedy selection to reduce multicollinearity:
    - sort by variance (desc)
    - add a feature if its max correlation with already-selected features is < threshold
    """
    if X.shape[1] <= 1:
        return X

    Xnum = X.select_dtypes(include=[np.number])
    cols = list(Xnum.columns)
    if len(cols) <= 1:
        return X.loc[:, cols]

    order = np.argsort(Xnum.var(ddof=1).values)[::-1]

    corr = Xnum.corr().to_numpy()
    if use_abs:
        corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)

    keep_mask = np.zeros(len(cols), dtype=bool)
    for j in order:
        if not keep_mask.any():
            keep_mask[j] = True
        else:
            if corr[j, keep_mask].max() < corr_thresh:
                keep_mask[j] = True

    keep_cols = [cols[i] for i in range(len(cols)) if keep_mask[i]]
    return X.loc[:, keep_cols]


def finite_columns_mask(X: pd.DataFrame) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    return np.isfinite(X.to_numpy()).all(axis=0)


# -----------------------------------------------------------------------------
# CV helpers
# -----------------------------------------------------------------------------


def make_master_ids(y: pd.Series, descriptors: Dict[str, pd.DataFrame]) -> pd.Index:
    common = y.index
    for X in descriptors.values():
        common = common.intersection(X.index)
    return common


def build_outer_splits(y_master: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    rskf = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=OUTER_SEED,
    )
    dummy_X = np.zeros((len(y_master), 1))
    return list(rskf.split(dummy_X, y_master.to_numpy()))


def fold_names(n_folds: int) -> List[str]:
    names: List[str] = []
    for i in range(n_folds):
        rep = i // OUTER_SPLITS + 1
        fold = i % OUTER_SPLITS + 1
        names.append(f"r{rep}_f{fold}")
    return names


# -----------------------------------------------------------------------------
# Training per fold
# -----------------------------------------------------------------------------


def empty_metrics(state: str, descriptor: str, fold: str) -> dict:
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
    pre_cols: Sequence[str],
    out_dir: Path,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
    filter_cfg: FeatureFilterConfig,
) -> dict:
    ensure_dir(out_dir)

    X_tr = X_master.iloc[tr_idx].copy()
    X_te = X_master.iloc[te_idx].copy()
    y_tr = y_master.iloc[tr_idx].copy()
    y_te = y_master.iloc[te_idx].copy()

    # Save ID lists for reproducibility
    pd.Series(X_tr.index).to_csv(out_dir / f"train_ids_{fold}.csv", index=False, header=False)
    pd.Series(X_te.index).to_csv(out_dir / f"test_ids_{fold}.csv", index=False, header=False)

    # Pre-split filter (computed once per descriptor/state on the full master table)
    if filter_cfg.pre_split_filter:
        if not pre_cols:
            LOGGER.warning("%s/%s/%s: no features after pre-split filtering.", state, descriptor, fold)
            return empty_metrics(state, descriptor, fold)
        X_tr = X_tr.loc[:, list(pre_cols)]
        X_te = X_te.loc[:, list(pre_cols)]

    # Drop non-finite columns in train
    finite_mask = finite_columns_mask(X_tr)
    X_tr = X_tr.loc[:, finite_mask]
    X_te = X_te.loc[:, finite_mask]
    if X_tr.shape[1] == 0:
        return empty_metrics(state, descriptor, fold)

    # Per-fold filter (to avoid leakage)
    if filter_cfg.per_fold_filter:
        keep = drop_low_variance_cols(X_tr, filter_cfg.var_thresh)
        X_tr = X_tr.loc[:, keep]
        X_te = X_te.loc[:, keep]
        if X_tr.shape[1] == 0:
            return empty_metrics(state, descriptor, fold)

        X_tr = select_low_correlation_cols_fullcorr(X_tr, filter_cfg.corr_thresh)
        X_te = X_te.loc[:, X_tr.columns]
        if X_tr.shape[1] == 0:
            return empty_metrics(state, descriptor, fold)

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=INNER_SEED,
    )

    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=OUTER_SEED,
    )

    grid = GridSearchCV(
        estimator=base,
        param_grid={"n_estimators": list(n_estimators_grid), "max_features": list(max_features_grid)},
        cv=inner_cv,
        scoring=make_scorer(matthews_corrcoef),
        n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)

    model: RandomForestClassifier = grid.best_estimator_

    ytr_pred = model.predict(X_tr)
    yte_pred = model.predict(X_te)

    metrics = {
        "state": state,
        "descriptor": descriptor,
        "fold": fold,
        "n_features": int(X_tr.shape[1]),
        "train_acc": float(accuracy_score(y_tr, ytr_pred)),
        "test_acc": float(accuracy_score(y_te, yte_pred)),
        "train_mcc": float(matthews_corrcoef(y_tr, ytr_pred)),
        "test_mcc": float(matthews_corrcoef(y_te, yte_pred)),
        "train_precision": float(precision_score(y_tr, ytr_pred, average="weighted", zero_division=0)),
        "test_precision": float(precision_score(y_te, yte_pred, average="weighted", zero_division=0)),
        "train_recall": float(recall_score(y_tr, ytr_pred, average="weighted", zero_division=0)),
        "test_recall": float(recall_score(y_te, yte_pred, average="weighted", zero_division=0)),
        "train_f1": float(f1_score(y_tr, ytr_pred, average="weighted", zero_division=0)),
        "test_f1": float(f1_score(y_te, yte_pred, average="weighted", zero_division=0)),
        "best_n_estimators": int(model.n_estimators),
        "best_max_features": None if model.max_features is None else str(model.max_features),
    }

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    pd.DataFrame(
        {"feature": X_tr.columns, "importance": model.feature_importances_}
    ).to_csv(out_dir / "importance.csv", index=False)

    return metrics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_max_features(values: Sequence[str]) -> List[object]:
    out: List[object] = []
    for v in values:
        v2 = v.strip()
        if v2.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(v2)
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nested-CV RandomForest for SI (5-fold × 3 repeats).")
    p.add_argument("--labels", type=Path, default=None, help="Path to FeN6-SSD_500_spin_labeled.csv")
    p.add_argument("--descriptors", type=Path, default=None, help="Path to descriptors/ directory")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: ML/.../results_rf)")
    p.add_argument(
        "--states",
        nargs="+",
        default=list(DEFAULT_STATES),
        help='Spin-state values to run (default: "high-spin low-spin")',
    )
    p.add_argument(
        "--n-estimators",
        nargs="+",
        type=int,
        default=list(DEFAULT_N_ESTIMATORS),
        help="Grid for n_estimators (default: 50 100 300)",
    )
    p.add_argument(
        "--max-features",
        nargs="+",
        default=list(DEFAULT_MAX_FEATURES),
        help='Grid for max_features (default: None sqrt log2). Use "None" for None.',
    )
    p.add_argument("--no-pre-filter", action="store_true", help="Disable pre-split filtering.")
    p.add_argument("--no-fold-filter", action="store_true", help="Disable per-fold filtering.")
    p.add_argument("--var-thresh", type=float, default=VAR_THRESH, help="Variance threshold.")
    p.add_argument("--corr-thresh", type=float, default=CORR_THRESH, help="Correlation threshold.")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO, WARNING, ...)")
    return p


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)

    labels_path = args.labels or DEFAULT_LABEL_CSV
    desc_root = args.descriptors or DEFAULT_DESC_DIR
    out_root = args.out_dir or DEFAULT_OUT_DIR

    ensure_dir(out_root)

    filter_cfg = FeatureFilterConfig(
        pre_split_filter=not args.no_pre_filter,
        per_fold_filter=not args.no_fold_filter,
        var_thresh=float(args.var_thresh),
        corr_thresh=float(args.corr_thresh),
    )

    n_estimators_grid = list(args.n_estimators)
    max_features_grid = parse_max_features(args.max_features)

    LOGGER.info("Repo root: %s", ROOT_DIR)
    LOGGER.info("Labels: %s", labels_path)
    LOGGER.info("Descriptors: %s", desc_root)
    LOGGER.info("Output: %s", out_root)
    LOGGER.info(
        "Grid: n_estimators=%s, max_features=%s",
        n_estimators_grid,
        [("None" if v is None else v) for v in max_features_grid],
    )
    LOGGER.info(
        "Filtering: pre_split=%s, per_fold=%s, var_thresh=%g, corr_thresh=%g",
        filter_cfg.pre_split_filter,
        filter_cfg.per_fold_filter,
        filter_cfg.var_thresh,
        filter_cfg.corr_thresh,
    )

    labels = read_labels(labels_path)
    descriptors = load_descriptors(desc_root)

    for state in args.states:
        LOGGER.info("State: %s", state)

        y_all = labels.loc[labels[COL_STATE] == state, COL_BEHAV].copy()
        y_all = y_all.replace({"nan": np.nan}).dropna()
        if y_all.empty:
            LOGGER.warning("No labeled rows found for state '%s'. Skipped.", state)
            continue

        master_ids = make_master_ids(y_all, descriptors)
        if len(master_ids) == 0:
            LOGGER.warning("No common IDs across labels and descriptors for state '%s'. Skipped.", state)
            continue

        y_master = y_all.loc[master_ids]
        splits = build_outer_splits(y_master)
        fnames = fold_names(len(splits))

        # Align descriptor tables to master_ids
        X_master_dict: Dict[str, pd.DataFrame] = {d: X.reindex(master_ids) for d, X in descriptors.items()}

        # Pre-split feature selection per descriptor/state (computed once; then applied to each fold)
        pre_cols_dict: Dict[str, List[str]] = {}
        if filter_cfg.pre_split_filter:
            for d, X in X_master_dict.items():
                Xp = X.copy()
                finite = finite_columns_mask(Xp)
                Xp = Xp.loc[:, finite]
                keep = drop_low_variance_cols(Xp, filter_cfg.var_thresh)
                Xp = Xp.loc[:, keep]
                Xp = select_low_correlation_cols_fullcorr(Xp, filter_cfg.corr_thresh)
                pre_cols_dict[d] = list(Xp.columns)
        else:
            for d, X in X_master_dict.items():
                pre_cols_dict[d] = list(X.columns)

        rows: List[dict] = []
        for desc_name, Xmat in X_master_dict.items():
            for (tr, te), fname in zip(splits, fnames):
                fold_out = out_root / state / desc_name / fname
                res = run_fold(
                    state=state,
                    descriptor=desc_name,
                    fold=fname,
                    X_master=Xmat,
                    y_master=y_master,
                    tr_idx=tr,
                    te_idx=te,
                    pre_cols=pre_cols_dict[desc_name],
                    out_dir=fold_out,
                    n_estimators_grid=n_estimators_grid,
                    max_features_grid=max_features_grid,
                    filter_cfg=filter_cfg,
                )
                rows.append(res)

        if rows:
            summary_path = out_root / f"summary_{state}.csv"
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            LOGGER.info("Saved: %s", summary_path)


if __name__ == "__main__":
    main()
