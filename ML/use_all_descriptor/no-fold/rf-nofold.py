#!/usr/bin/env python3
from __future__ import annotations

"""
Compute SHAP values for ECFP4-based RandomForest models (no outer CV).

Purpose
-------
This script retrains RandomForest classifiers and computes SHAP values for the
ECFP4-based contribution analysis reported in the manuscript (e.g., Figures 6 and S2).

Scope (intentionally limited)
-----------------------------
- Only ECFP4-family descriptors are supported:
    * ecfp4
    * ecfp4-fe
    * ecfp4_bind_otherbinary        (ECFP4 + env., binary)
    * ecfp4-fe_bind_otherbinary     (ECFP4-Fe + env., binary)
- This repository performs SHAP value calculation only. Projection of SHAP values
  onto molecular drawings and figure generation are done separately.

Method summary
--------------
- Repo-root discovery (search for descriptors/, ML/, and FeN6-SSD label CSV)
- ID/label normalization (strip whitespace)
- Preprocessing on the selected descriptor:
    * drop non-finite columns
    * low-variance filter
    * correlation filter (|r| >= 0.90)
- Hyperparameter selection via GridSearchCV with StratifiedKFold using MCC
- Train on the full dataset for the chosen spin state (no outer CV split)
- Compute SHAP values using shap.TreeExplainer

Outputs
-------
results_nofold/<state>/<descriptor>/
  - metrics.csv                  (key,value)
  - confusion_matrix.csv
  - features_used.txt
  - feature_importance.csv
  - best_params.csv              (best hyperparameters from GridSearchCV)
  - shap_values_all.csv

Notes
-----
- This script intentionally computes SHAP values on ALL available samples for each
  requested spin state. There is no subsampling option in order to avoid misleading
  command-line flags in a manuscript-reproducibility repository.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

THIS_FILE = Path(__file__).resolve()
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Repo / paths
# -----------------------------------------------------------------------------


def find_repo_root(start: Path) -> Path:
    """
    Find repository root by searching parent directories for:
      - descriptors/
      - ML/
      - FeN6-SSD/FeN6-SSD_500_spin_labeled.csv
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

DEFAULT_DESC_DIR = ROOT_DIR / "descriptors"
DEFAULT_LABEL_CSV = ROOT_DIR / "FeN6-SSD" / "FeN6-SSD_500_spin_labeled.csv"
DEFAULT_OUT_DIR = THIS_FILE.parent / "results_nofold"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COL_ID = "ccdc_id"
COL_STATE = "Spin-state of crystal structure"
COL_BEHAV = "Spin-state behaviour"

DEFAULT_STATES = ("high-spin", "low-spin")
POS_LABEL = "spin-crossover"

VAR_THRESH = 1e-12
CORR_THRESH = 0.90

DEFAULT_N_ESTIMATORS = (50, 100, 300)
DEFAULT_MAX_FEATURES = ("None", "sqrt", "log2")  # parsed to [None, "sqrt", "log2"]

# Only allow ECFP4-family descriptor folder names in this script.
ALLOWED_ECFP4_DESCRIPTORS = (
    "ecfp4",
    "ecfp4-fe",
    "ecfp4_bind_otherbinary",
    "ecfp4-fe_bind_otherbinary",
)


@dataclass(frozen=True)
class ShapConfig:
    """
    SHAP configuration.
    For manuscript reproducibility, SHAP is computed on ALL samples.
    """
    background_samples: int = 100
    model_output: str = "probability"
    feature_perturbation: str = "interventional"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_labels(path: Path) -> pd.DataFrame:
    """
    Read label CSV and normalize ID/state/behavior strings (strip).
    Returns DataFrame indexed by ccdc_id.
    """
    df = pd.read_csv(path)
    missing = [c for c in (COL_ID, COL_STATE, COL_BEHAV) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in label CSV: {missing}")

    df[COL_ID] = df[COL_ID].astype(str).str.strip()
    df[COL_STATE] = df[COL_STATE].astype(str).str.strip()
    df[COL_BEHAV] = df[COL_BEHAV].astype(str).str.strip()
    return df.set_index(COL_ID)


def load_one_descriptor(desc_root: Path, name: str) -> pd.DataFrame:
    """
    Load a single descriptor table from:
      <desc_root>/<name>/<name>.csv

    Cleans:
      - coerces non-numeric to NaN
      - keeps numeric columns only
      - drops columns containing NaN/inf
    """
    if name not in ALLOWED_ECFP4_DESCRIPTORS:
        raise ValueError(
            f"Unsupported descriptor '{name}'. Allowed (ECFP4-family only): {ALLOWED_ECFP4_DESCRIPTORS}"
        )

    sub = desc_root / name
    csv_path = sub / f"{name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Descriptor CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str).str.strip()

    # Keep numeric columns only; coerce non-numeric to NaN then drop NaN columns.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1)

    if df.shape[1] == 0:
        raise RuntimeError(f"Descriptor '{name}' has no usable numeric columns after cleaning.")

    LOGGER.info("Loaded descriptor '%s': %s", name, df.shape)
    return df


def finite_columns_mask(X: pd.DataFrame) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0, dtype=bool)
    return np.isfinite(X.to_numpy()).all(axis=0)


def drop_low_variance_cols(X: pd.DataFrame, var_thresh: float) -> List[str]:
    if X.shape[1] == 0:
        return []
    v = X.var(axis=0, ddof=1)
    return v[v > var_thresh].index.tolist()


def select_low_correlation_cols_fullcorr(
    X: pd.DataFrame,
    corr_thresh: float,
    use_abs: bool = True,
) -> pd.DataFrame:
    """
    Greedy selection based on variance ordering:
    keep a column if its max correlation with already-kept columns < corr_thresh.
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


def build_xy(
    labels: pd.DataFrame,
    X_desc: pd.DataFrame,
    spin_state: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build X/y for one spin state, restricting to IDs present in BOTH labels and descriptor.
    y is binarized to:
      POS_LABEL ('spin-crossover') vs spin_state ('high-spin' or 'low-spin').
    """
    y_state = labels.loc[labels[COL_STATE] == spin_state, COL_BEHAV].astype(str)

    master_ids = y_state.index.intersection(X_desc.index)
    if len(master_ids) == 0:
        raise RuntimeError(f"No common IDs between labels and descriptor for state='{spin_state}'.")

    y = y_state.loc[master_ids].copy()
    X = X_desc.reindex(master_ids).copy()

    # Binarize labels
    y = y.apply(lambda v: POS_LABEL if v == POS_LABEL else spin_state)

    # Drop rows with any NaN (safety)
    keep_rows = X.notna().all(axis=1)
    X = X.loc[keep_rows]
    y = y.loc[X.index]

    if X.empty:
        raise RuntimeError(f"All samples dropped after cleaning for state='{spin_state}'.")
    return X, y


def preprocess_X(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess features:
      1) drop non-finite columns
      2) low-variance filter
      3) correlation filter
    """
    X = X.loc[:, finite_columns_mask(X)]
    X = X.loc[:, drop_low_variance_cols(X, VAR_THRESH)]
    X = select_low_correlation_cols_fullcorr(X, CORR_THRESH)
    return X


def parse_max_features(values: Sequence[str]) -> List[object]:
    out: List[object] = []
    for v in values:
        v2 = v.strip()
        if v2.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(v2)
    return out


def find_pos_class_index(classes: Sequence[object], positive_label: str) -> int:
    """
    Find which class index corresponds to the positive label in predict_proba / shap outputs.
    Includes a few fallbacks for robustness.
    """
    cls = np.array(classes).astype(str)
    lower = np.char.lower(cls)
    target = positive_label.casefold()

    hit = np.where(lower == target)[0]
    if hit.size:
        return int(hit[0])

    # fallback heuristics
    hit2 = np.where([("spin" in c and "cross" in c) for c in lower])[0]
    if hit2.size:
        return int(hit2[0])

    hit3 = np.where([(c == "sc") or ("sco" in c) for c in lower])[0]
    if hit3.size:
        return int(hit3[0])

    if len(cls) == 2:
        return 1
    return 0


def stratified_background_sample(
    X: pd.DataFrame,
    y: pd.Series,
    n_total: int,
    seed: int,
) -> pd.DataFrame:
    """
    Stratified sample for SHAP background dataset.
    If X has <= n_total rows, returns X.
    """
    if len(X) <= n_total:
        return X

    n_cls = pd.Series(y).nunique()
    per = max(1, n_total // max(1, n_cls))

    Xbg = (
        X.assign(_y=y)
        .groupby("_y", group_keys=False)
        .apply(lambda d: d.sample(n=min(len(d), per), random_state=seed))
        .drop(columns=["_y"])
    )

    if len(Xbg) > n_total:
        Xbg = Xbg.sample(n=n_total, random_state=seed)

    return Xbg


# -----------------------------------------------------------------------------
# Training + SHAP
# -----------------------------------------------------------------------------


def train_rf_with_mcc_gridsearch(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a RandomForest model on the full dataset with MCC-based grid search.
    Returns the fitted model and best hyperparameters.
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=seed),
        param_grid={"n_estimators": list(n_estimators_grid), "max_features": list(max_features_grid)},
        cv=inner_cv,
        scoring=make_scorer(matthews_corrcoef),
        n_jobs=-1,
    )
    grid.fit(X, y)
    best_params = dict(grid.best_params_)

    model = RandomForestClassifier(
        **best_params,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)
    return model, best_params


def evaluate_and_save(
    model: RandomForestClassifier,
    best_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    out_dir: Path,
    seed: int,
    shap_cfg: ShapConfig,
) -> None:
    """
    Evaluate on the same dataset (no outer CV; for interpretation use) and save:
      - metrics
      - confusion matrix
      - feature importance
      - SHAP values on all samples
    """
    ensure_dir(out_dir)

    y_pred = model.predict(X)

    prob_pos: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) == 2:
        pos_idx = find_pos_class_index(np.array(model.classes_), POS_LABEL)
        prob_pos = model.predict_proba(X)[:, pos_idx]

    metrics = {
        "seed": int(seed),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "recall": float(recall_score(y, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "f1": float(f1_score(y, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
    }
    if prob_pos is not None:
        metrics["roc_auc"] = float(roc_auc_score((y == POS_LABEL).astype(int), prob_pos))

    pd.Series(metrics).to_csv(out_dir / "metrics.csv", header=False)
    pd.Series(best_params).to_csv(out_dir / "best_params.csv", header=False)

    classes = list(getattr(model, "classes_", sorted(pd.Series(y).unique())))
    cm = confusion_matrix(y, y_pred, labels=classes)
    pd.DataFrame(
        cm,
        index=[f"True {c}" for c in classes],
        columns=[f"Pred {c}" for c in classes],
    ).to_csv(out_dir / "confusion_matrix.csv")

    pd.Series(X.columns.astype(str)).to_csv(out_dir / "features_used.txt", index=False, header=False)
    pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).to_csv(
        out_dir / "feature_importance.csv",
        index=False,
    )

    # SHAP values on ALL samples
    Xbg = stratified_background_sample(X, y, shap_cfg.background_samples, seed)
    if Xbg.empty:
        Xbg = X

    explainer = shap.TreeExplainer(
        model,
        data=Xbg,
        model_output=shap_cfg.model_output,
        feature_perturbation=shap_cfg.feature_perturbation,
    )
    vals = explainer.shap_values(X, check_additivity=False)

    # shap may return a list per class; pick POS_LABEL class
    if isinstance(vals, list) and len(vals) >= 2:
        pos_idx2 = find_pos_class_index(np.array(getattr(model, "classes_", [])), POS_LABEL)
        vals = vals[pos_idx2]

    pd.DataFrame(vals, index=X.index, columns=X.columns).to_csv(out_dir / "shap_values_all.csv")


# -----------------------------------------------------------------------------
# CLI / Main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ECFP4-family RandomForest retraining + SHAP value calculation (Figures 6 / S2)."
    )
    p.add_argument("--labels", type=Path, default=None, help="Path to FeN6-SSD_500_spin_labeled.csv")
    p.add_argument("--descriptors", type=Path, default=None, help="Path to descriptors/ directory")
    p.add_argument(
        "--descriptor",
        choices=list(ALLOWED_ECFP4_DESCRIPTORS),
        default="ecfp4",
        help="ECFP4-family descriptor folder name",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: ./results_nofold)")
    p.add_argument("--states", choices=["both", "high-spin", "low-spin"], default="both")
    p.add_argument("--seed", type=int, default=0)

    # Keep only a background-sampling knob (not misleading for dataset size).
    p.add_argument("--shap_background_samples", type=int, default=100)

    p.add_argument("--n-estimators", nargs="+", type=int, default=list(DEFAULT_N_ESTIMATORS))
    p.add_argument("--max-features", nargs="+", default=list(DEFAULT_MAX_FEATURES))

    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)

    labels_path = args.labels or DEFAULT_LABEL_CSV
    desc_root = args.descriptors or DEFAULT_DESC_DIR
    out_root = args.out_dir or DEFAULT_OUT_DIR

    states = {
        "both": DEFAULT_STATES,
        "high-spin": ("high-spin",),
        "low-spin": ("low-spin",),
    }[args.states]

    ensure_dir(out_root)

    LOGGER.info("Repo root: %s", ROOT_DIR)
    LOGGER.info("Labels: %s", labels_path)
    LOGGER.info("Descriptors: %s", desc_root)
    LOGGER.info("Descriptor name: %s", args.descriptor)
    LOGGER.info("Output: %s", out_root)

    labels = read_labels(labels_path)
    X_desc = load_one_descriptor(desc_root, args.descriptor)

    n_estimators_grid = list(args.n_estimators)
    max_features_grid = parse_max_features(args.max_features)

    shap_cfg = ShapConfig(
        background_samples=int(args.shap_background_samples),
    )

    for state in states:
        try:
            X0, y0 = build_xy(labels, X_desc, state)
            X = preprocess_X(X0)
        except Exception as e:
            LOGGER.warning("State=%s: skipped (%s)", state, str(e))
            continue

        if X.shape[1] == 0:
            LOGGER.warning("State=%s: skipped (all features dropped).", state)
            continue

        out_dir = out_root / state / args.descriptor
        LOGGER.info("Run state=%s desc=%s: n=%d p=%d", state, args.descriptor, X.shape[0], X.shape[1])

        model, best_params = train_rf_with_mcc_gridsearch(
            X,
            y0.loc[X.index],
            seed=int(args.seed),
            n_estimators_grid=n_estimators_grid,
            max_features_grid=max_features_grid,
        )

        evaluate_and_save(
            model,
            best_params,
            X,
            y0.loc[X.index],
            out_dir=out_dir,
            seed=int(args.seed),
            shap_cfg=shap_cfg,
        )


if __name__ == "__main__":
    main()
