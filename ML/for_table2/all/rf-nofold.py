#!/usr/bin/env python3
from __future__ import annotations

import os

# -----------------------------------------------------------------------------
# Default thread caps
# - Apply only when not explicitly set from the shell/environment.
# - Must be done BEFORE importing numpy / pandas / sklearn / shap.
# -----------------------------------------------------------------------------
for _env_key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_key, "1")

"""
Train final RandomForest models on ALL data using hyperparameters aggregated from
nested-CV results in ML/for_table2/5-fold/results_rf, with optional SHAP outputs.

Design:
- Reuse summary_<state>.csv from 5-fold/results_rf
- Fit preprocessing on ALL data
- No imputation
- MBTR-safe streaming correlation pruning (no full corr matrix)
- Save trained final model (.joblib)
- Optionally compute SHAP on the final trained model

Default layout:
  this file: ML/for_table2/all/rf_final_from_5fold.py
  CV results: ML/for_table2/5-fold/results_rf/
  outputs   : ML/for_table2/all/results_rf/

Outputs:
<out_dir>/<state>/<descriptor>/
  - ids_all.csv
  - best_params.csv
  - apparent_metrics.csv
  - importance.csv
  - selected_features.csv
  - pruning_report.json
  - model.joblib
  - shap_mean_abs.csv                 # if --compute-shap
  - shap_top_features.csv             # if --compute-shap
  - shap_values.csv.gz                # if --compute-shap --save-shap-matrix

Note:
- apparent_metrics.csv is evaluated on the same data used for training.
  It is not an unbiased estimate.
- SHAP is also computed on the same data used for final training.
"""

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

THIS_FILE = Path(__file__).resolve()
LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Columns / constants
# -----------------------------------------------------------------------------

COL_ID = "ccdc_id"
COL_STATE = "Spin-state of crystal structure"
COL_TARGET = "Spin-state behaviour"

DEFAULT_STATES = ("high-spin", "low-spin")

VAR_THRESH_DEFAULT = 1e-12
CORR_THRESH_DEFAULT = 0.90
SEED = 42

FLOAT32_MAX = float(np.finfo(np.float32).max)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureFilterConfig:
    var_thresh: float = VAR_THRESH_DEFAULT
    corr_thresh: float = CORR_THRESH_DEFAULT
    use_abs_corr: bool = True
    corr_block_size: int = 512
    corr_keep_first: int = 1
    corr_max_keep: int = 0
    corr_compute_dtype: str = "float64"  # "float64" or "float32"

# -----------------------------------------------------------------------------
# Repo-root discovery
# -----------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
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
DEFAULT_CV_RESULTS = ROOT_DIR / "ML" / "for_table2" / "5-fold" / "results_rf"
DEFAULT_OUT_DIR = THIS_FILE.parent / "results_rf"

# -----------------------------------------------------------------------------
# Logging / helpers
# -----------------------------------------------------------------------------

def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_run_config(out_root: Path, args: argparse.Namespace) -> None:
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
        cfg["joblib"] = joblib.__version__
    except Exception:
        cfg["joblib"] = None
    try:
        import shap  # noqa
        cfg["shap"] = shap.__version__
    except Exception:
        cfg["shap"] = None

    (out_root / "run_config.json").write_text(
        pd.Series(cfg).to_json(indent=2, force_ascii=False),
        encoding="utf-8",
    )

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def read_labels(labels_path: Path) -> pd.DataFrame:
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

# -----------------------------------------------------------------------------
# Preprocessing helpers
# -----------------------------------------------------------------------------

def drop_any_nan_cols(X: pd.DataFrame) -> List[str]:
    if X.shape[1] == 0:
        return []
    ok = ~X.isna().any(axis=0)
    return X.columns[ok].tolist()

def drop_low_variance_cols(X: pd.DataFrame, var_thresh: float) -> List[str]:
    if X.shape[1] == 0:
        return []
    v = X.astype(np.float64, copy=False).var(axis=0, ddof=1)
    return v[v > var_thresh].index.tolist()

def _standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
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
) -> Tuple[List[str], dict]:
    report = {
        "method": "streaming_greedy",
        "corr_thresh": float(corr_thresh),
        "use_abs": bool(use_abs),
        "block_size": int(block_size),
        "keep_first": int(keep_first),
        "max_keep": int(max_keep),
        "compute_dtype": str(compute_dtype),
        "n_rows": int(X.shape[0]),
        "n_cols_in": int(X.shape[1]),
        "n_cols_out": 0,
    }

    if X.shape[1] == 0:
        return [], report
    if X.shape[1] == 1:
        cols = list(X.columns)
        report["n_cols_out"] = len(cols)
        return cols, report

    dtype = np.float64 if compute_dtype == "float64" else np.float32
    A = X.to_numpy(dtype=dtype, copy=False)
    cols_all = np.asarray(X.columns, dtype=object)

    finite_mask = np.isfinite(A).all(axis=0)
    A = A[:, finite_mask]
    cols = cols_all[finite_mask]
    if A.shape[1] <= 1:
        out_cols = cols.astype(str).tolist()
        report["n_cols_out"] = len(out_cols)
        return out_cols, report

    mu, sd = _standardize(A.astype(np.float64, copy=False))
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    A = A[:, ok]
    cols = cols[ok]
    sd = sd[ok]
    if A.shape[1] <= 1:
        out_cols = cols.astype(str).tolist()
        report["n_cols_out"] = len(out_cols)
        return out_cols, report

    Z = (A.astype(np.float64, copy=False) - mu[ok]) / sd
    order = np.argsort(sd)[::-1]
    denom = max(Z.shape[0] - 1, 1)

    initial_take = min(max(int(keep_first), 1), len(order))
    keep_idx: List[int] = order[:initial_take].tolist()
    K = Z[:, keep_idx].copy()

    if max_keep and len(keep_idx) >= max_keep:
        out_cols = cols[keep_idx].astype(str).tolist()
        report["n_cols_out"] = len(out_cols)
        return out_cols, report

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

    out_cols = cols[keep_idx].astype(str).tolist()
    report["n_cols_out"] = len(out_cols)
    return out_cols, report

def preprocess_final_X(X_raw: pd.DataFrame, filter_cfg: FeatureFilterConfig) -> Tuple[pd.DataFrame, dict]:
    report: dict = {"steps": {}}

    X = X_raw.replace([np.inf, -np.inf], np.nan)
    X = X.mask(np.abs(X.astype(np.float64, copy=False)) > FLOAT32_MAX, np.nan)
    report["steps"]["mask_huge_to_nan"] = {"float32_max": FLOAT32_MAX}

    keep_no_nan = drop_any_nan_cols(X)
    report["steps"]["drop_any_nan_cols"] = {"n_in": int(X.shape[1]), "n_out": int(len(keep_no_nan))}
    X = X.loc[:, keep_no_nan]
    if X.shape[1] == 0:
        return X, report

    keep_var = drop_low_variance_cols(X, filter_cfg.var_thresh)
    report["steps"]["variance_filter"] = {
        "var_thresh": float(filter_cfg.var_thresh),
        "n_in": int(X.shape[1]),
        "n_out": int(len(keep_var)),
    }
    X = X.loc[:, keep_var]
    if X.shape[1] == 0:
        return X, report

    keep_corr, corr_report = select_low_correlation_cols_streaming(
        X,
        filter_cfg.corr_thresh,
        use_abs=filter_cfg.use_abs_corr,
        block_size=int(filter_cfg.corr_block_size),
        keep_first=int(filter_cfg.corr_keep_first),
        max_keep=int(filter_cfg.corr_max_keep),
        compute_dtype=str(filter_cfg.corr_compute_dtype),
    )
    report["steps"]["corr_prune"] = corr_report
    X = X.loc[:, keep_corr]

    return X, report

# -----------------------------------------------------------------------------
# Hyperparameter aggregation
# -----------------------------------------------------------------------------

def parse_best_max_features(v: object) -> object:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    s = str(v).strip()
    if s.lower() in {"none", "null", "nan", "<na>"}:
        return None
    return s

def pick_final_params_from_summary(summary_csv: Path, *, descriptor: str) -> Tuple[int, object]:
    df = pd.read_csv(summary_csv)
    for c in ["descriptor", "best_n_estimators", "best_max_features"]:
        if c not in df.columns:
            raise KeyError(f"{summary_csv} missing column: {c}")

    d = df.loc[df["descriptor"] == descriptor].copy()
    if d.empty:
        raise RuntimeError(f"No rows for descriptor='{descriptor}' in {summary_csv}")

    d["best_n_estimators"] = pd.to_numeric(d["best_n_estimators"], errors="coerce").astype("Int64")
    d["best_max_features"] = d["best_max_features"].apply(parse_best_max_features).astype(object)
    d = d.dropna(subset=["best_n_estimators"])
    if d.empty:
        raise RuntimeError(f"All best_n_estimators are NaN for descriptor='{descriptor}' in {summary_csv}")

    grp = d.groupby(["best_n_estimators", "best_max_features"], dropna=False)
    counts = grp.size().rename("count").reset_index()

    max_count = int(counts["count"].max())
    top = counts.loc[counts["count"] == max_count].copy()

    def _normalize_pair(ne: object, mf: object) -> Tuple[int, object]:
        return int(ne), parse_best_max_features(mf)

    if len(top) == 1:
        row = top.iloc[0]
        return _normalize_pair(row["best_n_estimators"], row["best_max_features"])

    if "test_mcc" in d.columns:
        scored: List[Tuple[float, Tuple[int, object]]] = []
        for _, r in top.iterrows():
            ne, mf = _normalize_pair(r["best_n_estimators"], r["best_max_features"])
            sub = d.loc[
                (d["best_n_estimators"].astype("Int64") == ne) &
                (d["best_max_features"].apply(parse_best_max_features) == mf)
            ]
            x = pd.to_numeric(sub["test_mcc"], errors="coerce")
            mean_mcc = float(np.nanmean(x.to_numpy())) if len(x) else float("-inf")
            scored.append((mean_mcc, (ne, mf)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    top = top.sort_values(["best_n_estimators"], ascending=False)
    row = top.iloc[0]
    return _normalize_pair(row["best_n_estimators"], row["best_max_features"])

# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def apparent_metrics_binary(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    y_pred = model.predict(X)

    out: Dict[str, object] = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(y, y_pred)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "precision_weighted": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
    }

    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) == 2:
        proba = model.predict_proba(X)
        y_true = (y.astype(str) == str(model.classes_[1])).astype(int)
        out["roc_auc_(pos=classes_[1])"] = float(roc_auc_score(y_true, proba[:, 1]))

    classes = list(getattr(model, "classes_", sorted(pd.Series(y).unique())))
    cm = confusion_matrix(y, y_pred, labels=classes)
    out["classes"] = "|".join([str(c) for c in classes])
    out["confusion_matrix_flat"] = "|".join([str(int(v)) for v in cm.ravel()])
    return out

# -----------------------------------------------------------------------------
# SHAP helpers
# -----------------------------------------------------------------------------

def _subsample_for_shap(X: pd.DataFrame, n_samples: int, seed: int) -> pd.DataFrame:
    if n_samples <= 0 or len(X) <= n_samples:
        return X
    return X.sample(n=n_samples, random_state=seed)

def _extract_positive_class_shap_array(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    positive_label: Optional[str] = None,
) -> np.ndarray:
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

def compute_and_save_final_shap(
    *,
    model: RandomForestClassifier,
    X: pd.DataFrame,
    out_dir: Path,
    positive_label: Optional[str],
    save_shap_matrix: bool,
    shap_sample: int,
    seed: int,
) -> None:
    ensure_dir(out_dir)

    X_shap = _subsample_for_shap(X, n_samples=shap_sample, seed=seed)
    LOGGER.info("SHAP: using n=%d samples", int(X_shap.shape[0]))

    shap_arr = _extract_positive_class_shap_array(
        model=model,
        X=X_shap,
        positive_label=positive_label,
    )

    mean_abs = np.abs(shap_arr).mean(axis=0)
    shap_mean_df = pd.DataFrame(
        {
            "feature": X_shap.columns.astype(str),
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_mean_df.to_csv(out_dir / "shap_mean_abs.csv", index=False)
    shap_mean_df.head(50).to_csv(out_dir / "shap_top_features.csv", index=False)

    if save_shap_matrix:
        shap_mat = pd.DataFrame(
            shap_arr.astype(np.float32),
            index=X_shap.index.astype(str),
            columns=X_shap.columns.astype(str),
        )
        shap_mat.index.name = COL_ID
        shap_mat.to_csv(out_dir / "shap_values.csv.gz", compression="gzip")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train final RF models on ALL data using params from 5-fold/results_rf."
    )
    p.add_argument("--labels", type=Path, default=None, help="Path to FeN6-SSD_500_spin_labeled.csv")
    p.add_argument("--descriptors", type=Path, default=None, help="Path to descriptors/ directory")
    p.add_argument("--cv-results", type=Path, default=None, help="Path to 5-fold/results_rf/")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")

    p.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    p.add_argument("--descs", nargs="*", default=None)

    p.add_argument("--var-thresh", type=float, default=VAR_THRESH_DEFAULT)
    p.add_argument("--corr-thresh", type=float, default=CORR_THRESH_DEFAULT)
    p.add_argument("--signed-corr", action="store_true")
    p.add_argument("--no-corr-prune", action="store_true")

    p.add_argument("--corr-block-size", type=int, default=512)
    p.add_argument("--corr-keep-first", type=int, default=1)
    p.add_argument("--corr-max-keep", type=int, default=0)
    p.add_argument("--corr-dtype", choices=["float64", "float32"], default="float64")

    p.add_argument("--rf-n-jobs", type=int, default=1)
    p.add_argument("--float32-mode", action="store_true")

    p.add_argument("--compute-shap", action="store_true", help="Compute SHAP on final fitted model.")
    p.add_argument("--save-shap-matrix", action="store_true", help="Also save sample-wise SHAP matrix (.csv.gz).")
    p.add_argument(
        "--shap-sample",
        type=int,
        default=0,
        help="If >0, randomly subsample this many rows for SHAP. 0 means use all rows.",
    )
    p.add_argument(
        "--shap-positive-label",
        type=str,
        default=None,
        help="Positive class label for SHAP extraction. Default: classes_[1] if not specified.",
    )

    p.add_argument("--log-level", default="INFO")
    return p

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)

    labels_path = args.labels or DEFAULT_LABEL_CSV
    desc_root = args.descriptors or DEFAULT_DESC_DIR
    cv_root = args.cv_results or DEFAULT_CV_RESULTS
    out_root = args.out_dir or DEFAULT_OUT_DIR

    ensure_dir(out_root)
    write_run_config(out_root, args)

    filter_cfg = FeatureFilterConfig(
        var_thresh=float(args.var_thresh),
        corr_thresh=float(args.corr_thresh),
        use_abs_corr=not bool(args.signed_corr),
        corr_block_size=int(args.corr_block_size),
        corr_keep_first=int(args.corr_keep_first),
        corr_max_keep=int(args.corr_max_keep),
        corr_compute_dtype=str(args.corr_dtype),
    )

    LOGGER.info("Repo root   : %s", ROOT_DIR)
    LOGGER.info("Labels      : %s", labels_path)
    LOGGER.info("Descriptors : %s", desc_root)
    LOGGER.info("CV results  : %s", cv_root)
    LOGGER.info("Output      : %s", out_root)
    LOGGER.info(
        "Thread env  : OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s",
        os.environ.get("OMP_NUM_THREADS"),
        os.environ.get("MKL_NUM_THREADS"),
        os.environ.get("OPENBLAS_NUM_THREADS"),
        os.environ.get("NUMEXPR_NUM_THREADS"),
    )
    LOGGER.info("RF n_jobs   : %d", int(args.rf_n_jobs))
    LOGGER.info(
        "SHAP        : %s | save_matrix=%s | sample=%s | positive_label=%s",
        "ON" if args.compute_shap else "OFF",
        "ON" if args.save_shap_matrix else "OFF",
        ("ALL" if int(args.shap_sample) <= 0 else str(int(args.shap_sample))),
        str(args.shap_positive_label),
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
        LOGGER.info("Filtered descriptors: %d -> %s", len(descriptors), sorted(descriptors.keys()))
        if not descriptors:
            raise RuntimeError(f"--descs specified but none matched. Wanted={sorted(wanted)}")

    for state in args.states:
        LOGGER.info("State: %s", state)

        summary_csv = cv_root / f"summary_{state}.csv"
        if not summary_csv.is_file():
            LOGGER.warning("Missing summary for state '%s': %s (skipped)", state, summary_csv)
            continue

        y_all = labels.loc[labels[COL_STATE] == state, COL_TARGET].copy()
        y_all = y_all.replace({"nan": np.nan, "NaN": np.nan}).dropna()
        if y_all.empty:
            LOGGER.warning("No labeled rows for state '%s'. Skipped.", state)
            continue

        for desc_name, X_desc in descriptors.items():
            try:
                n_estimators, max_features = pick_final_params_from_summary(
                    summary_csv,
                    descriptor=desc_name,
                )
            except Exception as e:
                LOGGER.warning("State=%s desc=%s: params not found (%s). Skipped.", state, desc_name, str(e))
                continue

            max_features = parse_best_max_features(max_features)

            ids = y_all.index.intersection(X_desc.index)
            if len(ids) == 0:
                LOGGER.warning("State=%s desc=%s: no common IDs. Skipped.", state, desc_name)
                continue

            X0 = X_desc.reindex(ids).copy()
            y0 = y_all.loc[ids].copy()

            X0 = X0.replace([np.inf, -np.inf], np.nan)
            X0 = X0.mask(np.abs(X0.astype(np.float64, copy=False)) > FLOAT32_MAX, np.nan)

            keep_no_nan = drop_any_nan_cols(X0)
            X1 = X0.loc[:, keep_no_nan]

            if X1.shape[1] == 0:
                LOGGER.warning("State=%s desc=%s: all features dropped after NaN removal. Skipped.", state, desc_name)
                continue

            keep_var = drop_low_variance_cols(X1, filter_cfg.var_thresh)
            X2 = X1.loc[:, keep_var]

            if X2.shape[1] == 0:
                LOGGER.warning("State=%s desc=%s: all features dropped after variance filter. Skipped.", state, desc_name)
                continue

            if args.no_corr_prune:
                X = X2
                prune_report = {
                    "steps": {
                        "mask_huge_to_nan": {"float32_max": FLOAT32_MAX},
                        "drop_any_nan_cols": {"n_in": int(X0.shape[1]), "n_out": int(X1.shape[1])},
                        "variance_filter": {
                            "var_thresh": float(filter_cfg.var_thresh),
                            "n_in": int(X1.shape[1]),
                            "n_out": int(X2.shape[1]),
                        },
                        "corr_prune": {
                            "method": "disabled",
                            "n_cols_in": int(X2.shape[1]),
                            "n_cols_out": int(X2.shape[1]),
                        },
                    }
                }
            else:
                X, prune_report = preprocess_final_X(X0, filter_cfg)

            if X.shape[1] == 0:
                LOGGER.warning("State=%s desc=%s: all features dropped. Skipped.", state, desc_name)
                continue

            ok_rows = np.isfinite(X.to_numpy(dtype=np.float64, copy=False)).all(axis=1)
            X = X.loc[ok_rows]
            y = y0.loc[X.index]
            if X.empty:
                LOGGER.warning("State=%s desc=%s: empty after row filtering. Skipped.", state, desc_name)
                continue

            out_dir = out_root / state / desc_name
            ensure_dir(out_dir)

            pd.Series(X.index, name=COL_ID).to_csv(out_dir / "ids_all.csv", index=False)
            pd.Series(X.columns, name="feature").to_csv(out_dir / "selected_features.csv", index=False)
            (out_dir / "pruning_report.json").write_text(json.dumps(prune_report, indent=2), encoding="utf-8")

            model = RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_features=max_features,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=int(args.rf_n_jobs),
            )
            model.fit(X, y)

            pd.Series(
                {
                    "n_estimators": int(n_estimators),
                    "max_features": ("None" if max_features is None else str(max_features)),
                },
                name="value",
            ).to_csv(out_dir / "best_params.csv", header=False)

            pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).to_csv(out_dir / "importance.csv", index=False)

            mets = apparent_metrics_binary(model, X, y)
            pd.Series(mets, name="value").to_csv(out_dir / "apparent_metrics.csv", header=False)

            if args.compute_shap:
                try:
                    compute_and_save_final_shap(
                        model=model,
                        X=X,
                        out_dir=out_dir,
                        positive_label=args.shap_positive_label,
                        save_shap_matrix=bool(args.save_shap_matrix),
                        shap_sample=int(args.shap_sample),
                        seed=SEED,
                    )
                except Exception as e:
                    LOGGER.warning("State=%s desc=%s: SHAP failed (%s)", state, desc_name, str(e))

            model_bundle = {
                "model": model,
                "selected_features": list(X.columns.astype(str)),
                "state": state,
                "descriptor": desc_name,
                "n_estimators": int(n_estimators),
                "max_features": max_features,
                "filter_config": {
                    "var_thresh": float(filter_cfg.var_thresh),
                    "corr_thresh": float(filter_cfg.corr_thresh),
                    "use_abs_corr": bool(filter_cfg.use_abs_corr),
                    "corr_block_size": int(filter_cfg.corr_block_size),
                    "corr_keep_first": int(filter_cfg.corr_keep_first),
                    "corr_max_keep": int(filter_cfg.corr_max_keep),
                    "corr_compute_dtype": str(filter_cfg.corr_compute_dtype),
                },
                "classes_": list(model.classes_),
                "thread_env": {
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                    "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
                },
            }
            joblib.dump(model_bundle, out_dir / "model.joblib", compress=3)

            LOGGER.info(
                "Saved final: state=%s desc=%s n=%d p=%d params=(%s,%s)",
                state,
                desc_name,
                X.shape[0],
                X.shape[1],
                int(n_estimators),
                "None" if max_features is None else str(max_features),
            )

if __name__ == "__main__":
    main()