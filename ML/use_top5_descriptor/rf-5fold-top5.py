#!/usr/bin/env python3
from __future__ import annotations

"""
Top-5 RandomForest training for SI.

What this script does:
1) Build Top-K feature lists per (spin, descriptor) by aggregating RF importances from:
     ML/use_all_descriptor/5-fold/results_rf/<spin>/<descriptor>/<fold>/importance.csv

2) Train nested-CV RandomForest models using ONLY the Top-K features:
   - Outer CV: RepeatedStratifiedKFold (5 splits × 3 repeats)
   - Inner CV: GridSearchCV + StratifiedKFold (5 splits)
   - Scoring: MCC

3) Write minimal per-fold artifacts and per-spin summary CSVs.

Outputs (default):
  Top-K lists:
    ML/use_top5_descriptor/results_top5/top5_lists/top5_<spin>_<descriptor>.csv
      columns: feature, importance_mean

  Training outputs:
    ML/use_top5_descriptor/results_top5/<target_spin>/<descriptor>/topkmodel_<TAG>/<fold>/
      train_ids_used_<fold>.csv
      test_ids_used_<fold>.csv
      metrics_<fold>.csv
      importance_<fold>.csv

  Summaries:
    ML/use_top5_descriptor/results_top5/summary_<target_spin>.csv
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
# Repo root detection
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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

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
DEFAULT_MAX_FEATURES = ("None", "sqrt", "log2")  # parsed to [None, "sqrt", "log2"]


@dataclass(frozen=True)
class Paths:
    labels_csv: Path
    descriptors_dir: Path
    results_rf_dir: Path
    out_dir: Path
    topk_lists_dir: Path


# -----------------------------------------------------------------------------
# Logging / IO helpers
# -----------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s | %(message)s")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
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
    Keeps only numeric columns; drops any columns containing NaN/Inf.
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

        df = pd.read_csv(csv_path, index_col=0)
        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=1)

        if df.shape[1] == 0:
            LOGGER.warning("Descriptor '%s' has no usable numeric columns after cleaning. Skipped.", sub.name)
            continue

        out[sub.name] = df

    if not out:
        raise RuntimeError(f"No descriptor CSVs found under: {desc_root}")

    LOGGER.info("Loaded %d descriptors.", len(out))
    for k, v in out.items():
        LOGGER.info("  - %s: %s", k, v.shape)
    return out


# -----------------------------------------------------------------------------
# CV utilities
# -----------------------------------------------------------------------------


def make_master_ids(y_all: pd.Series, X_dict: Dict[str, pd.DataFrame]) -> pd.Index:
    common = y_all.index
    for X in X_dict.values():
        common = common.intersection(X.index)
    return y_all.index[y_all.index.isin(common)]


def build_outer_splits(y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=OUTER_SEED,
    )
    dummy = np.zeros((len(y), 1))
    return list(cv.split(dummy, y.to_numpy()))


def fold_name(i: int) -> str:
    rep = i // OUTER_SPLITS + 1
    f = i % OUTER_SPLITS + 1
    return f"r{rep}_f{f}"


def parse_max_features(values: Sequence[str]) -> List[object]:
    out: List[object] = []
    for v in values:
        v2 = v.strip()
        if v2.lower() in {"none", "null"}:
            out.append(None)
        else:
            out.append(v2)
    return out


# -----------------------------------------------------------------------------
# Top-K list builder
# -----------------------------------------------------------------------------


def _read_importance_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    cols = set(df.columns)
    if {"parameter", "importance"} <= cols:
        out = df[["parameter", "importance"]].copy()
        out = out.rename(columns={"parameter": "feature"})
        return out
    if {"feature", "importance"} <= cols:
        out = df[["feature", "importance"]].copy()
        return out

    return None


def build_topk_lists(
    *,
    results_rf_dir: Path,
    descriptors: Sequence[str],
    out_topk_dir: Path,
    topk: int,
) -> int:
    """
    Create topk_<spin>_<desc>.csv from importance.csv files.
    Aggregation: mean importance per feature across all folds.
    """
    ensure_dir(out_topk_dir)
    n_saved = 0

    for spin in SPIN_STATES:
        for desc in descriptors:
            base = results_rf_dir / spin / desc
            if not base.is_dir():
                LOGGER.warning("Missing importance source dir: %s", base)
                continue

            imp_files = sorted(base.rglob("importance.csv"))
            if not imp_files:
                LOGGER.warning("No importance.csv found under: %s", base)
                continue

            rows: List[pd.DataFrame] = []
            for f in imp_files:
                dfi = _read_importance_csv(f)
                if dfi is None:
                    continue
                dfi["feature"] = dfi["feature"].astype(str)
                dfi["importance"] = pd.to_numeric(dfi["importance"], errors="coerce")
                dfi = dfi.dropna()
                rows.append(dfi)

            if not rows:
                LOGGER.warning("No readable importance data for %s/%s.", spin, desc)
                continue

            all_imp = pd.concat(rows, ignore_index=True)
            top = (
                all_imp.groupby("feature", as_index=False)["importance"]
                .mean()
                .rename(columns={"importance": "importance_mean"})
                .sort_values("importance_mean", ascending=False)
                .head(topk)
            )

            out_csv = out_topk_dir / f"top{topk}_{spin}_{desc}.csv"
            top.to_csv(out_csv, index=False)
            n_saved += 1
            LOGGER.info("Saved Top-%d list: %s", topk, out_csv)

    if n_saved == 0:
        LOGGER.warning("No Top-%d lists were created. Check paths and importance.csv schema.", topk)

    return n_saved


def load_topk_features(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if "feature" in df.columns:
        col = "feature"
    elif "parameter" in df.columns:
        col = "parameter"
    else:
        raise KeyError(f"Top-K file missing 'feature' (or 'parameter') column: {path}")
    return df[col].astype(str).tolist()


# -----------------------------------------------------------------------------
# Training outputs (minimal)
# -----------------------------------------------------------------------------


def write_fold_artifacts(
    *,
    out_fold_dir: Path,
    fold: str,
    X_tr: pd.DataFrame,
    X_te: pd.DataFrame,
    y_tr: pd.Series,
    y_te: pd.Series,
    y_tr_pred: np.ndarray,
    y_te_pred: np.ndarray,
    model: RandomForestClassifier,
    best_params: dict,
) -> dict:
    ensure_dir(out_fold_dir)

    pd.Series(X_tr.index.astype(str)).to_csv(out_fold_dir / f"train_ids_used_{fold}.csv", index=False, header=False)
    pd.Series(X_te.index.astype(str)).to_csv(out_fold_dir / f"test_ids_used_{fold}.csv", index=False, header=False)

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
    pd.DataFrame([metrics_row]).to_csv(out_fold_dir / f"metrics_{fold}.csv", index=False)

    pd.DataFrame(
        {"feature": X_tr.columns.astype(str), "importance": model.feature_importances_}
    ).to_csv(out_fold_dir / f"importance_{fold}.csv", index=False)

    return metrics_row


# -----------------------------------------------------------------------------
# Core training (Top-K only)
# -----------------------------------------------------------------------------


def run_topk_model(
    *,
    X_master: pd.DataFrame,
    y_master: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    topk_features: Sequence[str],
    model_out_dir: Path,
    overwrite: bool,
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[object],
) -> List[dict]:
    cols = [c for c in topk_features if c in X_master.columns]
    if not cols:
        return []

    X = X_master.loc[:, cols]
    if not np.isfinite(X.to_numpy()).all():
        bad = (~np.isfinite(X.to_numpy())).sum()
        raise RuntimeError(f"Non-finite values found in Top-K matrix (count={bad}).")

    rows: List[dict] = []
    for i, (tr_idx, te_idx) in enumerate(splits):
        fold = fold_name(i)
        out_fold_dir = model_out_dir / fold
        metrics_path = out_fold_dir / f"metrics_{fold}.csv"

        if (not overwrite) and metrics_path.is_file():
            try:
                m = pd.read_csv(metrics_path).iloc[0].to_dict()
                m["fold"] = str(m.get("fold", fold))
                rows.append(m)
                continue
            except Exception:
                pass

        X_tr, X_te = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
        y_tr, y_te = y_master.iloc[tr_idx].copy(), y_master.iloc[te_idx].copy()

        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
        grid = GridSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=OUTER_SEED),
            param_grid={"n_estimators": list(n_estimators_grid), "max_features": list(max_features_grid)},
            cv=inner,
            scoring=make_scorer(matthews_corrcoef),
            n_jobs=-1,
        )
        grid.fit(X_tr, y_tr)

        model: RandomForestClassifier = grid.best_estimator_
        y_tr_pred = model.predict(X_tr)
        y_te_pred = model.predict(X_te)

        mrow = write_fold_artifacts(
            out_fold_dir=out_fold_dir,
            fold=fold,
            X_tr=X_tr,
            X_te=X_te,
            y_tr=y_tr,
            y_te=y_te,
            y_tr_pred=y_tr_pred,
            y_te_pred=y_te_pred,
            model=model,
            best_params=grid.best_params_,
        )
        rows.append(mrow)

    return rows


# -----------------------------------------------------------------------------
# CLI / Main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Top-K RF: build Top-K lists + nested-CV training (SI-ready).")
    p.add_argument("--labels", type=Path, default=None, help="Path to label CSV.")
    p.add_argument("--descriptors", type=Path, default=None, help="Path to descriptors/ directory.")
    p.add_argument("--results-rf-dir", type=Path, default=None, help="Path to results_rf from use_all_descriptor.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for results_top5.")
    p.add_argument("--topk", type=int, default=5, help="K for Top-K features (default: 5).")
    p.add_argument("--descs", nargs="*", default=None, help="Run only these descriptors (by folder name).")
    p.add_argument("--overwrite", action="store_true", help="Retrain even if fold outputs exist.")
    p.add_argument("--skip-build-topk", action="store_true", help="Do not build Top-K lists (assume existing).")
    p.add_argument("--n-estimators", nargs="+", type=int, default=list(DEFAULT_N_ESTIMATORS))
    p.add_argument("--max-features", nargs="+", default=list(DEFAULT_MAX_FEATURES))
    p.add_argument("--log-level", default="INFO")
    return p


def resolve_paths(args: argparse.Namespace) -> Paths:
    labels_csv = args.labels or (ROOT_DIR / "FeN6-SSD" / "FeN6-SSD_500_spin_labeled.csv")
    descriptors_dir = args.descriptors or (ROOT_DIR / "descriptors")
    results_rf_dir = args.results_rf_dir or (ROOT_DIR / "ML" / "use_all_descriptor" / "5-fold" / "results_rf")
    out_dir = args.out_dir or (ROOT_DIR / "ML" / "use_top5_descriptor" / "results_top5")
    topk_lists_dir = out_dir / "top5_lists"
    return Paths(
        labels_csv=labels_csv,
        descriptors_dir=descriptors_dir,
        results_rf_dir=results_rf_dir,
        out_dir=out_dir,
        topk_lists_dir=topk_lists_dir,
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv or sys.argv[1:])
    configure_logging(args.log_level)

    paths = resolve_paths(args)
    ensure_dir(paths.out_dir)
    ensure_dir(paths.topk_lists_dir)

    n_estimators_grid = list(args.n_estimators)
    max_features_grid = parse_max_features(args.max_features)

    LOGGER.info("Repo root: %s", ROOT_DIR)
    LOGGER.info("Labels: %s", paths.labels_csv)
    LOGGER.info("Descriptors: %s", paths.descriptors_dir)
    LOGGER.info("Importance source (results_rf): %s", paths.results_rf_dir)
    LOGGER.info("Output dir: %s", paths.out_dir)
    LOGGER.info("Top-K lists dir: %s", paths.topk_lists_dir)
    LOGGER.info("Grid: n_estimators=%s, max_features=%s", n_estimators_grid, max_features_grid)
    LOGGER.info("Top-K: %d", int(args.topk))

    labels = read_labels(paths.labels_csv)
    descriptors = load_descriptors(paths.descriptors_dir)

    if args.descs:
        wanted = set(args.descs)
        descriptors = {k: v for k, v in descriptors.items() if k in wanted}
        LOGGER.info("Filtered descriptors: %d", len(descriptors))

    if not args.skip_build_topk:
        build_topk_lists(
            results_rf_dir=paths.results_rf_dir,
            descriptors=list(descriptors.keys()),
            out_topk_dir=paths.topk_lists_dir,
            topk=int(args.topk),
        )
    else:
        LOGGER.info("Skipping Top-K list build (using existing files under %s).", paths.topk_lists_dir)

    for target_spin in SPIN_STATES:
        LOGGER.info("Target spin: %s", target_spin)

        y_all = labels.loc[labels[COL_STATE] == target_spin, COL_BEHAV].dropna().astype(str)
        y_bin = y_all.apply(lambda v: POSITIVE_LABEL if v == POSITIVE_LABEL else target_spin)

        master_ids = make_master_ids(y_bin, descriptors)
        if len(master_ids) == 0:
            LOGGER.warning("No common IDs across labels and descriptors for target_spin=%s. Skipped.", target_spin)
            continue

        y_master = y_bin.loc[master_ids]
        splits = build_outer_splits(y_master)

        summary_rows: List[dict] = []

        for desc, Xdesc in descriptors.items():
            X_master = Xdesc.reindex(master_ids)
            if X_master.shape[1] == 0 or X_master.shape[0] == 0:
                continue

            for source_spin in SPIN_STATES:
                topk_path = paths.topk_lists_dir / f"top{int(args.topk)}_{source_spin}_{desc}.csv"
                if not topk_path.is_file():
                    continue

                try:
                    topk_features = load_topk_features(topk_path)
                except Exception as e:
                    LOGGER.warning("Failed to read Top-K file %s (%s). Skipped.", topk_path, str(e))
                    continue

                tag = f"{target_spin[:2].upper()}using{source_spin[:2].upper()}"
                model_out_dir = paths.out_dir / target_spin / desc / f"topkmodel_{tag}"

                LOGGER.info("Run/reuse: desc=%s source=%s tag=%s", desc, source_spin, tag)

                try:
                    fold_metrics = run_topk_model(
                        X_master=X_master,
                        y_master=y_master,
                        splits=splits,
                        topk_features=topk_features,
                        model_out_dir=model_out_dir,
                        overwrite=bool(args.overwrite),
                        n_estimators_grid=n_estimators_grid,
                        max_features_grid=max_features_grid,
                    )
                except Exception as e:
                    LOGGER.warning("Training failed: target=%s desc=%s source=%s (%s)", target_spin, desc, source_spin, str(e))
                    continue

                for m in fold_metrics:
                    summary_rows.append(
                        {
                            "descriptor": desc,
                            "target_spin": target_spin,
                            "feature_source_spin": source_spin,
                            "fold": str(m.get("fold")),
                            "test_mcc": float(m.get("test_mcc")) if pd.notna(m.get("test_mcc")) else np.nan,
                            "test_accuracy": float(m.get("test_accuracy")) if pd.notna(m.get("test_accuracy")) else np.nan,
                            "test_f1": float(m.get("test_f1")) if pd.notna(m.get("test_f1")) else np.nan,
                            "n_features": int(m.get("n_features")) if pd.notna(m.get("n_features")) else np.nan,
                            "best_n_estimators": m.get("best_n_estimators"),
                            "best_max_features": m.get("best_max_features"),
                        }
                    )

        if summary_rows:
            out_csv = paths.out_dir / f"summary_{target_spin}.csv"
            pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
            LOGGER.info("Saved: %s", out_csv)
        else:
            LOGGER.warning("No models were run for target_spin=%s (missing Top-K lists or no matching columns).", target_spin)


if __name__ == "__main__":
    main()
