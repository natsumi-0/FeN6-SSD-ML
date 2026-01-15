# FeN6-SSD-ML

Machine-learning code for classifying spin-crossover (SCO) behavior in Fe(II)–N₆ coordination complexes,
supporting the accompanying manuscript submitted to *Journal of Chemical Information and Modeling*.

This repository provides:
- curated label/metadata tables (FeN6-SSD)
- precomputed descriptor CSVs
- Python scripts to reproduce the RandomForest experiments (nested CV, top-5, and SHAP value calculation for ECFP4-family descriptors)

---

## Repository structure

```text
FeN6-SSD-ML/
├─ .gitignore
├─ README.md
│
├─ descriptors/
│  ├─ mbtr/
│  │   └─ mbtr.csv
│  ├─ mbtr-fe/
│  │   └─ mbtr-fe.csv
│  ├─ mbtr_bind_otherbinary/                 # “+ env.”
│  │   └─ mbtr_bind_otherbinary.csv
│  ├─ mbtr-fe_bind_otherbinary/              # “+ env.”
│  │   └─ mbtr-fe_bind_otherbinary.csv
│  │
│  ├─ rac155/
│  │   └─ rac155.csv
│  ├─ rac155_nostereo/
│  │   └─ rac155_nostereo.csv
│  ├─ rac155_bind_otherbinary/               # “+ env.”
│  │   └─ rac155_bind_otherbinary.csv
│  ├─ rac155_nostereo_bind_otherbinary/      # “+ env.”
│  │   └─ rac155_nostereo_bind_otherbinary.csv
│  │
│  ├─ ecfp4/
│  │   ├─ ecfp4.csv
│  │   └─ SMARTS_hash_mapping.csv
│  ├─ ecfp4-fe/
│  │   └─ ecfp4-fe.csv
│  ├─ ecfp4_bind_otherbinary/                # “+ env.”
│  │   └─ ecfp4_bind_otherbinary.csv
│  ├─ ecfp4-fe_bind_otherbinary/             # “+ env.”
│  │   └─ ecfp4-fe_bind_otherbinary.csv
│  │
│  ├─ octadist/
│  │   └─ octadist.csv
│  └─ octadist_bind_otherbinary/             # “+ env.”
│      └─ octadist_bind_otherbinary.csv
│
├─ FeN6-SSD/
│  ├─ FeN6-SSD_500_spin_labeled.csv          # required label file (ccdc_id, spin state, SCO label)
│  ├─ FeN6-SSD_500_ligand_labeled.csv
│  ├─ FeN6-SSD_500_ligand_count.csv
│  ├─ FeN6-SSD_500_ligand_coord-no_count.csv
│  ├─ FeN6-SSD_500_ligand_charge.csv
│  ├─ FeN6-SSD_500_ion_list.csv
│  ├─ FeN6-SSD_500_solv_list.csv
│  ├─ FeN6-SSD_500_ion-solv_binary.csv
│  └─ FeN6-SSD_500_ion-solv_count.csv
│
└─ ML/
   ├─ use_all_descriptor/
   │  ├─ 5-fold/
   │  │   └─ rf-5fold.py                     # nested CV using all descriptors (Table 2)
   │  └─ no-fold/
   │      └─ rf-nofold.py                    # SHAP value calculation (ECFP4-family; Fig. 6 / Fig. S2)
   │
   └─ use_top5_descriptor/
      └─ rf-5fold-top5.py                    # top-5 feature models (Table 3)
````

---

## Requirements

The scripts use standard Python scientific packages:

* numpy
* pandas
* scikit-learn
* shap

Example installation with pip:

```bash
pip install -U numpy pandas scikit-learn shap
```

---

## Data / descriptors

* Labels: `FeN6-SSD/FeN6-SSD_500_spin_labeled.csv`
* Descriptors: `descriptors/<name>/<name>.csv`

Folders with the suffix `*_bind_otherbinary` correspond to the manuscript “+ env.” variants
(ion/solvent presence encoded as binary features and concatenated to the base descriptor).

---

## How to reproduce the experiments

All scripts automatically detect the repository root by locating:

* `descriptors/`
* `ML/`
* `FeN6-SSD/FeN6-SSD_500_spin_labeled.csv`

No manual path edits are required.

---

### 1) Nested CV with all descriptors (Table 2)

Run 3 × (5-fold) nested cross-validation for HS and LS, including feature importances.

```bash
cd ML/use_all_descriptor/5-fold
python rf-5fold.py
```

Outputs:

* `results_rf/summary_high-spin.csv`
* `results_rf/summary_low-spin.csv`
* fold-level artifacts under `results_rf/<spin_state>/<descriptor>/...`

---

### 2) Top-5 feature models (Table 3)

Build top-5 feature sets from step (1) and retrain/evaluate models.

```bash
cd ML/use_top5_descriptor
python rf-5fold-top5.py
```

Outputs:

* `results_top5/top5_lists/top5_<spin>_<descriptor>.csv`
* `results_top5/summary_high-spin.csv`
* `results_top5/summary_low-spin.csv`

---

### 3) SHAP value calculation for ECFP4 contribution analysis (Figures 6 and S2)

Compute SHAP values for the ECFP4-based contribution analysis used in the manuscript.
This step **calculates SHAP values only** (projection onto molecular drawings and figure generation were carried out separately).

```bash
cd ML/use_all_descriptor/no-fold
python rf-nofold.py --descriptor ecfp4
```

Outputs:

* `results_nofold/<spin_state>/<descriptor>/`

  * `metrics.csv`
  * `confusion_matrix.csv`
  * `feature_importance.csv`
  * `features_used.txt`
  * `best_params.csv`
  * `shap_values_all.csv`

Notes:

* `--descriptor` supports ECFP4-family descriptors:
  `ecfp4`, `ecfp4-fe`, `ecfp4_bind_otherbinary`, and `ecfp4-fe_bind_otherbinary`.

---

## Evaluation protocol (summary)

* Task: classify **SCO-undergoing vs non-SCO** complexes within the same spin state (HS or LS)
* Model: RandomForest (scikit-learn)
* Metrics: MCC and F1 (plus accuracy/precision/recall)
* Hyperparameters: selected by inner 5-fold CV using MCC
* Outer CV: 5 folds × 3 repeats
* Outer splits are shared across descriptors for each spin state

---

## Citation

If you use this repository, please cite the accompanying manuscript:

> Manuscript submitted to *Journal of Chemical Information and Modeling*.
> Citation details will be added upon publication.


