# FeN6-SSD-ML

Machine-learning code for classifying spin-crossover (SCO) behavior in Fe(II)–N₆ coordination complexes, supporting the accompanying manuscript submitted to *Journal of Chemical Information and Modeling*.

This repository provides:

* curated label/metadata tables (FeN6-SSD dataset)
* precomputed descriptor CSV files
* Python scripts to reproduce the RandomForest experiments reported in the manuscript

The scripts are organized primarily by **manuscript result section (Table / Figure number)** so that each script corresponds directly to a specific analysis workflow.

---

# Repository structure

```text
FeN6-SSD-ML/
├─ .gitignore
├─ README.md
│
├─ descriptors/
│  ├─ csd-param/
│  │   └─ csd-param.csv
│  │
│  ├─ ecfp4/
│  │   ├─ ecfp4.csv
│  │   └─ SMARTS_hash_mapping.csv
│  ├─ ecfp4-fe/
│  │   └─ ecfp4-fe.csv
│  ├─ ecfp4_bind_otherbinary/
│  │   └─ ecfp4_bind_otherbinary.csv
│  ├─ ecfp4-fe_bind_otherbinary/
│  │   └─ ecfp4-fe_bind_otherbinary.csv
│  │
│  ├─ mbtr/
│  │   └─ mbtr.csv
│  ├─ mbtr-cif/
│  │   └─ mbtr-cif.csv
│  ├─ mbtr-fe/
│  │   └─ mbtr-fe.csv
│  ├─ mbtr_bind_otherbinary/
│  │   └─ mbtr_bind_otherbinary.csv
│  ├─ mbtr-fe_bind_otherbinary/
│  │   └─ mbtr-fe_bind_otherbinary.csv
│  │
│  ├─ rac155/
│  │   └─ rac155.csv
│  ├─ rac155_nostereo/
│  │   └─ rac155_nostereo.csv
│  ├─ rac155_bind_otherbinary/
│  │   └─ rac155_bind_otherbinary.csv
│  ├─ rac155_nostereo_bind_otherbinary/
│  │   └─ rac155_nostereo_bind_otherbinary.csv
│  │
│  ├─ octadist/
│  │   └─ octadist.csv
│  └─ octadist_bind_otherbinary/
│      └─ octadist_bind_otherbinary.csv
│
├─ FeN6-SSD/
│  ├─ FeN6-SSD_500_spin_labeled.csv
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
   ├─ for_table2/
   │  ├─ 5-fold/
   │  │   └─ rf-5fold.py
   │  └─ all/
   │      └─ rf-nofold.py
   │
   ├─ for_table3/
   │  ├─ decide_top5/
   │  │   └─ rf-decide-top5.py
   │  └─ make_top5model/
   │      └─ rf-make-top5-model.py
   │
   ├─ for_tableS7/
   │  └─ rf-5fold_core-grouped.py
   │
   └─ for_tableS8/
      └─ rf-5-fold_leakfree.py
```

---

# Requirements

The scripts use standard Python scientific libraries.

Required packages:

```
numpy
pandas
scikit-learn
```

Optional (for SHAP analysis):

```
shap
```

Example installation:

```bash
pip install -U numpy pandas scikit-learn shap
```

---

# Data and descriptors

### Label file

Main label table:

```
FeN6-SSD/FeN6-SSD_500_spin_labeled.csv
```

This file contains:

* `ccdc_id`
* spin state
* SCO behavior labels
* FeN6 core identifier (`rename_id`)

### Descriptor tables

Descriptors are stored under

```
descriptors/<descriptor_name>/<descriptor_name>.csv
```

Examples:

```
descriptors/ecfp4/ecfp4.csv
descriptors/mbtr/mbtr.csv
descriptors/rac155/rac155.csv
descriptors/octadist/octadist.csv
```

Directories ending with

```
*_bind_otherbinary
```

correspond to the manuscript **“+ env.” descriptor variants**, where ion/solvent presence is encoded as additional binary features.

---

# How to reproduce the experiments

The scripts automatically detect the repository root by searching for:

```
descriptors/
ML/
FeN6-SSD/FeN6-SSD_500_spin_labeled.csv
```

Therefore **no manual path editing is required**.

---

# Table 2: Nested cross-validation with all descriptors

Run the main descriptor comparison used in **Table 2**.

```
cd ML/for_table2/5-fold
python rf-5fold.py
```

This script performs:

* repeated **5-fold outer cross-validation**
* inner CV hyperparameter optimization
* RandomForest training for each descriptor and spin state

Key features:

* training-only preprocessing
* variance filtering
* correlation pruning
* nested CV hyperparameter selection

Outputs:

```
results_rf/
  summary_high-spin.csv
  summary_low-spin.csv
```

Fold-level outputs are stored under

```
results_rf/<spin_state>/<descriptor>/<fold>/
```

---

# Figure 6: Final model fitting and SHAP analysis

Final models can be trained on the **full dataset** for interpretation purposes.

```
cd ML/for_table2/all
python rf-nofold.py
```

This step is mainly used for:

* SHAP analysis
* descriptor interpretation
* model export

Outputs include:

```
best_params.csv
apparent_metrics.csv
importance.csv
selected_features.csv
model.joblib
shap_mean_abs.csv
shap_values.csv.gz
```

Important note:

```
apparent_metrics.csv is evaluated on the training data itself and
should not be interpreted as an unbiased estimate of model performance.
```

The unbiased performance values reported in the manuscript come from the **nested CV results (Table 2)**.

---

# Table 3: Top-5 feature models

The Table 3 workflow consists of two steps.

---

## Step 1 — Determine Top-5 features

```
cd ML/for_table3/decide_top5
python rf-decide-top5.py
```

This script:

1. performs global feature filtering
2. runs repeated cross-validation
3. computes mean feature importance
4. determines Top-5 features for each descriptor

Outputs are written to:

```
results_globalfilter_topk/
```

Important note:

This step intentionally performs feature filtering on the **full dataset before cross-validation**, therefore it is **leaky** and used only for selecting candidate descriptor subsets.

---

## Step 2 — Train Top-5 models

```
cd ML/for_table3/make_top5model
python rf-make-top5-model.py
```

This script:

* loads Top-5 feature lists from Step 1
* reuses the saved train/test splits
* retrains RandomForest models
* evaluates model performance

Outputs are written to:

```
results_top5_fixed_exactsplits/
```

---

# Table S7: Group-based nested cross-validation

Run the group-based CV analysis used in **Table S7**.

```
cd ML/for_tableS7
python rf-5fold_core-grouped.py
```

This workflow uses **group-based splitting by FeN6 core identifier (`rename_id`)**.

Key properties:

* group-aware outer CV
* group-aware inner CV
* no train/test overlap of FeN6 cores
* training-only preprocessing

Outputs:

```
results_rf_group_leakfree/
```

Fold-level outputs include:

```
metrics.csv
importance.csv
train_ids.csv
test_ids.csv
train_groups.csv
test_groups.csv
```

Optional SHAP summaries may also be produced.

---

# Table S8: Leak-free fold-local Top-K evaluation

Run the fully leak-free Top-K evaluation used in **Table S8**.

```
cd ML/for_tableS8
python rf-5-fold_leakfree.py
```

For each outer fold the script performs two stages:

### Stage 1 — full descriptor model

* fold-local filtering
* RandomForest training
* hyperparameter tuning

### Stage 2 — fold-local Top-K model

* Top-K features selected **within the fold**
* retraining using only those features

All preprocessing and feature selection are performed using **training data only**.

Outputs:

```
results_outercv_full_topk/
```

Stage summaries are saved as:

```
summary_stage1_full_*.csv
summary_stage2_topk_samefold_*.csv
```

---

# Evaluation protocol

Task:

```
Binary classification of
spin-crossover vs non-SCO complexes
within the same spin state.
```

Model:

```
RandomForest (scikit-learn)
```

Metrics:

```
MCC
F1 score
accuracy
precision
recall
```

Hyperparameter optimization:

```
inner cross-validation using MCC
```

Different evaluation designs are used across the repository:

| workflow | description                                 |
| -------- | ------------------------------------------- |
| Table 2  | standard nested cross-validation            |
| Table 3  | Top-5 descriptor models                     |
| Table S7 | group-based CV by FeN6 core                 |
| Table S8 | fully leak-free fold-local Top-K evaluation |

---

# Citation

If you use this repository, please cite the accompanying manuscript:

```
Manuscript submitted to
Journal of Chemical Information and Modeling
```

Citation information will be updated upon publication.

