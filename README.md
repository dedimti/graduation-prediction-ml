# Graduation Prediction ML

**Comparative Analysis of Machine Learning Algorithms for Predicting On-Time Graduation of Undergraduate Students**

> Dedi Irawan, Sudarmaji  
> Informatics, Faculty of Computer Science, Muhammadiyah University of Metro, Indonesia  
> Published in: *Jurnal Teknik Informatika (JUTIF)*, 2026

---

## Overview

This repository contains the complete source code and experimental pipeline for the paper:

> *"Comparative Analysis of Machine Learning Algorithms for Predicting On-Time Graduation of Undergraduate Students"*

We compare four classifiers — **Random Forest**, **XGBoost**, **Support Vector Machine**, and **Logistic Regression** (baseline) — on the task of predicting whether undergraduate students will graduate on time.

---

## Key Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.8282** | 0.7996 | 0.8756 | **0.8359** | **0.9122** |
| Random Forest | 0.8181 | 0.7996 | 0.8485 | 0.8233 | 0.9036 |
| SVM | 0.8215 | 0.7840 | 0.8869 | 0.8323 | 0.8968 |
| Logistic Regression | 0.7932 | 0.7718 | 0.8413 | 0.8054 | 0.8641 |

XGBoost outperforms the LR baseline by **+3.5 pp accuracy** and **+0.048 AUC**, confirming non-linear feature interactions in the dataset.

---

## Dataset

- **Source**: UCI Machine Learning Repository — *Predict Students' Dropout and Academic Success*
- **DOI**: [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)
- **Records**: 4,424 students | **Features**: 36
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Target**: Binarized → Graduate (on-time) vs Non-Graduate (Dropout + Enrolled)

> The dataset is not included in this repository. Download it directly from UCI using the script below.

---

## Repository Structure

```
graduation-prediction-ml/
├── README.md
├── requirements.txt
├── main.py                    # Full pipeline: preprocess → train → evaluate
├── download_dataset.py        # Auto-download dataset from UCI
├── data/
│   └── .gitkeep              # Place dataset here after download
├── figures/
│   ├── Figure1_Performance_Comparison.png
│   ├── Figure2_Confusion_Matrices.png
│   ├── Figure3_Feature_Importance.png
│   └── Figure4_ROC_Curves.png
└── notebooks/
    └── analysis.ipynb         # Step-by-step exploratory notebook
```

---

## Setup

### 1. Clone repository
```bash
git clone https://github.com/dedimti/graduation-prediction-ml.git
cd graduation-prediction-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
```bash
python download_dataset.py
```

### 4. Run full pipeline
```bash
python main.py
```

---

## Environment

| Package | Version |
|---|---|
| Python | 3.10 |
| scikit-learn | 1.3.0 |
| xgboost | 2.0.0 |
| imbalanced-learn | 0.11.0 |
| pandas | ≥1.5.0 |
| numpy | ≥1.23.0 |
| matplotlib | ≥3.6.0 |
| seaborn | ≥0.12.0 |

---

## Reproducibility

All experiments use `random_state=42` throughout. No hyperparameter tuning was performed — all models run with Scikit-learn defaults to isolate algorithm-level differences.

To reproduce Table 5 in the paper exactly:
```bash
python main.py --seed 42 --no-tuning
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{irawab2026graduation,
  title   = {Comparative Analysis of Machine Learning Algorithms for Predicting
             On-Time Graduation of Undergraduate Students},
  author  = {Irawab, Dedi and Sudarmaji},
  journal = {Jurnal Teknik Informatika (JUTIF)},
  year    = {2026},
  volume  = {},
  pages   = {},
  doi     = {}
}
```

---

## License

This code is released under the **MIT License**. The dataset is subject to its own CC BY 4.0 license from UCI.
