# VisaRisk — Visa Certification Prediction

An advanced machine learning project that predicts U.S. visa certification outcomes (Certified vs. Denied) using 
ensemble methods, bagging and random forests, boosting, cross-validation and hyperparameter tuning.

---

## Business Context

The Office of Foreign Labor Certification (OFLC) processes hundreds of thousands of visa applications annually. This 
project builds and selects the best classification model to help prioritize applications and identify key factors that 
influence certification decisions, enabling more efficient review processes.

**Target variable:** `case_status` — `Certified` (1) or `Denied` (0)

---

## Dataset

`data/EasyVisa.csv` — 25,480 records with 12 features including:

| Feature | Description |
|---|---|
| `continent` | Applicant's continent of origin |
| `education_of_employee` | Education level (High School → Doctorate) |
| `has_job_experience` | Prior job experience (Y/N) |
| `requires_job_training` | Whether training is required (Y/N) |
| `no_of_employees` | Employer company size |
| `yr_of_estab` | Year employer was established |
| `region_of_employment` | U.S. region of the job |
| `prevailing_wage` | Offered wage (normalized to annual) |
| `unit_of_wage` | Wage frequency (Hour/Week/Month/Year) |
| `full_time_position` | Full-time vs part-time (Y/N) |
| `case_status` | Target: Certified or Denied |

---

## ML Pipeline

### 1. Data Preprocessing
- Imputed negative `no_of_employees` values with column median
- Normalized `prevailing_wage` to annual salary (Hour × 2076 / Week × 52.2 / Month × 12)
- Ordinal-encoded `education_of_employee` (High School=0 → Doctorate=3)
- Binned `no_of_employees` and `yr_of_estab` via quantile cuts
- One-hot encoded remaining categoricals; sanitized column names for XGBoost

### 2. Train / Validation / Test Split
- 60% train / 20% validation / 20% test (stratified on target)

### 3. Class Imbalance Handling
Three strategies were benchmarked:
- **Original data** (class_weight='balanced' in tree models)
- **SMOTE oversampling** (k=5, sampling_strategy=1.0)
- **Random undersampling**

### 4. Models Evaluated
| Model | Variants |
|---|---|
| Decision Tree | Baseline |
| Bagging (DT base) | Original / Over / Under |
| Random Forest | Original |
| Gradient Boosting (GBM) | Original / Over / Under |
| AdaBoost | Original / Over / Under |
| XGBoost | Original / Over / Under |

### 5. Hyperparameter Tuning
- **RandomizedSearchCV** (wide search) followed by **GridSearchCV** (narrow refinement) for GBM, AdaBoost, and XGBoost
- Scoring metric: **F1-score** (balances precision/recall for imbalanced classes)

### 6. Final Model
**Gradient Boosting Classifier (GBM) on original data** — selected based on best validation F1 score.

Key hyperparameters tuned: `n_estimators`, `learning_rate`, `subsample`, `max_features`, `init` estimator.

---

## Results

Final model: **Gradient Boosting Classifier (GBM) on original data**, selected by highest validation F1.

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Train | — | — | — | — |
| Validation | — | — | — | — |
| **Test** | — | — | — | — |

*Run `certify_visa-risk.ipynb` end-to-end to populate these metrics.*

---

## Project Structure

```
VisaRisk/
├── data/
│   └── EasyVisa.csv
├── models/
│   └── visa-risk.pkl          # saved best model (joblib)
├── src/
│   ├── preprocessing.py       # load_data(), preprocess(), encode_features()
│   ├── model_trainer.py       # get_base_models(), train_and_compare(), SMOTE/undersampling, tuning
│   ├── evaluate.py            # metrics, confusion matrix, feature importance, save/load model
│   └── viz_utils.py           # histogram_boxplot(), labeled_barplot(), stacked_barplot(), distribution_plot_wrt_target()
├── configs/
│   └── model_params.py        # hyperparameter grids for GBM, AdaBoost, XGBoost
├── certify_visa-risk.ipynb    # main notebook (uses src/ modules)
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn (RandomForest, GBM, AdaBoost, DecisionTree, Bagging)
- XGBoost
- imbalanced-learn (SMOTE, RandomUnderSampler)
- matplotlib, seaborn

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/<your-username>/VisaRisk.git
cd VisaRisk

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter lab certify_visa-risk.ipynb
```

The data path is resolved automatically from the project root — no manual edits needed.
