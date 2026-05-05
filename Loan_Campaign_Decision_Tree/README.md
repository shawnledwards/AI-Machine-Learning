# Personal Loan Campaign — Decision Tree Classifier

A machine learning project completed during the ML phase of my data science coursework. The goal is to help Branch Bank identify liability customers most likely to convert to personal loan customers, enabling targeted marketing campaigns.

## Problem Statement

Branch Bank wants to expand its base of personal loan customers. A prior campaign achieved a 9.6% conversion rate. This project builds a classification model to improve targeting — identifying which customers are most likely to accept a personal loan offer.

## Objective

Predict whether a liability customer will buy a personal loan, identify the customer attributes that drive purchases, and determine which customer segments to prioritize.

## Dataset

`data/Loan_Modelling.csv` — 5,000 Branch Bank customers with the following features:

| Feature | Description |
|---|---|
| `Age` | Customer age (years) |
| `Experience` | Years of professional experience |
| `Income` | Annual income (thousands USD) |
| `Family` | Family size |
| `CCAvg` | Avg. monthly credit card spend (thousands USD) |
| `Education` | 1: Undergrad, 2: Graduate, 3: Advanced/Professional |
| `Mortgage` | Value of home mortgage (thousands USD) |
| `Securities_Account` | Has securities account with bank |
| `CD_Account` | Has certificate of deposit account |
| `Online` | Uses internet banking |
| `CreditCard` | Has credit card from another bank |
| `Personal_Loan` | **Target** — accepted personal loan offer (1: Yes, 0: No) |

## Models Compared

| Model | Accuracy | Recall | Precision | F1 Score |
|---|---|---|---|---|
| Default Decision Tree | **0.980** | 0.886 | 0.910 | **0.898** |
| Balanced Weights | 0.975 | 0.859 | 0.883 | 0.871 |
| Pre-Pruned (hyperparameter tuning) | 0.779 | 1.000 | 0.310 | 0.474 |
| Post-Pruned (cost-complexity) | 0.949 | 0.993 | 0.661 | 0.794 |

**Selected model:** Default Decision Tree — highest F1 score (0.898) and best balance of precision and recall.

## Key Findings

- **Income** is the strongest predictor of loan purchase, followed by **Education**, **Family size**, and **CCAvg**
- Customers with income under $116,500 combined with CCAvg under $2,950 and some post-graduate education are the primary target segment
- For higher-income customers (>$116,500), post-graduate education and family size over two become the deciding factors
- The 9.6% class imbalance was addressed using `class_weight='balanced'` during model tuning

## Business Recommendations

- Target families with college-age children (back-to-school and tuition expenses)
- Offer credit card debt consolidation programs as a loan entry point
- Incentivize Securities and CD account holders with lower-rate loan products
- Run campaigns across both walk-up and online banking channels

## Notebook

`Loan_Campaign_Decision_Tree.ipynb`

## Setup

```bash
pip install numpy==1.25.2 pandas==1.5.3 matplotlib==3.7.1 seaborn==0.13.1 scikit-learn==1.2.2 sklearn-pandas==2.2.0
```

Run the notebook using Jupyter or directly in PyCharm. Data is expected at `data/Loan_Modelling.csv`.