"""
Hyperparameter search grids for all models.

Each entry is a dict with keys:
  random  - param_distributions for RandomizedSearchCV
  grid    - param_grid for GridSearchCV (narrowed after random search)
  n_iter  - iterations for RandomizedSearchCV
  cv      - cross-validation folds
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# ---------------------------------------------------------------------------
# Gradient Boosting (GBM)
# ---------------------------------------------------------------------------
GBM_PARAMS = {
    "random": {
        "init": [AdaBoostClassifier(random_state=1), DecisionTreeClassifier(random_state=1)],
        "n_estimators": np.arange(100, 175, 25).tolist(),
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "subsample": [0.7, 0.8, 0.9],
        "max_features": [0.5, 0.7, 1],
    },
    "grid": {
        "init": [AdaBoostClassifier(random_state=1), DecisionTreeClassifier(random_state=1)],
        "n_estimators": np.arange(100, 250, 25).tolist(),
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "subsample": [0.7, 0.8, 0.9],
        "max_features": [0.5, 0.7, 1],
    },
    "n_iter": 50,
    "cv": 5,
}

# Wider grid used for oversampling / undersampling runs
GBM_PARAMS_WIDE = {
    "random": {
        "init": [AdaBoostClassifier(random_state=1), DecisionTreeClassifier(random_state=1)],
        "n_estimators": np.arange(100, 300, 25).tolist(),
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "subsample": [0.7, 0.8, 0.9],
        "max_features": [0.3, 0.5, 0.7, 1],
    },
    "grid": {
        "init": [AdaBoostClassifier(random_state=1), DecisionTreeClassifier(random_state=1)],
        "n_estimators": np.arange(100, 300, 50).tolist(),
        "learning_rate": [0.1, 0.05, 0.01, 0.005],
        "subsample": [0.7, 0.8, 0.9],
        "max_features": [0.3, 0.5, 0.7, 1],
    },
    "n_iter": 50,
    "cv": 5,
}

# ---------------------------------------------------------------------------
# AdaBoost
# ---------------------------------------------------------------------------
ADA_PARAMS = {
    "random": {
        "n_estimators": [50, 75, 85, 100, 150],
        "learning_rate": [1.0, 0.5, 0.1, 0.01],
        "estimator": [
            DecisionTreeClassifier(max_depth=1, random_state=1),
            DecisionTreeClassifier(max_depth=2, random_state=1),
            DecisionTreeClassifier(max_depth=3, random_state=1),
            DecisionTreeClassifier(max_depth=4, random_state=1),
        ],
    },
    "grid": {
        "n_estimators": np.arange(65, 155, 10).tolist(),
        "learning_rate": [1.0, 0.5, 0.1, 0.01],
        "estimator": [
            DecisionTreeClassifier(max_depth=1, random_state=1),
            DecisionTreeClassifier(max_depth=2, random_state=1),
            DecisionTreeClassifier(max_depth=3, random_state=1),
            DecisionTreeClassifier(max_depth=4, random_state=1),
        ],
    },
    "n_iter": 30,
    "cv": 5,
}

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "random": {
        "n_estimators": [50, 75, 100, 125, 200],
        "scale_pos_weight": [2, 5, 10],
        "learning_rate": [0.01, 0.1, 0.2, 0.05, 0.005],
        "gamma": [0, 1, 3, 5, 8],
        "subsample": [0.5, 0.7, 0.8, 1.0],
        "max_depth": np.arange(1, 5, 1).tolist(),
        "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
        "colsample_bylevel": [0.3, 0.5, 0.7, 1.0],
        "reg_lambda": [5, 10],
    },
    "n_iter": 50,
    "cv": 10,
}

# Smaller grid for undersampling (faster)
XGB_PARAMS_UNDER = {
    "random": {
        "n_estimators": [50, 75, 100, 125],
        "scale_pos_weight": [2, 5, 10],
        "learning_rate": [0.01, 0.1, 0.2, 0.05],
        "gamma": [0, 1, 3, 5, 8],
        "subsample": [0.5, 0.7, 0.8, 1.0],
        "max_depth": np.arange(1, 5, 1).tolist(),
        "colsample_bytree": [0.3, 0.5, 0.7, 1.0],
        "colsample_bylevel": [0.3, 0.5, 0.7, 1.0],
        "reg_lambda": [5, 10],
    },
    "n_iter": 50,
    "cv": 10,
}
