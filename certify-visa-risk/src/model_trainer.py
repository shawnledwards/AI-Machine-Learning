"""
Model building, resampling, and hyperparameter tuning for VisaRisk.
"""

from sklearn import metrics
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ---------------------------------------------------------------------------
# Base model list
# ---------------------------------------------------------------------------

def get_base_models():
    """
    Return the standard list of (name, unfitted_model) tuples used for
    the initial untuned comparison across all sampling strategies.
    """
    return [
        ("Bagging", BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=1, class_weight="balanced"),
            random_state=1,
        )),
        ("Random Forest", RandomForestClassifier(random_state=1, class_weight="balanced")),
        ("GBM", GradientBoostingClassifier(random_state=1)),
        ("AdaBoost", AdaBoostClassifier(random_state=1)),
        ("Decision Tree", DecisionTreeClassifier(random_state=1, class_weight="balanced")),
        ("XGBoost", XGBClassifier(random_state=1, eval_metric="logloss")),
    ]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def train_and_compare(models, X_train, y_train, X_val, y_val, label=""):
    """
    Fit each model on (X_train, y_train) and print F1 train/val scores
    with the gap between them.

    models: list of (name, unfitted_model) tuples
    label: optional prefix printed above the table (e.g. 'Original', 'SMOTE')

    Returns the same list with fitted models (mutates in-place and returns).
    """
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"{'Model':<22} {'Train F1':>10} {'Val F1':>10} {'Gap':>10}")
    print("-" * 55)
    for name, model in models:
        model.fit(X_train, y_train)
        f1_train = f1_score(y_train, model.predict(X_train))
        f1_val = f1_score(y_val, model.predict(X_val))
        print(f"{name:<22} {f1_train:>10.4f} {f1_val:>10.4f} {f1_train - f1_val:>10.4f}")
    return models


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train, sampling_strategy=1, k_neighbors=5, random_state=1):
    """
    Apply SMOTE oversampling to the minority class.

    Returns (X_resampled, y_resampled).
    """
    print(f"Before SMOTE — class 1: {sum(y_train == 1):,}  class 0: {sum(y_train == 0):,}")
    sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_over, y_over = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE  — class 1: {sum(y_over == 1):,}  class 0: {sum(y_over == 0):,}")
    return X_over, y_over


def apply_undersampling(X_train, y_train, random_state=1):
    """
    Apply random undersampling to the majority class.

    Returns (X_resampled, y_resampled).
    """
    print(f"Before Under — class 1: {sum(y_train == 1):,}  class 0: {sum(y_train == 0):,}")
    rus = RandomUnderSampler(random_state=random_state)
    X_un, y_un = rus.fit_resample(X_train, y_train)
    print(f"After Under  — class 1: {sum(y_un == 1):,}  class 0: {sum(y_un == 0):,}")
    return X_un, y_un


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_random(model, param_distributions, X_train, y_train, n_iter=50, cv=5, random_state=1):
    """
    Fit a RandomizedSearchCV and return the best estimator.

    model: unfitted sklearn/XGBoost estimator
    param_distributions: dict of parameter distributions
    n_iter: number of random candidates to try (default 50)
    cv: cross-validation folds (default 5)
    """
    scorer = metrics.make_scorer(metrics.f1_score)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}  CV F1: {search.best_score_:.4f}")
    return search.best_estimator_


def tune_grid(model, param_grid, X_train, y_train, cv=5):
    """
    Fit a GridSearchCV and return the best estimator.

    model: unfitted sklearn/XGBoost estimator
    param_grid: dict of parameter lists
    cv: cross-validation folds (default 5)
    """
    scorer = metrics.make_scorer(metrics.f1_score)
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}  CV F1: {search.best_score_:.4f}")
    return search.best_estimator_
