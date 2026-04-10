"""
Model evaluation utilities for the VisaRisk classification task.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def model_performance_classification_sklearn(model, predictors, target):
    """
    Return a single-row DataFrame with Accuracy, Recall, Precision, and F1.

    model: fitted sklearn/XGBoost classifier
    predictors: feature DataFrame
    target: true labels Series
    """
    pred = model.predict(predictors)
    return pd.DataFrame(
        {
            "Accuracy": accuracy_score(target, pred),
            "Recall": recall_score(target, pred),
            "Precision": precision_score(target, pred),
            "F1": f1_score(target, pred),
        },
        index=[0],
    )


def confusion_matrix_sklearn(model, predictors, target):
    """
    Plot a confusion matrix annotated with counts and row percentages.

    model: fitted classifier
    predictors: feature DataFrame
    target: true labels Series
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def compare_models(results_train: pd.DataFrame, results_val: pd.DataFrame):
    """
    Display side-by-side training and validation metric tables for all tuned models.

    results_train / results_val: DataFrames where each column is a model name
                                  and rows are metric names (Accuracy, Recall, etc.)
    """
    print("=== Train Performance ===")
    display_or_print(results_train)
    print("\n=== Validation Performance ===")
    display_or_print(results_val)


def display_or_print(df):
    """Use IPython display if available, otherwise print."""
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df)


def select_best_model(results_val: pd.DataFrame, model_registry: list, metric: str = "F1"):
    """
    Pick the model with the highest validation metric score.

    results_val: DataFrame of metrics (rows) × model names (cols)
    model_registry: list of (name, fitted_model) tuples
    metric: row label to rank by (default 'F1')

    Returns (best_name, best_model).
    """
    best_name = results_val.loc[metric].idxmax()
    best_score = results_val.loc[metric, best_name]
    print(f"Best model by validation {metric}: {best_name} ({best_score:.4f})")

    registry_dict = dict(model_registry)
    best_model = registry_dict[best_name]
    return best_name, best_model


def plot_feature_importance(model, feature_names, figsize=(12, 12)):
    """
    Horizontal bar chart of feature importances.

    model: fitted tree-based model with feature_importances_ attribute
    feature_names: list/Index of feature names matching the training columns
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=figsize)
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color="violet", align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.show()


def save_model(model, filename: str = "visa-risk.pkl"):
    """
    Persist the trained model to the models/ directory via joblib.

    model: fitted classifier
    filename: file name (default 'gbm_easyvisa.pkl')
    """
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    path = models_dir / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path


def load_model(filename: str = "visa-risk.pkl"):
    """Load a previously saved model from the models/ directory."""
    path = Path(__file__).resolve().parents[1] / "models" / filename
    return joblib.load(path)
