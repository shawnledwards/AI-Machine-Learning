"""
Data loading and feature engineering for the EasyVisa dataset.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "EasyVisa.csv"

EDUCATION_ORDER = ["High School", "Bachelor's", "Master's", "Doctorate"]

WAGE_TO_ANNUAL = {
    "Hour": 173 * 12,   # 40 hrs/wk × ~4.33 wks/mo × 12 mo
    "Week": 52.2,
    "Month": 12,
    "Year": 1,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_data(path=None):
    """Load EasyVisa.csv and return a working copy."""
    filepath = Path(path) if path else DATA_PATH
    raw = pd.read_csv(filepath)
    return raw.copy()


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in order.

    1. Drop case_id (non-informative key)
    2. Fix negative no_of_employees → median
    3. Ordinal-encode education_of_employee
    4. Normalize prevailing_wage to annual
    5. Quantile-bin no_of_employees and yr_of_estab
    6. Encode target case_status → int (Certified=1, Denied=0)

    Returns a transformed copy; does not modify the input in-place.
    """
    df = data.copy()

    # 1. Drop key column
    df.drop(columns=["case_id"], inplace=True)

    # 2. Fix negative employee counts
    median_emp = df.loc[df["no_of_employees"] >= 0, "no_of_employees"].median()
    df.loc[df["no_of_employees"] < 0, "no_of_employees"] = median_emp

    # 3. Ordinal-encode education
    edu_map = {level: i for i, level in enumerate(EDUCATION_ORDER)}
    df["education_of_employee"] = df["education_of_employee"].map(edu_map)

    # 4. Normalize wages to annual
    for unit, multiplier in WAGE_TO_ANNUAL.items():
        mask = df["unit_of_wage"] == unit
        df.loc[mask, "prevailing_wage"] = df.loc[mask, "prevailing_wage"] * multiplier
        if unit != "Year":
            df.loc[mask, "unit_of_wage"] = f"{unit}_to_Year"

    # 5. Quantile-bin continuous columns
    df["no_of_employees"] = pd.qcut(df["no_of_employees"], q=4, labels=None)
    df["yr_of_estab"] = pd.qcut(df["yr_of_estab"], q=4, labels=None)

    # 6. Encode target
    df["case_status"] = df["case_status"].map({"Certified": 1, "Denied": 0}).astype(int)

    return df


def encode_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """
    One-hot encode categorical columns across all three splits and sanitize
    column names for XGBoost compatibility.

    Returns (X_train_enc, X_val_enc, X_test_enc) as float DataFrames.
    """
    X_train = pd.get_dummies(X_train, drop_first=True).astype(float)
    X_val = pd.get_dummies(X_val, drop_first=True).astype(float)
    X_test = pd.get_dummies(X_test, drop_first=True).astype(float)

    # Align val/test columns to train — fills any missing category columns with 0
    X_val  = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    def _sanitize(cols):
        return ["".join(c if c.isalnum() else "_" for c in str(col)) for col in cols]

    X_train.columns = _sanitize(X_train.columns)
    X_val.columns = _sanitize(X_val.columns)
    X_test.columns = _sanitize(X_test.columns)

    return X_train, X_val, X_test
