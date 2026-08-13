"""Fit-on-train-only feature encoding for window-instance tabular features.

Per the project rules, no statistic used for preprocessing (imputation
values, scaling mean/std, one-hot vocabulary) may be derived from
validation or test data. `TabularFeatureBuilder.fit` must therefore only
ever be called with the TRAIN split's window instances; `.transform` is
then reused, unchanged, on val/test.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TabularFeatureBuilder:
    def __init__(self, numeric_cols: List[str], categorical_cols: List[str]):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self._fitted = False

        numeric_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_pipe, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_pipe, categorical_cols))
        self.column_transformer = ColumnTransformer(transformers)

    def fit(self, train_windows_df: pd.DataFrame) -> "TabularFeatureBuilder":
        self.column_transformer.fit(train_windows_df)
        self._fitted = True
        return self

    def transform(self, windows_df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TabularFeatureBuilder.fit() (on TRAIN split only) must be called before transform().")
        return self.column_transformer.transform(windows_df)

    def fit_transform_train(self, train_windows_df: pd.DataFrame) -> np.ndarray:
        self.fit(train_windows_df)
        return self.transform(train_windows_df)

    @property
    def feature_names(self) -> List[str]:
        return list(self.column_transformer.get_feature_names_out())
