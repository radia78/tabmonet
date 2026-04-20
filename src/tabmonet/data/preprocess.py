import numpy as np
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.base import BaseEstimator, TransformerMixin
from autogluon.features import LabelEncoderFeatureGenerator

from typing import Optional


class DataPreprocessor:
    def __init__(
        self,
        problem_type: str,
        numerical_preprocessor: TransformerMixin,
        num_bins: Optional[int],
        cont_columns: Optional[list[str]] = None,
        cat_columns: Optional[list[str]] = None,
    ):
        # Per SOTA, change the preprocessing based on Robust-CLipping or something else
        self.problem_type = problem_type
        self.cont_encoder = numerical_preprocessor
        self.cat_encoder = LabelEncoderFeatureGenerator(verbosity=0)
        self.num_bins = num_bins

        if problem_type == "regression":
            self.target_encoder = StandardScaler()
        else:
            self.target_encoder = LabelEncoder()

        self.cat_columns = cat_columns if cat_columns is not None else None
        self.cont_columns = cont_columns if cont_columns is not None else None

        self.max_categories = 0

        self.num_cat_features = 0
        self.num_cont_features = 0

    def preprocess(self, X, y=None, is_train=False):
        cat_features = None
        cont_features = None
        y_processed = None
        bin_edges = None

        if is_train:
            if self.cont_columns is None:
                self.cont_columns = X.select_dtypes(include="number").columns.tolist()
            if self.cat_columns is None:
                self.cat_columns = X.select_dtypes(exclude="number").columns.tolist()
            self.num_cat_features = len(self.cat_columns)
            self.num_cont_features = len(self.cont_columns)

            # Fit them separately and retain the max-class inforamtion automatically
            if self.cat_columns:
                self.max_categories = []
                for c in self.cat_columns:
                    self.max_categories.append(X[c].cat.codes.max())

                cat_features = self.cat_encoder.fit_transform(
                    X[self.cat_columns].astype("category")
                ).to_numpy()

            if self.cont_columns:
                cont_features = self.cont_encoder.fit_transform(
                    X[self.cont_columns].astype("float32")
                )

                # Finding the bins and returning as needed
                if self.num_bins:
                    bin_edges = (
                        np.quantile(
                            cont_features,
                            np.array(
                                [i / self.num_bins for i in range(self.num_bins + 1)]
                            ),
                            axis=0,
                        )
                        .astype("float32")
                        .T
                    )
            if y is not None:
                if self.problem_type == "binary":
                    y_processed = self.target_encoder.fit_transform(
                        y.to_numpy().reshape(-1, 1)
                    )[:, np.newaxis]
                else:
                    y_processed = self.target_encoder.fit_transform(
                        y.to_numpy().reshape(-1, 1)
                    )

        else:
            if self.cat_columns:
                cat_features = self.cat_encoder.transform(
                    X[self.cat_columns].astype("category")
                ).to_numpy()

            if self.cont_columns:
                cont_features = self.cont_encoder.transform(
                    X[self.cont_columns].astype("float32")
                )
            if y is not None:
                if self.problem_type == "binary":
                    y_processed = self.target_encoder.transform(
                        y.to_numpy().reshape(-1, 1)
                    )[:, np.newaxis]
                else:
                    y_processed = self.target_encoder.transform(
                        y.to_numpy().reshape(-1, 1)
                    )

        return cat_features, cont_features, y_processed, bin_edges

    def inverse_transform_target(self, y):
        return self.target_encoder.inverse_transform(y)


class RobustScaleSmoothClipTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_array = X.to_numpy()
        self._median = np.median(X_array, axis=-2)
        quant_diff = np.quantile(X_array, 0.75, axis=-2) - np.quantile(
            X_array, 0.25, axis=-2
        )
        max_val = np.max(X_array, axis=-2)
        min_val = np.min(X_array, axis=-2)

        idxs = quant_diff == 0.0
        # On indexes where the quantile difference (IQR) is zero, fallback to a variation of min-max scaling
        quant_diff[idxs] = 0.5 * (max_val[idxs] - min_val[idxs])

        factors = 1.0 / (quant_diff + 1e-30)

        # If the feature is entirely constant, set factor to zero
        factors[quant_diff == 0.0] = 0.0

        self._factors = factors
        return self

    def transform(self, X, y=None):
        X_array = X.copy().to_numpy()
        # 1. Robust Scaling
        x_scaled = self._factors[None, :] * (X_array - self._median[None, :])

        # 2. Smooth Clipping
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)
