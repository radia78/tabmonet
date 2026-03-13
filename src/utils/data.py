from ucimlrepo import fetch_ucirepo
import torch
import numpy as np
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    QuantileTransformer,
    LabelEncoder,
    StandardScaler,
)
from autogluon.features import LabelEncoderFeatureGenerator

from typing import Optional


class TabularDataset(Dataset):
    def __init__(
        self,
        cat_features,
        cont_features,
        targets,
    ):
        # Initalize the variables
        self.cat_features = None
        self.cont_features = None

        if cat_features is not None:
            self.cat_features = torch.tensor(cat_features, dtype=torch.long)

        if cont_features is not None:
            self.cont_features = torch.tensor(cont_features, dtype=torch.float32)

        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        if self.cont_features is not None and self.cat_features is None:
            return (self.cont_features[idx], np.nan), self.targets[idx]

        elif self.cont_features is None and self.cat_features is not None:
            return (np.nan, self.cat_features[idx]), self.targets[idx]

        else:
            return (self.cont_features[idx], self.cat_features[idx]), self.targets[idx]


class DataPreprocessor:
    def __init__(self, problem_type: str, num_bins: Optional[int]):
        # Per SOTA, we turn the numerical features via Quantile transformation
        self.cont_encoder = QuantileTransformer(output_distribution="uniform")
        self.cat_encoder = LabelEncoderFeatureGenerator(verbosity=0)

        self.num_bins = num_bins
        self.problem_type = problem_type

        if problem_type == "regression":
            self.target_encoder = StandardScaler()
        else:
            self.target_encoder = LabelEncoder()

        self.cat_columns = []
        self.cont_columns = []

        self.num_cat_features = 0
        self.num_cont_features = 0

    def preprocess(self, X, y, is_train=False):
        cat_features = None
        cont_features = None

        bin_edges = None
        if is_train:
            self.cont_columns = X.select_dtypes(include="number").columns.tolist()
            self.cat_columns = X.select_dtypes(exclude="number").columns.tolist()

            self.num_cat_features = len(self.cat_columns)
            self.num_cont_features = len(self.cont_columns)

            # Fit them separately and retain the max-class inforamtion automatically
            if self.cat_columns:
                cat_features = self.cat_encoder.fit_transform(
                    X[self.cat_columns].astype("category")
                )
                self.cat_features = (
                    X[self.cat_columns].describe(include="all")["unique"].max()
                )

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

            y_processed = self.target_encoder.fit_transform(y.to_numpy().reshape(-1, 1))

        else:
            if self.cat_columns:
                cat_features = self.cat_encoder.transform(
                    X[self.cat_columns].astype("category")
                )

            if self.cont_columns:
                cont_features = self.cont_encoder.transform(
                    X[self.cont_columns].astype("float32")
                )

            y_processed = self.target_encoder.transform(y.to_numpy().reshape(-1, 1))

        return cat_features, cont_features, y_processed, bin_edges

    def inverse_transform_target(self, y):
        return self.target_encoder.inverse_transform(y)


def prepare_dataset(
    dataset_id: id, problem_type: str, test_size: float = 0.2, num_bins: int = 100
):
    # Load and split the dataset
    dataset = fetch_ucirepo(id=dataset_id)

    X, X_test, y, y_test = train_test_split(
        dataset.data.features, dataset.data.targets, test_size=test_size
    )

    # Always have an 80-20 split on validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Fit the preprocessor and get the processed outputs
    preprocessor = DataPreprocessor(problem_type=problem_type, num_bins=num_bins)

    # Preprocess all the train, val, test data
    cat_train, cont_train, y_train_proc, bin_edges = preprocessor.preprocess(
        X_train, y_train, is_train=True
    )
    cat_val, cont_val, y_val_proc, _ = preprocessor.preprocess(
        X_val, y_val, is_train=False
    )
    cat_test, cont_test, y_test_proc, _ = preprocessor.preprocess(
        X_test, y_test, is_train=False
    )

    if bin_edges is not None:
        bin_edges = torch.tensor(bin_edges)

    return (
        TabularDataset(
            cat_features=cat_train, cont_features=cont_train, targets=y_train_proc
        ),
        TabularDataset(
            cat_features=cat_val, cont_features=cont_val, targets=y_val_proc
        ),
        TabularDataset(
            cat_features=cat_test, cont_features=cont_test, targets=y_test_proc
        ),
        preprocessor,
        bin_edges,
    )
