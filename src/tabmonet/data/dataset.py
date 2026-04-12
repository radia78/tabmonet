import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tabmonet.data.preprocess import DataPreprocessor

from typing import Optional, List


class TabularDataset(Dataset):
    def __init__(self, cat_features, cont_features, problem_type, targets):
        # Initalize the variables
        self.cat_features = None
        self.cont_features = None

        if cat_features is not None:
            self.cat_features = torch.tensor(cat_features, dtype=torch.long)

        if cont_features is not None:
            self.cont_features = torch.tensor(cont_features, dtype=torch.float32)

        if problem_type == "regression" or problem_type == "binary":
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        if self.cont_features is not None and self.cat_features is None:
            return (self.cont_features[idx], np.nan), self.targets[idx]

        elif self.cont_features is None and self.cat_features is not None:
            return (np.nan, self.cat_features[idx]), self.targets[idx]

        else:
            return (self.cont_features[idx], self.cat_features[idx]), self.targets[idx]


def prepare_dataset(
    dataset_path: str,
    problem_type: str,
    rm_NA: bool,
    preprocessor: DataPreprocessor,
    target_column: str,
    numerical_columns: Optional[List[str]]=None,
    categorical_columns: Optional[List[str]]=None,
    test_size: float = 0.2,
    **kwargs
):
    # Load and split the dataset
    dataset = pd.read_csv(dataset_path)
    if rm_NA:
        print(f"Number of samples: {len(dataset)}")
        dataset.dropna(inplace=True)

        if problem_type == "regression":
            X, X_test, y, y_test = train_test_split(
                dataset[numerical_columns + categorical_columns],
                dataset[[target_column]],
                test_size=test_size,
            )
        else:
            X, X_test, y, y_test = train_test_split(
                dataset[numerical_columns + categorical_columns],
                dataset[[target_column]],
                test_size=test_size,
                stratify=dataset[[target_column]],
            )
        print(f"Number of samples after dropping null values: {len(dataset)}")
    else:
        if problem_type == "regression":
            X, X_test, y, y_test = train_test_split(
                dataset[numerical_columns + categorical_columns],
                dataset[[target_column]],
                test_size=test_size
            )
        else:
            X, X_test, y, y_test = train_test_split(
                dataset[numerical_columns + categorical_columns],
                dataset[[target_column]],
                test_size=test_size,
                stratify=dataset[[target_column]],
            )

    # Always have an 80-20 split on validation
    if problem_type == "regression":
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y.astype("category")
        )

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
            problem_type=problem_type,
            cat_features=cat_train,
            cont_features=cont_train,
            targets=y_train_proc,
        ),
        TabularDataset(
            problem_type=problem_type,
            cat_features=cat_val,
            cont_features=cont_val,
            targets=y_val_proc,
        ),
        TabularDataset(
            problem_type=problem_type,
            cat_features=cat_test,
            cont_features=cont_test,
            targets=y_test_proc,
        ),
        bin_edges,
    )
