from ucimlrepo import fetch_ucirepo
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    QuantileTransformer, 
    LabelEncoder,
    StandardScaler,
)

class TabularDataset(Dataset):
    def __init__(
            self,
            categorical_features,
            numerical_features,
            target,
            num_bins,
        ):
        # Initalize the variables
        self.categorical_features = None
        self.numerical_features = None
        self.bin_edges = None
        
        if categorical_features is not None:
            self.categorical_features = torch.tensor(
                categorical_features, 
                dtype=torch.long
            )

        if numerical_features is not None:
            self.numerical_features = torch.tensor(
                numerical_features,
                dtype=torch.float32
            )

        self.target = torch.tensor(
            target,
            dtype=torch.float32
        )

        if num_bins is not None:
            self.bin_edges = torch.quantile(
                torch.tensor(self.numerical_features, dtype=torch.float32), 
                torch.tensor([i / num_bins for i in range(num_bins + 1)]), 
                dim=0
            )

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        if self.numerical_features is not None and self.categorical_features is None:
            return self.numerical_features[idx], None, self.target[idx]
        
        elif self.numerical_features is None and self.categorical_features is not None:
            return None, self.categorical_features[idx], self.target[idx]
        
        else:
            return self.numerical_features[idx], self.categorical_features[idx], self.target[idx]

class DataPreprocessor:
    def __init__(self, task, max_class):
        # Per SOTA, we turn the numerical features via Quantile transformation
        self.numerical_feature_preprocessor = QuantileTransformer(output_distribution='normal')
        self.categorical_feature_preprocessor = OrdinalEncoder(max_categories=max_class)

        if task == "reg":
            self.target_preprocessor = StandardScaler()

        else:
            self.target_preprocessor = LabelEncoder()

        self.categorical_columns = []
        self.numerical_columns = []

        self.num_categorical_features = 0
        self.num_numerical_features = 0

    def fit(self, X, y):
        # Separate the features into categorical and non-categorical data
        self.numerical_columns = X.select_dtypes(include="number").columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

        # Fit them separately and retain the max-class inforamtion automatically
        if self.categorical_columns:
            self.categorical_feature_preprocessor.fit(X[self.categorical_columns])
            self.num_categorical_features = self.categorical_feature_preprocessor.n_features_in_
        
        if self.numerical_columns:
            self.numerical_feature_preprocessor.fit(X[self.numerical_columns])
            self.num_numerical_features = self.numerical_feature_preprocessor.n_features_in_

        # Fit the target transformer
        self.target_preprocessor.fit(y)

    def transform(self, X, y):
        cat_features = None
        num_features = None
        y_processed = None

        if self.categorical_columns:
            cat_features = self.categorical_feature_preprocessor.transform(X[self.categorical_columns])

        if self.numerical_columns:
            num_features = self.numerical_feature_preprocessor.transform(X[self.numerical_columns])

        y_processed = self.target_preprocessor.transform(y)

        return cat_features, num_features, y_processed
    
    
    def inverse_transform_target(self, y):
        return self.target_preprocessor.inverse_transform(y)

def load_dataset(
        dataset_id,
        task,
        max_class,
        test_size,
        num_bins
    ):
    # Load and split the dataset
    dataset = fetch_ucirepo(id=dataset_id)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data.features, dataset.data.targets, test_size=test_size
    )
    
    # Fit the preprocessor and get the processed outputs
    preprocessor = DataPreprocessor(task, max_class)
    preprocessor.fit(X_train, y_train)

    X_train_cat, X_train_num, y_train_processed = preprocessor.transform(X_train, y_train)
    X_test_cat, X_test_num, y_test_processed = preprocessor.transform(X_test, y_test)

    return (
        TabularDataset(X_train_cat, X_train_num, y_train_processed, num_bins), # Training dataset
        TabularDataset(X_test_cat, X_test_num, y_test_processed, num_bins=None), # Test dataset
        preprocessor
    )