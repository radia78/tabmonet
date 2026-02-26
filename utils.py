from scipy import sparse
from ucimlrepo import fetch_ucirepo
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    QuantileTransformer, 
    OrdinalEncoder, 
    StandardScaler,
    OneHotEncoder
)

def sparse_array_to_sparse_tensor(sparse_array):

    # Get all the necessary information for converting the sparse array
    crow_indices= torch.tensor(sparse_array.indptr, dtype=torch.long)
    col_indices = torch.tensor(sparse_array.indices, dtype=torch.long)
    values = torch.tensor(sparse_array.data, dtype=torch.float32)
    size_tuple = sparse_array.shape

    # Create a new sparse tensor
    return torch.sparse_csr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        size=size_tuple
    )

class DataPreprocessor:
    def __init__(self, task):
        # Per SOTA, we turn the numerical features via Quantile transformation
        self.numerical_feature_preprocessor = QuantileTransformer(output_distribution='normal')
        self.categorical_feature_preprocessor = OneHotEncoder(sparse_output=False)

        if task == "reg":
            self.target_preprocessor = StandardScaler()

        else:
            self.target_preprocessor = OrdinalEncoder()

        self.categorical_columns = []
        self.numerical_columns = []
        self.num_numerical_features = 0
        self.num_cateogrical_features = 0

    def fit(self, X, y):
        # Separate the features into categorical and non-categorical data
        self.numerical_columns = X.select_dtypes(include="number").columns.tolist()
        self.categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

        # Fit them separately and retain the max-class inforamtion automatically
        if self.categorical_columns:
            self.categorical_feature_preprocessor.fit(X[self.categorical_columns])
        
        if self.numerical_columns:
            self.numerical_feature_preprocessor.fit(X[self.numerical_columns])

        # Fit the target transformer
        self.target_preprocessor.fit(y)

    def transform(self, X, y):
        cat_features = None
        num_features = None
        y_processed = None

        if self.categorical_columns:
            cat_features = self.categorical_feature_preprocessor.transform(X[self.categorical_columns])
            self.num_cateogrical_features = cat_features.shape[-1]

        if self.numerical_columns:
            num_features = self.numerical_feature_preprocessor.transform(X[self.numerical_columns])
            self.num_numerical_features = num_features.shape[-1]

        y_processed = self.target_preprocessor.transform(y)

        return cat_features, num_features, y_processed
    
    def inverse_transform_target(self, y):
        return self.target_preprocessor.inverse_transform(y)


def load_dataset(
        dataset_id,
        task,
        test_size,
        batch_size,
        return_bin_edges,
        num_bins
        ):
    # Load and split the dataset
    dataset = fetch_ucirepo(id=dataset_id)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data.features, dataset.data.targets, test_size=test_size
    )
    
    # Fit the preprocessor and get the processed outputs
    preprocessor = DataPreprocessor(task)
    preprocessor.fit(X_train, y_train)

    print("Num of Categorical Features: ", len(preprocessor.categorical_columns))
    print("Num of Numerical Features: ", len(preprocessor.numerical_columns))

    X_train_cat, X_train_num, y_train_processed = preprocessor.transform(X_train, y_train)
    X_test_cat, X_test_num, y_test_processed = preprocessor.transform(X_test, y_test)
    
    # If categorical is missing only use the numerical
    if not preprocessor.categorical_columns:
        train_data = TensorDataset(
            torch.tensor(X_train_num, dtype=torch.float32),
            torch.tensor(y_train_processed, dtype=torch.float32)
        )

        test_data = TensorDataset(
            torch.Tensor(X_test_num, dtype=torch.float32), 
            torch.Tensor(y_test_processed, dtype=torch.float32)
        )

    # If numerical is missing only use categorical data
    if not preprocessor.numerical_columns:

        train_data = TensorDataset(
            torch.tensor(X_train_cat, dtype=torch.float32),
            torch.tensor(y_train_processed, dtype=torch.float32)
        )

        test_data = TensorDataset(
            torch.tensor(X_test_cat, dtype=torch.float32),
            torch.tensor(y_test_processed, dtype=torch.float32)
        )

    else:
        train_data = TensorDataset(
            torch.tensor(X_train_num, dtype=torch.float32),
            torch.tensor(X_train_cat, dtype=torch.float32),
            torch.tensor(y_train_processed, dtype=torch.float32)
        )

        test_data = TensorDataset(
            torch.tensor(X_test_num, dtype=torch.float32),
            torch.tensor(X_train_cat, dtype=torch.float32),
            torch.tensor(y_test_processed, dtype=torch.float32)
        )

    if return_bin_edges:
        bin_edges = torch.quantile(
            torch.tensor(X_train_num, dtype=torch.float32), 
            torch.tensor([i / num_bins for i in range(num_bins + 1)]), 
            dim=0
        )
        
        return (
            DataLoader(train_data, batch_size=batch_size),
            DataLoader(test_data, batch_size=batch_size),
            preprocessor,
            bin_edges,
        )

    else:
        return (
            DataLoader(train_data, batch_size=batch_size),
            DataLoader(test_data, batch_size=batch_size),
            preprocessor,
            None,
        )