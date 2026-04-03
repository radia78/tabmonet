from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from autogluon.features import LabelEncoderFeatureGenerator

import torch.nn as nn

from typing import Optional

@dataclass
class DataPreprocessorConfig:
    problem_type: str
    numerical_preprocessor: TransformerMixin
    num_bins: Optional[int]
    cont_columns: Optional[list[str]] = None
    cat_columns: Optional[list[str]] = None

class NumericalEmbeddingConfig:
    

class OptimizerConfig:
    name: str



class TabMONetConfig:
    neck: nn.Module
    backbone: nn.Module
    head: nn.Module
    numerical_encoder: Optional[nn.Module]
    categorical_encoder: Optional[nn.Module]
