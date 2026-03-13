import torch
import torch.nn as nn
from typing import Optional, Union

from utils.data import DataPreprocessor
from .layers.mlp import PolyMLP, MultiHeadPolyMLP
from .layers.embeddings import *

from functools import partial

EMB_REGISTRY = {
    "PBLD": PBLDEmbedding,
    "P": PeriodicEmbedding,
    "L": LinearEmbedding,
    "Q": QuantileEmbedding,
    "Q-NF": QuantileEmbeddingNoFraction,
    "NA": nn.Identity,
    "QV2": QuantileEmbeddingV2,
}

LAYER_REGISTRY = {"poly": PolyMLP, "mpoly": MultiHeadPolyMLP}


class PolyBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        expansion_factor=3,
        mlp_layer=Union[PolyMLP, MultiHeadPolyMLP],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.norm = norm_layer(self.embed_dim)
        self.mlp1 = mlp_layer(
            embed_dim,
            embed_dim,
            embed_dim,
        )
        self.mlp2 = mlp_layer(
            embed_dim,
            embed_dim * self.expansion_factor,
            embed_dim,
        )

    def forward(self, x):
        z = self.norm(x)
        z = self.mlp1(z)
        x = x + z
        z = self.norm(x)
        z = self.mlp2(z)
        x = x + z
        return x


class MultiLinearAdapter(nn.Module):
    def __init__(self, num_estimator: int, in_features: int, embed_dim: int, p=0.5):
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, embed_dim, bias=False),
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(p),
                )
                for _ in range(num_estimator)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        for adapter in self.adapters:
            nn.init.kaiming_normal_(adapter[0].weight)

    def forward(self, x):
        # Shape of input: (batch_size, in_features)
        o = torch.stack(
            [adapter(x.flatten(start_dim=1)) for adapter in self.adapters], dim=1
        )
        # Shape of output: (batch_size, num_estimator, embed_dim)

        return o


class TabMONet(nn.Module):
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        num_estimator: int,
        num_classes: int,
        embed_dim: int,
        expansion_factor: int,
        num_blocks: int,
        numerical_encoder: Optional[nn.Module],
        categorical_encoder: Optional[nn.Module],
    ):
        super().__init__()
        modules = []
        for _ in range(num_blocks):
            modules.append(PolyBlock(embed_dim, expansion_factor, mlp_layer=PolyMLP))

        adapters = []
        for _ in range(num_estimator):
            adapters.append(
                nn.Sequential(
                    nn.Linear(num_features * feature_dim, embed_dim, bias=False),
                )
            )
        self.adapters = nn.ModuleList(adapters)

        self.numerical_encoder = numerical_encoder
        self.categorical_encoder = categorical_encoder
        self.poly_blocks = nn.Sequential(*modules)
        self.head = nn.Linear(embed_dim * num_estimator, num_classes)

    def forward(self, x_num, x_cat):
        if x_num is None and x_cat is not None:
            e = self.categorical_encoder(x_cat)

        elif x_num is not None and x_cat is None:
            e = self.numerical_encoder(x_num)

        else:
            e = torch.cat(
                [
                    self.numerical_encoder(x_num),
                    self.categorical_encoder(x_cat),
                ],
                dim=1,
            )

        # (batch_size, num_features, feature_dim) -> (batch_size, num_estimator, embed_dim)
        k = torch.stack([adapter(e.flatten(1, -1)) for adapter in self.adapters], dim=1)
        h = self.poly_blocks(k)  # Shared Polynomial Weights
        o = self.head(h.flatten(1, -1))  # Flatten at the back for final prediction

        return o


def build_model(
    preprocessor: DataPreprocessor,
    cat_emb_config: dict,
    num_emb_config: dict,
    model_config: dict,
    bin_edges: Optional[torch.Tensor],
    compile: bool = False,
):
    num_embedding = None
    cat_embedding = None

    # Setup categorical and numerical embedding
    if preprocessor.cat_columns:
        cat_embedding = CategoricalEmbedding(
            num_features=preprocessor.num_cat_features,
            max_class=preprocessor.max_cat,
            embedding_size=cat_emb_config["emb_size"],
        )

    if preprocessor.cont_columns:
        # Only instatiate the numerical embedding if there are numerical columns
        # Retrieve the model and relevant components
        emb_model_name = num_emb_config.pop("name").strip()

        if emb_model_name == "Q" or emb_model_name == "Q-NF" or emb_model_name == "QV2":
            num_embedding = EMB_REGISTRY.get(emb_model_name)(
                **num_emb_config, bin_edges=bin_edges
            )

        else:
            num_embedding = EMB_REGISTRY.get(emb_model_name)(**num_emb_config)

    num_features = preprocessor.num_cat_features + preprocessor.num_cont_features
    feature_dim = num_emb_config["num_bins"]

    model = TabMONet(
        **model_config,
        num_features=num_features,
        feature_dim=feature_dim,
        categorical_encoder=cat_embedding,
        numerical_encoder=num_embedding,
    )

    if compile:
        model.compile(mode="reduce-overhead")

    return model
