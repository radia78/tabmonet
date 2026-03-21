import torch
import torch.nn as nn
from typing import Optional, Union

from utils.data import DataPreprocessor
from .layers.mlp import PolyMLP, MultiHeadPolyMlp
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

LAYER_REGISTRY = {"poly": PolyMLP, "mpoly": MultiHeadPolyMlp}


class PolyBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        expansion_factor: int = 3,
        mlp_layer: Union[PolyMLP, MultiHeadPolyMlp] = PolyMLP,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.norm = norm_layer(self.embed_dim)
        self.mlp1 = mlp_layer(
            embed_dim, embed_dim, embed_dim, use_spatial=False, bias=False
        )
        self.mlp2 = mlp_layer(
            in_features=embed_dim,
            hidden_features=embed_dim * expansion_factor,
            out_features=embed_dim,
            use_spatial=False,
            bias=False,
        )

    def forward(self, x):
        x_skip = x
        z = self.norm(x)
        z = self.mlp1(z)
        x = x + z
        z = self.norm(x)
        z = self.mlp2(z)
        x = x + z
        return x + x_skip


class PolyHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        embed_dim: int,
        expansion_factor: int = 3,
        mlp_layer: Union[PolyMLP, MultiHeadPolyMlp] = PolyMLP,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.norm1 = norm_layer(in_features)
        self.norm2 = norm_layer(embed_dim)
        self.mlp1 = mlp_layer(in_features, embed_dim, embed_dim, use_spatial=False)
        self.mlp2 = mlp_layer(
            in_features=embed_dim,
            hidden_features=embed_dim * expansion_factor,
            out_features=out_features,
            use_spatial=False,
        )

    def forward(self, x):
        x = self.norm1(x)
        x = self.mlp1(x)

        x = self.norm2(x)
        x = self.mlp2(x)

        return x


class MultiLinearAdapter(nn.Module):
    def __init__(self, num_estimator: int, in_features: int, embed_dim: int):
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features, embed_dim, bias=False)
                )
                for _ in range(num_estimator)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

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

        return self.norm(o)


class PyramidAdapter(nn.Module):
    def __init__(self, n_features: int, feature_dim: int, embed_dim: int):
        super().__init__()
        self.feature_subset = nn.Conv1d(
            in_channels=feature_dim, out_channels=embed_dim, kernel_size=3, bias=False
        )
        self.full_feature = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        # Initial shape is (batch size, feature_dim, n_features)
        subset = self.feature_subset(
            x
        )  # Shape is (batch_size, feature_dim, n_ensembles)
        full = self.full_feature(x)  # Shape is (batch_size, feature_dim, 1)
        return torch.cat([subset, full], dim=-1)


class TabMONet(nn.Module):
    def __init__(
        self,
        n_estimator: int,
        num_features: int,
        feature_dim: int,
        n_classes: int,
        embed_dim: int,
        expansion_factor: int,
        num_blocks: int,
        numerical_encoder: Optional[nn.Module],
        categorical_encoder: Optional[nn.Module],
    ):
        super().__init__()
        modules = []
        for _ in range(num_blocks):
            modules.append(
                PolyBlock(
                    embed_dim,
                    expansion_factor,
                    mlp_layer=PolyMLP,
                )
            )

        self.numerical_encoder = numerical_encoder
        self.categorical_encoder = categorical_encoder
        self.adapter = MultiLinearAdapter(
            n_estimator, num_features * feature_dim, embed_dim,
        )
        self.poly_blocks = nn.Sequential(*modules)
        self.head = PolyHead(
            in_features=embed_dim,
            out_features=n_classes,
            embed_dim=embed_dim
        )

    def forward(self, x_num, x_cat):
        if x_num is None and x_cat is not None:
            e = self.categorical_encoder(x_cat)

        elif x_num is not None and x_cat is None:
            e = self.numerical_encoder(x_num)

        elif x_num is not None and x_cat is not None:
            e = torch.cat(
                [
                    self.numerical_encoder(x_num),
                    self.categorical_encoder(x_cat),
                ],
                dim=1,
            )

        # (batch_size, num_features, feature_dim) -> (batch_size, num_estimator, embed_dim)
        # k = torch.stack([adapter(e.flatten(1, -1)) for adapter in self.adapters], dim=1)
        k = self.adapter(e.flatten(1, -1))
        h = self.poly_blocks(k)  # Shared Polynomial Weights
        o = self.head(h.mean(dim=1))

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
            max_class=preprocessor.max_categories,
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
