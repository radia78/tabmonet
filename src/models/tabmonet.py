from typing import List, Union
import torch
import torch.nn as nn
from typing import Optional, Union

from utils.data import DataPreprocessor
from .layers.mlp import PolyMLP, FeatureFilter
from .layers.embeddings import *

from functools import partial

EMB_REGISTRY = {
    "PBLD": PBLDEmbedding,
    "P": PeriodicEmbedding,
    "L": LinearEmbedding,
    "Q": QuantileEmbedding,
    "NA": nn.Identity,
}


class PolyBlock(nn.Module):
    def __init__(
        self,
        in_features: Union[int, List[int]],
        out_features: Union[int, List[int]],
        embed_dim: Union[int, List[int]],
        expansion_factor: int = 3,
        mlp_layer=PolyMLP,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        spatial_mix: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.norm = norm_layer(embed_dim)

        if isinstance(in_features, int):
            in_features = 2 * [in_features]
        if isinstance(embed_dim, int):
            embed_dim = 2 * [embed_dim]
        if isinstance(out_features, int):
            out_features = 2 * [out_features]

        self.mlp1 = mlp_layer(
            in_features=in_features[0],
            hidden_features=embed_dim[0],
            out_features=out_features[0],
            use_spatial=spatial_mix,
            bias=False,
        )
        self.mlp2 = mlp_layer(
            in_features=in_features[1],
            hidden_features=embed_dim[1] * expansion_factor,
            out_features=out_features[1],
            use_spatial=spatial_mix,
            bias=False,
        )

    def forward(self, x):
        x = self.mlp1(x)
        z = self.norm(x)
        z = self.mlp2(z)
        x = x + z
        return x


class PolyAttn(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.hidden_features = hidden_features
        self.w_qkv = nn.Linear(in_features, 3 * hidden_features, bias=False)
        self.w_o = nn.Linear(hidden_features, in_features, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # x shape: (batch_size, features, embed_dim)
        q, k, v = torch.split(self.w_qkv(x), 3 * [self.hidden_features], dim=-1)
        # Attention mechanism
        a = (self.alpha * (q @ k.transpose(1, -1)) + 1) ** 4
        a = a.div(torch.abs(a))
        o = self.w_o(a @ v)

        return o


class MultiLinearAdapter(nn.Module):
    def __init__(self, num_estimator: int, in_features: int, embed_dim: int):
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(in_features, embed_dim, bias=False))
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

        return o


class PyramidAdapter(nn.Module):
    def __init__(self, n_features: int, feature_dim: int, embed_dim: int):
        super().__init__()
        self.feature_subset = nn.Conv1d(
            in_channels=feature_dim, out_channels=embed_dim, kernel_size=3, bias=False
        )
        self.full_feature = nn.Linear(n_features, 1, bias=False)
        self.padding = nn.CircularPad1d((0, 2))

    def forward(self, x):
        # Initial shape is (batch size, feature_dim, n_features)
        subset = self.feature_subset(
            self.padding(x)
        )  # Shape is (batch_size, feature_dim, n_features)
        full = self.full_feature(x)  # Shape is (batch_size, feature_dim, 1)
        return torch.cat([subset, full], dim=-1)


class TabMONet(nn.Module):
    def __init__(
        self,
        num_features: int,
        n_classes: int,
        embed_dim: int,
        expansion_factor: int,
        num_blocks: int,
        numerical_encoder: Optional[nn.Module],
        categorical_encoder: Optional[nn.Module],
    ):
        super().__init__()
        modules = []
        for i in range(num_blocks):
            if i == 0:
                in_features =  [4, embed_dim]
            else:
                in_features = embed_dim
            modules.append(
                PolyBlock(
                    in_features=in_features,
                    expansion_factor=expansion_factor,
                    embed_dim=embed_dim,
                    out_features=embed_dim,
                )
            )

        self.soft_selection = nn.Parameter(torch.ones(num_features, 4))
        self.numerical_encoder = numerical_encoder
        self.categorical_encoder = categorical_encoder
        self.poly_blocks = nn.Sequential(*modules)
        self.head = PolyMLP(embed_dim * num_features, embed_dim, n_classes, bias=True)

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
        e = self.soft_selection * e
        e = self.poly_blocks(e)
        o = self.head(e.flatten(1, -1))

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
            feature_dim = num_emb_config["num_bins"]

        else:
            num_embedding = EMB_REGISTRY.get(emb_model_name)(**num_emb_config, num_features=preprocessor.num_cont_features)
            feature_dim = 4

    num_features = preprocessor.num_cat_features + preprocessor.num_cont_features

    model = TabMONet(
        **model_config,
        num_features=num_features,
        #feature_dim=feature_dim,
        categorical_encoder=cat_embedding,
        numerical_encoder=num_embedding,
    )

    if compile:
        model.compile(mode="reduce-overhead")

    return model
