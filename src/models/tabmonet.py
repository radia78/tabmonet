from typing import Optional
import torch
import torch.nn as nn

from utils.data import DataPreprocessor
from .embeddings import *
from .layers import PolyMLP, PolyBlock


EMB_REGISTRY = {
    "PBLD": PBLDEmbedding,
    "P": PeriodicEmbedding,
    "L": LinearEmbedding,
    "Q": QuantileEmbedding,
    "NA": nn.Identity,
}


class TabMONet(nn.Module):
    def __init__(
        self,
        neck: nn.Module,
        backbone: nn.Module,
        head: nn.Module,
        numerical_encoder: Optional[nn.Module],
        categorical_encoder: Optional[nn.Module],
    ):
        super().__init__()
        self.neck = neck
        self.backbone = backbone
        self.numerical_encoder = numerical_encoder
        self.categorical_encoder = categorical_encoder
        self.head = head

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
        #e = self.soft_selection * e.flatten(1, -1)
        e = self.neck(e)
        e = self.backbone(e)
        o = self.head(e)

        return o
    

class TabMONet(nn.Module):
    def __init__(
        self,
        num_features: int,
        n_classes: int,
        embed_dim: int,
        numerical_encoder: Optional[nn.Module],
        categorical_encoder: Optional[nn.Module],
    ):
        super().__init__()
        modules = []
        for _ in range(3):
            modules.append(
                PolyBlock(
                    in_features=embed_dim,
                    expansion_factor=1,
                    embed_dim=embed_dim,
                    out_features=embed_dim,
                    spatial_mix=True,
                )
            )

        self.soft_selection = nn.Parameter(torch.ones(num_features))
        self.adapter = nn.Linear(4, embed_dim, bias=False)
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
        #e = self.soft_selection * e.flatten(1, -1)
        e = self.adapter(e)
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
