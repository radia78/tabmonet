from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from tabmonet.layers.layer import (
    PolyBlock,
    EnsembleAdapter,
    ConvolutionEnsemble,
    PolyMLP,
)
from tabmonet.layers.embedding import Embedding


class TabMONetBase(nn.Module):
    def __init__(
        self,
        problem_type: str,
        n_class: Optional[int] = None,
        numerical_encoder: Optional[Embedding] = None,
        categorical_encoder: Optional[Embedding] = None,
    ):
        super().__init__()
        self.problem_type = problem_type
        match problem_type:
            case "regression":
                self.criterion = nn.MSELoss()
            case "binary":
                self.criterion = nn.BCEWithLogitsLoss()
            case "multiclass":
                self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                assert n_class is not None, (
                    "multiclass requires n_class to be an `int` > 1"
                )

        self.numerical_encoder = numerical_encoder
        self.categorical_encoder = categorical_encoder

    def encode(self, x_num, x_cat):
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

        return e

    def predict(self, x_num, x_cat, threshold=0.5):
        match self.problem_type:
            case "binary":
                pred = (self.predict_proba(x_num, x_cat) > threshold).float()
            case "multiclass":
                pred = self.predict_proba(x_num, x_cat).argmax(dim=-1)
            case "regression":
                pred = self.predict_proba(x_num, x_cat)

        return pred

    def predict_proba(self, x_num, x_cat):
        out, _ = self(x_num, x_cat)
        match self.problem_type:
            case "binary":
                prob = F.sigmoid(out)
            case "multiclass":
                prob = F.softmax(out, dim=-1)
            case "regression":
                prob = out

        return prob


class TabMONetV1(TabMONetBase):
    def __init__(
        self,
        n_estimator: int,
        n_blocks: int,
        expansion_factor: int,
        n_features: int,
        feature_dim: int,
        emb_dim: int,
        problem_type: str,
        n_class: Optional[int] = None,
        numerical_encoder: Optional[Embedding] = None,
        categorical_encoder: Optional[Embedding] = None,
        layer_norm: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            problem_type=problem_type,
            numerical_encoder=numerical_encoder,
            categorical_encoder=categorical_encoder,
            n_class=n_class,
        )
        self.neck = EnsembleAdapter(
            n_estimator=n_estimator,
            in_features=n_features * feature_dim,
            out_features=emb_dim,
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                PolyBlock(
                    emb_dim, expansion_factor, spatial_mix=False, layer_norm=layer_norm
                )
            )
        self.blocks = nn.Sequential(*blocks)
        if problem_type in ("regression", "binary"):
            self.head = PolyMLP(
                n_estimator * emb_dim, emb_dim, 1, layer_norm=layer_norm
            )
        else:
            self.head = PolyMLP(
                n_estimator * emb_dim, emb_dim, n_class, layer_norm=layer_norm
            )

    def forward(self, x_num, x_cat, y=None):
        e = self.encode(x_num, x_cat)
        e = self.neck(e)
        e = self.blocks(e)
        o = self.head(e.flatten(1, -1))

        loss = self.criterion(o, y) if y is not None else None

        return o, loss


class TabMONetV2(TabMONetBase):
    def __init__(
        self,
        n_blocks: int,
        expansion_factor: int,
        n_features: int,
        feature_dim: int,
        emb_dim: int,
        n_class: int,
        problem_type: str,
        numerical_encoder: Optional[Embedding] = None,
        categorical_encoder: Optional[Embedding] = None,
        layer_norm: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            problem_type=problem_type,
            numerical_encoder=numerical_encoder,
            categorical_encoder=categorical_encoder,
            n_class=n_class,
        )
        self.neck = ConvolutionEnsemble(n_features, feature_dim, emb_dim)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                PolyBlock(
                    emb_dim, expansion_factor, spatial_mix=True, layer_norm=layer_norm
                )
            )
        self.blocks = nn.Sequential(*blocks)
        if problem_type in ("regression", "binary"):
            self.head = PolyMLP(n_features * emb_dim, emb_dim, 1, layer_norm=layer_norm)
        else:
            self.head = PolyMLP(
                n_features * emb_dim, emb_dim, n_class, layer_norm=layer_norm
            )

    def forward(self, x_num, x_cat, y=None):
        e = self.encode(x_num, x_cat)
        e = self.neck(e)
        e = self.blocks(e)
        o = self.head(e.flatten(1, -1))

        loss = self.criterion(o, y) if y is not None else None

        return o, loss


class RealTabMONet(TabMONetBase):
    def __init__(
        self,
        n_blocks: int,
        expansion_factor: int,
        n_features: int,
        feature_dim: int,
        emb_dim: int,
        n_class: int,
        problem_type: int,
        numerical_encoder: Optional[Embedding] = None,
        categorical_encoder: Optional[Embedding] = None,
        layer_norm: nn.Module = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            problem_type=problem_type,
            numerical_encoder=numerical_encoder,
            categorical_encoder=categorical_encoder,
            n_class=n_class,
        )
        self.soft_selection = nn.Parameter(torch.ones(n_features, 1))
        self.neck = nn.Linear(feature_dim, emb_dim, bias=False)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                PolyBlock(
                    emb_dim, expansion_factor, spatial_mix=True, layer_norm=layer_norm
                )
            )
        self.blocks = nn.Sequential(*blocks)
        if problem_type in ("regression", "binary"):
            self.head = PolyMLP(n_features * emb_dim, emb_dim, 1, layer_norm=layer_norm)
        else:
            self.head = PolyMLP(
                n_features * emb_dim, emb_dim, n_class, layer_norm=layer_norm
            )

    def forward(self, x_num, x_cat, y=None):
        e = self.encode(x_num, x_cat) * self.soft_selection
        e = self.neck(e)
        e = self.blocks(e)
        o = self.head(e.flatten(1, -1))

        loss = self.criterion(o, y) if y is not None else None

        return o, loss
