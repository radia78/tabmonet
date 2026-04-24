import logging
import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional
from functools import partial

from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel
from autogluon.common.space import Int
from autogluon.common.utils.pandas_utils import get_approximate_df_mem_usage

from tabarena.utils.config_utils import ConfigGenerator

from tabmonet.models.base import TabMONetV1, TabMONetV2, RealTabMONet
from tabmonet.data.preprocess import DataPreprocessor, RobustScaleSmoothClipTransform
from tabmonet.data.dataset import TabularDataset
from tabmonet.optimizers import configure_optimizer
from tabmonet.schedulers.wd_scheduler import FlatCosineLR
from tabmonet.schedulers.lr_scheduler import CosineLogLR
from tabmonet.layers.embedding import (
    LinearEmbedding,
    PeriodicEmbedding,
    PBLDEmbedding,
    QuantileEmbedding,
    CategoricalEmbedding,
)
from tabmonet.layers.layer import NewtonRaphsonLayerNorm

from omegaconf import OmegaConf
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class TabMONetModel(AbstractTorchModel):
    """
    AutoGluon model wrapper for TabMONet models.
    Supports TabMONetV1, TabMONetV2, and RealTabMONet.
    """

    ag_key = "TABMONET"
    ag_name = "TABMONET"
    ag_priority = 75
    seed_name = "random_state"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = None
        self.model = None
        self.bin_edges = None
        self._imputer = None
        self._features_to_impute = None

    def _set_default_params(self):
        default_params = {
            "model_type": "v1",  # Options: 'v1', 'v2', 'real'
            "n_estimator": 4,
            "n_blocks": 3,
            "expansion_factor": 1,
            "emb_dim": 256,
            "feature_dim": 32,
            "numerical_encoder_type": "quantile",  # 'linear', 'periodic', 'pbld', 'quantile'
            "layer_norm_type": "full",
            "device": "cpu",
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_layer_norm(self, layer_norm_type: str):
        if layer_norm_type == "full":
            return partial(nn.LayerNorm, eps=1e-6)
        elif layer_norm_type == "newton_raphson":
            return NewtonRaphsonLayerNorm

    def _get_numerical_encoder(
        self,
        n_features: int,
        feature_dim: int,
        encoder_type: str,
        bin_edges: Optional[torch.Tensor] = None,
    ):
        if encoder_type == "linear":
            return LinearEmbedding(n_features, feature_dim, bias=True)
        elif encoder_type == "periodic":
            return PeriodicEmbedding(n_features, feature_dim, sigma=0.1, bias=True)
        elif encoder_type == "pbld":
            return PBLDEmbedding(n_features, feature_dim)
        elif encoder_type == "quantile":
            if bin_edges is None:
                raise ValueError("Quantile encoder requires bin_edges")
            return QuantileEmbedding(n_features, feature_dim, bin_edges)
        else:
            raise ValueError(f"Unknown numerical_encoder_type: {encoder_type}")

    def _preprocess(
        self, X: pd.DataFrame, is_train: bool = False, **kwargs
    ) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        X = X.copy(deep=True)

        # Identify features for imputation
        if is_train:
            self._features_to_impute = self._feature_metadata.get_features(
                valid_raw_types=["int", "float"], invalid_special_types=["category"]
            )
            if self._features_to_impute:
                self._imputer = SimpleImputer(strategy="mean")
                self._imputer.fit(X[self._features_to_impute])

        if self._imputer is not None and self._features_to_impute:
            X[self._features_to_impute] = self._imputer.transform(
                X[self._features_to_impute]
            )

        return X

    # Prepare DataLoaders
    @staticmethod
    def create_loader(cat, cont, y, batch_size, problem_type, shuffle=False):
        dataset = TabularDataset(cat, cont, problem_type, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ):
        start_time = time.time()
        params = self._get_model_params()
        self.device = params["device"]

        # Preprocess using internal AG _preprocess (handles imputation we added)
        X = self.preprocess(X, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val, is_train=False)

        # Identify features for DataPreprocessor
        cont_columns = self._feature_metadata.get_features(
            valid_raw_types=["int", "float"], invalid_special_types=["category"]
        )
        cat_columns = self._feature_metadata.get_features(valid_raw_types=["category"])

        # Initialize DataPreprocessor
        num_bins = (
            params["feature_dim"]
            if params["numerical_encoder_type"] == "quantile"
            else None
        )
        self.preprocessor = DataPreprocessor(
            problem_type=self.problem_type,
            numerical_preprocessor=RobustScaleSmoothClipTransform(),
            num_bins=num_bins,
            cont_columns=cont_columns,
            cat_columns=cat_columns,
        )

        # Preprocess training data for the model
        cat_train, cont_train, y_train, bin_edges = self.preprocessor.preprocess(
            X, y, is_train=True
        )
        if bin_edges is not None:
            self.bin_edges = torch.from_numpy(bin_edges).float()

        # Preprocess validation data
        if X_val is not None and y_val is not None:
            cat_val, cont_val, y_val, _ = self.preprocessor.preprocess(
                X_val, y_val, is_train=False
            )
        else:
            cat_val, cont_val, y_val = None, None, None

        # Model Initialization
        n_features = len(cont_columns) + len(cat_columns)
        n_class = self.num_classes if self.problem_type != "regression" else 1

        # Setup encoders
        num_encoder = None
        if cont_columns:
            num_encoder = self._get_numerical_encoder(
                len(cont_columns),
                params["feature_dim"],
                params["numerical_encoder_type"],
                self.bin_edges,
            )

        cat_encoder = None
        if cat_columns:
            cat_encoder = CategoricalEmbedding(
                cardinalities=self.preprocessor.max_categories,
                emb_dim=params["feature_dim"],
            )

        layer_norm = self._get_layer_norm(params["layer_norm_type"])

        model_type = params["model_type"]
        model_kwargs = {
            "n_features": n_features,
            "feature_dim": params["feature_dim"],
            "emb_dim": params["emb_dim"],
            "n_blocks": params["n_blocks"],
            "expansion_factor": params["expansion_factor"],
            "problem_type": self.problem_type,
            "n_class": n_class,
            "numerical_encoder": num_encoder,
            "categorical_encoder": cat_encoder,
            "layer_norm": layer_norm,
        }

        if model_type == "v1":
            model_kwargs["n_estimator"] = params["n_estimator"]
            self.model = TabMONetV1(**model_kwargs)
        elif model_type == "v2":
            self.model = TabMONetV2(**model_kwargs)
        elif model_type == "real":
            self.model = RealTabMONet(**model_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.to(self.device)
        self.model.compile(mode="reduce-overhead")

        # Training Setup
        training_kwargs = OmegaConf.create(
            {
                "batch_size": 256,
                "epochs": 256,
                "optimizer": {
                    "muon": {
                        "_target_": "torch.optim.Muon",
                        "lr": 1e-3,
                        "weight_decay": 0.1,
                    },
                    "adamw": {
                        "_target_": "torch.optim.AdamW",
                        "lr": 1e-3,
                        "weight_decay": 0.01,
                        "betas": [0.9, 0.95],
                        "fused": True,
                    },
                },
            }
        )

        optimizers = configure_optimizer(training_kwargs, self.model)
        wd_schedulers = [
            FlatCosineLR(opt, T_max=training_kwargs["epochs"]) for opt in optimizers
        ]
        lr_schedulers = [
            CosineLogLR(opt, k=4, T_max=training_kwargs["epochs"]) for opt in optimizers
        ]

        train_loader = self.create_loader(
            cat_train,
            cont_train,
            y_train,
            training_kwargs["batch_size"],
            problem_type=self.problem_type,
            shuffle=True,
        )
        val_loader = (
            self.create_loader(
                cat_val,
                cont_val,
                y_val,
                training_kwargs["batch_size"],
                problem_type=self.problem_type,
            )
            if cat_val is not None or cont_val is not None
            else None
        )

        # Training Loop
        best_val_loss = float("inf")
        best_weights = None

        for epoch in range(training_kwargs["epochs"]):
            # Time limit check
            if time_limit is not None:
                if time.time() - start_time > time_limit:
                    logger.warning(
                        f"\tStopping training early due to time limit ({time_limit}s reached)"
                    )
                    break

            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_cat, batch_cont = None, None

                # Attach to device based on if it is not empty
                if not batch_X[1].isnan().any():
                    batch_cat = batch_X[1].to(self.device)

                if not batch_X[0].isnan().any():
                    batch_cont = batch_X[0].to(self.device)

                batch_y = batch_y.to(self.device)

                for opt in optimizers:
                    opt.zero_grad()

                _, loss = self.model(batch_cont, batch_cat, batch_y)
                loss.backward()

                for opt in optimizers:
                    opt.step()

                train_loss += loss.item()

            for sch in lr_schedulers:
                sch.step()

            for sch in wd_schedulers:
                sch.step()

            if val_loader:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in train_loader:
                        batch_cat, batch_cont = None, None

                        # Attach to device based on if it is not empty
                        if not batch_X[1].isnan().any():
                            batch_cat = batch_X[1].to(self.device)

                        if not batch_X[0].isnan().any():
                            batch_cont = batch_X[0].to(self.device)

                        batch_y = batch_y.to(self.device)
                        _, loss = self.model(batch_cont, batch_cat, batch_y)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.model.state_dict()

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{training_kwargs['epochs']}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {best_val_loss:.4f}"
                )

        if best_weights:
            self.model.load_state_dict(best_weights)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        # Preprocess using internal AG _preprocess (handles imputation)
        X = self.preprocess(X, is_train=False)

        self.model.eval()
        cat_features, cont_features, _, _ = self.preprocessor.preprocess(
            X, is_train=False
        )

        cont_tensor = (
            torch.from_numpy(cont_features).float().to(self.device)
            if cont_features is not None
            else None
        )
        cat_tensor = (
            torch.from_numpy(cat_features).long().to(self.device)
            if cat_features is not None
            else None
        )

        with torch.no_grad():
            outputs = self.model.predict_proba(cont_tensor, cat_tensor)

        outputs_np = outputs.detach().cpu().numpy()

        # Inverse transform target for regression
        if self.problem_type == "regression":
            outputs_np = self.preprocessor.inverse_transform_target(outputs_np)

        if self.problem_type in ("regression", "binary"):
            return outputs_np.flatten()

        else:
            return outputs_np

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        hyperparameters = self._get_model_params()
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=hyperparameters,
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls, *, X: pd.DataFrame, hyperparameters: dict = None, **kwargs
    ) -> int:
        if hyperparameters is None:
            hyperparameters = {}

        emb_dim = hyperparameters.get("emb_dim", 256)
        n_blocks = hyperparameters.get("n_blocks", 3)
        expansion_factor = hyperparameters.get("expansion_factor", 1)
        batch_size = hyperparameters.get("batch_size", 256)

        # Model parameter estimate (rough)
        # Each block has multiple linear/poly layers. Rough upper bound:
        # Params ~ n_blocks * (emb_dim * emb_dim * expansion_factor)
        # 4 bytes per float32
        model_params_mem = n_blocks * 5 * (emb_dim * emb_dim * expansion_factor) * 4

        # Activation memory estimate (rough peak during backprop)
        # Activations ~ batch_size * n_blocks * emb_dim * expansion_factor
        activations_mem = batch_size * n_blocks * 10 * (emb_dim * expansion_factor) * 4

        # Dataset memory
        dataset_size_mem_est = 5 * get_approximate_df_mem_usage(X).sum()

        # Baseline overhead (PyTorch, CUDA, etc.)
        baseline_overhead = 5e8  # 500MB

        mem_estimate = (
            model_params_mem
            + activations_mem
            + dataset_size_mem_est
            + baseline_overhead
        )

        return int(mem_estimate)

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}

    @classmethod
    def supported_problem_types(cls) -> List[str]:
        return ["binary", "multiclass", "regression"]

    def _get_default_stopping_metric(self):
        return self.eval_metric

    def get_device(self) -> str:
        return self.device

    def _set_device(self, device: str):
        self.model.to(device)


def get_configs_tabmonetv1(*, num_random_configs: int = 1):
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    manual_configs = [
        {
            "model_type": "v1",  # Options: 'v1', 'v2', 'real'
            "numerical_encoder_type": "quantile",  # 'linear', 'periodic', 'pbld', 'quantile'
            "layer_norm_type": "full",
            "device": device,
        },
    ]
    search_space = {
        "n_estimator": Int(1, 4),
        "n_blocks": Int(1, 3),
        "expansion_factor": Int(1, 3),
        "emb_dim": Int(64, 256),
        "feature_dim": Int(16, 64),
    }

    gen_custom_tabmonetv1 = ConfigGenerator(
        model_cls=TabMONetModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen_custom_tabmonetv1.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )


def get_configs_tabmonetv1_fhe(*, num_random_configs: int = 1):
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    manual_configs = [
        {
            "model_type": "v1",  # Options: 'v1', 'v2', 'real'
            "numerical_encoder_type": "quantile",  # 'linear', 'periodic', 'pbld', 'quantile'
            "layer_norm_type": "newton_raphson",
            "emb_dim": 256,
            "feature_dim": 16,
            "expansion_factor": 1,
            "n_blocks": 3,
            "device": device,
        },
    ]
    search_space = {
        "n_estimator": Int(1, 4),
    }

    gen_custom_tabmonetv1 = ConfigGenerator(
        model_cls=TabMONetModel,
        manual_configs=manual_configs,
        search_space=search_space,
    )
    return gen_custom_tabmonetv1.generate_all_bag_experiments(
        num_random_configs=num_random_configs, fold_fitting_strategy="sequential_local"
    )
