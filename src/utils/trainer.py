from tqdm import tqdm
import os
import torch
import numpy as np
import scipy as sp
import pickle
import time

import matplotlib.pyplot as plt

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import root_mean_squared_error, roc_auc_score, log_loss
from typing import Optional, List

from utils.data import DataPreprocessor
from utils.optimizer import WDScheduler


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: List[Optimizer],
        lr_scheduler: Optional[List[LRScheduler]],
        wd_scheduler: Optional[List[WDScheduler]],
        criterion: torch.nn.Module,
        preprocessor: DataPreprocessor,
        problem_type: str,
        run_config: dict,
        log_dir: str,
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.wd_scheduler = wd_scheduler
        self.criterion = criterion
        self.preprocessor = preprocessor
        self.problem_type = problem_type
        self.device = device

        self.run_config = run_config
        self.best_model_weights = None
        self.best_val_score = 0
        self.best_epoch = 0

        # Initialize TensorBoard Writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_step(self, X, y):
        cat_features, cont_features = None, None
        targets = y.to(self.device)

        # Attach to device based on if it is not empty
        if not X[1].isnan().any():
            cat_features = X[1].to(self.device)

        if not X[0].isnan().any():
            cont_features = X[0].to(self.device)

        targets = targets.to(self.device)

        for opt in self.optimizer:
            opt.zero_grad()

        outputs = self.model(cont_features, cat_features)
        loss = self.criterion(outputs, targets)
        loss.backward()

        for opt in self.optimizer:
            opt.step()

        # Log the loss and metircs
        self.train_running_loss += loss.item()

        train_metrics = self.prepare_metrics_inputs(
            targets=targets.detach(), outputs=outputs.detach()
        )
        self.train_running_results["labels"].append(train_metrics[0])
        self.train_running_results["preds"].append(train_metrics[1])

    def train_epoch(self, data: DataLoader):
        self.model.train()

        self.train_running_loss = 0.0
        self.train_running_results = {"labels": [], "preds": []}

        for X, y in data:
            self.train_step(X=X, y=y)

        if self.lr_scheduler is not None:
            for sch in self.lr_scheduler:
                sch.step()

        if self.wd_scheduler is not None:
            for wd in self.lr_scheduler:
                wd.step()

        avg_loss = self.train_running_loss / len(data)
        targets = np.concatenate(self.train_running_results["labels"], axis=0)
        if self.problem_type == "multiclass":
            outputs = sp.special.softmax(
                np.concatenate(self.train_running_results["preds"], axis=0), axis=-1
            )
        else:
            outputs = np.concatenate(self.train_running_results["preds"], axis=0)

        # Stack and compute the metric
        avg_metric = self.log_metrics(targets=targets, outputs=outputs)

        # Log training loss to TensorBoard
        return avg_loss, avg_metric

    def val_step(self, X, y):
        cat_features, cont_features = None, None
        targets = y.to(self.device)

        # Attach to device based on if it is not empty
        if not X[1].isnan().any():
            cat_features = X[1].to(self.device)

        if not X[0].isnan().any():
            cont_features = X[0].to(self.device)

        outputs = self.model(cont_features, cat_features)
        loss = self.criterion(outputs, targets)

        self.val_running_loss += loss.item()

        # Add the targets and outputs for metric logging
        val_metrics = self.prepare_metrics_inputs(
            targets=targets.detach(), outputs=outputs.detach()
        )
        self.val_running_results["labels"].append(val_metrics[0])
        self.val_running_results["preds"].append(val_metrics[1])

    def val_epoch(self, data: DataLoader, epoch: Optional[int] = None):
        self.model.eval()

        self.val_running_loss = 0.0
        self.val_running_results = {"labels": [], "preds": []}

        with torch.no_grad():
            for X, y in data:
                self.val_step(X=X, y=y)

        avg_loss = self.val_running_loss / len(data)
        # Stack and compute the metric
        targets = np.concatenate(self.val_running_results["labels"], axis=0)
        if self.problem_type == "multiclass":
            outputs = sp.special.softmax(
                np.concatenate(self.val_running_results["preds"], axis=0), axis=-1
            )
        else:
            outputs = np.concatenate(self.val_running_results["preds"], axis=0)

        # Stack and compute the metric
        avg_metric = self.log_metrics(targets=targets, outputs=outputs)

        if epoch and self.problem_type == "multiclass":
            fig = self.plot_learned_separation(outputs=outputs, targets=targets)
            self.writer.add_figure(f"Plot", fig, epoch)

        # Log training loss to TensorBoard
        return avg_loss, avg_metric

    def run(
        self,
        num_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ):
        self.model.to(self.device)
        prog_bar = tqdm(range(num_epochs))
        for epoch in prog_bar:
            train_loss, train_metric = self.train_epoch(train_loader)
            if (epoch + 1) % 4 == 0:
                val_loss, val_metric = self.val_epoch(val_loader, epoch)

            else:
                val_loss, val_metric = self.val_epoch(val_loader)

            # Needs to be changed
            prog_bar.set_description(
                desc=f"Epoch [{epoch + 1}/{num_epochs}] (Score/Loss): Train {train_metric:.3f}/{train_loss:.4f} | Val {val_metric:.3f}/{val_loss:.4f}"
            )

            self.writer.add_scalars(
                "Loss", {"Train": train_loss, "Val": val_loss}, epoch
            )
            self.writer.add_scalars(
                "Score", {"Train": train_metric, "Val": val_metric}, epoch
            )

            if epoch == 0:
                self.best_model_weights = self.model.state_dict()
                self.best_val_score = val_metric
                self.best_epoch = epoch + 1

            else:
                caching_best = (
                    val_metric > self.best_val_score
                    if self.problem_type == "binary"
                    else val_metric < self.best_val_score
                )
                if caching_best:
                    self.best_model_weights = self.model.state_dict()
                    self.best_val_score = val_metric
                    self.best_epoch = epoch + 1

        print(f"Best model is at epoch {self.best_epoch}")
        self.model.load_state_dict(self.best_model_weights, assign=True)

        # Benchmark the model at test data
        t0 = time.time()
        _, test_metric = self.val_epoch(test_loader)
        t1 = time.time()
        inference_time = t1 - t0
        self.writer.add_hparams(self.run_config, {"Test Score": test_metric, "Inference Time": inference_time})

        # Close the writer when finished
        self.writer.flush()
        self.writer.close()

    def prepare_metrics_inputs(self, targets, outputs):
        if self.problem_type == "regression":
            # If we are performing regression, we have to inverse transform for a fair assesment
            cleaned_targets = self.preprocessor.inverse_transform_target(
                targets.cpu().numpy().reshape(-1, 1)
            )
            cleaned_outputs = self.preprocessor.inverse_transform_target(
                outputs.cpu().numpy().reshape(-1, 1)
            )

            return cleaned_targets, cleaned_outputs

        elif self.problem_type == "binary":
            # Inverse transform is not necessary for prediction since things will be encoded as 0-1 (bin-cls)
            cleaned_outputs = outputs.sigmoid().cpu().numpy()

            return targets.cpu().numpy(), cleaned_outputs

        else:
            cleaned_outputs = outputs.cpu().numpy()

            return targets.cpu().numpy(), cleaned_outputs

    def plot_learned_separation(self, outputs: torch.Tensor, targets: torch.Tensor):
        # Create a 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the 3D scatter
        sc = ax.scatter(
            outputs[:, 0], outputs[:, 1], outputs[:, 2], c=targets, cmap="viridis", s=50
        )

        # Set labels
        ax.set_xlabel("Logits of Class 1")
        ax.set_ylabel("Logits of Class 2")
        ax.set_zlabel("Logits of Class 3")

        # Set title
        ax.set_title("Logits Plot of 3-Class Classification")
        ax.view_init(elev=30, azim=30)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Class Label")

        return fig

    def log_metrics(self, targets, outputs):
        if self.problem_type == "regression":
            avg_score = root_mean_squared_error(targets, outputs)
        elif self.problem_type == "binary":
            avg_score = roc_auc_score(targets, outputs)
        else:
            avg_score = log_loss(targets, outputs)

        return avg_score

    def save_model(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)

        # Load the object from the file
        with open(f"{model_path}/preprocessor.pkl", "wb") as file:
            pickle.dump(self.preprocessor, file)

        torch.save(
            self.best_model_weights,
            f"{model_path}/tabmonet.pt",
        )
