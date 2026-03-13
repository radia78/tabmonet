from tqdm import tqdm
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import root_mean_squared_error, roc_auc_score
from typing import Optional, List, Union

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
        self.train_running_metric += self.log_metrics(
            train_metrics[0], train_metrics[1]
        )

    def train_epoch(self, data: DataLoader):
        self.model.train()

        self.train_running_loss = 0.0
        self.train_running_metric = 0.0

        for X, y in data:
            self.train_step(X=X, y=y)

        if self.lr_scheduler is not None:
            for sch in self.lr_scheduler:
                sch.step()

        if self.wd_scheduler is not None:
            for wd in self.lr_scheduler:
                wd.step()

        avg_loss = self.train_running_loss / len(data)
        avg_metric = self.train_running_metric / len(data)

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

        # Add the targets and outputs for metric logging
        val_metrics = self.prepare_metrics_inputs(
            targets=targets.detach(), outputs=outputs.detach()
        )
        self.val_running_metric += self.log_metrics(val_metrics[0], val_metrics[1])

        self.val_running_loss += loss.item()

    def val_epoch(self, data: DataLoader):
        self.model.eval()

        self.val_running_loss = 0.0
        self.val_running_metric = 0.0

        with torch.no_grad():
            for X, y in data:
                self.val_step(X=X, y=y)

        avg_loss = self.val_running_loss / len(data)
        avg_metric = self.val_running_metric / len(data)

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
                if val_metric < self.best_val_score:
                    self.best_model_weights = self.model.state_dict()
                    self.best_val_score = val_metric
                    self.best_epoch = epoch + 1

        print(f"Best model is at epoch {self.best_epoch}")
        self.model.load_state_dict(self.best_model_weights, assign=True)

        # Benchmark the model at test data
        _, test_metric = self.val_epoch(test_loader)
        self.writer.add_hparams(self.run_config, {"Test Score": test_metric})

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

            return targets, cleaned_outputs

        else:
            cleaned_outputs = outputs.softmax().cpu().numpy()

            return targets, cleaned_outputs

    def log_metrics(self, targets, outputs):
        if self.problem_type == "regression":
            avg_score = root_mean_squared_error(targets, outputs)

        else:
            avg_score = roc_auc_score(targets, outputs)

        return avg_score

    def save_model(self, model_path: str):
        os.makedirs(model_path, exist_ok=True)

        torch.save(
            self.best_model_weights.state_dict(),
            f"{model_path}/tabmonet.pt",
        )
