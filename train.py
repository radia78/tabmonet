import os

import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import yaml
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from embedding import *
from utils import *
from model import TabularMONet

from typing import Union, List, Tuple, Any

_embedding_models = [
    "PBLD",
    "Q",
    "P",
    "L",
    "NA",
    "Q-NF",
    "QV2",
]

_embedding_model_mapping = {
    "PBLD": PBLDEmbedding,
    "P": PeriodicEmbedding,
    "L": LinearEmbedding,
    "Q": QuantileEmbedding,
    "Q-NF": QuantileEmbeddingNoFraction,
    "NA": nn.Identity,
    "QV2": QuantileEmbeddingV2,
}

_loss_mapping = {
    "reg": nn.MSELoss, 
    "bin-cls": nn.BCEWithLogitsLoss,
    "multi-cls": nn.CrossEntropyLoss
}

_scheduler_mapping = {
    "cos-anneal": lr_scheduler.CosineAnnealingLR,
    "cos-anneal-wr": lr_scheduler.CosineAnnealingWarmRestarts,
    "NA": None,
}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Union[optim.Optimizer, List[optim.Optimizer], Tuple[optim.Optimizer], Any],
        scheduler: Union[lr_scheduler.LRScheduler, List[lr_scheduler.LRScheduler], Tuple[lr_scheduler.LRScheduler]],
        preprocessor,
        task: str,
        log_dir: str="runs/experiment_1",
        device: str="cpu",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.preprocessor = preprocessor
        self.task = task
        self.device = device

        self.log_dir = log_dir

        # Initialize TensorBoard Writer
        self.writer = SummaryWriter(log_dir=log_dir)
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        running_loss = 0.0

        # TODO: Make this adaptable to cases where there is missing categorical/numerical features
        for num_features, cat_features, targets in train_loader:

            # Attaching data to device
            num_features, cat_features = num_features.to(self.device), cat_features.to(self.device)
            targets = targets.to(self.device)

            # There might be cases where we use multiple optimizers 
            if isinstance(self.optimizer, list) or isinstance(self.optimzier, tuple):
                for opt in self.optimizer:
                    opt.zero_grad()
            else:
                self.optimizer.zero_grad()

            outputs = self.model(num_features, cat_features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # There might be cases where we use multiple optimizers 
            if isinstance(self.optimizer, list) or isinstance(self.optimzier, tuple):
                for opt in self.optimizer:
                    opt.step()
            else:
                self.optimizer.step()

            running_loss += loss.item()

        if self.scheduler is not None:
            if isinstance(self.scheduler, list) or isinstance(self.scheduler, tuple):
                for sch in self.scheduler:
                    sch.step()

            else:
                self.scheduler.step() 

        avg_loss = running_loss / len(train_loader)

        # Log training loss to TensorBoard
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        return avg_loss

    def validate(self, val_loader, epoch):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for num_features, cat_features, targets in val_loader:

                num_features, cat_features = num_features.to(self.device), cat_features.to(self.device)
                targets = self.preprocessor.inverse_transform_target(targets.numpy())

                if self.task == "reg":
                    outputs = self.preprocessor.target_preprocessor.inverse_transform(
                        self.model(
                            num_features,
                            cat_features
                        ).cpu().numpy()
                    )
                    running_loss += root_mean_squared_error(targets, outputs)

                else:
                    outputs = self.preprocessor.target_preprocessor.inverse_transform(
                        self.model(
                            num_features,
                            cat_features
                        ).sigmoid().cpu().numpy()
                    )
                    running_loss += roc_auc_score(targets, outputs)

        avg_loss = running_loss / len(val_loader)

        # Log validation loss to TensorBoard
        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
        return avg_loss

    def fit(
        self,
        model_path,
        model_name,
        train_loader,
        val_loader,
        epochs=10,
        save_model=False,
    ):
        prog_bar = tqdm(range(epochs))
        for epoch in prog_bar:
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            prog_bar.set_description(
                desc=f"Epoch [{epoch + 1}/{epochs}] Loss: {train_loss:.4f} | Val: {val_loss:.4f}"
            )

        # Close the writer when finished
        self.writer.close()

        if save_model:
            print(os.path.exists(model_path))
            os.makedirs(model_path, exist_ok=True)

            torch.save(
                self.model.state_dict(),
                f"model_cache/{model_path}/tabular-monet-{model_name}.pt",
            )


def main(config):
    model_config = config["model"]
    num_emb_config = config["embedding"]["numerical"]
    optim_config = config['optim']
    lr_config = config['lr']

    emb_model_name = num_emb_config.pop("name").strip()
    assert emb_model_name in _embedding_models, (
        f"Embedding '{emb_model_name}' does not exist"
    )

    if emb_model_name == "Q" or emb_model_name == "Q-NF" or emb_model_name == "QV2":
        num_bins = num_emb_config["num_bins"]
        return_bin_edges = True

    else:
        num_bins = 0
        return_bin_edges = False

    train_loader, test_loader, preprocessor, bin_edges = load_dataset(
        dataset_id=config["id"],
        task=config["task"],
        test_size=config["test_size"],
        batch_size=config["batch_size"],
        return_bin_edges=return_bin_edges,
        num_bins=num_bins,
    )

    if config['numerical_variables']:
        # Get bin edges if we use quantile embedding        
        if return_bin_edges:
            num_embedding = _embedding_model_mapping.get(emb_model_name)(
                **num_emb_config, bin_edges=bin_edges
            )

        else:
            num_embedding = _embedding_model_mapping.get(emb_model_name)(**num_emb_config)


    model = TabularMONet(
        **model_config,
        num_embedding=num_embedding,
    )

    # Configuring the optimizer
    optim_name = optim_config.pop('name')
    if optim_name == 'muon':
        # Obtain all 2D parameters from the model except the bias for the embedding
        muon_params = [
            p
            for name, p in model.named_parameters()
            if ("weight" in name) and (p.dim() > 1)
        ]
        adam_params = [p for name, p in model.named_parameters() if "weight" not in name]

        # For some reason its read as a string so convert to float
        optim_config['muon_config']['lr'] = float(optim_config['muon_config']['lr'])
        optim_config['adam_config']['lr'] = float(optim_config['adam_config']['lr'])

        optimizer = [
            torch.optim.Muon(muon_params, **optim_config['muon_config']),
            torch.optim.AdamW(adam_params, **optim_config['adam_config'])
        ]

    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_config)

    # Configuring the scheduler
    lr_name = lr_config.pop('name')
    if lr_name == "NA":
        scheduler = None

    else: 
        if optim_name == 'muon':
            scheduler = [
                _scheduler_mapping[lr_name](op, **lr_config) for op in optimizer
            ]

        else:
            scheduler = _scheduler_mapping[lr_name](optimizer, **lr_config)

    critertion = _loss_mapping[config["task"]]()

    # Run name are formatted as the following
    # runs/[DATA_ID]/[MODEL_CONFIG]-[EMBEDDING_CONFIG]-[OPTIMIZER_CONFIG]-[SCHEDULER_CONFIG]
    model_config_name = "-".join(f"{k}={v}" for k, v in model_config.items())
    num_emb_config_name = "-".join(f"{k}={v}" for k, v in num_emb_config.items())
    #cat_emb_config_name = "-".join(f"{k}={v}" for k, v in cat_emb_config.items())
    optim_config_name = "-".join(f"{k}={v}" for k, v in optim_config.items())
    lr_config_name = "-".join(f"{k}={v}" for k, v in lr_config.items())

    # The name gets annoying but this allows us to view the specific settings of the model during experimentation
    run_name = "-".join(
        [
            model_config_name, 
            num_emb_config_name, 
            #cat_emb_config_name,
        ])

    trainer = Trainer(
        model=model,
        criterion=critertion,
        optimizer=optimizer,
        scheduler=scheduler,
        preprocessor=preprocessor,
        log_dir=f"runs/{config['id']}/{run_name}",
        task=config["task"],
        device="mps",
    )

    trainer.fit(
        model_path=f"{config['id']}/{emb_model_name}",
        model_name=run_name,
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=config["epochs"],
        save_model=config["save_model"],
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
