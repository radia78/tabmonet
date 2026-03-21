import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.data import DataLoader

import yaml
from datetime import datetime

from utils.data import *
from utils.trainer import Trainer
from utils.optimizer import CosineLogLR, FlatCosineWD, AdamWMuonWrapper
from models.tabmonet import build_model

LOSS_REGISTRY = {
    "regression": nn.MSELoss,
    "binary": nn.BCEWithLogitsLoss,
    "multiclass": nn.CrossEntropyLoss,
}

LR_SCHEDULE_REGISTRY = {
    "cos-anneal": lr_scheduler.CosineAnnealingLR,
    "cos-log": CosineLogLR,
    "cos-anneal-wr": lr_scheduler.CosineAnnealingWarmRestarts,
    "NA": None,
}

WD_SCHEDULE_REGISTRY = {"flat-cos": FlatCosineWD, "NA": None}


def main(run_name, config):
    model_config = config["model"]

    num_emb_config = config["numerical_embedding"]
    cat_emb_config = config["categorical_embedding"]
    optim_config = config["optimizer"]
    lr_config = config["lr_schedule"]
    wd_config = config["wd_schedule"]

    # Load the dataset
    train_dataset, val_dataset, test_dataset, preprocessor, bin_edges = prepare_dataset(
        rm_NA=True,
        dataset_id=config["dataset_id"],
        problem_type=config["problem_type"],
        num_bins=num_emb_config["num_bins"],
        test_size=config["test_size"],
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=config["batch_size"], shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Build the model
    model = build_model(
        preprocessor=preprocessor,
        cat_emb_config=cat_emb_config,
        num_emb_config=num_emb_config,
        model_config=model_config,
        bin_edges=bin_edges,
        compile=config["compile"],
    )

    # Configuring the optimizer
    optim_name = optim_config.pop("name")
    if optim_name == "muon":
        # Obtain all 2D parameters from the model except the bias for the embedding
        muon_params = [
            p
            for name, p in model.named_parameters()
            if ("weight" in name) and (p.dim() == 2)
        ]
        adam_params = [
            p for name, p in model.named_parameters() if "weight" not in name
        ]
        optimizer = [
            torch.optim.Muon(muon_params, **optim_config["muon_config"]),
            torch.optim.AdamW(adam_params, **optim_config["adam_config"]),
        ]

    elif optim_name == "adamw":
        optimizer = [torch.optim.AdamW(model.parameters(), **optim_config)]

    # Configuring the scheduler
    lr_name = lr_config.pop("name")
    if lr_name == "NA":
        lr_scheduler = None

    else:
        if optim_name == "muon":
            lr_scheduler = [
                LR_SCHEDULE_REGISTRY.get(lr_name)(op, **lr_config) for op in optimizer
            ]

        else:
            lr_scheduler = [LR_SCHEDULE_REGISTRY.get(lr_name)(optimizer, **lr_config)]

    # Setup the weight-decay scheduler
    wd_name = wd_config.pop("name")
    if wd_name == "NA":
        wd_scheduler = None
    else:
        wd_scheduler = [WD_SCHEDULE_REGISTRY.get(wd_name)(optimizer, **wd_config)]

    # Flatten the dictionary for logging purposes
    # Reinsert names
    config["optimizer"]["name"] = optim_name
    config["lr_schedule"]["name"] = lr_name
    config["wd_schedule"]["name"] = wd_name

    run_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            for u, w in v.items():
                if isinstance(w, list):
                    run_config[f"{k}/{u}{i}"] = str(j)
                elif isinstance(w, dict):
                    for i, j in w.items():
                        if isinstance(j, list):
                            run_config[f"{k}/{u}/{i}"] = str(j)
                        else:
                            run_config[f"{k}/{u}/{i}"] = str(j)
                else:
                    run_config[f"{k}/{u}"] = w
        else:
            run_config[k] = v

    critertion = LOSS_REGISTRY.get(config["problem_type"])()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        wd_scheduler=wd_scheduler,
        criterion=critertion,
        preprocessor=preprocessor,
        problem_type=config["problem_type"],
        run_config=run_config,
        log_dir=f"runs/{config['dataset_id']}/{run_name}-{datetime.now()}",
        device=config["device"],
    )

    trainer.run(
        num_epochs=config["num_epochs"],
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    if config["save_model"]:
        trainer.save_model(
            f"model_cache/{config['dataset_id']}/{run_name}-{datetime.now()}"
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    run_name = args.config.split("/")[-1].split(".")[0]

    main(run_name, config)
