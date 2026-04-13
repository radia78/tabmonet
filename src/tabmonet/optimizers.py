import torch.nn as nn
from hydra.utils import instantiate


def configure_optimizer(cfg, model: nn.Module):
    # Find out if soft-selection is part of the model parameters
    is_soft_selection = False
    for name, _ in model.named_parameters():
        if "soft_selection" in name:
            is_soft_selection = True
            break

    # Configure non-Muon
    if len(cfg.optimizer.values()) >= 2:
        muon_params = [
            p
            for name, p in model.named_parameters()
            if ("weight" in name) and (p.dim() == 2) and ("soft_selection" not in name)
        ]
        adam_params = [
            p
            for name, p in model.named_parameters()
            if ("weight" not in name) and ("soft_selection" not in name)
        ]

        if is_soft_selection:
            muon_params = [
                p
                for name, p in model.named_parameters()
                if ("weight" in name)
                and (p.dim() == 2)
                and ("soft_selection" not in name)
            ]
            adam_params = [
                p
                for name, p in model.named_parameters()
                if ("weight" not in name) and ("soft_selection" not in name)
            ]
            optimizer = [
                instantiate(cfg.optimizer.muon, params=muon_params),
                instantiate(
                    cfg.optimizer.adamw,
                    [
                        {
                            "params": adam_params,
                            "lr": cfg.optimizer.adamw.lr,
                            "betas": cfg.optimizer.adamw.betas,
                            "weight_decay": cfg.optimizer.adamw.weight_decay,
                        },
                        {
                            "params": [model.soft_selection],
                            "lr": cfg.optimizer.adamw.lr * 10,
                            "betas": cfg.optimizer.adamw.betas,
                            "weight_decay": cfg.optimizer.adamw.weight_decay,
                        },
                    ],
                ),
            ]

        else:
            muon_params = [
                p
                for name, p in model.named_parameters()
                if ("weight" in name) and (p.dim() == 2)
            ]
            adam_params = [
                p for name, p in model.named_parameters() if ("weight" not in name)
            ]
            params = [muon_params, adam_params]
            optimizer = [
                instantiate(opt_config, params=param)
                for opt_config, param in zip(cfg.optimizer.values(), params)
            ]

    else:
        if is_soft_selection:
            optimizer = [
                instantiate(
                    cfg.optimizer.adamw,
                    [
                        {
                            "params": adam_params,
                            "lr": cfg.optimizer.adamw.lr,
                            "betas": cfg.optimizer.adamw.betas,
                            "weight_decay": cfg.optimizer.adamw.weight_decay,
                        },
                        {
                            "params": [model.soft_selection],
                            "lr": cfg.optimizer.adamw.lr * 10,
                            "betas": cfg.optimizer.adamw.betas,
                            "weight_decay": cfg.optimizer.adamw.weight_decay,
                        },
                    ],
                )
            ]

        else:
            optimizer = [
                instantiate(opt_config, params=model.parameters())
                for opt_config in cfg.optimizer.values()
            ]

    return optimizer
