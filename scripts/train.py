import hydra
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from tabmonet.data.dataset import prepare_dataset
from tabmonet.trainer import Trainer
from tabmonet.optimizers import configure_optimizer


@hydra.main(version_base=None, config_path="../confs", config_name="config")
def train(cfg):
    preprocessor = instantiate(cfg.preprocessor)
    train_dataset, val_dataset, test_dataset, bin_edges = prepare_dataset(
        **cfg.dataset, preprocessor=preprocessor
    )

    # Build the encoder
    if "QuantileEmbedding" in cfg.model.numerical_encoder._target_:
        numerical_encoder = instantiate(
            cfg.model.numerical_encoder, bin_edges=bin_edges
        )
    else:
        numerical_encoder = instantiate(cfg.model.numerical_encoder)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    # Build the model
    model = instantiate(cfg.model, numerical_encoder=numerical_encoder)
    if cfg.compile:
        model.compile(mode="reduce-overhead")

    # Configuring the optimizer
    # Obtain all 2D parameters from the model except the bias for the embedding
    optimizers = configure_optimizer(cfg, model)

    # Setup the weight-decay scheduler
    wd_schedulers = [instantiate(cfg.wd_scheduler, optimizer=opt) for opt in optimizers]

    # Setup the learning-rate scheduler
    lr_schedulers = [instantiate(cfg.lr_scheduler, optimizer=opt) for opt in optimizers]

    trainer = Trainer(
        model=model,
        optimizer=optimizers,
        lr_scheduler=lr_schedulers,
        wd_scheduler=wd_schedulers,
        preprocessor=preprocessor,
        problem_type=cfg.dataset.problem_type,
        log_dir=HydraConfig.get().runtime.output_dir,
        device=cfg.device,
        log=cfg.log,
    )

    trainer.run(
        num_epochs=cfg.epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    if cfg.log:
        trainer.save_model(f"{HydraConfig.get().runtime.output_dir}")


if __name__ == "__main__":
    train()
