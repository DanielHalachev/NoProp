from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.models.model_wrapper import NoPropModelWrapper


def get_scheduler(optimizer, total_steps, warmup_steps=5000) -> LRScheduler:
    """
    Creates a learning rate scheduler that combines warmup and cosine annealing.

    :param optimizer: The optimizer for which to create the scheduler.
    :param total_steps: Total number of training steps.
    :param warmup_steps: Number of warmup steps.
    :return: A chained scheduler with warmup and cosine annealing.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    return torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])


def train(
    wrapper: NoPropModelWrapper,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    train_loader: DataLoader,
    validation_loader: DataLoader,
):
    """
    Trains the model for a specified number of epochs, validating after each epoch.

    :param wrapper: Model wrapper containing the model training configuration.
    :param optimizer: Optimizer for the model parameters.
    :param scheduler: Learning rate scheduler.
    :param train_loader: DataLoader for the training dataset.
    :param validation_loader: DataLoader for the validation dataset.
    """

    # wandb records
    columns = ["Epoch", "Image", "Ground Truth", "Prediction"]
    all_records = []
    patience_counter = 0

    total_epochs = wrapper.train_config.epochs
    best_loss = float("inf")

    with tqdm(total=total_epochs, desc="Training Progress", unit="epoch") as epoch_bar:
        for epoch in range(1, total_epochs + 1):
            train_loss = wrapper.model.train_epoch(
                optimizer,
                scheduler,
                train_loader,
                wrapper.train_config.eta,
                epoch,
                total_epochs,
            )
            validation_acc, records = wrapper.model.validate_epoch(
                validation_loader,
                epoch,
                total_epochs,
                wrapper.train_config.logs_per_epoch,
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation:acc": validation_acc,
                }
            )

            all_records.extend(
                [
                    (epoch, wandb.Image(img), label, pred)
                    for epoch, img, label, pred in records
                ]
            )
            table = wandb.Table(columns=columns, data=all_records)
            wandb.log({"validation_samples": table})

            epoch_bar.set_postfix(
                {
                    "Train Loss": f"{train_loss:.4f}",
                    "Validation Acc": f"{validation_acc:.4f}",
                }
            )

            # ATTENTION:
            # keep the best model but not on validation loss
            # but on training loss, as the paper suggests
            # this is because there is no traditional backpropagation
            # we teach each block in the model to denoise an input
            # with a specific noise level instead
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                wrapper.save_model(wrapper.train_config.model_path)
                wandb.save(str(Path(wrapper.train_config.model_path)))
            else:
                patience_counter += 1
            if patience_counter >= wrapper.train_config.patience:
                print("Early stopping triggered")
                break

            epoch_bar.update(1)
