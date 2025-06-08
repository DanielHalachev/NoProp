import random
from pathlib import Path

import torch
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.model_wrapper import NoPropModelWrapper


def get_scheduler(optimizer, total_steps, warmup_steps=5000):
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


def train_epoch(
    wrapper: NoPropModelWrapper, optimizer, scheduler, dataloader: DataLoader
):
    """
    Trains the model for one epoch.

    :param wrapper: Model wrapper containing the model and device information.
    :param optimizer: Optimizer for the model parameters.
    :param scheduler: Learning rate scheduler.
    :param criterion: Loss function to optimize.
    :param dataloader: DataLoader for the training dataset.
    :return: Average training loss for the epoch.
    """

    wrapper.model.train()
    training_loss = 0.0
    for src, trg in tqdm(dataloader):
        src = src.to(wrapper.device)
        trg = trg.to(wrapper.device)

        training_loss += wrapper.model.train_step(
            src, trg, optimizer, wrapper.train_config.eta
        ) * src.size(0)
    avg_loss = training_loss / len(dataloader.dataset)  # type: ignore
    return avg_loss


def validate_epoch(
    epoch: int,
    wrapper: NoPropModelWrapper,
    dataloader: DataLoader,
):
    """
    Validates the model for one epoch and logs the results.

    :param epoch: Current epoch number.
    :param wrapper: Model wrapper containing the model and device information.
    :param dataloader: DataLoader for the validation dataset.
    :return: Tuple containing the average validation loss, accuracy, and records of predictions.
    """
    validation_loss = 0.0
    correct = 0
    total = 0
    wrapper.model.eval()

    logged_samples = 0
    total_samples_to_log = wrapper.train_config.logs_per_epoch

    records: list[tuple[int, object, str, str]] = []

    with torch.no_grad():
        for src, trg in tqdm(dataloader):
            src = src.to(wrapper.device)
            trg = trg.to(wrapper.device)

            validation_loss_temp, correct_temp, total_temp, predictions = (
                wrapper.model.validate_step(
                    src,
                    trg,
                    wrapper.train_config.eta,
                    wrapper.train_config.inference_number_of_steps,
                )
            )
            validation_loss += validation_loss_temp
            correct += correct_temp
            total += total_temp

            # Calculate the probability of logging a sample
            remaining_samples_to_log = total_samples_to_log - logged_samples
            if remaining_samples_to_log > 0:
                probability = remaining_samples_to_log / (total - logged_samples)

                for i, (pred, label) in enumerate(zip(predictions.tolist(), trg.toli)):
                    if (
                        logged_samples < total_samples_to_log
                        and random.random() < probability
                    ):
                        records.append((epoch, src[i], label.item(), pred))
                        logged_samples += 1

    avg_loss = validation_loss / len(dataloader.dataset)  # type: ignore
    return (
        validation_loss,
        correct / total,
        records,
    )


def train(
    wrapper: NoPropModelWrapper,
    optimizer,
    scheduler,
    train_loader,
    validation_loader,
):
    """
    Trains the model for a specified number of epochs, validating after each epoch.

    :param wrapper: Model wrapper containing the model training configuration.
    :param optimizer: Optimizer for the model parameters.
    :param criterion: Loss function to optimize.
    :param scheduler: Learning rate scheduler.
    :param train_loader: DataLoader for the training dataset.
    :param validation_loader: DataLoader for the validation dataset.
    """

    columns = ["Epoch", "Image", "Ground Truth", "Prediction"]
    all_records = []
    patience_counter = 0
    best_loss = float("inf")

    for epoch in range(wrapper.train_config.epochs):
        print(
            f"====================================== EPOCH {(epoch + 1)} ======================================"
        )
        train_loss = train_epoch(wrapper, optimizer, scheduler, train_loader)
        validation_loss, validation_acc, records = validate_epoch(
            epoch, wrapper, validation_loader
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
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

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={validation_loss:.4f}"
        )
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
