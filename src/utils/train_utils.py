import torch
from tqdm import tqdm
from pathlib import Path
import wandb
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from src.models.base_model import NoPropModel
from src.models.model_wrapper import NoPropModelWrapper
from utils.test_utils import test  # type:ignore
import random


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
    wrapper: NoPropModelWrapper, optimizer, scheduler, criterion, dataloader: DataLoader
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

    # TODO maybe drop autocast and clipping
    wrapper.model.train()
    training_loss = 0.0
    counter = 0
    scaler = GradScaler()
    for src, _, trg in tqdm(dataloader):
        src = src.to(wrapper.device)
        trg = trg.to(wrapper.device)

        counter += 1

        optimizer.zero_grad()
        with autocast(wrapper.device.type):
            output = wrapper.model(src, trg[:-1])
            loss = criterion(output.view(-1, output.shape[-1]), trg[1:, :].view(-1))
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        training_loss += loss.item()
    return training_loss / counter


def calculate_validation_loss(model: NoPropModel, criterion, src, trg):
    """
    Calculates the validation loss for a given batch of data.

    :param model: The model to evaluate.
    :param criterion: The loss function to use.
    :param src: Source input tensor.
    :param trg: Target output tensor.
    :return: The validation loss for the batch.
    """

    output = model(src, trg[:-1, :])
    loss = criterion(
        output.view(-1, output.shape[-1]), torch.reshape(trg[1:, :], (-1,))
    )
    return loss.item()


def validate_epoch(
    epoch: int,
    wrapper: NoPropModelWrapper,
    criterion,
    dataloader: DataLoader,
):
    """
    Validates the model for one epoch and logs the results.

    :param epoch: Current epoch number.
    :param wrapper: Model wrapper containing the model and device information.
    :param criterion: Loss function to evaluate the model.
    :param dataloader: DataLoader for the validation dataset.
    :return: Average validation loss and a list of records for logging.
    """

    validation_loss = 0.0
    counter = 0
    wrapper.model.eval()

    logged_samples = 0
    total_samples_to_log = wrapper.train_config.logs_per_epoch

    records: list[tuple[int, object, str, str]] = []

    with torch.no_grad():
        for src, trg in tqdm(dataloader):
            src = src.to(wrapper.device)
            trg = trg.to(wrapper.device)

            batch_size = len(trg)
            counter += batch_size

            validation_loss += (
                calculate_validation_loss(wrapper.model, criterion, src, trg)
                * batch_size
            )

            predictions = wrapper.predict(src)  # List of predicted classes

            # Calculate the probability of logging a sample
            remaining_samples_to_log = total_samples_to_log - logged_samples
            if remaining_samples_to_log > 0:
                probability = remaining_samples_to_log / (counter - logged_samples)

                for i, (pred, label) in enumerate(zip(predictions, trg)):
                    if (
                        logged_samples < total_samples_to_log
                        and random.random() < probability
                    ):
                        records.append((epoch, src[i], label.item(), pred))
                        logged_samples += 1

    return (
        validation_loss / counter,
        records,
    )


def train(
    wrapper: NoPropModelWrapper,
    optimizer,
    criterion,
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
        train_loss = train_epoch(wrapper, optimizer, scheduler, criterion, train_loader)
        validation_loss, records = validate_epoch(
            epoch, wrapper, criterion, validation_loader
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
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
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience_counter = 0
            wrapper.save_model(wrapper.train_config.model_path)
            wandb.save(str(Path(wrapper.train_config.model_path)))
        else:
            patience_counter += 1
        if patience_counter >= wrapper.train_config.patience:
            print("Early stopping triggered")
            break
