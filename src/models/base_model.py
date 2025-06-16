import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.utils.model_config import NoPropBaseModelConfig


class BaseNoPropModel(ABC, torch.nn.Module):
    """
    Base class for all NoProp models.
    """

    def __init__(
        self,
        config: NoPropBaseModelConfig,
        device: torch.device,
    ) -> None:
        """
        Initializes the NoPropModel with the specified device.

        :param config: Configuration object containing model parameters.
        :param label_encoder: Label encoder for encoding labels into embeddings.
        :param noise_scheduler: Noise scheduler for managing noise levels during training.
        :param concatenator: Concatenator for combining features from different sources.
        :param device: The device (CPU or GPU) on which the model will run.
        """

        super().__init__()

        self.W_Embed = nn.Parameter(
            torch.zeros(config.num_classes, config.embedding_dimension)
        )
        self.classifier = nn.Linear(config.embedding_dimension, config.num_classes)
        self.device = device

    @abstractmethod
    def alpha_bar(self, *args, **kwargs) -> torch.Tensor:
        """
        Abstract method to get the alpha_bar values from the noise scheduler.
        :return: Tensor representing the alpha_bar values."""
        pass

    def save_model(self, path: os.PathLike):
        """
        Saves the model state dictionary to the specified path.

        :param path: Path where the model state dictionary will be saved.
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path: os.PathLike):
        """
        Loads the model state dictionary from the specified path.

        :param path: Path where the model state dictionary will be loaded from.
        """
        self.load_state_dict(torch.load(path))

    @abstractmethod
    def forward_denoise(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for inference. Must be implemented by subclasses.

        :param x: Input tensor.
        :return: Inferred output tensor.
        """
        pass

    @abstractmethod
    def train_epoch(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        dataloader: DataLoader,
        eta: float,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """
        Trains the model for one epoch.

        :param optimizer: Optimizer for the model parameters.
        :param scheduler: Learning rate scheduler.
        :param dataloader: DataLoader for the training dataset.
        :param eta: Learning rate.
        :return: Average training loss for the epoch.
        """
        pass

    @abstractmethod
    def validate_epoch(
        self,
        dataloader: DataLoader,
        eta: float,
        epoch: int,
        total_epochs: int,
        logs_per_epoch: int,
    ) -> tuple[float, list[tuple[int, object, int, int]]]:
        """
        Abstract method for validation step. Must be implemented by subclasses.

        The NoProp paper uses the train loss for optimizing the training.
        Validation loss calculation may still be helpful to measure overfitting though.

        :param src: Source tensor (input data).
        :param trg: Target tensor (labels).
        :param eta: Hyperparameter (if applicable).
        :param inference_number_of_steps: Number of steps for inference (if applicable).
        :return: Tuple of (validation loss, number of correct predictions, total number of predictions, predictions).
        """
        pass

    def test(self, dataloader: DataLoader) -> tuple[int, int, list[torch.Tensor]]:
        """
        Tests the model on the provided DataLoader.

        :param dataloader: DataLoader for the test dataset.
        :return: Tuple of (number of correct predictions, total number of predictions, predictions).
        """
        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            predictions = []

            for src, trg in dataloader:
                src = src.to(self.device)
                trg = trg.to(self.device)
                c, t, preds = self.test_step(src, trg)
                correct += c
                total += t
                predictions.append(preds)

            return (correct, total, predictions)

    def test_step(
        self, src: torch.Tensor, trg: torch.Tensor, *args, **kwargs
    ) -> tuple[int, int, torch.Tensor]:
        """
        Abstract method for test step. Must be implemented by subclasses.

        :param src: Source tensor (input data).
        :param trg: Target tensor (labels).
        :param inference_number_of_steps: Number of steps for inference (if applicable).
        :return: Tuple of (correct predictions, total predictions, predictions).
        """

        correct = 0
        predictions = self.infer(src).argmax(dim=1)
        correct += int((predictions == trg).sum().item())
        return correct, trg.size(0), predictions
