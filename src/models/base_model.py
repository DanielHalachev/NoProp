import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.components.backbone import Backbone
from src.components.concatenator import Concatenator
from src.components.noise_scheduler import NoiseScheduler
from src.embeddings.label_encoder import LabelEncoder
from src.utils.model_config import NoPropBaseModelConfig


class BaseNoPropModel(ABC, torch.nn.Module):
    """
    Base class for all NoProp models.
    """

    def __init__(
        self,
        config: NoPropBaseModelConfig,
        label_encoder: LabelEncoder,
        noise_scheduler: NoiseScheduler,
        concatenator: Concatenator,
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
        self.backbone = Backbone(
            config.backbone_resnet_type, config.embedding_dimension
        )
        self.label_encoder = label_encoder
        self.noise_scheduler = noise_scheduler
        self.concatenator = concatenator
        self.W_Embed = nn.Parameter(
            torch.zeros(config.num_classes, config.embedding_dimension)
        )
        self.device = device
        self.to(device)

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
    def forward_denoise(self, *args, **kwargs) -> torch.Tensor:
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

    @abstractmethod
    def train_step(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        eta: float,
    ) -> float:
        """
        Abstract method for training step. Must be implemented by subclasses.

        :param src: Source tensor (input data).
        :param trg: Target tensor (labels).
        :param optimizer: Optimizer for updating model parameters.
        :param eta: Hyperparameter.
        :return: Loss value for the training step over the whole batch.
        """
        pass

    @abstractmethod
    def validate_step(
        self, src: torch.Tensor, trg: torch.Tensor, eta: float, *args, **kwargs
    ) -> tuple[float, int, int, torch.Tensor]:
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

    @abstractmethod
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
        pass
