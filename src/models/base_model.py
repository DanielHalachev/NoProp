import os
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.components.backbone import Backbone
from src.components.concatenator import Concatenator
from src.components.noise_scheduler import NoiseScheduler
from src.embeddings.label_encoder import LabelEncoder
from src.utils.model_config import NoPropBaseModelConfig


class BaseNoPropModel(ABC, torch.nn.Module):
    """
    Base class for all NoProp models.
    """

    def __init__(self, config: NoPropBaseModelConfig, device: torch.device) -> None:
        """
        Initializes the NoPropModel with the specified device.

        :param config: Configuration object containing model parameters.
        :param device: The device (CPU or GPU) on which the model will run.
        """

        super().__init__()
        self.config = config
        self.backbone = Backbone(
            config.backbone_resnet_type, config.embedding_dimension
        )
        self.label_encoder = LabelEncoder(config.embedding_dimension)
        self.noise_scheduler = NoiseScheduler(config.noise_scheduler_hidden_dimension)
        self.concatenator = Concatenator(config.embedding_dimension, config.num_classes)
        self.W_Embed = nn.Parameter(
            torch.zeros(config.num_classes, config.embedding_dimension)
        )
        self.device = device
        self.to(device)

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
