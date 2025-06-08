import os
from pathlib import Path

from src.data.dataset_manager import DatasetType


class NoPropTrainConfig:
    """
    Configuration class for training NoProp models.
    This class encapsulates all the necessary parameters for training, such as batch size, number of epochs,
    learning rate, weight decay, patience for early stopping, number of workers for data loading, and logging frequency.
    """

    def __init__(
        self,
        model_path: os.PathLike,
        dataset_type: DatasetType,
        dataset_path: Path,
        batch_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 0.01,
        weight_decay: float = 0,
        samples_per_class: int = 10,
        eta: float = 1.0,
        inference_number_of_steps=40,
        patience: int = 5,
        workers: int = 8,
        logs_per_epoch: int = 5,
    ) -> None:
        """
        Initializes the training configuration for NoProp models.

        :param dataset_type: Type of dataset to be used for training (e.g., MNIST, CIFAR-10).
        :param dataset_path: Path to the dataset directory.
        :param model_path: Path to save the trained model.
        :param batch_size: Number of samples per batch during training.
        :param epochs: Total number of epochs for training.
        :param learning_rate: Learning rate for the optimizer.
        :param weight_decay: Weight decay (L2 regularization) for the optimizer.
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param workers: Number of worker threads for data loading.
        :param logs_per_epoch: Number of logs to record per epoch.
        """

        self.model_path = model_path
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.samples_per_class = samples_per_class
        self.eta = eta
        self.inference_number_of_steps = inference_number_of_steps
        self.patience = patience
        self.workers = workers
        self.logs_per_epoch = logs_per_epoch
