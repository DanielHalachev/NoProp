import os


class NoPropTrainConfig:
    """
    Configuration class for training NoProp models.
    This class encapsulates all the necessary parameters for training, such as batch size, number of epochs,
    learning rate, weight decay, patience for early stopping, number of workers for data loading, and logging frequency.
    """

    def __init__(
        self,
        model_path: os.PathLike,
        batch_size: int = 32,
        epochs: int = 20,
        learning_rate: float = 0.01,
        weight_decay: float = 0,
        patience: int = 5,
        workers: int = 8,
        logs_per_epoch: int = 5,
    ) -> None:
        """
        Initializes the training configuration for NoProp models.

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
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.workers = workers
        self.logs_per_epoch = logs_per_epoch
