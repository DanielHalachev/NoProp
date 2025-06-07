import os
import torch
from src.models.base_model import NoPropModel
from src.utils.train_config import NoPropTrainConfig


class NoPropModelWrapper:
    """
    Wrapper around any NoProp model to store device and training configuration in order to avoid passing too many parameters.
    """

    def __init__(
        self, model: NoPropModel, train_config: NoPropTrainConfig, device: torch.device
    ) -> None:
        """
        Initializes the NoPropModelWrapper with an instance of a model, training configuration, and device.

        :param model: An instance of NoPropModel.
        :param train_config: An instance of NoPropTrainConfig containing training parameters.
        :param device: The device (CPU or GPU) on which the model will run.
        """
        self.model = model
        self.train_config = train_config
        self.device = device

    def save_model(self, path: os.PathLike | None = None):
        """
        Saves the model to the specified path or to the default path in the training configuration.

        :param path: Optional path to save the model. If None, uses the path from train_config.
        """
        save_path = self.train_config.model_path if path is None else path
        self.model.save_model(save_path)

    def predict(self, img) -> list[str]:
        """
        Makes a prediction using the model.

        :param img: A batch of image inputs for prediction.
        :return: A list of predicted class labels.
        """
        # TODO make it work for batches
        return []
