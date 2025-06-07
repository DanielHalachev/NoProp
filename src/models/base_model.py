import os
import torch


class NoPropModel(torch.nn.Module):
    """
    Base class for all NoProp models.
    """

    def __init__(self, device: torch.device) -> None:
        """
        Initializes the NoPropModel with the specified device.

        :param device: The device (CPU or GPU) on which the model will run.
        """
        super().__init__()
        self.device = device
        self.to(device)

    def save_model(self, path: os.PathLike):
        """
        Saves the model state dictionary to the specified path.

        :param path: Path where the model state dictionary will be saved.
        """
        torch.save(self.state_dict(), path)
