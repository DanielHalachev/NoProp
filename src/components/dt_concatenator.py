import torch
import torch.nn as nn

from src.components.concatenator import Concatenator


class DTConcatenator(Concatenator):
    """
    Concatenator is a neural network module that combines features from different sources
    (fx, fz) into a single output of shape [batch_size, num_classes].
    fx is the image tensor,
    fz are the features from the previous noise block,
    """

    def __init__(self, embedding_dimension: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dimension * 2, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, num_classes),
        )

    @property
    def output_features(self):
        """
        Returns the number of output features of the last layer in the network.
        This is useful for determining the output dimension of the model.
        """

        return self.net[-1].out_features

    def forward(self, fx: torch.Tensor, fz: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FuseHead module.
        :param fx: Tensor representing the image features.
        :param fz: Tensor representing the previous noise block features.
        :param ft: Tensor representing the time embedding features (only for NoPropCT).
        :return: Output tensor of size [batch_size, num_classes]
        """
        x = torch.cat([fx, fz], dim=1)
        return self.net(x)
