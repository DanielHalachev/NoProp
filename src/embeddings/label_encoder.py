from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class LabelEncoder(ABC, nn.Module):
    """
    Encodes a label-embedding vector z_t of shape [batch_size, embed_dim] via a small FC net with skip connection.
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LabelEncoder.

        :param z: Input tensor of shape [batch_size, embedding_dimension], which represents the label-embedding vector.
        :return: Output tensor of the same shape as input z, after passing through the fully connected layers with a skip connection.
        """
        pass
