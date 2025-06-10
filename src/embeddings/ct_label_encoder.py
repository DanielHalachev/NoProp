import torch
import torch.nn as nn

from src.embeddings.label_encoder import LabelEncoder


class CTLabelEncoder(LabelEncoder):
    """
    Encodes a label-embedding vector z_t of shape [batch_size, embedding_dimension] via a small FC net with skip connection.
    """

    def __init__(self, embedding_dimension: int, hidden_dim: int):
        """
        Initializes the LabelEncoder with a specified embedding output dimension and hidden dimension.

        :param embedding_dimension: Dimension of the embedding vector that will be output.
        :param hidden_dim: Dimension of the hidden layer in the fully connected network.
        """

        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(embedding_dimension, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, embedding_dimension)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LabelEncoder.

        :param z: Input tensor of shape [batch_size, embedding_dimension], which represents the label-embedding vector.
        :return: Output tensor of the same shape as input z, after passing through the fully connected layers with a skip connection.
        """

        x = self.fc1(z)
        x = self.relu(x)
        x = self.fc2(x)
        out = x + z
        return out
