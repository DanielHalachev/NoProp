import torch
import torch.nn as nn

from src.embeddings.label_encoder import LabelEncoder


class CTLabelEncoder(LabelEncoder):
    """
    Encodes a label-embedding vector z_t of shape [batch_size, embedding_dimension] via a small FC net with skip connection.
    """

    def __init__(self, embedding_dimension: int, label_encoder_hidden_dimension: int):
        """
        Initializes the LabelEncoder with a specified embedding output dimension and hidden dimension.

        :param embedding_dimension: Dimension of the embedding vector that will be output.
        :param label_encoder_hidden_dimension: Dimension of the hidden layer in the fully connected network.
        """

        super().__init__()
        self.hidden_dimension = label_encoder_hidden_dimension
        self.output_dimension = embedding_dimension
        self.fc1 = nn.Linear(embedding_dimension, self.hidden_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dimension, embedding_dimension)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LabelEncoder.

        :param z: Input tensor of shape [batch_size, embedding_dimension], which represents the label-embedding vector.
        :return: Output tensor of the shape [batch_size, embedding_dimension].
        """

        x = self.fc1(z)
        x = self.relu(x)
        if self.hidden_dimension != self.output_dimension:
            # If the hidden dimension is not equal to the embedding dimension, we need to project it back
            x = self.fc2(x)
        return x
