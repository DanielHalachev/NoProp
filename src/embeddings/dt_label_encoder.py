import torch
import torch.nn as nn

from src.embeddings.label_encoder import LabelEncoder


class DTLabelEncoder(LabelEncoder):
    """
    Encodes a label-embedding vector z_t of shape [batch_size, embedding_dimension] via a net with skip connection.
    """

    def __init__(self, embedding_dimension: int, label_encoder_hidden_dimension: int):
        """
        Initializes the DTLabelEncoder with a specified embedding output dimension and hidden dimension.

        :param embedding_dimension: Dimension of the embedding vector that will be output.
        :param hidden_dim: Dimension of the hidden layers in the fully connected network.
        """

        super().__init__()
        self.hidden_dimension = label_encoder_hidden_dimension
        self.output_dimension = embedding_dimension
        self.seg1 = nn.Sequential(
            nn.Linear(embedding_dimension, label_encoder_hidden_dimension),
            nn.BatchNorm1d(label_encoder_hidden_dimension),
            nn.ReLU(),
        )
        self.seg2 = nn.Sequential(
            nn.Linear(label_encoder_hidden_dimension, label_encoder_hidden_dimension),
            nn.BatchNorm1d(label_encoder_hidden_dimension),
            nn.ReLU(),
            nn.Linear(label_encoder_hidden_dimension, label_encoder_hidden_dimension),
            nn.BatchNorm1d(label_encoder_hidden_dimension),
        )

        self.project = nn.Linear(label_encoder_hidden_dimension, embedding_dimension)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LabelEncoder.

        :param z: Input tensor of shape [batch_size, embedding_dimension], which represents the label-embedding vector.
        :return: Output tensor of the same shape as input z, after passing through the fully connected layers with a skip connection.
        """

        input = self.seg1(z)
        x = input
        y = self.seg2(input)
        output = x + y
        if self.hidden_dimension != self.output_dimension:
            # If the hidden dimension is not equal to the embedding dimension, we need to project it back
            output = self.project(output)
        return output
