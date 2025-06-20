import torch
import torch.nn as nn

from src.embeddings.sinusoidal_embedding import sinusoidal_embedding


class TimeEncoder(nn.Module):
    """
    Encodes a timestamp t of shape [batch_size ,1] into an embedding of shape [batch_size, output_embedding_dim].
    """

    def __init__(self, time_encoder_hidden_dimension: int, embedding_dim: int):
        """
        Initializes the TimeEncoder with the specified time embedding dimension and output embedding dimension.

        :param time_emb_dim: Dimension of the time embedding.
        :param output_embedding_dim: Dimension of the output embedding.
        """

        super().__init__()
        self.hidden_dimension = time_encoder_hidden_dimension
        self.fc = nn.Sequential(
            nn.Linear(time_encoder_hidden_dimension, embedding_dim), nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, timestamp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TimeEncoder.
        Takes a timestamp tensor, creates a sinusoidal time embedding, and passes it through a fully connected layer.
        The output is an encoded time embedding in the specified output dimension.

        [batch_size,1] -> [batch_size, time_emb_dim] -> [batch_size, output_embedding_dim]

        :param timestamp: Input tensor of shape [batch_size, 1], representing the timestamp.
        :return: Output tensor of shape [batch_size, embedding_dim], which is the encoded time embedding.
        """

        # [batch_size,1] -> [batch_size, time_emb_dim]
        time_embedding = sinusoidal_embedding(timestamp, self.hidden_dimension)
        # [batch_size,time_emb_dim] -> [batch_size, embedding_dim]
        return self.relu(self.fc(time_embedding))
