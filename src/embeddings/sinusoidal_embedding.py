import math

import torch


def sinusoidal_embedding(tensor: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Computes sinusoidal embeddings for a given tensor.

    :param tensor: Input tensor of shape (batch_size, sequence_length).
    :param dimension: The dimension of the sinusoidal embedding.
    :return: Sinusoidal embeddings of shape (batch_size, sequence_length, dimension).
    """

    half = dimension // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=tensor.device, dtype=tensor.dtype)
        / (half - 1)
    )
    args = tensor * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)
