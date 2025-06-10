from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Concatenator(ABC, nn.Module):
    """
    Concatenator is a neural network module that combines features from different sources
    into a single output of shape [batch_size, num_classes].
    fx is the image tensor,
    fz are the features from the previous noise block,
    ft is the time embedding (only for NoPropCT).
    """

    @abstractmethod
    def forward(
        self, fx: torch.Tensor, fz: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the FuseHead module.
        :param fx: Tensor representing the image features.
        :param fz: Tensor representing the previous noise block features.
        :param ft: Tensor representing the time embedding features (only for NoPropCT).
        :return: Output tensor of size [batch_size, num_classes]
        """
