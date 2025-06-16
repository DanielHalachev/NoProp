from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NoiseScheduler(ABC, nn.Module):
    """
    Abstract base class for noise schedulers in diffusion models.
    This class defines the interface for noise schedulers, which are responsible for
    computing the alpha_bar values used in the diffusion process.
    It is expected that subclasses will implement the alpha_bar method to compute
    the alpha_bar values based on a given time tensor.
    """

    @abstractmethod
    def alpha_bar(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes the alpha_bar value.

        :return: A tensor representing the alpha_bar values,
                which are used in diffusion models to control the noise schedule.
        """
