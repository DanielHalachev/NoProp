from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NoiseScheduler(ABC, nn.Module):
    """
    NoiseScheduler is a module that emits a learnable continuous noise schedule.
    It uses a parameterized gamma function to adjust the alpha_bar values,
    which are used in diffusion models to control the noise schedule during training and inference.

    The gamma function is parameterized by two learnable parameters, gamma0 and gamma1,
    and a neural network that maps the time tensor t to a normalized gamma value.
    """

    @abstractmethod
    def alpha_bar(self, *args, **kwargs) -> torch.Tensor:
        """
        Computes the alpha_bar value.

        :param t: A tensor representing time, typically in the range [0, 1].
        :return: A tensor representing the alpha_bar values,
                which are used in diffusion models to control the noise schedule.
        """
