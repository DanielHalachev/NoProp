import torch
import torch.nn as nn

from src.components.noise_scheduler import NoiseScheduler


class CTNoiseScheduler(NoiseScheduler):
    """
    CTNoiseScheduler is a module that emits a learnable continuous noise schedule.
    It uses a parameterized gamma function to adjust the alpha_bar values,
    which are used in diffusion models to control the noise schedule during training and inference.

    The gamma function is parameterized by two learnable parameters, gamma0 and gamma1,
    and a neural network that maps the time tensor t to a normalized gamma value.
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Initializes the CTNoiseScheduler with a hidden dimension for the gamma function.

        :param hidden_dim: The dimension of the hidden layer in the gamma function.
        """

        super().__init__()
        self.gamma_tilde = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.gamma0 = nn.Parameter(torch.tensor(-7.0))
        self.gamma1 = nn.Parameter(torch.tensor(7.0))

    def _gamma_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the normalized gamma value for a given time tensor t.
        The gamma value is normalized between 0 and 1 based on the gamma_tilde function.

        :param t: A tensor representing time, typically in the range [0, 1].
        :return: A tensor of the same shape as t, with values normalized between 0 and 1.
        """

        g0 = self.gamma_tilde(torch.zeros_like(t))
        g1 = self.gamma_tilde(torch.ones_like(t))
        return ((self.gamma_tilde(t) - g0) / (g1 - g0 + 1e-8)).clamp(0, 1)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the alpha_bar value for a given time tensor t.
        The alpha_bar value is computed using the parameterized gamma function,
        which adjusts the noise schedule based on the time tensor.

        :param t: A tensor representing time, typically in the range [0, 1].
        :return: A tensor of the same shape as t, with values representing the alpha_bar
        values, which are used in diffusion models to control the noise schedule.
        """

        gamma_t = self.gamma0 + (self.gamma1 - self.gamma0) * (1 - self._gamma_bar(t))
        return torch.sigmoid(-gamma_t / 2).clamp(0.01, 0.99)
