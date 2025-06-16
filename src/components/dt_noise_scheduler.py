import torch

from src.components.noise_scheduler import NoiseScheduler


class DTNoiseScheduler(NoiseScheduler):
    """
    DTNoiseScheduler is a module that emits a uniform noise schedule.

    """

    def __init__(self, steps: int):
        super().__init__()

        t = torch.arange(1, steps + 1)
        alpha_t = torch.square(torch.cos(t / steps * (torch.pi / 2)))
        alpha_bar = torch.cumprod(alpha_t, dim=0)

        self.register_buffer("alpha_bar_buff", alpha_bar)

    def alpha_bar(self) -> torch.Tensor:
        """
        Computes the alpha_bar value for a given time tensor t.
        :param t: A tensor representing time, with values in the range [0, 1].
        """

        # Index into the pre-computed schedule
        return torch.as_tensor(self.alpha_bar_buff)
