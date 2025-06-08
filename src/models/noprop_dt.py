import torch

from src.models.base_model import BaseNoPropModel
from src.utils.model_config import NoPropDTConfig


class NoPropDT(BaseNoPropModel):
    """
    NoProp Discrete Time Model.
    """

    def __init__(self, config: NoPropDTConfig, device: torch.device) -> None:
        super().__init__(config, device)

    def forward_denoise(
        self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def train_step(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        eta: float,
    ) -> float:
        raise NotImplementedError

    def validate_step(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        eta: float,
    ) -> tuple[float, int, int, torch.Tensor]:
        raise NotImplementedError

    def test_step(
        self, src: torch.Tensor, trg: torch.Tensor
    ) -> tuple[int, int, torch.Tensor]:
        raise NotImplementedError
