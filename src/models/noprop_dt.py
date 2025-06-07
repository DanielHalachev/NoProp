import torch

from src.models.base_model import NoPropModel


class NoPropDT(NoPropModel):
    """
    NoProp Discrete Time Model.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__(device)
