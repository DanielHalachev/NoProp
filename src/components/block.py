from abc import ABC

import torch.nn as nn


class DenoiseBlock(ABC, nn.Module):
    """
    Abstract base class for denoise blocks in the model.
    This class serves as a template for implementing specific denoise blocks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
