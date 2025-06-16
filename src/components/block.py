from abc import ABC

import torch.nn as nn


class DenoiseBlock(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
