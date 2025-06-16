import torch
import torch.nn as nn

from components.backbone import Backbone
from components.block import DenoiseBlock
from components.dt_concatenator import DTConcatenator
from embeddings.dt_label_encoder import DTLabelEncoder
from utils.model_config import NoPropDTConfig


class DTDenoiseBlock(DenoiseBlock):
    """
    Denoise block for NoPropDT model, which processes images and labels.
    This block consists of a backbone for feature extraction, a label encoder for class labels,
    and a concatenator to combine these features.
    """

    def __init__(self, config: NoPropDTConfig) -> None:
        """Initializes the DTDenoiseBlock with a backbone, label encoder, and concatenator.
        :param config: Model configuration for NoPropDT.
        """
        super().__init__()
        self.backbone = Backbone(
            config.backbone_resnet_type, config.embedding_dimension
        )
        self.after_backbone = nn.BatchNorm1d(config.embedding_dimension)

        self.label_encoder = DTLabelEncoder(
            config.embedding_dimension, config.label_encoder_hidden_dimension
        )

        self.concatenator = DTConcatenator(
            config.embedding_dimension, config.num_classes
        )

    def forward(self, x: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DTDenoiseBlock - concatenates x and z_t
        and passes them through several linear layers.

        :param x: Input tensor representing the image features.
        :param z_t: Input tensor representing the label-embedding vector.
        :return: Output tensor of shape [batch_size, num_classes].
        """
        fx = self.backbone(x)
        fx = self.after_backbone(fx)
        fz = self.label_encoder(z_t)
        return self.concatenator(fx, fz)
