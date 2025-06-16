import torch
import torch.nn as nn

from components.backbone import Backbone
from components.block import DenoiseBlock
from components.ct_concatenator import CTConcatenator
from embeddings.ct_label_encoder import CTLabelEncoder
from embeddings.time_encoder import TimeEncoder
from utils.model_config import NoPropCTConfig


class CTDenoiseBlock(DenoiseBlock):
    """
    Denoise block for NoPropCT model, which processes images, labels, and time embeddings.
    This block consists of a backbone for feature extraction, a label encoder for class labels,
    and a time encoder for temporal information, followed by a concatenator to combine these features.
    """

    def __init__(self, config: NoPropCTConfig) -> None:
        super().__init__()
        self.backbone = Backbone(
            config.backbone_resnet_type, config.embedding_dimension
        )
        self.after_backbone = nn.ReLU()

        self.label_encoder = CTLabelEncoder(
            config.embedding_dimension, config.label_encoder_hidden_dimension
        )

        self.concatenator = CTConcatenator(
            config.embedding_dimension, config.num_classes
        )

        self.time_encoder = TimeEncoder(
            config.time_encoder_hidden_dimension, config.embedding_dimension
        )

    def forward(
        self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the CTDenoiseBlock - concatenates x, z_t and t
        and passes them through several linear layer.

        :param x: Input tensor representing the image features.
        :param z_t: Input tensor representing the label-embedding vector.
        :param t: Input tensor representing the time embedding.
        :return: Output tensor of shape [batch_size, num_classes].
        """
        fx = self.backbone(x)
        fx = self.after_backbone(fx)
        fz = self.label_encoder(z_t)
        ft = self.time_encoder(t)
        return self.concatenator(fx, fz, ft)
