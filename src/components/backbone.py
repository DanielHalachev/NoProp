import torch
import torch.nn as nn
from torchvision import models  # type: ignore

from src.components.resnet_type import ResNetType, ResNetWeights


class Backbone(nn.Module):
    """
    CNN Backbone serving as a feature extractor and basis for the denoise block.
    """

    def __init__(
        self,
        backbone_type: ResNetType,
        output_feature_dimension: int,
        use_resnet: bool = False,
        pretrained: bool = False,
    ):
        """
        Initialize the CNN backbone.

        :param backbone_type: Type of the backbone to use, e.g., ResNet18, ResNet34, etc
        :param output_feature_dimension: Output feature dimension.
        :param use_resnet: Whether to use a ResNet backbone or a custom CNN, as in the paper.
        :param pretrained: Whether to use pre-trained weights, if using a ResNet.
        """
        super(Backbone, self).__init__()
        self.use_resnet = use_resnet
        self.output_feature_dimension = output_feature_dimension
        if use_resnet:
            supported_backbones = [e.value for e in ResNetType]
            if backbone_type not in supported_backbones:
                raise ValueError(
                    f"Unsupported backbone_type: {backbone_type.value}. Choose from {supported_backbones}"
                )

            if output_feature_dimension <= 0:
                raise ValueError(
                    f"output_feature_dimension must be a positive number, got {output_feature_dimension}"
                )

            weights = (
                getattr(models, ResNetWeights.weights[backbone_type]).DEFAULT
                if pretrained
                else None
            )

            cnn = getattr(models, backbone_type.value)(weights=weights)

            # extract all layers except the final fully connected layer
            self.features = nn.Sequential(*list(cnn.children())[:-1])
            feat_dim = cnn.fc.in_features  # Dimension of features before the fc layer
            # add a projection layer to match the desired output feature dimension
            self.proj = nn.Linear(feat_dim, output_feature_dimension)

        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(0.2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, output_feature_dimension),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet backbone.

        :param x: Input tensor of shape [B, C, H, W],
            where B is batch size,
            C is the number of channels,
            H is the input height,
            W is the input width.
        :return (Tensor): Output tensor of shape [B, output_feature_dimension].
        """
        if self.use_resnet:
            batch_size = x.size(0)
            # extract features and flatten
            x = self.features(x).view(batch_size, -1)  # [B, C, 1, 1] -> [B, C]
            # project to the desired output feature dimension
            return self.proj(x)  # [batch_size, output_feature_dimension]
        else:
            return self.features(x)
