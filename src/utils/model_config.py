import json
import os
from abc import ABC

from src.components.resnet_type import ResNetType
from src.models.model_type import NoPropModelType

DEFAULTS = {
    "model_type": NoPropModelType.NO_PROP_DT,
    "backbone_resnet_type": ResNetType.RESNET18,
    "num_classes": 10,
    "embedding_dimension": 128,
    "label_encoder_hidden_dimension": 64,
}


class NoPropBaseModelConfig(ABC):
    """
    Configuration class for NoProp models parameters.

    :ivar model_type: The type of NoProp model (Discrete Time or Continuous Time).
    :ivar backbone_resnet_type: The type of backbone ResNet.
    :ivar num_classes: The number of output classes for classification tasks.
    :ivar embedding_dimension: The dimension of the embedding layer.
    :ivar label_encoder_hidden_dimension: The hidden dimension for the label encoder.
    """

    def __init__(
        self,
        model_type: NoPropModelType = NoPropModelType.NO_PROP_DT,
        backbone_resnet_type: ResNetType = ResNetType.RESNET50,
        num_classes: int = 10,
        embedding_dimension: int = 128,
        label_encoder_hidden_dimension: int = 64,
    ) -> None:
        self.model_type = model_type
        self.backbone_resnet_type = backbone_resnet_type
        self.num_classes = num_classes
        self.embedding_dimension = embedding_dimension
        self.label_encoder_hidden_dimension = label_encoder_hidden_dimension

    def from_file(self, file_path: os.PathLike) -> "NoPropBaseModelConfig":
        """
        Updates the current NoPropBaseModelConfig instance with values from a JSON file.
        Only updates the attributes present in the file, keeping the rest unchanged.

        :param file_path: Path to the JSON configuration file.
        """
        with open(file_path, "r") as file:
            config_data = json.load(file)

        # update only the attributes present in the config file
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

        return self


class NoPropDTConfig(NoPropBaseModelConfig):
    """
    Configuration class for NoProp Discrete Time Model parameters.
    Inherits from NoPropModelConfig.

    :ivar model_type: The type of NoProp model (Discrete Time or Continuous Time).
    :ivar backbone_resnet_type: The type of backbone ResNet.
    :ivar num_classes: The number of output classes for classification tasks.
    :ivar embedding_dimension: The dimension of the embedding layer.
    :ivar label_encoder_hidden_dimension: The hidden dimension for the label encoder.
    :ivar number_of_timesteps: The number of timesteps for the discrete time model.
    """

    def __init__(
        self,
        backbone_resnet_type: ResNetType = ResNetType.RESNET50,
        num_classes: int = 10,
        embedding_dimension: int = 128,
        label_encoder_hidden_dimension: int = 64,
        number_of_timesteps: int = 1000,
    ) -> None:
        super().__init__(
            model_type=NoPropModelType.NO_PROP_DT,
            backbone_resnet_type=backbone_resnet_type,
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            label_encoder_hidden_dimension=label_encoder_hidden_dimension,
        )
        self.number_of_timesteps = number_of_timesteps

    def from_file(self, file_path: os.PathLike) -> "NoPropDTConfig":
        """
        Updates the current NoPropDTConfig instance with values from a JSON file.
        Only updates the attributes present in the file, keeping the rest unchanged.

        :param file_path: Path to the JSON configuration file.
        """
        super().from_file(file_path)
        return self


class NoPropCTConfig(NoPropBaseModelConfig):
    """
    Configuration class for NoProp Continuous Time Model parameters.
    Inherits from NoPropModelConfig.

    :ivar model_type: The type of NoProp model (Discrete Time or Continuous Time).
    :ivar backbone_resnet_type: The type of backbone ResNet.
    :ivar num_classes: The number of output classes for classification tasks.
    :ivar embedding_dimension: The dimension of the embedding layer.
    :ivar label_encoder_hidden_dimension: The hidden dimension for the label encoder.
    :ivar time_embedding_dimension: The dimension of the time embedding layer.
    :ivar noise_scheduler_hidden_dimension: The hidden dimension for the noise scheduler.
    """

    def __init__(
        self,
        backbone_resnet_type: ResNetType = ResNetType.RESNET50,
        num_classes: int = 10,
        embedding_dimension: int = 128,
        label_encoder_hidden_dimension: int = 64,
        noise_scheduler_hidden_dimension: int = 64,
        time_embedding_dimension: int = 128,
    ) -> None:
        super().__init__(
            model_type=NoPropModelType.NO_PROP_CT,
            backbone_resnet_type=backbone_resnet_type,
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            label_encoder_hidden_dimension=label_encoder_hidden_dimension,
        )
        self.noise_scheduler_hidden_dimension = noise_scheduler_hidden_dimension
        self.time_embedding_dimension = time_embedding_dimension

    def from_file(self, file_path: os.PathLike) -> "NoPropCTConfig":
        """
        Updates the current NoPropCTConfig instance with values from a JSON file.
        Only updates the attributes present in the file, keeping the rest unchanged.

        :param file_path: Path to the JSON configuration file.
        """
        super().from_file(file_path)
        return self
