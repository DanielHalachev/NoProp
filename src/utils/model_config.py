from src.models.model_type import NoPropModelType


class NoPropModelConfig:
    """
    Configuration class for NoProp models parameters.
    """

    def __init__(self, type: NoPropModelType = NoPropModelType.NO_PROP_DT) -> None:
        self.type = type
