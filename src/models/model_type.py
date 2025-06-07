import enum


class NoPropModelType(str, enum.Enum):
    """
    Enum identifier for different model types in NoProp project.
    """

    NO_PROP_CT = "CT"
    NO_PROP_DT = "DT"
    NO_PROP_FM = "FM"
