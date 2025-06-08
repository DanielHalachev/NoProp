from enum import Enum


class ResNetType(str, Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"


class ResNetWeights:
    weights = {
        ResNetType.RESNET18: "ResNet18_Weights",
        ResNetType.RESNET34: "ResNet34_Weights",
        ResNetType.RESNET50: "ResNet50_Weights",
        ResNetType.RESNET101: "ResNet101_Weights",
        ResNetType.RESNET152: "ResNet152_Weights",
    }
