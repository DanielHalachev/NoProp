from pathlib import Path

import medmnist
import torchvision  # type:ignore
from medmnist import INFO
from torch.utils.data import Dataset


def repeat_channels(x):
    """
    Converts a 1-channel grayscale image to 3 channels.
    """
    return x.repeat(3, 1, 1)

def identity(x):
    """
    Returns the input tensor unchanged.
    """
    return x


class MedMNISTDatasetManager:
    @classmethod
    def get_datasets(
        cls, data_root: Path, dataset_name: str = "bloodmnist"
    ) -> tuple[Dataset, Dataset, Dataset, int]:
        """
        Retrieves the MedMNIST datasets for training, validation, and testing at 128x128 resolution.

        :param data_root: Path to the directory where MedMNIST data is stored.
        :param dataset_name: Name of the MedMNIST dataset (e.g., 'pathmnist', 'breastmnist').
        :return: A tuple containing three datasets (train_dataset, validation_dataset, test_dataset)
                 and the number of classes.
        """
        dataset_info = INFO[dataset_name]
        num_channels = dataset_info["n_channels"]
        num_classes = len(dataset_info["label"])

        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(repeat_channels) if num_channels == 1 else torchvision.transforms.Lambda(identity),
                torchvision.transforms.Normalize(
                    (0.5920, 0.3192, 0.3927), # Mean for BLOODMNIST
                    (0.4515, 0.5172, 0.1914)  # Std for BLOODMNIST
                ),
            ]
        )
        validation_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                 torchvision.transforms.Lambda(repeat_channels) if num_channels == 1 else torchvision.transforms.Lambda(identity),
                torchvision.transforms.Normalize(
                    (0.5920, 0.3192, 0.3927), # Mean for BLOODMNIST
                    (0.4515, 0.5172, 0.1914)  # Std for BLOODMNIST
                ),
                torchvision.transforms.Normalize(
                    (0.5,) * num_channels, (0.5,) * num_channels
                ),
            ]
        )

        size = 128

        dataset_class = getattr(medmnist, dataset_info["python_class"])

        train_dataset = dataset_class(
            split="train",
            root=data_root,
            download=True,
            transform=train_transform,
            size=size,
            mmap_mode="r",
        )
        train_dataset.labels = train_dataset.labels.squeeze()  

        validation_dataset = dataset_class(
            split="val",
            root=data_root,
            download=True,
            transform=validation_transform,
            size=size,
            mmap_mode="r",
        )
        validation_dataset.labels = validation_dataset.labels.squeeze()
        
        test_dataset = dataset_class(
            split="test",
            root=data_root,
            download=True,
            transform=validation_transform,
            size=size,
            mmap_mode="r",
        )
        test_dataset.labels = test_dataset.labels.squeeze()  

        return train_dataset, validation_dataset, test_dataset, num_classes
