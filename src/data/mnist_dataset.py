from pathlib import Path

import torchvision  # type:ignore
from torch.utils.data import Dataset


def repeat_channels(x):
    """
    Converts a 1-channel grayscale image to 3 channels.
    """
    return x.repeat(3, 1, 1)


class MNISTDatasetManager:
    @classmethod
    def get_datasets(cls, data_root: Path) -> tuple[Dataset, Dataset, Dataset, int]:
        """
        Retrieves the MNIST datasets for training, validation, and testing.

        The paper does not use validation loss for training optimization,
        so the validation and test sets may be the same.

        :return: A tuple containing three datasets:
            train_dataset,
            validation_dataset,
            test_dataset.
        """

        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # MNIST is 1-channel grayscale, convert to 3 channels
                torchvision.transforms.Lambda(repeat_channels),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # Normalize MNIST data
            ]
        )
        validation_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                # MNIST is 1-channel grayscale, convert to 3 channels
                torchvision.transforms.Lambda(repeat_channels),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # Normalize MNIST data
            ]
        )

        num_classes = 10  # MNIST has 10 classes (digits 0-9)

        train_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform,
        )
        validation_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=validation_transform,
        )
        test_dataset = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=validation_transform,
        )

        return train_dataset, validation_dataset, test_dataset, num_classes
